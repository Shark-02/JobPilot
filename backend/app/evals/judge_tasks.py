"""Direct judgments scoped by actual source permissions and observation time.

Each semantic check gets its own request; code owns text restoration, counts,
exact equality, Gold thresholds and the final case verdict.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
from typing import Any

from app.evals.judge import JudgeSchemaError, answer_spans, strict_json_loads, validate_response
from app.evals.judge_scheduler import JudgeTask
from pydantic import create_model

from app.evals.schema import (
    StrictModel, JudgeCheckResult, ProjectSelectionExtraction, ClassificationExtraction,
    BulletsExtraction, QuestionsExtraction, TopicSetExtraction, MemoryTopicsExtraction,
)

PROMPT_VERSION = "direct_semantics_1"
ANSWER_GROUNDING_SHARD_SIZE = 4


class SemanticSubtaskResult(StrictModel):
    items: list[JudgeCheckResult]


EXTRACTION_MODELS = {
    "project_selection": ProjectSelectionExtraction,
    "classification": ClassificationExtraction,
    "bullets": BulletsExtraction,
    "questions": QuestionsExtraction,
    "topic_set": TopicSetExtraction,
    "memory_topics": MemoryTopicsExtraction,
}


def authoritative_evidence(payload: dict) -> list[dict]:
    """Preserve every actually-visible raw excerpt, including distinct overlaps."""
    return [deepcopy(item) for item in payload.get("evidence", [])
            if item.get("visible_to_agent") is True and item.get("authoritative_for_grounding") is True]


def _historical_memory(snapshot: dict, *, mutation: bool) -> dict:
    required = {"id", "kind", "content", "attributes", "source_refs"}
    if mutation:
        required.add("status")
    if (not isinstance(snapshot, dict) or not required.issubset(snapshot) or not isinstance(snapshot["source_refs"], list)
            or any(not isinstance(ref, str) for ref in snapshot["source_refs"])
            or any(not isinstance(snapshot[key], str) for key in ("id", "kind", "content"))
            or not isinstance(snapshot["attributes"], dict)
            or (mutation and not isinstance(snapshot["status"], str))):
        raise JudgeSchemaError("Historical Memory snapshot is incomplete")
    # A recall has no status field. Never fill it (or any other field) from the
    # final snapshot: it may describe a later version of the same entity.
    return {key: deepcopy(value) for key, value in snapshot.items() if key != "bound_sources"}


def _bind_historical_sources(memories: list[dict], payload: dict, ledger: list[dict]) -> None:
    sources: dict[str, dict[str, dict]] = {}
    ids = {m["id"] for m in memories}

    def collect(memory_id, rows):
        for raw in rows:
            if not isinstance(raw.get("source_ref"), str) or not isinstance(raw.get("content"), str):
                raise JudgeSchemaError("Historical Memory source observation is incomplete")
            identity = "bound_source_" + sha256((raw["source_ref"] + raw["content"]).encode()).hexdigest()[:20]
            sources.setdefault(memory_id, {})[raw["source_ref"]] = {
                **deepcopy(raw), "id": raw.get("id", identity),
                "visible_to_agent": False, "authoritative_for_grounding": False,
            }

    # Seed annotations precede memory_before; the baseline may have resolved a
    # newer version of a bound document. Later committed annotations follow it.
    for event in ledger:
        if event.get("phase") == "seed":
            collect(event.get("result_memory_id"), event.get("sources", []))
    for memory in payload.get("memory_before", []):
        if memory["id"] in ids:
            collect(memory["id"], memory.get("bound_sources", []))
    for event in ledger:
        if event.get("phase") != "seed":
            collect(event.get("result_memory_id"), event.get("sources", []))
    for memory in memories:
        available = sources.get(memory["id"], {})
        if any(ref not in available for ref in memory["source_refs"]):
            raise JudgeSchemaError("Historical Memory bound source content is unavailable at the observed time")
        memory["bound_sources"] = [deepcopy(available[ref]) for ref in dict.fromkeys(memory["source_refs"])]


def _recall_prior_turns(payload: dict, step: str, turn: str, searches: list[dict]) -> set[str]:
    # Memory cases carry authored steps as an ordered list. Dictionary insertion
    # order in step_turn_ids is not a temporal contract (replays may reorder it).
    earlier = set()
    steps = payload.get("task")
    if isinstance(steps, list):
        for item in steps:
            if item.get("id") == step:
                break
            if item.get("id") in payload.get("step_turn_ids", {}):
                earlier.add(payload["step_turn_ids"][item["id"]])
        else:
            earlier.clear()
    # Actual tool observations are also appended in execution order. Only turns
    # confined to the prefix before the first recall establish a prior turn.
    tools = payload.get("tool_events", [])
    if searches:
        boundary = next(i for i, event in enumerate(tools) if event is searches[0])
        earlier.update({e.get("turn_id") for e in tools[:boundary]} -
                       {e.get("turn_id") for e in tools[boundary:]})
    return earlier - {turn, None}


def _bound_payload(plan: Any, payload: dict, check: Any) -> dict:
    step = check.target_step_id
    turn = payload.get("step_turn_ids", {}).get(step)
    if not step or not turn:
        raise JudgeSchemaError("Memory semantic task requires its observed target turn")
    events = []
    searches = []
    all_events = payload.get("memory_events", [])
    if check.observation_scope == "target_mutation":
        policy = next((c.expected for c in plan.checks if c.evaluator_kind == "memory_mutation"
                       and c.expected.get("target_step_id") == step), None)
        if not policy:
            raise JudgeSchemaError("Memory task has no matching mutation contract")
        target_events = [e for e in all_events
                  if e.get("turn_id") == turn and e.get("phase", "target") == "target" and e.get("success")]
        events = target_events
        ids = {e.get("result_memory_id") for e in events}
        if None in ids:
            raise JudgeSchemaError("Historical Memory mutation has no result ID")
        # The target turn is a whole-step boundary. Include its final committed
        # state, even if multiple mutations touched the same Memory in that turn.
        boundary = max((i for i, e in enumerate(all_events) if any(e is target for target in target_events)), default=-1)
        ledger = [e for e in all_events[:boundary + 1] if e.get("success") and e.get("result_memory_id") in ids]
        by_id = {}
        for event in target_events:
            if not isinstance(event.get("after"), list):
                raise JudgeSchemaError("Historical Memory mutation after snapshot is unavailable")
            for snapshot in event["after"]:
                if snapshot.get("id") in ids:
                    by_id[snapshot["id"]] = _historical_memory(snapshot, mutation=True)
        if ids != set(by_id):
            raise JudgeSchemaError("Historical Memory target snapshot is unavailable")
        memories = list(by_id.values())
    elif check.observation_scope == "memory_recall":
        searches = [e for e in payload.get("tool_events", [])
                    if e.get("turn_id") == turn and e.get("tool_name") == "search_memory"
                    and e.get("execution_kind") == "real" and e.get("result", {}).get("ok")]
        memories = []
        for event in searches:
            recalled = (event["result"].get("agent_payload") or {}).get("memories")
            if not isinstance(recalled, list):
                raise JudgeSchemaError("Historical Memory recall snapshot is unavailable")
            for snapshot in recalled:
                memory = _historical_memory(snapshot, mutation=False)
                # Keep distinct recalled versions of the same ID. The semantic
                # scope is the actual recalls of the target turn, not its end.
                if memory not in memories:
                    memories.append(memory)
        ids = {m["id"] for m in memories}
        earlier_turns = _recall_prior_turns(payload, step, turn, searches)
        ledger = [e for e in all_events if e.get("success") and e.get("result_memory_id") in ids
                  and (e.get("phase") == "seed" or e.get("turn_id") in earlier_turns)]
    else:
        raise JudgeSchemaError("Bound-source semantic task has no explicit observation scope")
    _bind_historical_sources(memories, payload, ledger)
    if check.observation_scope == "memory_recall":
        for memory in memories:
            observed_sources = {s["source_ref"]: s["content"] for s in memory["bound_sources"]}
            for event in all_events:
                if not (event.get("success") and event.get("turn_id") == turn
                        and event.get("result_memory_id") == memory["id"]):
                    continue
                # A mutable raw document may have been reread in the same turn.
                # These two ledgers have no shared call ordering; an older raw
                # version cannot stand in for the version bound at recall time.
                if any(s.get("source_ref") in observed_sources
                       and s.get("content") != observed_sources[s["source_ref"]]
                       for s in event.get("sources", [])):
                    raise JudgeSchemaError("Historical Memory source order is ambiguous within the recall turn")
    answer = payload.get("final_answers_by_step", {}).get(step, "")
    expected_spans = {s["id"]: s["text"] for s in answer_spans(answer, prefix=f"step_{step}")}
    if answer == payload.get("final_answer"):
        expected_spans.update({s["id"]: s["text"] for s in answer_spans(answer)})
    spans = [s for s in payload.get("answer_spans", []) if expected_spans.get(s["id"]) == s["text"]]
    task = payload.get("task")
    if isinstance(task, list):
        task = [deepcopy(item) for item in task if item.get("id") == step]
    return {
        "task": task, "source_policy": payload.get("source_policy"),
        "final_answer": answer, "answer_spans": spans, "final_answers_by_step": {step: answer},
        "evidence": [], "memories": memories,
        "memory_before": [m for m in payload.get("memory_before", []) if m["id"] in ids],
        "memory_events": events if events else ledger, "tool_events": searches,
        "fixture_source_refs": payload.get("fixture_source_refs", {}),
        "step_turn_ids": {step: turn}, "step_source_refs": {step: payload.get("step_source_refs", {}).get(step)},
    }


def observed_answer_spans(payload: dict) -> list[dict[str, str]]:
    """Remove only a proven final/last-step alias; distinct steps stay distinct."""
    spans = deepcopy(payload.get("answer_spans", []))
    steps = payload.get("final_answers_by_step", {})
    if not steps:
        return spans
    if len(steps) == 1:
        step, text = next(iter(steps.items()))
    else:
        task = payload.get("task")
        if not isinstance(task, list) or not task or not isinstance(task[-1], dict):
            return spans
        step = task[-1].get("id")
        if step not in steps:
            return spans
        text = steps[step]
    if text != payload.get("final_answer"):
        return spans
    final = {s["id"]: s["text"] for s in answer_spans(text)}
    historical = {s["id"]: s["text"] for s in answer_spans(text, prefix=f"step_{step}")}
    actual = {s["id"]: s["text"] for s in spans}
    if any(actual.get(key) != value for key, value in historical.items()):
        return spans
    return [s for s in spans if final.get(s["id"]) != s["text"]]


def _answer_grounding_shards(data: dict) -> list[dict]:
    """Split the answer view without widening its factual support scope."""
    spans = observed_answer_spans(data)
    chunks = [spans[index:index + ANSWER_GROUNDING_SHARD_SIZE]
              for index in range(0, len(spans), ANSWER_GROUNDING_SHARD_SIZE)] or [[]]
    shards = []
    for chunk in chunks:
        shard = deepcopy(data)
        # The view and validator must both be restricted to this shard. Remove
        # aliases before _semantic_view runs so it cannot restore a full answer.
        shard["answer_spans"] = deepcopy(chunk)
        shard.pop("final_answer", None)
        shard.pop("final_answers_by_step", None)
        shards.append(shard)
    return shards


def _semantic_view(data: dict, *, fixture_sources: bool = False) -> tuple[dict, dict[str, str]]:
    """Present original answer spans and only the already-resolved factual scope."""
    view = {key: deepcopy(data[key]) for key in (
        "task", "source_policy", "fixture_source_refs", "step_turn_ids",
        "step_source_refs", "evidence", "memories") if key in data}
    view["answer_spans"] = observed_answer_spans(data)
    evidence_aliases: dict[str, str] = {}
    evidence_ids: dict[str, str] = {}
    bound_ids: dict[str, str] = {}

    def alias(item: dict, *, prefix: str, ids: dict[str, str]) -> None:
        original = item["id"]
        if original not in ids:
            ids[original] = f"{prefix}{len(ids) + 1}"
            evidence_aliases[ids[original]] = original
        item["id"] = ids[original]

    # Eligibility and provenance have already selected this whole raw source
    # set. Keep their full records in data for validation; they are not another
    # reasoning task for the model. A fixture dependency still needs its source
    # identity so the model can associate the required fixture with an excerpt.
    evidence_keys = {"id", "content", "source_ref"} if fixture_sources else {"id", "content"}
    if (data.get("source_policy") or {}).get("fact_time_policy") == "preserve_event_time":
        # Conflicting raw observations must retain their actual observation
        # time. Never infer a missing time from a later snapshot or source row.
        evidence_keys.update({"turn_id", "phase"})
    view["evidence"] = [{key: value for key, value in item.items() if key in evidence_keys}
                        for item in view.get("evidence", [])]
    for item in view["evidence"]:
        alias(item, prefix="e", ids=evidence_ids)
    for memory in view.get("memories", []):
        if "bound_sources" not in memory:
            continue
        memory["bound_sources"] = [
            {key: value for key, value in source.items() if key in {"id", "content", "source_ref"}}
            for source in memory.get("bound_sources", [])
        ]
        for source in memory.get("bound_sources", []):
            alias(source, prefix="b", ids=bound_ids)
    return view, evidence_aliases


class JudgeTaskPlanner:
    def plan(self, plan: Any, payload: dict) -> list[JudgeTask]:
        base = {key: deepcopy(payload[key]) for key in (
            "task", "source_policy", "final_answer", "answer_spans", "final_answers_by_step",
            "fixture_source_refs", "step_turn_ids", "step_source_refs") if key in payload}
        groups: list[dict] = []
        # Retain the original full answer observation for structural
        # extractions.  A grounding shard must never make an existing
        # extraction validate only its first four spans.
        answer_grounding_data: dict | None = None
        answer_observation_group: dict | None = None
        for check in plan.semantic_checks:
            if check.support_mode == "bound_memory_source":
                data = _bound_payload(plan, payload, check)
            else:
                data = {**deepcopy(base), "evidence": authoritative_evidence(payload)
                        if check.support_mode == "authoritative_visible" else [], "memories": []}
            if check.id == "answer_grounding":
                answer_grounding_data = deepcopy(data)
                for index, shard in enumerate(_answer_grounding_shards(data), 1):
                    groups.append({"checks": [check], "specs": [], "data": shard,
                                   "task_id": f"semantic:answer_grounding:shard:{index:03d}",
                                   "answer_grounding_shard": True})
            else:
                groups.append({"checks": [check], "specs": [], "data": data})
        for spec in plan.extractions:
            if spec.kind not in EXTRACTION_MODELS:
                raise JudgeSchemaError(f"Unregistered observation kind: {spec.kind}")
            if spec.kind == "memory_topics":
                memories = [{key: deepcopy(value) for key, value in m.items() if key != "bound_sources"}
                            for m in payload.get("memories", [])]
                group = next((group for group in groups if [
                    {key: value for key, value in m.items() if key != "bound_sources"}
                    for m in group["data"].get("memories", [])] == memories), None)
                if group is None:
                    data = {**deepcopy(base), "evidence": [], "memories": memories}
                    group = {"checks": [], "specs": [], "data": data}
                    groups.append(group)
            else:
                # Recognition of answer structure can share the answer semantic
                # request; it never changes which raw facts support its checks.
                group = next((group for group in groups if any(
                    check.observation_scope == "answer" for check in group["checks"])), None)
                if group is not None and group.get("answer_grounding_shard"):
                    # Preserve the prior complete answer view in exactly one
                    # observation task.  Attaching here would silently limit
                    # existing extraction validation to shard 001.
                    if answer_observation_group is None:
                        if answer_grounding_data is None:
                            raise JudgeSchemaError("answer_grounding shard has no full observation")
                        answer_observation_group = {"checks": [], "specs": [],
                                                    "data": deepcopy(answer_grounding_data)}
                        groups.append(answer_observation_group)
                    group = answer_observation_group
                if group is None:
                    data = {**deepcopy(base), "evidence": [], "memories": []}
                    group = {"checks": [], "specs": [], "data": data}
                    groups.append(group)
            group["specs"].append(spec)
        return [self._task(group["checks"], group["specs"], group["data"], task_id=group.get("task_id"))
                for group in groups]

    def _task(self, checks: list[Any], specs: list[Any], data: dict, *, task_id: str | None = None) -> JudgeTask:
        if len(checks) > 1:
            raise JudgeSchemaError("Semantic JudgeTask must contain at most one check")
        child = SimpleNamespace(semantic_checks=checks, extractions=specs)
        view, evidence_aliases = _semantic_view(data, fixture_sources=any(
            check.required_fixture_id for check in checks))
        check_aliases = {f"c{index}": check.id for index, check in enumerate(checks, 1)}
        if specs:
            models = list(dict.fromkeys(EXTRACTION_MODELS[spec.kind] for spec in specs))
            item_type = models[0]
            for model in models[1:]:
                item_type = item_type | model
            model = create_model("DirectSubtaskResult", __base__=SemanticSubtaskResult,
                                 observations=(list[item_type], ...))
        else:
            model = SemanticSubtaskResult

        def validate(raw):
            body = model.model_validate(strict_json_loads(raw)).model_dump(mode="json")
            actual = [item["id"] for item in body["items"]]
            if len(actual) != len(set(actual)) or set(actual) != set(check_aliases):
                raise JudgeSchemaError("Semantic items must contain every requested check alias exactly once")
            by_alias = {item["id"]: item for item in body["items"]}
            restored = []
            for check_alias, check_id in check_aliases.items():
                item = by_alias[check_alias]
                if any(eid not in evidence_aliases for eid in item["evidence_ids"]):
                    raise JudgeSchemaError("Unknown evidence alias")
                restored.append({**item, "id": check_id,
                                 "evidence_ids": [evidence_aliases[eid] for eid in item["evidence_ids"]]})
            response = validate_response(json.dumps({"checks": restored,
                                         "extractions": body.get("observations", [])}), plan=child, payload=data)
            return response.model_dump(mode="json")

        prompt = (
            "Judge each criterion independently; return only JSON using the supplied minimal schema. "
            "Return each short c-ID exactly once. Give PASS, FAIL or UNKNOWN and a short reason; "
            "UNKNOWN means insufficient information to judge, not a missing required answer. "
            "Assess substantive task completion and fidelity to actual observations, including project attribution, "
            "numeric meaning, units, causal relationships and explicit boundaries. Do not infer support merely "
            "because an evidence ID is eligible. Gold is a rubric, never factual evidence. "
            "For authoritative_visible criteria, use only submitted actually-visible raw evidence; "
            "assistant text and derived summaries are not sources. For answer_only criteria, judge the answer. "
            "For technical_rubric criteria, use the supplied technical reference. Bound_memory_source criteria "
            "use only the submitted historical Memories and their bound sources at the target mutation/recall; "
            "select the requested topic from those observations without borrowing a later version, and "
            "a non-absence bound-memory PASS must cite a short ID of its bound source. "
            "Missing required facts fail; conditional claims pass when absent and must be correct when present. "
            "Evidence IDs and answer_span_id are optional except explicit fixture/item bindings: a fixture-bound "
            "PASS must cite its exact source identity; an item-bound PASS identifies that delivered item's span. "
            "When supplied, cite only short e/b evidence aliases and actual answer span IDs. "
            "For requested observations, identify all actual delivered items or classifications, including "
            "additional/conflicting items; do not fill omissions from Gold or calculate counts, thresholds or verdicts. "
            "Select existing span IDs for bullets, main questions and follow-ups; never recopy their text. "
            "Memory topic mappings cover every submitted Memory, including other topics.\n"
        )
        contract_checks = [{**check.model_dump(mode="json", exclude={"id", "gold_path"}), "id": alias}
                           for alias, check in zip(check_aliases, checks, strict=True)]
        contract = {"checks": contract_checks, "response_schema": model.model_json_schema()}
        if specs:
            contract["observations"] = [spec.model_dump(mode="json") for spec in specs]
        resolved_task_id = task_id or (f"semantic:{checks[0].id}" if checks else f"observation:{specs[0].id}")
        kind = "semantic" if checks else "observation"
        return JudgeTask(id=resolved_task_id, kind=kind, stage=kind,
                         messages=self._messages(prompt, contract, view), validate=validate,
                         schema_version=PROMPT_VERSION)

    @staticmethod
    def _messages(prompt: str, contract: dict, data: dict) -> list[dict]:
        return [{"role": "system", "content": prompt + json.dumps(contract, ensure_ascii=False)},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]
