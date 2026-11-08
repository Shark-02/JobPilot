"""Execute compiled checks and aggregate outcomes deterministically."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any, Callable

from app.evals.compiler import ExecutableEvalPlan
from app.evals.evidence import EvidencePack
from app.evals.evaluators import EvalClassification, EvalFailureClassifier, EvalOutcome, case_result


class EvaluationContractError(ValueError):
    pass


@dataclass
class EvaluationContext:
    data: dict[str, Any] = field(default_factory=dict)
    evidence_pack: EvidencePack = field(default_factory=EvidencePack)
    fixture_source_refs: dict[str, str] = field(default_factory=dict)
    step_turn_ids: dict[str, str] = field(default_factory=dict)
    step_source_refs: dict[str, str] = field(default_factory=dict)
    final_answers_by_step: dict[str, str] = field(default_factory=dict)
    memory_before: list[dict[str, Any]] = field(default_factory=list)
    memory_after: list[dict[str, Any]] = field(default_factory=list)
    memory_events: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    fault_events: list[dict[str, Any]] = field(default_factory=list)
    source_refs_valid: bool = True
    invalid_source_refs: list[str] = field(default_factory=list)
    retrieval_result: dict[str, Any] | None = None

    @property
    def final_answer(self) -> str:
        answers = self.data.get("final_answers") or []
        return answers[-1] if answers else ""


@dataclass(frozen=True)
class CheckActual:
    passed: bool
    actual: Any
    evidence_ids: tuple[str, ...] = ()
    reason: str | None = None


def _revision_normal(text: str) -> str:
    # Local revision equality preserves identifiers, punctuation and symbols.
    # Only whitespace differences are formatting, never semantic edits.
    return re.sub(r"\s+", "", text)


def _enum(value: Any) -> Any:
    return getattr(value, "value", value)


def _extraction(expected: dict, extractions: dict) -> dict:
    identifier = expected.get("extraction_id")
    if identifier not in extractions:
        raise EvaluationContractError(f"Missing extraction: {identifier}")
    return extractions[identifier]


def _terminal(expected, ctx, extractions, semantic):
    turns = ctx.data.get("turns") or []
    states = [_enum(turn.status) for turn in turns]
    return CheckActual(bool(states) and all(state == "COMPLETED" for state in states), states)


def _answer(expected, ctx, extractions, semantic):
    return CheckActual(bool(ctx.final_answer.strip()), {"characters": len(ctx.final_answer)})


def _sources(expected, ctx, extractions, semantic):
    return CheckActual(ctx.source_refs_valid, ctx.invalid_source_refs)


def _capture(expected, ctx, extractions, semantic):
    if not ctx.evidence_pack.capture_complete:
        raise EvaluationContractError("Incomplete actual-visible evidence capture: " + ",".join(ctx.evidence_pack.errors))
    return CheckActual(True, {"complete": True, "evidence_count": len(ctx.evidence_pack.items)})


def _project_selection(expected, ctx, extractions, semantic):
    selected = _extraction(expected, extractions)["items"]
    actual = set(selected)
    required = set(expected["required"])
    valid = required <= actual and len(selected) == len(actual)
    if "optional" in expected:
        valid = valid and actual <= required | set(expected["optional"]) and len(actual) <= expected["max_selected"]
    if expected.get("any_of"):
        valid = valid and bool(actual & set(expected["any_of"]))
    return CheckActual(valid, selected)


def _classification(expected, ctx, extractions, semantic):
    items = _extraction(expected, extractions)["items"]
    identities = [item["id"] for item in items]
    actual = {item["id"]: item["label"] for item in items}
    required = {item["id"]: item["label"] for item in expected["items"]}
    return CheckActual(len(identities) == len(set(identities)) and actual == required, actual)


def _bullet_count(expected, ctx, extractions, semantic):
    items = _extraction(expected, extractions)["items"]
    return CheckActual(0 < len(items) <= expected["max"], len(items))


def _local_revision(expected, ctx, extractions, semantic):
    items = _extraction(expected, extractions)["items"]
    draft = expected["draft"]
    actual_ids = [item["id"] for item in items]
    expected_ids = [item["id"] for item in draft]
    original = {item["id"]: item["text"] for item in draft}
    target = expected["target_id"]
    shape_valid = len(items) == len(draft) and actual_ids == expected_ids
    unchanged = {item["id"]: _revision_normal(item["text"]) == _revision_normal(original[item["id"]])
                 for item in items if item["id"] in original and item["id"] != target}
    targets = [item for item in items if item["id"] == target]
    target_changed = len(targets) == 1 and _revision_normal(targets[0]["text"]) != _revision_normal(original[target])
    valid = shape_valid and len(unchanged) == len(draft) - 1 and all(unchanged.values()) and target_changed
    return CheckActual(valid, {"bullet_ids": actual_ids, "target_changed": target_changed, "non_target_unchanged": unchanged},
                       reason=None if valid else "Target must change; all other final bullets, count and order must remain identical")


def _question_limits(expected, ctx, extractions, semantic):
    items = _extraction(expected, extractions)["items"]
    counts = [len(item["followups"]) for item in items]
    valid = len(items) == expected["main_questions"] and all(count <= expected["max_followups_per_question"] for count in counts)
    return CheckActual(valid, {"main_questions": len(items), "followup_counts": counts})


def _topic_coverage(expected, ctx, extractions, semantic):
    extraction = _extraction(expected, extractions)
    topics = ([topic for question in extraction["items"] for topic in question["topic_ids"]]
              if extraction["kind"] == "questions" else extraction["items"])
    required = set(expected["topic_ids"])
    actual = set(topics)
    valid = len(actual & required) >= expected["min"]
    if expected.get("exact_set"):
        valid = valid and actual == required and len(topics) == len(actual)
    return CheckActual(valid, sorted(actual))


def _fixture_visibility(expected, ctx, extractions, semantic):
    """Exact source identity is code-observable; material meaning is semantic."""
    visible = {item.id: item for item in ctx.evidence_pack.items if item.visible_to_agent and item.authoritative_for_grounding}
    required_ref = ctx.fixture_source_refs.get(expected["fixture_id"])
    identities = {item.source_ref for item in visible.values() if item.source_ref == required_ref}
    # The runner maps one exact message identity per fixture and rejects double injection.
    return CheckActual(bool(required_ref) and len(identities) == 1,
                       {"required_ref": required_ref, "visible_refs": sorted(identities)})


def _target_events(expected, ctx):
    turn_id = ctx.step_turn_ids.get(expected["target_step_id"])
    if not turn_id:
        raise EvaluationContractError("Missing target turn identity")
    return [event for event in ctx.memory_events if event["turn_id"] == str(turn_id) and event.get("phase", "target") == "target"]


def _memory_topics(extractions):
    return extractions.get("memory_topics", {}).get("items", [])


def _relevant_events(expected, ctx, extractions):
    events = _target_events(expected, ctx)
    topic_items = _memory_topics(extractions)
    if not topic_items:
        return events
    target_topics = set(expected.get("target_topic_ids") or [])
    ids = {item["memory_id"] for item in topic_items if not target_topics or item["topic_id"] in target_topics}
    return [event for event in events if event.get("target_memory_id") in ids or event.get("result_memory_id") in ids]


def _memory_mutation(expected, ctx, extractions, semantic):
    events = _relevant_events(expected, ctx, extractions) if expected["target_existing"] else _target_events(expected, ctx)
    successful = [event for event in events if event["success"]]
    if not expected["should_write"]:
        return CheckActual(not events, {"attempts": len(events), "successful_mutations": len(successful)})
    valid = bool(successful)
    if expected.get("expected_count") is not None:
        valid = valid and len(successful) == expected["expected_count"]
    details = []
    for event in successful:
        after = [item for item in event["after"] if item["id"] == event.get("result_memory_id")]
        before = [item for item in event["before"] if item["id"] == event.get("target_memory_id")]
        event_valid = event["operation"] in expected["allowed_operations"] and bool(after) and after[0]["kind"] in expected["allowed_kinds"]
        if expected["target_existing"]:
            event_valid = event_valid and bool(before) and before[0]["status"] == "ACTIVE" and event["operation"] in {"MERGE", "SUPERSEDE"}
        if expected["require_new_source"]:
            old_refs = {ref for item in before for ref in item.get("source_refs", [])}
            new_refs = set(event["new_source_refs"])
            event_valid = event_valid and bool(new_refs - old_refs)
        valid = valid and event_valid
        details.append({"operation": event["operation"], "target_memory_id": event.get("target_memory_id"),
                        "result_memory_id": event.get("result_memory_id"), "valid": event_valid})
    return CheckActual(valid, {"target_turn_id": ctx.step_turn_ids[expected["target_step_id"]], "mutations": details})


def _memory_state(expected, ctx, extractions, semantic):
    mapping = _extraction(expected, extractions)["items"]
    after = {item["id"]: item for item in ctx.memory_after}
    mapped_ids = [item["memory_id"] for item in mapping]
    pairs = [(item["memory_id"], item["topic_id"]) for item in mapping]
    if len(pairs) != len(set(pairs)) or set(mapped_ids) != set(after):
        raise EvaluationContractError("Memory topic extraction must cover every observed Memory with unique entity/topic pairs, including other topics")
    events = _target_events(expected, ctx)
    details = []
    valid = True
    for topic in expected["topics"]:
        relevant = [item for item in mapping if item["topic_id"] == topic["topic_id"] and item["memory_id"] in after]
        active = [item for item in relevant if after[item["memory_id"]]["status"] == "ACTIVE"]
        expected_active = [item for item in active if item["state"] == topic["state"]]
        topic_valid = len(expected_active) == topic["active_count"]
        if topic["reject_conflicting_active"]:
            topic_valid = topic_valid and len(active) == len(expected_active)
        if topic["preserve_history"]:
            final_ids = {item["memory_id"] for item in expected_active}
            evolutions = [event for event in events if event["success"] and event.get("result_memory_id") in final_ids
                          and event["operation"] in {"MERGE", "SUPERSEDE"} and event["before"]]
            history_preserved = False
            for event in evolutions:
                old = next((item for item in event["before"] if item["id"] == event["target_memory_id"]), None)
                new = after.get(event.get("result_memory_id"))
                if event["operation"] == "SUPERSEDE":
                    previous = after.get(event["target_memory_id"])
                    history_preserved |= bool(previous and previous["status"] in {"SUPERSEDED", "ARCHIVED"} and old and previous["content"] == old["content"])
                elif old and new:
                    # MERGE retains original message evidence and a committed
                    # before/after event, rather than inventing an old DB row.
                    history_preserved |= set(old.get("source_refs", [])) <= set(new.get("source_refs", [])) and old["content"] != new["content"]
            topic_valid = topic_valid and history_preserved
        valid = valid and topic_valid
        details.append({"topic_id": topic["topic_id"], "active": active, "valid": topic_valid})
    return CheckActual(valid, details)


def _memory_source(expected, ctx, extractions, semantic):
    events = [event for event in _relevant_events(expected, ctx, extractions) if event["success"]]
    required_ref = (ctx.fixture_source_refs.get(expected.get("fixture_id")) if expected["mode"] == "fixture_message"
                    else ctx.step_source_refs.get(expected["target_step_id"]))
    valid = bool(events)
    details = []
    for event in events:
        refs = set(event["new_source_refs"])
        sources = {source["source_ref"] for source in event.get("sources", []) if source.get("content")}
        if expected["mode"] == "bound_section_or_chunk":
            event_valid = any(ref.startswith("document://") for ref in refs) and refs <= sources
        else:
            event_valid = bool(required_ref) and required_ref in refs and required_ref in sources
        valid = valid and event_valid
        details.append({"result_memory_id": event.get("result_memory_id"), "source_refs": sorted(refs), "valid": event_valid})
    return CheckActual(valid, {"required_ref": required_ref, "bindings": details})


def _memory_recall(expected, ctx, extractions, semantic):
    turn_id = ctx.step_turn_ids.get(expected["step_id"])
    if not turn_id:
        raise EvaluationContractError("Missing cross-conversation turn identity")
    topic_items = _memory_topics(extractions)
    active_ids = {item["id"] for item in ctx.memory_after if item["status"] == "ACTIVE"}
    by_topic = {topic: {item["memory_id"] for item in topic_items if item["topic_id"] == topic and item["memory_id"] in active_ids}
                for topic in expected["required_topics"]}
    searches = [event for event in ctx.tool_events if event["turn_id"] == str(turn_id) and event["tool_name"] == "search_memory"
                and event["result"].get("ok") and event["execution_kind"] == "real"]
    recalled_ids = {item["id"] for event in searches for item in (event["result"].get("agent_payload") or {}).get("memories", [])}
    conversations = {event.get("conversation") for event in searches}
    writes = [event for event in ctx.memory_events if event["success"] and event.get("result_memory_id") in active_ids]
    target_conversations = {event.get("conversation") for event in writes}
    cross_conversation = bool(conversations) and any(label not in target_conversations for label in conversations)
    answer = ctx.final_answers_by_step.get(expected["step_id"], "")
    valid = bool(searches) and cross_conversation and bool(answer.strip()) and all(ids & recalled_ids for ids in by_topic.values())
    return CheckActual(valid, {"turn_id": str(turn_id), "recalled_ids": sorted(recalled_ids), "topics": {key: sorted(value) for key, value in by_topic.items()},
                               "cross_conversation": cross_conversation, "final_answer": answer})


def _fault_triggered(expected, ctx, extractions, semantic):
    injected = [event for event in ctx.tool_events if event["execution_kind"] == "injected"]
    if not injected:
        raise EvaluationContractError("Declared fault did not trigger; runtime experiment conditions were not established")
    valid = all(event["tool_name"] in expected["target_tools"] and not event["result"].get("ok")
                and (event["result"].get("error") or {}).get("code") == expected["error_code"]
                and (event["result"].get("error") or {}).get("retryable") == expected["retryable"] for event in injected)
    for tool in expected["target_tools"]:
        actual_calls = [event for event in ctx.tool_events if event["tool_name"] == tool and event["execution_kind"] in {"real", "injected"}]
        if expected.get("trigger") == "every_call":
            valid = valid and all(event["execution_kind"] == "injected" for event in actual_calls)
        elif expected.get("trigger") == "occurrence":
            injected_positions = [index for index, event in enumerate(actual_calls, 1) if event["execution_kind"] == "injected"]
            valid = valid and injected_positions == [expected["occurrence"]]
    return CheckActual(valid, [{"call_id": event["call_id"], "tool_name": event["tool_name"], "result": event["result"]} for event in injected])


def _fault_recovery(expected, ctx, extractions, semantic):
    injected = [event for event in ctx.tool_events if event["execution_kind"] == "injected"]
    pairs = []
    for failed in injected:
        index = ctx.tool_events.index(failed)
        for later in ctx.tool_events[index + 1:]:
            if later["tool_name"] != failed["tool_name"] or later["turn_id"] != failed["turn_id"] or later["execution_kind"] != "real" or not later["result"].get("ok"):
                continue
            payload = later["result"].get("agent_payload") or {}
            if later["tool_name"] == "search_knowledge" and not any(item.get("content", "").strip() for item in payload.get("modules", [])):
                continue
            if later["tool_name"] == "read_source" and not any(item.get("content", "").strip() for item in payload.get("sources", [])):
                continue
            pairs.append({"failed_call_id": failed["call_id"], "successful_call_id": later["call_id"]})
            break
    errors = [event for event in ctx.tool_events if not event["result"].get("ok")]
    terminal = _terminal(True, ctx, extractions, semantic).passed
    valid = bool(injected) and len(pairs) == len(injected) and len(errors) == expected["tool_error_count"] and terminal
    return CheckActual(valid, {"recovery_lineage": pairs, "tool_error_count": len(errors), "terminal_completed": terminal})


def _fault_attempts(expected, ctx, extractions, semantic):
    counts = {tool: sum(event["tool_name"] == tool for event in ctx.tool_events) for tool in expected["target_tools"]}
    return CheckActual(all(count <= expected["max_attempts"] for count in counts.values()), counts)


def _single_answer(expected, ctx, extractions, semantic):
    count = sum(bool(answer.strip()) for answer in ctx.data.get("final_answers", []))
    return CheckActual(count == 1, count)


def _bound_memory_support(definition, actual, plan, ctx, extractions):
    """Bound-source facts may only support the target mutation or actual recall."""
    if definition.observation_scope == "target_mutation":
        policy = next((check.expected for check in plan.checks if check.evaluator_kind == "memory_mutation"), None)
        if policy is None or policy["target_step_id"] != definition.target_step_id:
            raise EvaluationContractError("Memory semantic scope has no target mutation contract")
        # Current topic mapping describes the final snapshot; it cannot decide
        # which entities were observed by an earlier semantic task.
        events = [event for event in _target_events(policy, ctx) if event["success"]]
        memory_ids = {event.get("result_memory_id") for event in events}
        boundary = max((index for index, event in enumerate(ctx.memory_events)
                        if any(event is target for target in events)), default=-1)
        source_events = ctx.memory_events[:boundary + 1]
        observed_refs = {snapshot["id"]: set(snapshot.get("source_refs", []))
                         for event in events for snapshot in event.get("after", [])
                         if snapshot["id"] in memory_ids}
        source_snapshots = ctx.memory_before
    elif definition.observation_scope == "memory_recall":
        step_id = definition.target_step_id
        turn_id = ctx.step_turn_ids.get(step_id)
        searches = [event for event in ctx.tool_events if event["turn_id"] == turn_id and event["tool_name"] == "search_memory"
                    and event["execution_kind"] == "real" and event["result"].get("ok")]
        memory_ids = {item["id"] for event in searches for item in (event["result"].get("agent_payload") or {}).get("memories", [])}
        answer = ctx.final_answers_by_step.get(step_id, "")
        span = actual.get("answer_span_id")
        span_valid = span is None or span.startswith(f"step_{step_id}_") or (
            answer == ctx.final_answer and span.startswith("answer_"))
        if not searches or not answer.strip() or not span_valid:
            return False
        source_events = ctx.memory_events
        source_snapshots = ctx.memory_after
        observed_refs = None
    else:
        raise EvaluationContractError("Bound-memory semantic check requires explicit observation scope")
    sources = {source["id"]: source for memory in source_snapshots if memory["id"] in memory_ids
               for source in memory.get("bound_sources", [])
               if observed_refs is None or source["source_ref"] in observed_refs.get(memory["id"], set())}
    # Target mutation uses its committed prefix; later same-ID versions cannot
    # expand the set of sources supporting an earlier judgment.
    for event in source_events:
        if event.get("result_memory_id") not in memory_ids or not event["success"]:
            continue
        for source in event.get("sources", []):
            if observed_refs is not None and source["source_ref"] not in observed_refs.get(event["result_memory_id"], set()):
                continue
            identity = "bound_source_" + sha256((source["source_ref"] + source["content"]).encode()).hexdigest()[:20]
            sources[identity] = source
    used = actual.get("evidence_ids", [])
    if definition.presence == "absence" and not used:
        return True
    if not used or any(identifier not in sources for identifier in used):
        return False
    if definition.required_fixture_id:
        required_ref = ctx.fixture_source_refs.get(definition.required_fixture_id)
        if not required_ref or not any(sources[identifier]["source_ref"] == required_ref for identifier in used):
            return False
    return True


def _retrieval_metrics(expected, ctx, extractions, semantic):
    result = ctx.retrieval_result
    if result is None or result.get("retrieval_contract_version") != expected["contract_version"]:
        raise EvaluationContractError("Retrieval metric contract identity mismatch")
    facts = {fact["id"] for fact in expected["facts"]}
    satisfied, missing = set(result["satisfied_fact_ids"]), set(result["missing_fact_ids"])
    if (result["answerability"] != expected["answerability"] or satisfied & missing
            or satisfied | missing != facts):
        raise EvaluationContractError("Retrieval metric observations do not match the compiled facts")
    return CheckActual(True, result["metrics"], tuple(result["covered_evidence_ids"]))


EXECUTOR_REGISTRY: dict[str, Callable] = {
    "terminal_completed": _terminal, "final_answer_nonempty": _answer, "source_refs_valid": _sources,
    "evidence_capture_complete": _capture, "project_selection": _project_selection,
    "classification": _classification, "bullet_count": _bullet_count, "local_revision": _local_revision,
    "question_limits": _question_limits, "topic_coverage": _topic_coverage,
    "fixture_visibility": _fixture_visibility, "retrieval_metrics": _retrieval_metrics,
    "memory_mutation": _memory_mutation, "memory_target_state": _memory_state,
    "memory_source_binding": _memory_source, "memory_recall": _memory_recall,
    "fault_triggered": _fault_triggered, "fault_recovery": _fault_recovery,
    "fault_attempts": _fault_attempts, "single_final_answer": _single_answer,
}


def validate_executor_mapping(plan: ExecutableEvalPlan) -> None:
    for check in plan.checks:
        if check.evaluator_kind not in EXECUTOR_REGISTRY and check.evaluator_kind not in {"semantic", "semantic_group"}:
            raise EvaluationContractError(f"Scoring field {check.gold_path} has no executor: {check.evaluator_kind}")


def execute_plan(plan: ExecutableEvalPlan, context: EvaluationContext, judge_response: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_executor_mapping(plan)
    if plan.suite == "retrieval":
        from app.evals.retrieval_contract import RETRIEVAL_CONTRACT_VERSION
        result = context.retrieval_result
        if (plan.contract_version != RETRIEVAL_CONTRACT_VERSION
                or len(plan.checks) != 1 or plan.checks[0].evaluator_kind != "retrieval_metrics"
                or result is None
                or result.get("retrieval_contract_version") != RETRIEVAL_CONTRACT_VERSION
                or result.get("outcome") != "SCORED"):
            raise EvaluationContractError("Current retrieval metric observations are missing")
        # The pure scorer is the sole owner of support/metric semantics.
        actual = EXECUTOR_REGISTRY["retrieval_metrics"](plan.checks[0].expected, context, {}, {})
        return {**result, "check_results": [{
            "id": "retrieval_metrics", "gold_path": "facts+answerability",
            "evaluator_kind": "retrieval_metrics", "required": True,
            "expected": plan.checks[0].expected, "actual": actual.actual,
            "evidence_ids": list(actual.evidence_ids), "result": "SCORED", "reason": None}]}
    judge_response = judge_response or {}
    interrupted = EvalFailureClassifier.judge_interruption(judge_response)
    interrupted = interrupted or EvalFailureClassifier.agent_interruption(context.data)
    semantic = {item["id"]: item for item in judge_response.get("checks", [])}
    extractions = {item["id"]: item for item in judge_response.get("extractions", [])}
    check_results = []
    partial_by_id = {}
    semantic_definitions = {item.id: item for item in plan.semantic_checks}
    if not context.evidence_pack.capture_complete:
        check_results.append({"id": "harness_evidence_capture", "gold_path": "observable.evidence_capture",
                              "evaluator_kind": "evidence_capture_complete", "required": True,
                              "expected": {"capture_complete": True}, "actual": context.evidence_pack.errors,
                              "evidence_ids": [], "result": "ERROR", "reason": "Actual-visible evidence capture is incomplete"})
    for check in plan.checks:
        record = {"id": check.id, "gold_path": check.gold_path, "evaluator_kind": check.evaluator_kind,
                  "required": check.required, "expected": check.expected, "actual": None, "evidence_ids": [],
                  "result": "SKIPPED", "reason": None}
        try:
            if interrupted and check.evaluator_kind in {"fault_triggered", "fault_recovery", "fault_attempts", "memory_target_state", "memory_recall", "memory_source_binding"}:
                record["reason"] = interrupted.reason
                check_results.append(record)
                partial_by_id[check.id] = record
                continue
            if check.evaluator_kind == "semantic":
                if interrupted:
                    record["reason"] = interrupted.reason
                    check_results.append(record)
                    partial_by_id[check.id] = record
                    continue
                if check.id not in semantic:
                    raise EvaluationContractError(f"Required Judge check missing: {check.id}")
                actual = semantic[check.id]
                if actual["result"] not in {"PASS", "FAIL", "UNKNOWN"}:
                    raise EvaluationContractError(f"Illegal Judge result: {actual['result']}")
                passed = actual["result"] == "PASS"
                definition = semantic_definitions[check.id]
                if passed:
                    span = actual.get("answer_span_id")
                    if definition.support_mode == "authoritative_visible" and actual.get("evidence_ids"):
                        evidence = {item.id: item for item in context.evidence_pack.items}
                        referenced = actual["evidence_ids"]
                        if any(identifier not in evidence or not evidence[identifier].visible_to_agent
                               or not evidence[identifier].authoritative_for_grounding for identifier in referenced):
                            raise EvaluationContractError("Factual semantic references must be authoritative actually-visible evidence")
                    if definition.required_extraction_id:
                        extraction = extractions.get(definition.required_extraction_id)
                        if extraction is None:
                            raise EvaluationContractError("Semantic scope extraction is missing")
                        scoped_items = [item for item in extraction["items"] if item.get("id") == definition.required_item_id]
                        if len(scoped_items) != 1 or not span or scoped_items[0].get("answer_span_id") != span:
                            passed = False
                    if definition.support_mode == "bound_memory_source":
                        passed = passed and _bound_memory_support(definition, actual, plan, context, extractions)
                if definition.required_fixture_id and passed:
                    required_ref = context.fixture_source_refs.get(definition.required_fixture_id)
                    evidence = {item.id: item for item in context.evidence_pack.items}
                    used = [evidence[identifier] for identifier in actual.get("evidence_ids", []) if identifier in evidence]
                    # Memory checks receive bound-source evidence in the Judge pack.
                    if definition.support_mode != "bound_memory_source":
                        passed = bool(required_ref) and any(item.source_ref == required_ref and item.authoritative_for_grounding for item in used)
                outcome = CheckActual(passed, actual["result"], tuple(actual.get("evidence_ids", [])), actual["reason"])
            elif check.evaluator_kind == "semantic_group":
                group = check.expected
                records = [partial_by_id.get(identifier) for identifier in group["check_ids"]]
                if any(record is None for record in records):
                    raise EvaluationContractError("Semantic group references an unexecuted check")
                if any(record["result"] == "ERROR" for record in records):
                    raise EvaluationContractError("Semantic group child contract error")
                if interrupted and any(record["result"] == "SKIPPED" for record in records):
                    record["reason"] = interrupted.reason
                    check_results.append(record)
                    partial_by_id[check.id] = record
                    continue
                count = sum(record["result"] == "PASS" for record in records)
                if group["mode"] == "min_count":
                    passed = count >= group["min"]
                elif group["mode"] == "all_of":
                    passed = count == len(records)
                elif group["mode"] == "any_of":
                    passed = count > 0
                else:
                    raise EvaluationContractError(f"Unknown semantic aggregation policy: {group['mode']}")
                outcome = CheckActual(passed, {"passed_count": count, "check_ids": group["check_ids"]})
            else:
                outcome = EXECUTOR_REGISTRY[check.evaluator_kind](check.expected, context, extractions, semantic)
            record.update(actual=outcome.actual, evidence_ids=list(outcome.evidence_ids), result="PASS" if outcome.passed else "FAIL", reason=outcome.reason)
        except (EvaluationContractError, KeyError, TypeError, ValueError) as exc:
            if interrupted and check.evaluator_kind in {"project_selection", "classification", "bullet_count", "local_revision", "question_limits", "topic_coverage", "memory_target_state", "memory_recall"}:
                record.update(result="SKIPPED", reason=interrupted.reason)
            else:
                record.update(result="ERROR", reason=str(exc))
        check_results.append(record)
        partial_by_id[check.id] = record
    expected_semantic = set(semantic_definitions)
    if not interrupted and (set(semantic) != expected_semantic or len(semantic) != len(judge_response.get("checks", []))):
        check_results.append({"id": "judge_check_contract", "gold_path": "compiled.semantic_checks", "evaluator_kind": "schema",
                              "required": True, "expected": sorted(expected_semantic), "actual": list(semantic), "evidence_ids": [],
                              "result": "ERROR", "reason": "Judge check IDs must appear completely and exactly once"})
    if any(item["result"] == "ERROR" for item in check_results):
        classification = EvalFailureClassifier.harness_error("EVAL_CHECK_CONTRACT_ERROR")
    elif interrupted:
        classification = interrupted
    elif any(item["required"] and item["result"] != "PASS" for item in check_results):
        classification = EvalClassification(EvalOutcome.FAIL, "REQUIRED_CHECK_FAILED", "DETERMINISTIC_AGGREGATOR")
    else:
        classification = EvalClassification(EvalOutcome.PASS)
    return case_result(classification, check_results=check_results, final_answer=context.final_answer)
