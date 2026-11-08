"""Parallel bounded Judge execution; final scoring belongs to the existing executor."""
from __future__ import annotations

import json
from hashlib import sha256
import re
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config import Settings
from app.evals.schema import JudgeResponse
from app.infra.provider_errors import classify_provider_error_code
from app.infra.token_count import TokenCounter

JUDGE_CONTRACT_VERSION = "direct_semantics_1"
_ANSWER_GROUNDING_SHARD_TASK = re.compile(r"semantic:answer_grounding:shard:\d{3,}$")


class JudgeSchemaError(ValueError):
    pass


def answer_spans(answer: str, *, prefix: str = "answer") -> list[dict[str, str]]:
    """Lossless line spans; IDs refer to text, never an inferred answer sidecar."""
    return [{"id": f"{prefix}_{index}", "text": text}
            for index, text in enumerate(answer.splitlines(), 1) if text.strip()]


def _exact_ids(actual: list[str], expected: list[str], label: str) -> None:
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise JudgeSchemaError(f"{label} must contain all and only expected IDs exactly once")


def strict_json_loads(raw: str) -> Any:
    """Reject duplicate fields, prose/fences and non-finite JSON in every task."""
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise JudgeSchemaError(f"Duplicate JSON field: {key}")
            value[key] = item
        return value
    try:
        return json.loads(raw, object_pairs_hook=object_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(JudgeSchemaError("Non-finite JSON value")))
    except (ValueError, TypeError) as exc:
        raise JudgeSchemaError("Judge response does not satisfy the typed JSON contract") from exc


def _is_answer_grounding_shard(task_id: str) -> bool:
    return _ANSWER_GROUNDING_SHARD_TASK.fullmatch(task_id) is not None


def _aggregate_answer_grounding(shards: list[tuple[str, dict]]) -> dict:
    """Apply ALL semantics after each shard has passed its local validator."""
    if not shards:
        raise JudgeSchemaError("answer_grounding has no completed shards")
    rows = []
    evidence_ids = []
    seen_evidence = set()
    for task_id, body in sorted(shards, key=lambda row: int(row[0].rsplit(":", 1)[-1])):
        checks = body.get("checks", [])
        if len(checks) != 1 or checks[0].get("id") != "answer_grounding":
            raise JudgeSchemaError("answer_grounding shard returned an invalid semantic result")
        item = checks[0]
        if item["result"] not in {"PASS", "FAIL", "UNKNOWN"}:
            raise JudgeSchemaError("answer_grounding shard returned an invalid result")
        rows.append((task_id, item))
        for evidence_id in item.get("evidence_ids", []):
            if evidence_id not in seen_evidence:
                seen_evidence.add(evidence_id)
                evidence_ids.append(evidence_id)
    status = "FAIL" if any(item["result"] == "FAIL" for _, item in rows) else (
        "UNKNOWN" if any(item["result"] == "UNKNOWN" for _, item in rows) else "PASS")
    decisive_task, decisive = next((task_id, item) for task_id, item in rows if item["result"] == status)
    summary = ", ".join(f"{task_id.rsplit(':', 1)[-1]}={item['result']}" for task_id, item in rows)
    return {"id": "answer_grounding", "result": status,
            "reason": f"answer_grounding ALL shards ({summary}); decisive shard {decisive_task.rsplit(':', 1)[-1]}: {decisive['reason']}",
            "answer_span_id": None, "evidence_ids": evidence_ids}


def validate_response(raw: str, *, plan: Any, payload: dict[str, Any]) -> JudgeResponse:
    """Strict JSON and typed schema, followed by request-specific referential checks."""
    try:
        response = JudgeResponse.model_validate(strict_json_loads(raw))
    except (ValueError, TypeError, ValidationError) as exc:
        raise JudgeSchemaError("Judge response does not satisfy the typed JSON contract") from exc
    _exact_ids([item.id for item in response.checks], [item.id for item in plan.semantic_checks], "checks")
    _exact_ids([item.id for item in response.extractions], [item.id for item in plan.extractions], "extractions")
    evidence_ids = {item["id"] for item in payload.get("evidence", [])
                    if item.get("visible_to_agent") is True and item.get("authoritative_for_grounding") is True}
    spans = {item["id"]: item["text"] for item in payload.get("answer_spans", [])}
    span_ids = set(spans)
    memory_ids = {item["id"] for item in payload.get("memories", [])}
    # Final assembly retains source identities observed at earlier target
    # times. Task-local parsers already enforce each narrower historical scope.
    bound_source_ids = {source["id"] for memory in [*payload.get("memory_before", []),
                                                   *payload.get("memories", [])]
                        for source in memory.get("bound_sources", [])}
    bound_source_ids.update(source.get("id") or "bound_source_" + sha256(
        (source["source_ref"] + source["content"]).encode()).hexdigest()[:20]
        for event in payload.get("memory_events", []) for source in event.get("sources", []))

    def check_refs(item: Any, allowed_evidence: set[str]) -> None:
        if isinstance(item, dict):
            if "answer_span_id" in item:
                span = item["answer_span_id"]
                if span is not None and span not in span_ids:
                    raise JudgeSchemaError("Unknown answer span ID")
            if "evidence_ids" in item:
                if any(eid not in allowed_evidence for eid in item["evidence_ids"]):
                    raise JudgeSchemaError("Unknown evidence ID")
                if len(item["evidence_ids"]) != len(set(item["evidence_ids"])):
                    raise JudgeSchemaError("Duplicate evidence ID")
            for child in item.values():
                check_refs(child, allowed_evidence)
        elif isinstance(item, list):
            for child in item:
                check_refs(child, allowed_evidence)
    checks = {item.id: item for item in plan.semantic_checks}
    for item in response.checks:
        mode = getattr(checks[item.id], "support_mode", "authoritative_visible")
        allowed_evidence = bound_source_ids if mode == "bound_memory_source" else (
            evidence_ids if mode == "authoritative_visible" else set())
        check_refs(item.model_dump(), allowed_evidence)
    for item in response.extractions:
        check_refs(item.model_dump(), evidence_ids)
    specs = {item.id: item for item in plan.extractions}
    for extraction in response.extractions:
        spec = specs[extraction.id]
        if extraction.kind != spec.kind:
            raise JudgeSchemaError("Observation kind differs from compiled contract")
        allowed = set(spec.allowed_ids)
        if extraction.kind in {"project_selection", "topic_set"}:
            if len(extraction.items) != len(set(extraction.items)):
                raise JudgeSchemaError("Duplicate observed entity ID")
            if any(item not in allowed and item != "unknown" for item in extraction.items):
                raise JudgeSchemaError("Unknown observed entity ID")
        elif extraction.kind == "classification":
            ids = [item.id for item in extraction.items]
            if len(ids) != len(set(ids)) or any(item not in allowed for item in ids):
                raise JudgeSchemaError("Unknown or repeated classification ID")
            if any(item.label not in spec.labels for item in extraction.items):
                raise JudgeSchemaError("Invalid classification label")
        elif extraction.kind == "questions":
            if len({item.id for item in extraction.items}) != len(extraction.items):
                raise JudgeSchemaError("Repeated question ID")
            for item in extraction.items:
                if any(topic not in allowed for topic in item.topic_ids):
                    raise JudgeSchemaError("Unknown question topic ID")
                if any(span not in spans for span in item.followup_span_ids):
                    raise JudgeSchemaError("Unknown follow-up span ID")
        elif extraction.kind == "memory_topics":
            ids = [item.memory_id for item in extraction.items]
            pairs = [(item.memory_id, item.topic_id) for item in extraction.items]
            if set(ids) != memory_ids or len(pairs) != len(set(pairs)):
                raise JudgeSchemaError("Memory topics must cover every observed memory with unique memory/topic pairs")
            if any(item.topic_id not in allowed and item.topic_id != "other" for item in extraction.items):
                raise JudgeSchemaError("Unknown memory topic ID")
        elif extraction.kind == "bullets":
            if any(item.id is not None and item.id not in allowed for item in extraction.items):
                raise JudgeSchemaError("Unknown bullet ID")
    return response


def restore_observation_text(extractions: list[dict], payload: dict) -> list[dict]:
    """Recover delivered text from immutable observations, never model copies."""
    spans = {span["id"]: span["text"] for span in payload.get("answer_spans", [])}

    def bullet_text(item: dict) -> str:
        text = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", spans[item["answer_span_id"]])
        identifier = item.get("id")
        if identifier is not None:
            # The selected ID was validated against the compiled draft. Its
            # exact display label is structure, while the remainder is text.
            for colon in (":", "："):
                prefix = identifier + colon
                if text.startswith(prefix):
                    return text[len(prefix):]
        return text

    restored = []
    for extraction in extractions:
        value = dict(extraction)
        if value["kind"] == "bullets":
            value["items"] = [{**item, "text": bullet_text(item)} for item in value["items"]]
        elif value["kind"] == "questions":
            value["items"] = [{**item, "text": spans[item["answer_span_id"]],
                               "followups": [spans[sid] for sid in item["followup_span_ids"]]}
                              for item in value["items"]]
        restored.append(value)
    return restored


class StructuredJudge:
    """Direct semantic judgments and minimal observations over real task scopes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.eval_judge_base_url or settings.llm_base_url
        self.api_key = settings.eval_judge_api_key or settings.llm_api_key
        self.model = settings.eval_judge_model or settings.llm_model
        self.counter = TokenCounter(self.model)
        self._scheduler = None

    async def judge(self, *, plan: Any, payload: dict[str, Any]) -> dict[str, Any]:
        from app.evals.judge_scheduler import JudgeScheduler
        from app.evals.judge_tasks import JudgeTaskPlanner
        from app.evals.schema import CONTRACT_VERSION

        started = perf_counter()
        result: dict[str, Any] = {"contract_version": JUDGE_CONTRACT_VERSION, "model": self.model}
        tasks_run: list[dict] = []
        semantic_results: list[dict] = []
        durations: dict[str, float] = {}
        identity = {"contract_version": JUDGE_CONTRACT_VERSION,
                    "plan": plan.model_dump(mode="json") if hasattr(plan, "model_dump") else {
                        "semantic_checks": [c.model_dump(mode="json") for c in plan.semantic_checks],
                        "extractions": [e.model_dump(mode="json") for e in plan.extractions]},
                    "payload": payload}
        try:
            if getattr(plan, "contract_version", CONTRACT_VERSION) != CONTRACT_VERSION:
                raise JudgeSchemaError("Frozen plan contract is obsolete; recreate its observations with the current contract")
            # Resolve historical scope before constructing a client or performing I/O.
            tasks = JudgeTaskPlanner().plan(plan, payload)
            if tasks and (not self.base_url or not self.api_key or not self.model):
                result["execution_error"] = {"code": "JUDGE_NOT_CONFIGURED"}
            else:
                if tasks:
                    if self._scheduler is None:
                        self._scheduler = JudgeScheduler(self.settings, model=self.model, base_url=self.base_url,
                                                         api_key=self.api_key, counter=self.counter,
                                                         client_factory=lambda **kw: AsyncOpenAI(**kw))
                    stage_started = perf_counter()
                    tasks_run = await self._scheduler.run(tasks, identity=identity)
                    durations["direct_judge"] = perf_counter() - stage_started
                shard_task_ids = {task.id for task in tasks if _is_answer_grounding_shard(task.id)}
                errors = [row for row in tasks_run if "execution_error" in row]
                shard_rows = [(row["task_id"], row["body"])
                              for row in tasks_run if row["task_id"] in shard_task_ids
                              and "execution_error" not in row]
                # Keep the pre-existing partial-run diagnostics for ordinary
                # semantic tasks even when a different task fails.  Shards are
                # one logical check and are only exposed after aggregation.
                semantic_results = [item for row in tasks_run if "execution_error" not in row
                                    and row["task_id"] not in shard_task_ids
                                    for item in row["body"].get("checks", [])]
                shard_has_fail = any(item["result"] == "FAIL"
                                     for _, body in shard_rows for item in body.get("checks", []))
                fatal_errors = [row for row in errors if row["task_id"] not in shard_task_ids
                                or not classify_provider_error_code(row["execution_error"]["code"])]
                shard_provider_errors = [row for row in errors if row["task_id"] in shard_task_ids
                                         and classify_provider_error_code(row["execution_error"]["code"])]
                if fatal_errors or (shard_provider_errors and not shard_has_fail):
                    selected_errors = fatal_errors or shard_provider_errors
                    selected = next((row for row in selected_errors if not classify_provider_error_code(
                        row["execution_error"]["code"])), selected_errors[0])
                    result["execution_error"] = {**selected["execution_error"], "task_id": selected["task_id"],
                                                 "failed_task_ids": [row["task_id"] for row in errors]}
                else:
                    extraction_results = [item for row in tasks_run if "execution_error" not in row
                                          for item in row["body"].get("extractions", [])]
                    checks_by_id = {item["id"]: item for item in semantic_results}
                    if shard_task_ids:
                        checks_by_id["answer_grounding"] = _aggregate_answer_grounding(shard_rows)
                    extractions_by_id = {item["id"]: item for item in extraction_results}
                    semantic_results = [checks_by_id[check.id] for check in plan.semantic_checks]
                    extraction_results = [extractions_by_id[spec.id] for spec in plan.extractions]
                    response = validate_response(json.dumps({"checks": semantic_results, "extractions": extraction_results},
                                                             ensure_ascii=False), plan=plan, payload=payload)
                    body = response.model_dump(mode="json")
                    body["extractions"] = restore_observation_text(body["extractions"], payload)
                    result.update(body)
        except Exception as exc:
            result["execution_error"] = {"code": getattr(exc, "code", "JUDGE_SCHEMA_ERROR" if isinstance(exc, ValueError)
                                                        else "JUDGE_EVALUATOR_ERROR"),
                                         "detail": str(exc), "exception_type": type(exc).__name__, "task_id": "planner:direct"}
        result["subtasks"] = tasks_run
        result["telemetry"] = self._telemetry(tasks_run, perf_counter() - started, result)
        failures_by_stage: dict[str, int] = {}
        for row in tasks_run:
            stage = row["telemetry"]["stage"]
            failures_by_stage[stage] = failures_by_stage.get(stage, 0) + int("execution_error" in row)
        result["summary"] = {
            "total_subtasks": len(tasks_run), "concurrency": self.settings.eval_judge_concurrency,
            "stage_latency_seconds": durations, "total_latency_seconds": perf_counter() - started,
            "input_token_estimate": result["telemetry"]["input_token_estimate"],
            "output_token_estimate": result["telemetry"]["content_token_estimate"],
            "failures_by_stage": failures_by_stage,
            "semantic_result_counts": {state: sum(c["result"] == state for c in semantic_results)
                                       for state in ("PASS", "FAIL", "UNKNOWN")},
            "cache_hits": sum(bool(row["telemetry"].get("cache_hit")) for row in tasks_run),
        }
        return result

    def _telemetry(self, rows: list[dict], elapsed: float, result: dict) -> dict:
        items = [r["telemetry"] for r in rows]
        total = {key: sum(t.get(key, 0) or 0 for t in items) for key in (
            "input_token_estimate", "reasoning_chunks", "reasoning_chars", "content_chunks", "content_chars", "content_token_estimate")}
        for key in ("first_chunk_latency_seconds", "first_reasoning_latency_seconds", "first_content_latency_seconds"):
            values = [t[key] for t in items if t.get(key) is not None]
            total[key] = min(values) if values else None
        for key in ("provider_input_tokens", "provider_output_tokens"):
            # A missing gateway usage value stays unavailable, never inferred
            # from the configured maximum or treated as a measured zero.
            values = [t.get(key) for t in items if not t.get("cache_hit")]
            complete = all(t.get("provider_usage_complete") for t in items if not t.get("cache_hit"))
            total[key] = sum(values) if values and complete and all(v is not None for v in values) else None
        total.update(total_latency_seconds=elapsed, stream=True,
                     max_retries=self.settings.eval_judge_transport_retries,
                     max_output_tokens=self.settings.eval_judge_max_output_tokens,
                     finish_reason="stop" if "execution_error" not in result else next(
                         (t.get("finish_reason") for t in items if t.get("error_type")), None),
                     http_status=next((t["http_status"] for t in items if t.get("http_status")), None),
                     provider_error=result.get("execution_error", {}).get("code"))
        return total
