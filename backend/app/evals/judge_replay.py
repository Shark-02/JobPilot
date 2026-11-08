"""One-shot, file-only Judge replay over a frozen Agent run.

No Agent, dataset compiler, database session or suite runner is used. The trace
wrappers forward original requests/chunks/parsers unchanged to the current Judge.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.evals.compiler import ExecutableEvalPlan
from app.evals.evidence import EvidencePack
from app.evals.executor import EvaluationContext, execute_plan
from app.evals.judge import (JudgeSchemaError, StructuredJudge, answer_spans, restore_observation_text,
                             strict_json_loads, validate_response)
from app.evals.judge_scheduler import JudgeScheduler
from app.evals.schema import CONTRACT_VERSION, SourcePolicy
from app.infra.provider_errors import classify_provider_error_code

_ATTEMPT: ContextVar[tuple[str, int] | None] = ContextVar("replay_attempt", default=None)
REPLAY_VERSION = "frozen_judge_replay_2"


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def require_current_contract(plan_data: Any) -> None:
    """Old scoring plans are historical evidence, never a migration input."""
    version = plan_data.get("contract_version") if isinstance(plan_data, dict) else None
    if version != CONTRACT_VERSION:
        raise ValueError(
            f"Incompatible frozen evaluation contract; current contract is {CONTRACT_VERSION}. "
            "Preserve this historical run and capture a new current-contract run; frozen plans are not recompiled."
        )


def load_frozen_inputs(run_dir: Path, case_id: str, metadata_path: Path):
    rows = [strict_json_loads(line) for line in (run_dir / "case_results.jsonl").read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError("Frozen run must contain exactly one selected case")
    row = matches[0]
    if row["runtime"]["run_failed"] is not False:
        raise ValueError("Judge-only replay requires a completed frozen Agent")
    plans = strict_json_loads((run_dir / "compiled_plans.json").read_text())
    saved = [p for p in plans if p["case_id"] == case_id]
    if len(saved) != 1 or saved[0] != row["compiled_plan"]:
        raise ValueError("Frozen compiled plan copies disagree")
    require_current_contract(row["compiled_plan"])
    plan = ExecutableEvalPlan.model_validate(row["compiled_plan"])
    metadata = strict_json_loads(metadata_path.read_text())
    if metadata["case_id"] != case_id or not isinstance(metadata["task"], (str, list)) or not isinstance(metadata["source_policy"], dict):
        raise ValueError("Historical task/source policy metadata is invalid")
    SourcePolicy.model_validate(metadata["source_policy"])
    obs = row["observable"]
    pack = EvidencePack.model_validate(obs["evidence"])
    if not pack.capture_complete or pack.errors:
        raise ValueError("Frozen actual-visible evidence capture is incomplete")
    answer = obs["final_answer"]
    spans = answer_spans(answer)
    for step, text in obs["final_answers_by_step"].items():
        spans.extend(answer_spans(text, prefix=f"step_{step}"))
    payload = {
        "task": metadata["task"], "source_policy": metadata["source_policy"],
        "final_answer": answer, "answer_spans": spans, "evidence": pack.judge_items(),
        "memories": obs["memory_after"],
        **{key: obs[key] for key in ("final_answers_by_step", "fixture_source_refs", "step_turn_ids",
                                   "step_source_refs", "memory_before", "memory_events", "tool_events")},
    }
    checks = {r["evaluator_kind"]: r for r in row["result"]["check_results"]}
    states = checks.get("terminal_completed", {}).get("actual")
    source_check = checks.get("source_refs_valid")
    if not isinstance(states, list) or len(states) != len(row["runtime"]["turns"]) or source_check is None:
        raise ValueError("Frozen terminal/source validation observations are unavailable")
    if not states or any(status != "COMPLETED" for status in states):
        raise ValueError("Frozen Agent terminal states must actually be completed")
    invalid = source_check["actual"]
    if not isinstance(invalid, list):
        raise ValueError("Frozen source validation observation is invalid")
    context = EvaluationContext(
        data={**deepcopy(row["runtime"]), "final_answers": [answer], "tool_events": obs["tool_events"],
              "turns": [SimpleNamespace(id=identifier, status=status, error_code=None)
                        for identifier, status in zip(row["runtime"]["turns"], states, strict=True)]},
        evidence_pack=pack, memory_after=obs["memory_after"], fault_events=obs["fault_events"],
        source_refs_valid=source_check["result"] == "PASS", invalid_source_refs=invalid,
        **{key: obs[key] for key in ("fixture_source_refs", "step_turn_ids", "step_source_refs",
                                    "final_answers_by_step", "memory_before", "memory_events", "tool_events")},
    )
    provenance = {
        "case_id": case_id, "run_dir": str(run_dir.resolve()), "metadata": metadata,
        "input_file_sha256": {name: sha256((run_dir / name).read_bytes()).hexdigest()
                               for name in ("case_results.jsonl", "compiled_plans.json")},
        "metadata_sha256": sha256(metadata_path.read_bytes()).hexdigest(),
        "answer_sha256": sha256(answer.encode()).hexdigest(), "plan_sha256": _digest(row["compiled_plan"]),
        "evidence_sha256": _digest(payload["evidence"]), "payload_sha256": _digest(payload),
        "answer_span_count": len(spans), "frozen_evidence_count": len(pack.items),
        "judge_evidence_count": len(payload["evidence"]), "terminal_states": states,
        "source_refs_valid_from_frozen_record": context.source_refs_valid,
        "historical_case_outcome": row["result"]["outcome"],
    }
    return plan, payload, context, provenance


class ReplayTrace:
    def __init__(self, root: Path):
        self.root = root
        self.tasks: dict[str, dict] = {}
        self.validations: dict[str, list[dict]] = {}

    def task_path(self, identifier: str) -> Path:
        return self.root / "tasks" / sha256(identifier.encode()).hexdigest()[:24]

    def register(self, task):
        item = {"task_id": task.id, "kind": task.kind, "stage": task.stage,
                "schema_version": task.schema_version, "messages": task.messages}
        self.tasks[task.id] = item
        _write(self.task_path(task.id) / "request.json", item)

    def validate(self, task, raw):
        entries = self.validations.setdefault(task.id, [])
        path = self.task_path(task.id) / f"validation_{len(entries) + 1:02}.json"
        entry = {"raw": raw, "valid": False}
        entries.append(entry)
        try:
            body = task.validate(raw)
            entry.update(valid=True, body=body)
            return body
        except Exception as exc:
            causes = []
            cause = exc
            while cause is not None:
                causes.append({"type": type(cause).__name__, "detail": str(cause)})
                cause = cause.__cause__
            entry.update(error_code=getattr(exc, "code", "JUDGE_SCHEMA_ERROR"), causes=causes)
            raise
        finally:
            _write(path, entry)


class RecordingScheduler(JudgeScheduler):
    def __init__(self, *args, trace: ReplayTrace, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = trace

    async def _run_one(self, task, identity):
        self.trace.register(task)
        wrapped = replace(task, validate=lambda raw: self.trace.validate(task, raw))
        result = await super()._run_one(wrapped, identity)
        _write(self.trace.task_path(task.id) / "result.json", result)
        print(json.dumps({"task_id": task.id, "status": result["telemetry"]["status"],
                          "error": result.get("execution_error"), "cache_hit": result["telemetry"]["cache_hit"]}, ensure_ascii=False), flush=True)
        return result

    async def _attempt(self, task, sequence):
        token = _ATTEMPT.set((task.id, sequence))
        try:
            raw, telemetry, error = await super()._attempt(task, sequence)
            _write(self.trace.task_path(task.id) / f"attempt_{sequence:02}.json",
                   {"raw": raw, "telemetry": telemetry, "execution_error": error})
            return raw, telemetry, error
        finally:
            _ATTEMPT.reset(token)


class RecordingClient:
    def __init__(self, factory, trace: ReplayTrace, **kwargs):
        self.real = factory(**kwargs)
        self.trace = trace
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        identifier, sequence = _ATTEMPT.get()
        path = self.trace.task_path(identifier)
        _write(path / f"provider_request_{sequence:02}.json", kwargs)
        stream = await self.real.chat.completions.create(**kwargs)
        return RecordingStream(stream, path / f"stream_{sequence:02}.jsonl")

    async def close(self):
        await self.real.close()


class RecordingStream:
    def __init__(self, real, path):
        self.real = real
        self.output = path.open("w", encoding="utf-8")
        path.chmod(0o600)
        self.started = perf_counter()

    async def __aiter__(self):
        count = 0
        async for chunk in self.real:
            usage = getattr(chunk, "usage", None)
            item = {"elapsed_seconds": perf_counter() - self.started, "choices": [],
                    "usage": usage.model_dump() if hasattr(usage, "model_dump") else (
                        vars(usage) if hasattr(usage, "__dict__") else None)}
            for choice in chunk.choices or []:
                item["choices"].append({"finish_reason": choice.finish_reason,
                    "content": getattr(choice.delta, "content", None),
                    "reasoning_content": getattr(choice.delta, "reasoning_content", None)})
            self.output.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if count % 50 == 0:
                self.output.flush()
            yield chunk

    async def close(self):
        try:
            await self.real.close()
        finally:
            self.output.close()


def audit_trace(trace: ReplayTrace, plan: ExecutableEvalPlan, response: dict) -> dict:
    """Record direct result coverage and execution errors, not a semantic proof."""
    validation_errors = [
        {"task_id": identifier, **{key: value for key, value in entry.items() if key != "raw"}}
        for identifier, entries in trace.validations.items() for entry in entries if not entry["valid"]
    ]
    errors = [{"task_id": row["task_id"], **row["execution_error"]}
              for row in response.get("subtasks", []) if "execution_error" in row]
    classifications = Counter()
    for error in errors:
        code = error["code"]
        category = "provider_interruption" if classify_provider_error_code(code) else (
            "model_output_or_contract_error" if code in {
                "JUDGE_SCHEMA_ERROR", "JUDGE_OUTPUT_TRUNCATED", "JUDGE_INCOMPLETE_OUTPUT"}
            else "execution_error")
        classifications[category] += 1
    checks, observations = response.get("checks", []), response.get("extractions", [])
    expected_checks = {check.id for check in plan.semantic_checks}
    expected_observations = {spec.id for spec in plan.extractions}
    check_ids = [check["id"] for check in checks]
    observation_ids = [observation["id"] for observation in observations]
    covered = (set(check_ids) == expected_checks and len(check_ids) == len(expected_checks)
               and set(observation_ids) == expected_observations
               and len(observation_ids) == len(expected_observations))
    return {
        "status": "DIRECT_JUDGE_COMPLETE" if covered and "execution_error" not in response else "NOT_ESTABLISHED",
        "semantic_check_ids": check_ids, "observation_ids": observation_ids,
        "missing_semantic_check_ids": sorted(expected_checks - set(check_ids)),
        "missing_observation_ids": sorted(expected_observations - set(observation_ids)),
        "semantic_result_counts": dict(Counter(check["result"] for check in checks)),
        "validation_errors": validation_errors, "task_errors": errors,
        "error_categories": dict(classifications),
        "coverage_method": "declared_check_and_observation_ids",
        "limitation": "Complete result IDs establish protocol coverage, not semantic correctness or Judge stability.",
    }


async def run_replay(run_dir: Path, case_id: str, metadata_path: Path, output_dir: Path,
                     cache_dir: Path, settings: Settings, *, client_factory=None, resume=False):
    entry_started = perf_counter()
    plan, payload, context, provenance = load_frozen_inputs(run_dir, case_id, metadata_path)
    run_dir, output_dir, cache_dir = run_dir.resolve(), output_dir.resolve(), cache_dir.resolve()
    if (output_dir.is_relative_to(run_dir) or run_dir.is_relative_to(output_dir)
            or cache_dir.is_relative_to(run_dir) or run_dir.is_relative_to(cache_dir)
            or output_dir.is_relative_to(cache_dir) or cache_dir.is_relative_to(output_dir)):
        raise ValueError("Replay output/cache must be independent of frozen input and each other")
    if output_dir.exists() and not resume:
        raise ValueError("Output already exists; explicit --resume is required")
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.chmod(0o700)
    cache_dir.chmod(0o700)
    settings = settings.model_copy(update={"eval_judge_cache_dir": cache_dir})
    judge = StructuredJudge(settings)
    app_root = Path(__file__).resolve().parents[1]
    manifest = {"replay_version": REPLAY_VERSION, "provenance": provenance,
        "model": judge.model, "endpoint_sha256": _digest(judge.base_url),
        "judge_source_sha256": {str(p.relative_to(app_root)): sha256(p.read_bytes()).hexdigest()
                                for p in sorted((app_root / "evals").glob("*.py"))},
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in settings.model_dump().items() if key.startswith("eval_judge_")
                     and key not in {"eval_judge_api_key", "eval_judge_base_url"}},
        "agent_runs": 0, "database_access": False, "judge_invocations_per_attempt": 1}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and strict_json_loads(manifest_path.read_text()) != manifest:
        raise ValueError("Resume identity changed; use a new output directory")
    _write(manifest_path, manifest)
    invocations = output_dir / "invocations"
    invocations.mkdir(exist_ok=True, mode=0o700)
    invocation = invocations / f"{len(list(invocations.iterdir())) + 1:03}"
    invocation.mkdir(mode=0o700)
    _write(invocation / "frozen_plan.json", plan.model_dump(mode="json"))
    _write(invocation / "frozen_payload.json", payload)
    trace = ReplayTrace(invocation)
    factory = client_factory or AsyncOpenAI
    judge._scheduler = RecordingScheduler(settings, model=judge.model, base_url=judge.base_url,
        api_key=judge.api_key, counter=judge.counter, trace=trace,
        client_factory=lambda **kw: RecordingClient(factory, trace, **kw))
    started = perf_counter()
    _write(invocation / "state.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    response = await judge.judge(plan=plan, payload=payload)
    elapsed = perf_counter() - started
    validation = {"valid": False, "execution_error": response.get("execution_error")}
    try:
        merged = {key: response.get(key, []) for key in ("checks", "extractions")}
        # The executor sees code-restored text. The model wire never contains
        # text/followups, so validate that wire and recompute the derived fields.
        wire = deepcopy(merged)
        for extraction in wire["extractions"]:
            for item in extraction.get("items", []):
                if extraction["kind"] in {"bullets", "questions"}:
                    item.pop("text", None)
                if extraction["kind"] == "questions":
                    item.pop("followups", None)
        validated = validate_response(json.dumps(wire, ensure_ascii=False), plan=plan, payload=payload)
        restored = validated.model_dump(mode="json")
        restored["extractions"] = restore_observation_text(restored["extractions"], payload)
        if restored != merged:
            raise JudgeSchemaError("Judge result differs from its original-observation text restoration")
        validation["valid"] = "execution_error" not in response
        validation["response"] = validated.model_dump(mode="json")
    except ValueError as exc:
        validation["detail"] = str(exc)
        response.setdefault("execution_error", {"code": "JUDGE_SCHEMA_ERROR"})
        validation["execution_error"] = response["execution_error"]
    _write(invocation / "judge_result.json", response)
    _write(invocation / "strict_validation.json", validation)
    result = execute_plan(plan, context, response)
    _write(invocation / "evaluation_result.json", result)
    audit = audit_trace(trace, plan, response)
    _write(invocation / "audit.json", audit)
    input_drift = [name for name, digest in provenance["input_file_sha256"].items()
                   if sha256((run_dir / name).read_bytes()).hexdigest() != digest]
    summary = {"case_id": case_id, "outcome": result["outcome"], "reason": result.get("reason"),
        "wall_seconds": perf_counter() - entry_started, "judge_seconds": elapsed,
        "strict_contract_valid": validation["valid"],
        "audit_status": audit["status"], "error_categories": audit["error_categories"],
        "input_drift": input_drift, "invocation_dir": str(invocation), "cache_dir": str(cache_dir),
        "telemetry": response.get("telemetry"), "pipeline_summary": response.get("summary"),
        "quality_fail_established": result["outcome"] == "FAIL" and validation["valid"],
        "agent_runs": 0, "database_access": False}
    _write(invocation / "summary.json", summary)
    _write(invocation / "state.json", {"status": "finished", "outcome": result["outcome"]})
    _write(output_dir / "latest.json", summary)
    report = [f"# Judge-only Replay: {case_id}", "",
        f"Outcome: {summary['outcome']}; reason: {summary['reason']}",
        f"Strict contract valid: {summary['strict_contract_valid']}; audit: {summary['audit_status']}",
        f"Wall time: {summary['wall_seconds']:.3f}s; Judge call: {elapsed:.3f}s", "",
        "Exactly one StructuredJudge.judge invocation; Agent runs 0; database access 0.",
        "Input hashes and configuration: manifest.json; task raw/telemetry: tasks/.",
        "Declared semantic result and observation coverage: audit.json.",
        "An execution ERROR/BLOCKED cannot establish a genuine evaluation quality FAIL.",
        "Complete protocol coverage does not prove semantic correctness or Judge stability.", "",
        "Telemetry / Token Estimates (provider usage may be unavailable):", "",
        "```json", json.dumps(response.get("telemetry", {}), ensure_ascii=False, indent=2), "```", ""]
    (invocation / "report.md").write_text("\n".join(report), encoding="utf-8")
    (invocation / "report.md").chmod(0o600)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = asyncio.run(run_replay(args.run_dir, args.case_id, args.metadata, args.output_dir,
                                    args.cache_dir, get_settings(), resume=args.resume))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["strict_contract_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
