"""Run explicitly selected Judge tasks over frozen inputs.

This task-level diagnostic never invokes the Agent or final scoring. By default
requests and source must match the recorded replay; the explicit current-tasks
mode permits a new task implementation while keeping its input plan frozen.
A validated task result establishes protocol validity, never a case verdict.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.evals.compiler import ExecutableEvalPlan
from app.evals.judge import strict_json_loads
from app.evals.judge_replay import (RecordingClient, RecordingScheduler, ReplayTrace,
                                    _digest, _write, require_current_contract)
from app.evals.judge_scheduler import JudgeScheduler, JudgeTask
from app.evals.judge_tasks import JudgeTaskPlanner
from app.infra.token_count import TokenCounter

PROBE_VERSION = "frozen_judge_task_probe_4"
_CONTROLLED_SETTINGS = {
    "eval_judge_timeout_seconds", "eval_judge_max_output_tokens",
    "eval_judge_concurrency", "eval_judge_transport_retries", "eval_judge_cache_dir",
}


def _file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Any:
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("A required frozen input file is missing or invalid") from exc


def _request(task: JudgeTask) -> dict:
    return {"task_id": task.id, "kind": task.kind, "stage": task.stage,
            "schema_version": task.schema_version, "messages": task.messages}


def _independent(output: Path, original: Path) -> None:
    if output.is_relative_to(original) or original.is_relative_to(output):
        raise ValueError("Probe output must be independent of all original replay inputs")


@dataclass(frozen=True)
class PreparedProbe:
    tasks: list[JudgeTask]
    settings: Settings
    model: str
    base_url: str
    api_key: str
    identity: dict
    manifest: dict


def prepare_probe(invocation_dir: Path, task_ids: list[str], output_dir: Path,
                  settings: Settings, *, use_current_tasks: bool = False) -> PreparedProbe:
    """Validate every selected request and all frozen identities before I/O to a model."""
    invocation_dir, output_dir = invocation_dir.resolve(), output_dir.resolve()
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Select at least one unique explicit task ID")
    if output_dir.exists():
        raise ValueError("Probe output already exists; use a new output directory")
    replay_dir = invocation_dir.parent.parent
    _independent(output_dir, replay_dir)
    manifest_path = replay_dir / "manifest.json"
    original = _read(manifest_path)
    payload_path, plan_path = invocation_dir / "frozen_payload.json", invocation_dir / "frozen_plan.json"
    payload, plan_data = _read(payload_path), _read(plan_path)
    require_current_contract(plan_data)
    try:
        provenance = original["provenance"]
        expected = {
            "payload_sha256": _digest(payload), "plan_sha256": _digest(plan_data),
            "answer_sha256": sha256(payload["final_answer"].encode()).hexdigest(),
            "evidence_sha256": _digest(payload["evidence"]),
        }
        if any(provenance[key] != value for key, value in expected.items()):
            raise ValueError("Frozen payload or plan disagrees with the replay manifest")
        metadata = provenance["metadata"]
        if payload["task"] != metadata["task"] or payload["source_policy"] != metadata["source_policy"]:
            raise ValueError("Frozen task/source policy disagrees with replay metadata")
        source_run = Path(provenance["run_dir"]).resolve()
        _independent(output_dir, source_run)
        for name, digest in provenance["input_file_sha256"].items():
            if name not in {"case_results.jsonl", "compiled_plans.json"}:
                raise ValueError("Unexpected frozen run file identity")
            if _file_digest(source_run / name) != digest:
                raise ValueError("Original frozen run file hash changed")
        app_root = Path(__file__).resolve().parents[1]
        if not original["judge_source_sha256"]:
            raise ValueError("Frozen Judge source identity is missing")
        current_source_hashes = {}
        for name, digest in original["judge_source_sha256"].items():
            source = (app_root / name).resolve()
            if not source.is_relative_to(app_root / "evals"):
                raise ValueError("Frozen Judge source path is outside the evaluation source directory")
            if not source.is_file():
                raise ValueError("Recorded Judge source no longer exists; use a new current-contract frozen run")
            current_source_hashes[name] = _file_digest(source)
            if not use_current_tasks and current_source_hashes[name] != digest:
                raise ValueError("Judge source differs from the original request implementation")
        current_source_hashes.update({str(source.relative_to(app_root)): _file_digest(source)
                                     for source in sorted((app_root / "evals").glob("*.py"))})
        current = settings.model_dump()
        controlled_settings = _CONTROLLED_SETTINGS
        for key, value in original["settings"].items():
            if not key.startswith("eval_judge_") or key in {"eval_judge_api_key", "eval_judge_base_url"}:
                raise ValueError("Unexpected setting in frozen Judge identity")
            if key == "eval_judge_cache_dir" and value:
                _independent(output_dir, Path(value).resolve())
            if key not in controlled_settings and current.get(key) != value:
                raise ValueError("A non-budget Judge setting differs from the original replay")
        model = settings.eval_judge_model or settings.llm_model
        base_url = settings.eval_judge_base_url or settings.llm_base_url
        api_key = settings.eval_judge_api_key or settings.llm_api_key
        if not model or not base_url or not api_key:
            raise ValueError("Judge model, endpoint and API key must be configured")
        if model != original["model"] or _digest(base_url) != original["endpoint_sha256"]:
            raise ValueError("Judge model or endpoint differs from the original replay")
        plan = ExecutableEvalPlan.model_validate(plan_data)
        if plan.case_id != provenance["case_id"] or metadata["case_id"] != plan.case_id:
            raise ValueError("Frozen case identities disagree")
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        raise ValueError("Frozen replay identity is incomplete or invalid") from exc
    except ValueError as exc:
        # Do not expose source data via Pydantic validation details.
        if type(exc) is not ValueError:
            raise ValueError("Frozen compiled plan or identity is invalid") from exc
        raise
    probe_settings = settings.model_copy(update={"eval_judge_concurrency": 1,
        "eval_judge_transport_retries": 0, "eval_judge_cache_dir": None})
    tasks = JudgeTaskPlanner().plan(plan, payload)
    by_id = {task.id: task for task in tasks}
    selected, requests = [], {}
    for identifier in task_ids:
        task = by_id.get(identifier)
        if task is None:
            raise ValueError("Selected task is not an initial task in the current frozen plan")
        path = invocation_dir / "tasks" / sha256(identifier.encode()).hexdigest()[:24] / "request.json"
        if not use_current_tasks and _read(path) != _request(task):
            raise ValueError("Current task request differs from its recorded frozen request")
        requests[identifier] = {"recorded_request_sha256": _file_digest(path) if path.exists() else None,
            "request_sha256": _digest(_request(task)), "schema_version": task.schema_version,
            "kind": task.kind, "stage": task.stage}
        selected.append(task)
    request_mode = "current_tasks" if use_current_tasks else "recorded_request"
    effective_settings = probe_settings.model_dump(mode="json")
    settings_delta = {key: {"recorded": value, "current": effective_settings[key]}
                      for key, value in original["settings"].items() if value != effective_settings[key]}
    identity = {"probe_version": PROBE_VERSION, "request_mode": request_mode,
        "original_manifest_sha256": _file_digest(manifest_path), "plan": plan_data, "payload": payload,
        "current_source_sha256": current_source_hashes, "settings_delta": settings_delta}
    manifest = {"probe_version": PROBE_VERSION, "invocation_dir": str(invocation_dir),
        "request_mode": request_mode,
        "original_manifest_sha256": identity["original_manifest_sha256"],
        "frozen_input_sha256": {"frozen_payload.json": _file_digest(payload_path),
                                "frozen_plan.json": _file_digest(plan_path)},
        "selected_tasks": requests, "model": model, "endpoint_sha256": _digest(base_url),
        "recorded_source_sha256": original["judge_source_sha256"],
        "current_source_sha256": current_source_hashes, "settings_delta": settings_delta,
        "budget": {"timeout_seconds": probe_settings.eval_judge_timeout_seconds,
                   "max_output_tokens": probe_settings.eval_judge_max_output_tokens},
        "concurrency": 1, "transport_retries": 0, "cache": None,
        "agent_runs": 0, "database_access": False, "case_scoring": False}
    return PreparedProbe(selected, probe_settings, model, base_url, api_key, identity, manifest)


class TaskProbeScheduler(RecordingScheduler):
    async def _run_one(self, task, identity):
        self.trace.register(task)
        wrapped = replace(task, validate=lambda raw: self.trace.validate(task, raw))
        # RecordingScheduler._attempt and RecordingClient retain the original
        # replay trace; only the task-level stdout status differs.
        result = await JudgeScheduler._run_one(self, wrapped, identity)
        _write(self.trace.task_path(task.id) / "result.json", result)
        print(json.dumps({"task_id": task.id,
            "status": "TASK_FAILED" if "execution_error" in result else "TASK_VALIDATED",
            "error": result.get("execution_error"), "telemetry": result["telemetry"]},
            ensure_ascii=False), flush=True)
        return result


async def run_probe(invocation_dir: Path, task_ids: list[str], output_dir: Path,
                    settings: Settings, *, client_factory=None, use_current_tasks: bool = False) -> dict:
    prepared = prepare_probe(invocation_dir, task_ids, output_dir, settings,
                             use_current_tasks=use_current_tasks)
    output_dir = output_dir.resolve()
    # exist_ok=False also protects against another process creating this path
    # between the read-only preparation and the first write.
    output_dir.mkdir(parents=True, mode=0o700)
    output_dir.chmod(0o700)
    _write(output_dir / "manifest.json", prepared.manifest)
    _write(output_dir / "state.json", {"status": "running",
        "started_at": datetime.now(timezone.utc).isoformat()})
    trace = ReplayTrace(output_dir)
    factory = client_factory or AsyncOpenAI
    scheduler = TaskProbeScheduler(prepared.settings, model=prepared.model,
        base_url=prepared.base_url, api_key=prepared.api_key, counter=TokenCounter(prepared.model),
        trace=trace, client_factory=lambda **kwargs: RecordingClient(factory, trace, **kwargs))
    started = perf_counter()
    rows = await scheduler.run(prepared.tasks, identity=prepared.identity)
    summaries = [{"task_id": row["task_id"],
        "status": "TASK_FAILED" if "execution_error" in row else "TASK_VALIDATED",
        "error": row.get("execution_error"), "telemetry": row["telemetry"]} for row in rows]
    summary = {"status": "TASK_VALIDATED" if all(item["status"] == "TASK_VALIDATED"
        for item in summaries) else "TASK_FAILED", "tasks": summaries,
        "request_mode": prepared.manifest["request_mode"],
        "wall_seconds": perf_counter() - started, "output_dir": str(output_dir),
        "agent_runs": 0, "database_access": False, "case_scoring": False}
    _write(output_dir / "summary.json", summary)
    _write(output_dir / "state.json", {"status": summary["status"]})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocation-dir", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-current-tasks", action="store_true",
        help="Use current task source/prompts over the same current-contract frozen plan and payload")
    parser.add_argument("--prepare-only", action="store_true",
        help="Validate frozen identities and selected requests locally without writing output or calling a provider")
    args = parser.parse_args()
    try:
        settings = get_settings()
        if args.prepare_only:
            prepared = prepare_probe(args.invocation_dir, args.task_id, args.output_dir, settings,
                                     use_current_tasks=args.use_current_tasks)
            summary = {"status": "TASK_PREPARED", "task_ids": [task.id for task in prepared.tasks],
                       "request_mode": prepared.manifest["request_mode"], "provider_calls": 0,
                       "agent_runs": 0, "database_access": False, "case_scoring": False}
        else:
            summary = asyncio.run(run_probe(args.invocation_dir, args.task_id, args.output_dir, settings,
                                           use_current_tasks=args.use_current_tasks))
    except ValueError as exc:
        print(json.dumps({"status": "TASK_FAILED", "error": {"code": "TASK_PROBE_INPUT_INVALID",
            "detail": str(exc)}, "provider_calls": 0}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["status"] in {"TASK_VALIDATED", "TASK_PREPARED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
