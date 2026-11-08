"""Frozen task probes validate exact requests without running final scoring."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.evals.compiler import CompiledCheck, ExecutableEvalPlan, SemanticCheck
from app.evals.judge_replay import _digest, _write
from app.evals.judge_task_probe import prepare_probe, run_probe
from app.evals.judge_tasks import JudgeTaskPlanner

SECRET = "probe-never-record-this-api-key"
ANSWER = "Falcon 使用三层上下文；Java 17。"


def options(**changes):
    values = dict(eval_judge_model="mock", eval_judge_base_url="https://mock.invalid/v1",
        eval_judge_api_key=SECRET, eval_judge_timeout_seconds=2,
        eval_judge_max_output_tokens=8192, eval_judge_concurrency=2,
        eval_judge_transport_retries=1)
    return Settings(**(values | changes))


@pytest.fixture
def frozen_tasks(tmp_path):
    replay = tmp_path / "replay"
    invocation = replay / "invocations" / "001"
    run = tmp_path / "original_run"
    run.mkdir()
    for name in ("case_results.jsonl", "compiled_plans.json"):
        (run / name).write_text("{}\n")
    plan = ExecutableEvalPlan(case_id="opaque", suite="agent_tasks", task_type="profile_summary",
        checks=[CompiledCheck(id="fact", gold_path="gold.fact", evaluator_kind="semantic", expected="PASS"),
                CompiledCheck(id="prohibition", gold_path="gold.prohibition", evaluator_kind="semantic", expected="PASS")],
        semantic_checks=[
            SemanticCheck(id="fact", gold_path="gold.fact", statement="The context has three layers"),
            SemanticCheck(id="prohibition", gold_path="gold.prohibition", statement="No nonexistent job",
                          support_mode="answer_only"),
        ], extractions=[])
    payload = {"task": "Describe observed project work", "source_policy": {
        "fact_time_policy": "current_snapshot", "support": "actual_visible_sources",
        "allow_other_supported_material_claims": True}, "final_answer": ANSWER,
        "answer_spans": [{"id": "answer_1", "text": ANSWER}], "memories": [],
        "evidence": [{"id": "raw", "content": "Falcon 使用三层上下文。",
            "visible_to_agent": True, "authoritative_for_grounding": True}]}
    settings = options()
    plan_data = plan.model_dump(mode="json")
    _write(invocation / "frozen_payload.json", payload)
    _write(invocation / "frozen_plan.json", plan_data)
    app_root = Path(__file__).resolve().parents[2] / "app"
    manifest = {"model": settings.eval_judge_model, "endpoint_sha256": _digest(settings.eval_judge_base_url),
        "provenance": {"case_id": "opaque", "run_dir": str(run),
            "metadata": {"case_id": "opaque", "task": payload["task"],
                         "source_policy": payload["source_policy"]},
            "payload_sha256": _digest(payload), "plan_sha256": _digest(plan_data),
            "answer_sha256": sha256(ANSWER.encode()).hexdigest(),
            "evidence_sha256": _digest(payload["evidence"]),
            "input_file_sha256": {name: sha256((run / name).read_bytes()).hexdigest()
                                  for name in ("case_results.jsonl", "compiled_plans.json")}},
        "judge_source_sha256": {str(path.relative_to(app_root)): sha256(path.read_bytes()).hexdigest()
                                for path in (app_root / "evals").glob("*.py")},
        "settings": {key: value for key, value in settings.model_dump(mode="json").items()
                     if key.startswith("eval_judge_") and key not in {"eval_judge_api_key", "eval_judge_base_url"}}}
    _write(replay / "manifest.json", manifest)
    tasks = JudgeTaskPlanner().plan(plan, payload)
    for task in tasks:
        path = invocation / "tasks" / sha256(task.id.encode()).hexdigest()[:24] / "request.json"
        _write(path, {"task_id": task.id, "kind": task.kind, "stage": task.stage,
                      "schema_version": task.schema_version, "messages": task.messages})
    return invocation, tasks, settings


class FakeProvider:
    def __init__(self, *, finish_reason="stop", malformed=False):
        self.requests, self.constructors = [], []
        self.finish_reason, self.malformed = finish_reason, malformed

    def factory(self, **constructor):
        self.constructors.append(constructor)
        provider = self

        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)

            async def create(self, **kwargs):
                provider.requests.append(kwargs)
                data = json.loads(kwargs["messages"][-1]["content"])
                contract = json.loads(kwargs["messages"][0]["content"].split("\n", 1)[1])
                body = {"items": [{"id": check["id"], "result": "PASS",
                                  "reason": "The requested check was judged."}
                                 for check in contract["checks"]]}
                if provider.malformed:
                    body = {"items": []}
                return FakeStream(json.dumps(body, ensure_ascii=False), provider.finish_reason)

            async def close(self):
                pass

        return Client()


class FakeStream:
    def __init__(self, raw, finish_reason):
        self.raw, self.finish_reason = raw, finish_reason

    async def __aiter__(self):
        yield SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=self.finish_reason,
            delta=SimpleNamespace(content=self.raw, reasoning_content="synthetic reason"))])

    async def close(self):
        pass


async def test_only_explicit_selected_initial_tasks_run_with_original_validators(frozen_tasks, tmp_path,
                                                                                monkeypatch, capsys):
    import app.evals.executor
    import app.evals.judge

    def forbidden(*args, **kwargs):
        raise AssertionError("A probe must not run complete Judge or case scoring")

    monkeypatch.setattr(app.evals.judge.StructuredJudge, "judge", forbidden)
    monkeypatch.setattr(app.evals.executor, "execute_plan", forbidden)
    invocation, tasks, settings = frozen_tasks
    selected = tasks
    provider, output = FakeProvider(), tmp_path / "probe"
    summary = await run_probe(invocation, [task.id for task in selected], output, settings,
                              client_factory=provider.factory)
    assert summary["status"] == "TASK_VALIDATED" and not summary["case_scoring"]
    assert summary["request_mode"] == "recorded_request"
    assert summary["agent_runs"] == 0 and summary["database_access"] is False
    assert [r["messages"] for r in provider.requests] == [t.messages for t in selected]
    assert len(provider.requests) == 2 and all(r["max_tokens"] == 8192 for r in provider.requests)
    assert all(r["max_retries"] == 0 for r in provider.constructors)
    assert all(r["telemetry"]["retry_count"] == 0 and not r["telemetry"]["cache_hit"] for r in summary["tasks"])
    validations = [json.loads(path.read_text()) for path in output.rglob("validation_01.json")]
    assert len(validations) == 2 and all(item["valid"] for item in validations)
    assert validations[0]["body"] and validations[1]["body"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.rglob("*") if path.is_file())
    assert all(SECRET not in path.read_text() for path in output.rglob("*") if path.is_file())
    stdout = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {row["status"] for row in stdout} == {"TASK_VALIDATED"}
    assert all("outcome" not in row for row in stdout)


async def test_even_complete_json_with_length_finish_is_failed_without_validation(frozen_tasks, tmp_path):
    invocation, _, settings = frozen_tasks
    provider, output = FakeProvider(finish_reason="length"), tmp_path / "truncated"
    summary = await run_probe(invocation, ["semantic:prohibition"], output, settings,
                              client_factory=provider.factory)
    assert summary["status"] == "TASK_FAILED"
    assert summary["tasks"][0]["error"]["code"] == "JUDGE_OUTPUT_TRUNCATED"
    assert summary["tasks"][0]["telemetry"]["finish_reason"] == "length"
    assert len(provider.requests) == 1 and not list(output.rglob("validation_*.json"))
    result = json.loads(next(output.rglob("result.json")).read_text())
    assert "body" not in result


async def test_complete_response_still_must_pass_original_task_validator(frozen_tasks, tmp_path):
    invocation, _, settings = frozen_tasks
    output = tmp_path / "malformed"
    summary = await run_probe(invocation, ["semantic:prohibition"], output, settings,
                              client_factory=FakeProvider(malformed=True).factory)
    assert summary["status"] == "TASK_FAILED"
    assert summary["tasks"][0]["error"]["code"] == "JUDGE_SCHEMA_ERROR"
    assert json.loads(next(output.rglob("validation_01.json")).read_text())["valid"] is False


async def test_one_changed_recorded_request_blocks_all_targets_before_client(frozen_tasks, tmp_path):
    invocation, tasks, settings = frozen_tasks
    selected = tasks
    changed = selected[1]
    path = invocation / "tasks" / sha256(changed.id.encode()).hexdigest()[:24] / "request.json"
    body = json.loads(path.read_text())
    body["messages"][-1]["content"] = "changed frozen input"
    _write(path, body)
    provider, output = FakeProvider(), tmp_path / "mismatch"
    with pytest.raises(ValueError, match="differs from its recorded"):
        await run_probe(invocation, [task.id for task in selected], output, settings,
                        client_factory=provider.factory)
    assert provider.requests == provider.constructors == [] and not output.exists()


@pytest.mark.parametrize("use_current_tasks", [False, True])
@pytest.mark.parametrize("defect", ["payload", "plan", "original_run", "metadata", "model", "endpoint", "temperature"])
async def test_frozen_identity_or_nonbudget_setting_drift_blocks_provider(frozen_tasks, tmp_path, defect,
                                                                       use_current_tasks):
    invocation, _, settings = frozen_tasks
    if defect in {"payload", "plan"}:
        path = invocation / ("frozen_payload.json" if defect == "payload" else "frozen_plan.json")
        body = json.loads(path.read_text())
        body["final_answer" if defect == "payload" else "case_id"] = "changed"
        _write(path, body)
    elif defect == "original_run":
        (tmp_path / "original_run" / "compiled_plans.json").write_text("changed")
    elif defect == "metadata":
        path = invocation.parent.parent / "manifest.json"
        body = json.loads(path.read_text())
        body["provenance"]["metadata"]["task"] = "changed task"
        _write(path, body)
    elif defect == "model":
        settings = settings.model_copy(update={"eval_judge_model": "other"})
    elif defect == "endpoint":
        settings = settings.model_copy(update={"eval_judge_base_url": "https://other.invalid/v1"})
    else:
        settings = settings.model_copy(update={"eval_judge_temperature": 0.5})
    provider, output = FakeProvider(), tmp_path / "changed"
    with pytest.raises(ValueError):
        await run_probe(invocation, ["semantic:prohibition"], output, settings, client_factory=provider.factory,
                        use_current_tasks=use_current_tasks)
    assert provider.constructors == [] and not output.exists()


def test_budget_can_change_but_recovery_and_overwriting_are_forbidden(frozen_tasks, tmp_path):
    invocation, _, settings = frozen_tasks
    output = tmp_path / "probe"
    prepared = prepare_probe(invocation, ["semantic:prohibition"], output,
        settings.model_copy(update={"eval_judge_timeout_seconds": 600, "eval_judge_max_output_tokens": 16384,
            "eval_judge_cache_dir": tmp_path / "ignored_cache", "eval_judge_concurrency": 8,
            "eval_judge_transport_retries": 5}))
    assert prepared.settings.eval_judge_concurrency == 1
    assert prepared.settings.eval_judge_transport_retries == 0
    assert prepared.settings.eval_judge_cache_dir is None
    assert prepared.manifest["budget"] == {"timeout_seconds": 600, "max_output_tokens": 16384}
    output.mkdir()
    (output / "keep.txt").write_text("preserved")
    with pytest.raises(ValueError, match="already exists"):
        prepare_probe(invocation, ["semantic:prohibition"], output, settings)
    assert (output / "keep.txt").read_text() == "preserved"


def test_source_change_requires_explicit_current_tasks_mode(frozen_tasks, tmp_path, monkeypatch):
    import app.evals.judge_task_probe as probe

    invocation, _, settings = frozen_tasks
    source = Path(probe.__file__).parent / "judge_tasks.py"
    old_digest = probe._file_digest(source)
    real_digest = probe._file_digest
    new_digest = "a" * 64
    monkeypatch.setattr(probe, "_file_digest", lambda path: new_digest if path == source else real_digest(path))
    with pytest.raises(ValueError, match="source differs"):
        prepare_probe(invocation, ["semantic:prohibition"], tmp_path / "default", settings)
    prepared = prepare_probe(invocation, ["semantic:prohibition"], tmp_path / "opt_in", settings,
                             use_current_tasks=True)
    assert prepared.manifest["request_mode"] == "current_tasks"
    assert prepared.manifest["recorded_source_sha256"]["evals/judge_tasks.py"] == old_digest
    assert prepared.manifest["current_source_sha256"]["evals/judge_tasks.py"] == new_digest
    assert prepared.manifest["case_scoring"] is False


def test_current_tasks_can_use_new_messages_but_default_still_rejects(frozen_tasks, tmp_path):
    invocation, tasks, settings = frozen_tasks
    selected = next(task for task in tasks if task.id == "semantic:prohibition")
    path = invocation / "tasks" / sha256(selected.id.encode()).hexdigest()[:24] / "request.json"
    old = json.loads(path.read_text())
    old["schema_version"] = "historical_schema"
    old["messages"][0]["content"] = "historical prompt"
    _write(path, old)
    with pytest.raises(ValueError, match="differs from its recorded"):
        prepare_probe(invocation, [selected.id], tmp_path / "default", settings)
    prepared = prepare_probe(invocation, [selected.id], tmp_path / "current", settings, use_current_tasks=True)
    assert prepared.tasks[0].messages == selected.messages
    assert prepared.manifest["selected_tasks"][selected.id]["recorded_request_sha256"] == sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("use_current_tasks", [False, True])
def test_source_path_safety_is_preserved_in_both_modes(frozen_tasks, tmp_path, use_current_tasks):
    invocation, _, settings = frozen_tasks
    path = invocation.parent.parent / "manifest.json"
    original = json.loads(path.read_text())
    original["judge_source_sha256"] = {"config.py": "a" * 64}
    _write(path, original)
    with pytest.raises(ValueError, match="source path is outside"):
        prepare_probe(invocation, ["semantic:prohibition"], tmp_path / "probe", settings,
                      use_current_tasks=use_current_tasks)


@pytest.mark.parametrize("use_current_tasks", [False, True])
async def test_historical_contract_blocks_before_removed_source_hash_or_output(
        frozen_tasks, tmp_path, monkeypatch, use_current_tasks):
    import app.evals.judge_task_probe as probe

    invocation, _, settings = frozen_tasks
    plan_path = invocation / "frozen_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["contract_version"] = "eval_contract_v2"
    _write(plan_path, plan)
    manifest_path = invocation.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["plan_sha256"] = _digest(plan)
    manifest["judge_source_sha256"] = {"evals/judge_claims.py": "historical-deleted-source"}
    _write(manifest_path, manifest)

    def forbidden(path):
        raise AssertionError("Old plan must be rejected before deleted source file hash lookup")

    monkeypatch.setattr(probe, "_file_digest", forbidden)
    provider, output = FakeProvider(), tmp_path / "historical-output"
    with pytest.raises(ValueError, match="Incompatible frozen evaluation contract"):
        await run_probe(invocation, ["semantic:prohibition"], output, settings,
                        client_factory=provider.factory, use_current_tasks=use_current_tasks)
    assert provider.constructors == [] and not output.exists()


def test_prepare_only_cli_validates_locally_without_output_or_provider(frozen_tasks, tmp_path, monkeypatch, capsys):
    import app.evals.judge_task_probe as probe

    invocation, _, settings = frozen_tasks
    output = tmp_path / "prepared-only"
    monkeypatch.setattr(probe, "get_settings", lambda: settings)
    monkeypatch.setattr("sys.argv", ["judge_task_probe", "--invocation-dir", str(invocation),
        "--task-id", "semantic:prohibition", "--output-dir", str(output), "--prepare-only"])

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepare-only must not create a provider")

    monkeypatch.setattr(probe, "AsyncOpenAI", forbidden)
    assert probe.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "TASK_PREPARED" and summary["provider_calls"] == 0
    assert summary["case_scoring"] is False and not output.exists()
