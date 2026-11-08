"""File-only replay checks: no Agent, database or real provider is invoked."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings
from app.evals.compiler import CompiledCheck, ExecutableEvalPlan, ExtractionSpec, SemanticCheck
from app.evals.judge_replay import load_frozen_inputs, run_replay


ANSWER = "🔧 JobPilot 使用三层上下文体系；Java 17。"
SOURCE = "JobPilot 使用三层上下文体系。"
SECRET = "never-record-this-api-key"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    plan = ExecutableEvalPlan(
        case_id="opaque", suite="agent_tasks", task_type="profile_summary",
        checks=[
            CompiledCheck(id="terminal", gold_path="checks", evaluator_kind="terminal_completed", expected=True),
            CompiledCheck(id="sources", gold_path="checks", evaluator_kind="source_refs_valid", expected=True),
            CompiledCheck(id="capture", gold_path="checks", evaluator_kind="evidence_capture_complete", expected=True),
            CompiledCheck(id="fact", gold_path="gold.fact", evaluator_kind="semantic", expected="PASS"),
        ],
        semantic_checks=[SemanticCheck(id="fact", gold_path="gold.fact", statement="JobPilot structure")],
        extractions=[],
    ).model_dump(mode="json")
    row = {
        "case_id": "opaque", "suite": "agent_tasks", "compiled_plan": plan,
        "runtime": {"run_failed": False, "turns": ["frozen-turn"]},
        "result": {"outcome": "PASS", "check_results": [
            {"evaluator_kind": "terminal_completed", "actual": ["COMPLETED"], "result": "PASS"},
            {"evaluator_kind": "source_refs_valid", "actual": [], "result": "PASS"},
        ]},
        "observable": {
            "final_answer": ANSWER, "final_answers_by_step": {}, "fixture_source_refs": {},
            "step_turn_ids": {}, "step_source_refs": {}, "memory_before": [], "memory_after": [],
            "memory_events": [], "tool_events": [], "fault_events": [],
            "evidence": {"capture_complete": True, "errors": [], "items": [
                {"id": "raw", "origin_type": "knowledge_module", "content": SOURCE,
                 "visible_to_agent": True, "authoritative_for_grounding": True, "turn_id": "frozen-turn"},
                {"id": "hidden", "origin_type": "knowledge_module", "content": "private unseen text",
                 "visible_to_agent": False, "authoritative_for_grounding": True, "turn_id": "frozen-turn"},
                {"id": "summary", "origin_type": "assistant_summary", "content": "non-authoritative summary",
                 "visible_to_agent": True, "authoritative_for_grounding": False, "turn_id": "frozen-turn"},
            ]},
        },
    }
    metadata = tmp_path / "historical_metadata.json"
    dump(metadata, {"case_id": "opaque", "task": "frozen historical task",
                    "source_policy": {"fact_time_policy": "current_snapshot", "support": "actual_visible_sources",
                                      "allow_other_supported_material_claims": True},
                    "git_revision": "historical", "source_path": "frozen-case.yaml", "source_sha256": "abc"})
    dump(root / "compiled_plans.json", [plan])
    (root / "case_results.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return root, metadata, row


def options(**changes):
    return Settings(eval_judge_model="mock", eval_judge_base_url="https://mock.invalid/v1",
                    eval_judge_api_key=SECRET, eval_judge_timeout_seconds=2,
                    eval_judge_transport_retries=1, eval_judge_retry_backoff_seconds=0,
                    **changes)


class MockProvider:
    def __init__(self, *, verdict="PASS", error=None, unknown_id=False, malformed=False):
        self.verdict, self.error, self.unknown_id, self.malformed = verdict, error, unknown_id, malformed
        self.requests = []
        self.closed = 0

    def factory(self, **constructor):
        backend = self

        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)

            async def create(self, **kwargs):
                backend.requests.append(kwargs)
                data = json.loads(kwargs["messages"][-1]["content"])
                system = kwargs["messages"][0]["content"]
                contract = json.loads(system.split("\n", 1)[1])
                body = {"items": [{"id": check["id"], "result": backend.verdict,
                                  "reason": "Frozen source relation."}
                                 for check in contract["checks"]]}
                if contract.get("observations"):
                    body["observations"] = []
                    for spec in contract["observations"]:
                        spans = data["answer_spans"]
                        if spec["kind"] == "bullets":
                            items = [{"answer_span_id": span["id"]} for span in spans]
                        else:
                            items = [{"id": "q1", "answer_span_id": spans[0]["id"],
                                      "followup_span_ids": [spans[1]["id"]]},
                                     {"id": "q2", "answer_span_id": spans[2]["id"]}]
                        body["observations"].append({"id": spec["id"], "kind": spec["kind"], "items": items})
                if backend.unknown_id:
                    body["items"][0]["id"] = "unknown-check"
                if backend.malformed:
                    body = {"items": []}
                return MockStream(json.dumps(body, ensure_ascii=False), backend.error)

            async def close(self):
                backend.closed += 1

        return Client()


class MockStream:
    def __init__(self, raw, error):
        self.raw, self.error = raw, error

    async def __aiter__(self):
        yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
            finish_reason=None, delta=SimpleNamespace(reasoning_content="mock reasoning", content=None))])
        if self.error:
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
                finish_reason=None, delta=SimpleNamespace(content='{"partial":'))])
            raise self.error
        yield SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30), choices=[SimpleNamespace(
            finish_reason="stop", delta=SimpleNamespace(content=self.raw))])

    async def close(self):
        pass


def frozen_hashes(root, metadata):
    return {str(p): sha256(p.read_bytes()).hexdigest() for p in [*root.iterdir(), metadata]}


def test_load_uses_original_compiled_plan_observation_and_authority(frozen, monkeypatch):
    import app.evals.compiler
    import app.evals.datasets

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not load or recompile mutable Gold")

    monkeypatch.setattr(app.evals.compiler, "compile_case", forbidden)
    monkeypatch.setattr(app.evals.datasets, "load_dataset", forbidden)
    root, metadata, row = frozen
    before = frozen_hashes(root, metadata)
    plan, payload, context, provenance = load_frozen_inputs(root, "opaque", metadata)
    assert plan.model_dump(mode="json") == row["compiled_plan"]
    assert payload["final_answer"] == ANSWER and payload["task"] == "frozen historical task"
    assert payload["answer_spans"] == [{"id": "answer_1", "text": ANSWER}]
    assert [e["id"] for e in payload["evidence"]] == ["raw"]
    assert context.final_answer == ANSWER and context.data["turns"][0].status == "COMPLETED"
    assert provenance["answer_sha256"] == sha256(ANSWER.encode()).hexdigest()
    assert frozen_hashes(root, metadata) == before


@pytest.mark.parametrize("defect", ["plan_mismatch", "agent_failure", "capture_incomplete", "missing_terminal"])
def test_invalid_frozen_inputs_fail_before_provider(frozen, defect):
    root, metadata, row = frozen
    if defect == "plan_mismatch":
        plan = deepcopy(row["compiled_plan"])
        plan["semantic_checks"][0]["statement"] = "changed statement"
        dump(root / "compiled_plans.json", [plan])
    elif defect == "agent_failure":
        row["runtime"]["run_failed"] = True
    elif defect == "capture_incomplete":
        row["observable"]["evidence"]["capture_complete"] = False
    else:
        row["result"]["check_results"].pop(0)
    (root / "case_results.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_frozen_inputs(root, "opaque", metadata)


async def test_replay_preserves_direct_coverage_and_success_cache(frozen, tmp_path):
    root, metadata, row = frozen
    before = frozen_hashes(root, metadata)
    provider = MockProvider()
    output, cache = tmp_path / "output", tmp_path / "task-cache"
    result = await run_replay(root, "opaque", metadata, output, cache, options(), client_factory=provider.factory)
    invocation = Path(result["invocation_dir"])
    assert result["outcome"] == "PASS" and result["strict_contract_valid"]
    assert result["audit_status"] == "DIRECT_JUDGE_COMPLETE"
    assert result["wall_seconds"] > 0
    audit = json.loads((invocation / "audit.json").read_text())
    assert audit["semantic_check_ids"] == ["fact"] and audit["observation_ids"] == []
    assert audit["missing_semantic_check_ids"] == audit["missing_observation_ids"] == []
    assert audit["semantic_result_counts"] == {"PASS": 1}
    assert audit["coverage_method"] == "declared_check_and_observation_ids"
    assert "unicode_offsets" not in audit and "quantification_clues" not in audit
    assert result["telemetry"]["input_token_estimate"] > 0
    assert result["telemetry"]["provider_input_tokens"] == 100
    assert result["agent_runs"] == 0 and result["database_access"] is False
    assert frozen_hashes(root, metadata) == before
    assert stat.S_IMODE(output.stat().st_mode) == stat.S_IMODE(cache.stat().st_mode) == 0o700
    assert all(SECRET not in p.read_text() for p in output.rglob("*") if p.is_file())
    resumed_provider = MockProvider(error=AssertionError("Cache recovery must avoid provider"))
    resumed = await run_replay(root, "opaque", metadata, output, cache, options(),
                               client_factory=resumed_provider.factory, resume=True)
    assert resumed["outcome"] == "PASS" and resumed["pipeline_summary"]["cache_hits"] == 1
    assert resumed_provider.requests == [] and resumed["telemetry"]["input_token_estimate"] == 0
    assert frozen_hashes(root, metadata) == before


async def test_valid_semantic_fail_is_quality_fail_with_strict_contract(frozen, tmp_path):
    root, metadata, _ = frozen
    result = await run_replay(root, "opaque", metadata, tmp_path / "out", tmp_path / "cache",
                              options(), client_factory=MockProvider(verdict="FAIL").factory)
    assert result["outcome"] == "FAIL" and result["strict_contract_valid"]
    assert result["quality_fail_established"] and not result["error_categories"]


async def test_unknown_check_preserves_invalid_raw_and_no_quality_verdict(frozen, tmp_path):
    root, metadata, _ = frozen
    result = await run_replay(root, "opaque", metadata, tmp_path / "out", tmp_path / "cache",
                              options(), client_factory=MockProvider(unknown_id=True).factory)
    assert result["outcome"] == "ERROR" and not result["strict_contract_valid"]
    assert not result["quality_fail_established"] and result["audit_status"] == "NOT_ESTABLISHED"
    audit = json.loads((Path(result["invocation_dir"]) / "audit.json").read_text())
    assert audit["validation_errors"] and audit["missing_semantic_check_ids"] == ["fact"]
    validations = list(Path(result["invocation_dir"]).glob("tasks/*/validation_*.json"))
    assert any(not json.loads(p.read_text())["valid"] for p in validations)


async def test_midstream_network_failure_keeps_all_attempts_and_partial_trace(frozen, tmp_path):
    root, metadata, _ = frozen
    provider = MockProvider(error=httpx.RemoteProtocolError("peer closed stream"))
    result = await run_replay(root, "opaque", metadata, tmp_path / "out", tmp_path / "cache",
                              options(), client_factory=provider.factory)
    assert result["outcome"] == "BLOCKED" and not result["strict_contract_valid"]
    assert result["error_categories"] == {"provider_interruption": 1}
    assert not result["quality_fail_established"] and len(provider.requests) == 2
    invocation = Path(result["invocation_dir"])
    attempts = list(invocation.glob("tasks/*/attempt_*.json"))
    assert len(attempts) == 2
    assert all(json.loads(p.read_text())["raw"] == '{"partial":' for p in attempts)
    judge_result = json.loads((invocation / "judge_result.json").read_text())
    assert all(row["telemetry"]["attempt_count"] == 2 and "body" not in row for row in judge_result["subtasks"])
    assert not list((tmp_path / "cache").glob("*.json"))
    assert len(list(invocation.glob("tasks/*/stream_*.jsonl"))) == 2


@pytest.mark.parametrize("unsafe", ["output_in_source", "cache_in_source", "same_output_cache"])
async def test_replay_refuses_writes_into_frozen_run(frozen, tmp_path, unsafe):
    root, metadata, _ = frozen
    output, cache = tmp_path / "out", tmp_path / "cache"
    if unsafe == "output_in_source":
        output = root / "out"
    elif unsafe == "cache_in_source":
        cache = root / "cache"
    else:
        cache = output
    provider = MockProvider()
    with pytest.raises(ValueError, match="independent"):
        await run_replay(root, "opaque", metadata, output, cache, options(), client_factory=provider.factory)
    assert provider.requests == []


async def test_resume_rejects_changed_model_identity_before_provider(frozen, tmp_path):
    root, metadata, _ = frozen
    output, cache = tmp_path / "out", tmp_path / "cache"
    await run_replay(root, "opaque", metadata, output, cache, options(), client_factory=MockProvider().factory)
    provider = MockProvider()
    changed = options().model_copy(update={"eval_judge_max_output_tokens": 7000})
    with pytest.raises(ValueError, match="Resume identity changed"):
        await run_replay(root, "opaque", metadata, output, cache, changed,
                         client_factory=provider.factory, resume=True)
    assert provider.requests == []


async def test_schema_invalid_output_is_still_auditable(frozen, tmp_path):
    root, metadata, _ = frozen
    result = await run_replay(root, "opaque", metadata, tmp_path / "out", tmp_path / "cache",
                              options(), client_factory=MockProvider(malformed=True).factory)
    assert result["outcome"] == "ERROR" and not result["strict_contract_valid"]
    assert result["audit_status"] == "NOT_ESTABLISHED"


async def test_historical_contract_rejected_before_output_cache_or_provider(frozen, tmp_path):
    root, metadata, row = frozen
    row["compiled_plan"]["contract_version"] = "eval_contract_v2"
    dump(root / "compiled_plans.json", [row["compiled_plan"]])
    (root / "case_results.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    provider, output, cache = MockProvider(), tmp_path / "new-output", tmp_path / "new-cache"
    with pytest.raises(ValueError, match="Incompatible frozen evaluation contract"):
        await run_replay(root, "opaque", metadata, output, cache, options(), client_factory=provider.factory)
    assert provider.requests == [] and not output.exists() and not cache.exists()


@pytest.mark.parametrize("kind", ["bullets", "questions"])
@pytest.mark.parametrize("forged", [False, True])
async def test_replay_validates_span_only_wire_and_checks_code_restored_text(
        frozen, tmp_path, monkeypatch, kind, forged):
    from app.evals.judge import StructuredJudge

    root, metadata, row = frozen
    lines = (["- First delivered bullet.", "- Second delivered bullet."] if kind == "bullets" else
             ["1. What changed?", "- What was measured?", "2. How do retries stop?"])
    row["observable"]["final_answer"] = "\n".join(lines)
    compiled = ExecutableEvalPlan.model_validate(row["compiled_plan"])
    compiled.extractions = [ExtractionSpec(id="delivered", kind=kind, instructions="Identify every delivered item.")]
    compiled.checks.append(CompiledCheck(id="shape", gold_path="gold.shape",
        evaluator_kind="bullet_count" if kind == "bullets" else "question_limits",
        expected=({"extraction_id": "delivered", "max": 2} if kind == "bullets" else
                  {"extraction_id": "delivered", "main_questions": 2, "max_followups_per_question": 1})))
    row["compiled_plan"] = compiled.model_dump(mode="json")
    dump(root / "compiled_plans.json", [row["compiled_plan"]])
    (root / "case_results.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    if forged:
        original = StructuredJudge.judge

        async def altered(self, **kwargs):
            result = await original(self, **kwargs)
            item = result["extractions"][0]["items"][0]
            item["text" if kind == "bullets" else "followups"] = (
                "Invented delivered text" if kind == "bullets" else ["Invented follow-up"])
            return result

        monkeypatch.setattr(StructuredJudge, "judge", altered)
    result = await run_replay(root, "opaque", metadata, tmp_path / "out", tmp_path / "cache",
                              options(), client_factory=MockProvider().factory)
    invocation = Path(result["invocation_dir"])
    body = json.loads((invocation / "judge_result.json").read_text())
    strict = json.loads((invocation / "strict_validation.json").read_text())
    assert result["strict_contract_valid"] is not forged
    if forged:
        assert result["outcome"] == "ERROR" and not result["quality_fail_established"]
        assert "original-observation text restoration" in strict["detail"]
    else:
        assert result["outcome"] == "PASS"
        item = body["extractions"][0]["items"][0]
        assert item["text"] == ("First delivered bullet." if kind == "bullets" else lines[0])
        if kind == "questions":
            assert item["followups"] == [lines[1]]
        wire_item = strict["response"]["extractions"][0]["items"][0]
        assert "text" not in wire_item and "followups" not in wire_item
