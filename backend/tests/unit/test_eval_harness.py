from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.agent.loop import ThinAgentLoop
from app.evals.datasets import DatasetError, load_dataset
from app.evals.evaluators import (
    EvalClassification,
    EvalFailureClassifier,
    EvalOutcome,
    case_result,
    summarize,
)
from app.evals.faults import ToolFaultPlan
from app.evals.runner import EvalRunner
from app.evals.retrieval_contract import METRIC_NAMES, RETRIEVAL_CONTRACT_VERSION


def test_dataset_is_the_frozen_27_case_contract() -> None:
    dataset = load_dataset()

    assert len(dataset.cases) == 27
    assert {case.suite for case in dataset.cases.values()} == {
        "retrieval", "agent_tasks", "memory", "runtime_failure"
    }
    assert len(dataset.manifest.frozen_documents) == 6


def test_retrieval_case_contains_task_requirements_separate_from_protocol_and_evidence() -> None:
    dataset = load_dataset()
    cases = dataset.by_suite("retrieval")
    assert len(cases) == 9
    assert all(case.facts for case in cases)
    assert not {"gold", "checks", "setup", "source_policy", "pass_policy", "max_modules", "max_tokens"} & set(cases[0].model_dump())
    assert dataset.retrieval_evidence.evidence
    assert dataset.retrieval_protocol.retrieval_k == 5
    assert dataset.retrieval_protocol.max_context_tokens == 8000
    assert set(dataset.retrieval_protocol.metrics) == set(METRIC_NAMES)


def test_retrieval_summary_averages_answerable_metrics_without_turning_scores_into_passes() -> None:
    results = [{"case_id": f"query-{index}", "suite": "retrieval", "result": {
        "outcome": "SCORED", "answerability": "answerable",
        "metrics": dict.fromkeys(METRIC_NAMES, value),
    }} for index, value in enumerate([0.0, 1.0])]
    results.append({"case_id": "no-corpus-proof", "suite": "retrieval", "result": {
        "outcome": "SCORED", "answerability": "unanswerable",
        "metrics": dict.fromkeys(METRIC_NAMES, None),
    }})
    summary = summarize(results)
    retrieval = summary["suites"]["retrieval"]
    assert summary["run_status"] == "COMPLETED"
    assert retrieval["scored"] == retrieval["scorable_case_count"] == 3
    assert retrieval["passed"] == retrieval["failed"] == 0
    assert retrieval["pass_rate"] is None
    assert retrieval["answerable_case_count"] == 2 and retrieval["unanswerable_case_count"] == 1
    assert retrieval["metrics"] == dict.fromkeys(METRIC_NAMES, 0.5)
    assert retrieval["abstention"] == "NOT_EVALUATED"



def _runtime_data(
    *, model_error_codes: tuple[str, ...] = (), turn_error_codes: tuple[str, ...] = (), tool_error_codes: tuple[str, ...] = ()
) -> dict[str, object]:
    return {
        "model_calls": [SimpleNamespace(status="FAILED", error_code=code) for code in model_error_codes],
        "turns": [SimpleNamespace(error_code=code) for code in turn_error_codes],
        "tool_calls": [SimpleNamespace(status="FAILED", error_code=code) for code in tool_error_codes],
    }


def test_provider_402_blocks_the_case_and_is_excluded_from_pass_rate() -> None:
    class ProviderError(Exception):
        status_code = 402

    assert ThinAgentLoop._error_code(ProviderError()) == "PROVIDER_HTTP_402"
    assert ThinAgentLoop._error_code(asyncio.TimeoutError()) == "MODEL_TIMEOUT"
    assert ThinAgentLoop._error_code(OSError("DNS lookup failed")) == "PROVIDER_CONNECTION_ERROR"
    classification = EvalFailureClassifier.agent_interruption(
        _runtime_data(model_error_codes=("PROVIDER_HTTP_402",))
    )

    assert classification == EvalClassification(EvalOutcome.BLOCKED, "MODEL_PROVIDER_BILLING", "MAIN_AGENT")


@pytest.mark.parametrize(
    ("error_code", "reason"),
    [
        ("MODEL_TIMEOUT", "MODEL_PROVIDER_TIMEOUT"),
        ("PROVIDER_HTTP_429", "MODEL_PROVIDER_RATE_LIMITED"),
        ("PROVIDER_HTTP_503", "MODEL_PROVIDER_UNAVAILABLE"),
        ("PROVIDER_CONNECTION_ERROR", "MODEL_PROVIDER_UNAVAILABLE"),
    ],
)
def test_model_provider_interruptions_block_the_case(error_code: str, reason: str) -> None:
    classification = EvalFailureClassifier.agent_interruption(
        _runtime_data(model_error_codes=(error_code,))
    )

    assert classification == EvalClassification(EvalOutcome.BLOCKED, reason, "MAIN_AGENT")


@pytest.mark.parametrize(
    ("error_code", "reason"),
    [
        ("TURN_LIMIT", "AGENT_TURN_LIMIT"),
        ("TOOL_LIMIT", "AGENT_TOOL_LIMIT"),
        ("CONTEXT_OVERFLOW", "AGENT_CONTEXT_OVERFLOW"),
        ("DEADLINE_EXCEEDED", "AGENT_TOTAL_DEADLINE"),
    ],
)
def test_agent_budget_and_context_failures_remain_capability_failures(error_code: str, reason: str) -> None:
    data = _runtime_data(turn_error_codes=(error_code,))

    assert EvalFailureClassifier.agent_interruption(data) is None
    assert EvalFailureClassifier.agent_failure(data) == EvalClassification(EvalOutcome.FAIL, reason, "MAIN_AGENT")


def test_recorded_provider_interruption_overrides_a_terminal_deadline() -> None:
    classification = EvalFailureClassifier.agent_interruption(
        _runtime_data(model_error_codes=("MODEL_TIMEOUT",), turn_error_codes=("DEADLINE_EXCEEDED",))
    )

    assert classification == EvalClassification(EvalOutcome.BLOCKED, "MODEL_PROVIDER_TIMEOUT", "MAIN_AGENT")


def test_invalid_tool_arguments_are_an_agent_failure_when_the_case_does_not_recover() -> None:
    data = _runtime_data(tool_error_codes=("VALIDATION_ERROR",))

    assert EvalFailureClassifier.agent_interruption(data) is None
    assert EvalFailureClassifier.agent_failure(data) == EvalClassification(
        EvalOutcome.FAIL, "AGENT_TOOL_ARGUMENT_ERROR", "MAIN_AGENT"
    )


def test_judge_provider_timeout_blocks_but_a_normal_judge_fail_is_a_fail() -> None:
    blocked = EvalFailureClassifier.judge_interruption(
        {"execution_error": {"code": "JUDGE_TIMEOUT"}}
    )
    normal_verdict = EvalFailureClassifier.judge_interruption({"checks": [{"id": "atomic", "result": "FAIL"}], "extractions": []})
    failed = case_result(
        EvalFailureClassifier.pass_or_fail(False, failure_reason="JUDGE_REJECTED", source="JUDGE")
    )

    assert blocked == EvalClassification(EvalOutcome.BLOCKED, "JUDGE_PROVIDER_TIMEOUT", "JUDGE")
    assert normal_verdict is None
    assert failed == {"outcome": "FAIL", "reason": "JUDGE_REJECTED", "source": "JUDGE"}


def test_harness_failure_is_an_error_not_an_agent_failure() -> None:
    result = EvalRunner._harness_case_result(
        load_dataset().cases["A1"].model_copy(update={"id": "A-fixture"}),
        DatasetError("fixture reference cannot be resolved"),
    )["result"]

    assert result == {
        "outcome": "ERROR",
        "reason": "EVAL_DATASET_ERROR",
        "source": "EVAL_HARNESS",
        "harness_error": "DatasetError",
    }


def test_summary_excludes_blocked_and_error_cases_from_the_pass_rate() -> None:
    results = [
        {
            "case_id": f"A{index}",
            "suite": "agent_tasks",
            "result": case_result(EvalClassification(EvalOutcome.PASS)),
        }
        for index in range(1, 8)
    ]
    results.extend(
        [
            {
                "case_id": "A8",
                "suite": "agent_tasks",
                "result": case_result(EvalClassification(EvalOutcome.FAIL, "AGENT_TASK_FAILED", "MAIN_AGENT")),
            },
            {
                "case_id": "A9",
                "suite": "agent_tasks",
                "result": case_result(EvalClassification(EvalOutcome.FAIL, "JUDGE_REJECTED", "JUDGE")),
            },
            {
                "case_id": "A10",
                "suite": "agent_tasks",
                "result": case_result(EvalClassification(EvalOutcome.BLOCKED, "MODEL_PROVIDER_TIMEOUT", "MAIN_AGENT")),
            },
            {
                "case_id": "A11",
                "suite": "agent_tasks",
                "result": case_result(EvalClassification(EvalOutcome.ERROR, "EVAL_FIXTURE_ERROR", "EVAL_HARNESS")),
            },
        ]
    )

    summary = summarize(results)
    outcome = summary["suites"]["agent_tasks"]

    assert summary["case_count"] == 11
    assert summary["run_status"] == "ERROR"
    assert outcome == {
        "case_count": 11,
        "scorable_case_count": 9,
        "passed": 7,
        "failed": 2,
        "blocked_case_count": 1,
        "error_case_count": 1,
        "pass_rate": 7 / 9,
    }


def test_eval_artifacts_record_pinned_profile_instead_of_global_model(tmp_path) -> None:
    import json

    from app.evals.runner import RunOptions, write_artifacts

    runner = object.__new__(EvalRunner)
    runner.options = RunOptions(embedding_profile="candidate")
    runner.embedding_profile = SimpleNamespace(id=uuid4(), name="candidate", model="different-model", dimension=1024)
    summary = {"run_status": "COMPLETED", "outcomes": {}, "suites": {}, "runtime": {}, "case_count": 1}
    results = [{"case_id": "R1", "suite": "retrieval", "result": {
        "outcome": "SCORED", "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
        "answerability": "answerable", "metrics": dict.fromkeys(METRIC_NAMES, 1.0),
    }}]
    judges = [{"case_id": "R1", "checks": [], "extractions": []}]

    runner._bind_metadata(summary, results, judges)
    settings = SimpleNamespace(llm_model="agent", eval_judge_model=None, eval_judge_max_output_tokens=8192)
    artifact = write_artifacts(
        package_root=tmp_path, run_id=uuid4(), summary=summary, results=results,
        judges=judges, settings=settings, embedding_profile_metadata=runner.profile_metadata(),
    )

    expected = {
        "embedding_profile_id": str(runner.embedding_profile.id), "embedding_profile_name": "candidate",
        "embedding_model": "different-model", "embedding_dimension": 1024,
    }
    for path in ("run.json", "summary.json", "case_results.jsonl", "judge_results.jsonl"):
        data = json.loads((artifact / path).read_text())
        assert {key: data[key] for key in expected} == expected


def test_eval_profile_resolution_failure_records_unknown_identity_without_guessing() -> None:
    from app.evals.runner import RunOptions

    runner = object.__new__(EvalRunner)
    runner.options = RunOptions(embedding_profile="missing-profile")
    runner.embedding_profile = None

    assert runner.profile_metadata() == {
        "embedding_profile_id": None, "embedding_profile_name": "missing-profile",
        "embedding_model": None, "embedding_dimension": None,
    }



async def test_eval_rejects_same_space_with_different_grandfathered_profile_uuid_before_commit(monkeypatch) -> None:
    from dataclasses import asdict, replace

    import app.evals.datasets as datasets
    from app.embedding.profiles import ProfileDefinition, canonical_json

    initial = ProfileDefinition(
        name="same-legacy-label", base_url="https://source.example/v1", model="synthetic-model",
        dimension=1024, identity_version=1,
    )
    source_definition = replace(initial, legacy_identity_json=canonical_json(initial.legacy_payload()))
    other_initial = replace(initial, base_url="https://another-equivalent.example/v1")
    other_definition = replace(other_initial, legacy_identity_json=canonical_json(other_initial.legacy_payload()))
    assert source_definition.space_id == other_definition.space_id
    assert source_definition.id != other_definition.id
    source = SimpleNamespace(id=source_definition.id, **asdict(source_definition))

    class ProfileService:
        async def register(self, definition):
            return SimpleNamespace(id=other_definition.id)

        async def ensure_indexes(self, profile):
            raise AssertionError("UUID mismatch must fail before index mutation")

    monkeypatch.setattr(datasets, "EmbeddingProfileService", lambda *args: ProfileService())
    # A session without commit/write methods also proves no target commit occurs.
    with pytest.raises(DatasetError, match="profile UUID differs from source"):
        await datasets._copy_profile(object(), source, object())
