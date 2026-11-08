from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.evals import comparison as comparison_module
from app.evals.comparison import COMPARISON_VERSION, compare_runs, comparison_contract, comparison_report, retrieval_case_diff
from app.evals.datasets import load_dataset
from app.evals.retrieval_contract import METRIC_NAMES, RETRIEVAL_CONTRACT_VERSION, RetrievalProtocol


def _module(section: str, rank: int) -> dict:
    return {
        "section_id": section, "rank": rank,
        "logical_document_key": "document", "section_path": [section],
        "source_refs": [f"document://document/section/{section}"],
    }


def _result(*, unanswerable=False, metrics=None, evidence=None, facts=None, modules=None):
    return {
        "outcome": "SCORED", "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
        "answerability": "unanswerable" if unanswerable else "answerable",
        "metrics": metrics if metrics is not None else dict.fromkeys(METRIC_NAMES, None if unanswerable else 1.0),
        "covered_evidence_ids": evidence or [], "satisfied_fact_ids": facts or [],
        "missing_fact_ids": [], "fact_witnesses": {}, "modules": modules or [],
    }


def test_retrieval_diff_preserves_gold_and_reports_coverage_facts_and_ranks() -> None:
    before = {"case_id": "R8", "result": _result(
        evidence=["eu.school", "eu.stage"], facts=["school", "stage"],
        modules=[_module("education", 1), _module("internship", 2)],
    )}
    after = {"case_id": "R8", "result": _result(
        metrics={"evidence_recall": 0.5, "mrr": 0.5, "fact_recall": 0.5, "complete_support": 0.0},
        evidence=["eu.school"], facts=["school"],
        modules=[_module("distractor", 1), _module("education", 2)],
    )}
    original = json.dumps([before, after], sort_keys=True)

    diff = retrieval_case_diff(before, after)

    assert diff["comparison_scorable"] and diff["metrics_applicable"]
    assert diff["covered_evidence_lost"] == ["eu.stage"]
    assert diff["satisfied_facts_lost"] == ["stage"]
    assert diff["metric_deltas"] == {"evidence_recall": -0.5, "mrr": -0.5, "fact_recall": -0.5, "complete_support": -1.0}
    assert any(item["baseline_rank"] == 1 and item["candidate_rank"] == 2 for item in diff["ranking_changes"])
    assert not {"new_false_positives", "critical_identity_case", "regression"} & set(diff)
    assert json.dumps([before, after], sort_keys=True) == original


def test_unanswerable_top_k_is_observed_without_abstention_or_normal_metrics() -> None:
    before = {"case_id": "R9", "result": _result(unanswerable=True)}
    after = {"case_id": "R9", "result": _result(unanswerable=True, modules=[_module("resume", 1)])}

    diff = retrieval_case_diff(before, after)

    assert diff["comparison_scorable"] and not diff["metrics_applicable"]
    assert diff["metric_deltas"] == {}
    assert diff["ranking_changes"]

    inconclusive = retrieval_case_diff(before, {"case_id": "R9", "result": {"outcome": "ERROR"}})
    assert inconclusive["comparison_scorable"] is False
    assert inconclusive["metric_deltas"] == {} and inconclusive["covered_evidence_lost"] == []


@pytest.mark.parametrize("result", [
    {"outcome": "PASS", "matched_evidence_groups": {"g1": 1}},
    {"outcome": "FAIL", "negative_clean": False},
    {"outcome": "SCORED", "metrics": dict.fromkeys(METRIC_NAMES, 1.0)},
    {**_result(), "retrieval_contract_version": "old"},
])
def test_legacy_retrieval_case_artifacts_are_rejected(result):
    with pytest.raises(ValueError, match="Legacy or incompatible retrieval artifact"):
        retrieval_case_diff({"case_id": "R1", "result": result}, {"case_id": "R1", "result": _result()})


def _run(root: Path, profile: str, contract: dict, *, lexical: list | None = None) -> None:
    root.mkdir()
    (root / "run.json").write_text(json.dumps({
        "eval_run_id": profile + "-run", "embedding_profile_id": profile, "embedding_profile_name": profile,
        "embedding_model": profile + "-model", "embedding_dimension": 1024, "comparison_contract": contract,
        "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
    }))
    (root / "case_results.jsonl").write_text(json.dumps({
        "case_id": "R9", "suite": "retrieval", "result": _result(unanswerable=True),
        "retrieval_trace": {"input": {"query": "fixed"}, "output": {"lexical": lexical or []}},
    }) + "\n")
    (root / "summary.json").write_text(json.dumps({"suites": {"retrieval": {"metrics": dict.fromkeys(METRIC_NAMES, None)}}}))


def test_comparison_rejects_changed_controls_and_lexical_results(tmp_path) -> None:
    contract = {"version": COMPARISON_VERSION, "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
                "case_ids": ["R9"], "settings": {"budget": 12}}
    _run(tmp_path / "qwen", "qwen", contract)
    _run(tmp_path / "bge", "bge", contract)
    assert compare_runs(tmp_path / "qwen", tmp_path / "bge")["controlled_ab"] is True
    run_path = tmp_path / "bge" / "run.json"
    run = json.loads(run_path.read_text())
    run["comparison_contract"]["settings"]["budget"] = 13
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="controls differ: settings"):
        compare_runs(tmp_path / "qwen", tmp_path / "bge")
    run["comparison_contract"] = contract
    run_path.write_text(json.dumps(run))
    case_path = tmp_path / "bge" / "case_results.jsonl"
    case = json.loads(case_path.read_text())
    case["retrieval_trace"]["output"]["lexical"] = [{"chunk_id": "unexpected", "rank": 1}]
    case_path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="lexical results differ for R9"):
        compare_runs(tmp_path / "qwen", tmp_path / "bge")


def test_comparison_reports_only_current_metrics_and_rejects_legacy_contract(tmp_path):
    contract = {"version": COMPARISON_VERSION, "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION}
    _run(tmp_path / "qwen", "qwen", contract)
    _run(tmp_path / "bge", "bge", contract)
    comparison = compare_runs(tmp_path / "qwen", tmp_path / "bge")
    assert set(comparison["metrics"]) == set(METRIC_NAMES)
    assert all(values == {"baseline": None, "candidate": None} for values in comparison["metrics"].values())
    assert "critical_identity_regressions" not in comparison
    assert "Unanswerable query metrics are not applicable" in comparison_report(comparison)
    for root in (tmp_path / "qwen", tmp_path / "bge"):
        path = root / "run.json"
        run = json.loads(path.read_text())
        run["comparison_contract"] = {"version": 3}
        path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="Legacy comparison contract"):
        compare_runs(tmp_path / "qwen", tmp_path / "bge")


def test_contract_captures_actual_controls_without_credentials_or_urls() -> None:
    settings = Settings(_env_file=None, llm_api_key="secret-value", llm_base_url="https://gateway.example/v1", llm_model="same-agent")
    contract = comparison_contract(settings, load_dataset(), ["R8", "R9"], {"chunk_count": 7})
    encoded = json.dumps(contract)

    assert "secret-value" not in encoded
    assert "https://gateway.example" not in encoded
    assert contract["settings"]["agent_max_model_turns"] == settings.agent_max_model_turns
    assert contract["retrieval"] == {"rrf_k": 60, "max_modules": 5, "max_tokens": 8000, "k": 5}
    assert contract["version"] == 4 and contract["retrieval_contract_version"] == RETRIEVAL_CONTRACT_VERSION
    assert len(contract["case_package_sha256"]) == len(contract["implementation_sha256"]) == 64


@pytest.mark.parametrize("name", ["retrieval_evidence.yaml", "retrieval_cases.yaml", "retrieval_protocol.yaml"])
def test_each_root_retrieval_yaml_is_hashed_as_an_experiment_control(tmp_path, name):
    for filename in ("retrieval_evidence.yaml", "retrieval_cases.yaml", "retrieval_protocol.yaml"):
        (tmp_path / filename).write_text("initial: true\n")
    dataset = SimpleNamespace(root=tmp_path, retrieval_protocol=RetrievalProtocol())
    settings = Settings(_env_file=None)
    before = comparison_contract(settings, dataset, ["R1"], {})
    (tmp_path / name).write_text("changed: true\n")
    after = comparison_contract(settings, dataset, ["R1"], {})
    assert before["case_package_sha256"] != after["case_package_sha256"]
    assert {key: value for key, value in before.items() if key != "case_package_sha256"} == {
        key: value for key, value in after.items() if key != "case_package_sha256"
    }


def test_retrieval_budget_comes_from_protocol(tmp_path):
    for name in ("retrieval_evidence.yaml", "retrieval_cases.yaml", "retrieval_protocol.yaml"):
        (tmp_path / name).write_text("{}\n")
    protocol = RetrievalProtocol(retrieval_k=3, max_context_tokens=4000)
    dataset = SimpleNamespace(root=tmp_path, retrieval_protocol=protocol)
    contract = comparison_contract(Settings(_env_file=None), dataset, [], {})
    assert contract["retrieval"] == {"rrf_k": 60, "max_modules": 3, "max_tokens": 4000, "k": 3}


@pytest.mark.parametrize("setting,value", [
    ("eval_judge_concurrency", 3),
    ("eval_judge_transport_retries", 0), ("eval_judge_retry_backoff_seconds", 1.0),
    ("eval_judge_cache_dir", Path("private-cache")),
])
def test_parallel_judge_controls_cannot_silently_change_between_comparable_runs(setting, value):
    settings = Settings(_env_file=None)
    dataset = load_dataset()
    before = comparison_contract(settings, dataset, [], {})
    after = comparison_contract(settings.model_copy(update={setting: value}), dataset, [], {})
    assert before["settings"] != after["settings"]
    assert "private-cache" not in json.dumps(after)


@pytest.mark.parametrize("relative_path,constant", [
    ("evals/judge.py", "JUDGE_CONTRACT_VERSION"),
    ("evals/judge_tasks.py", "PROMPT_VERSION"),
    ("evals/schema.py", "CONTRACT_VERSION"),
    ("evals/schema.py", "SCHEMA_VERSION"),
])
def test_judge_contract_prompt_and_schema_changes_invalidate_controlled_comparison(
    tmp_path, monkeypatch, relative_path, constant,
):
    source_root = Path(comparison_module.__file__).resolve().parents[1]
    app_root = tmp_path / "app"
    for path in ("evals/comparison.py", "evals/judge.py", "evals/judge_tasks.py",
                 "evals/schema.py"):
        copy = app_root / path
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((source_root / path).read_bytes())
    monkeypatch.setattr(comparison_module, "__file__", str(app_root / "evals/comparison.py"))
    settings = Settings(_env_file=None)
    dataset = load_dataset()
    before = comparison_contract(settings, dataset, ["R9"], {})

    version_path = app_root / relative_path
    lines = version_path.read_text().splitlines(keepends=True)
    assignments = [index for index, line in enumerate(lines) if line.startswith(f"{constant} = ")]
    assert len(assignments) == 1
    lines[assignments[0]] = f'{constant} = "test_changed_version"\n'
    version_path.write_text("".join(lines))
    after = comparison_contract(settings, dataset, ["R9"], {})

    assert before["implementation_sha256"] != after["implementation_sha256"]
    assert {key: value for key, value in before.items() if key != "implementation_sha256"} == {
        key: value for key, value in after.items() if key != "implementation_sha256"
    }
    _run(tmp_path / "baseline", "qwen", before)
    _run(tmp_path / "candidate", "bge", after)
    with pytest.raises(ValueError, match="controls differ: implementation_sha256"):
        compare_runs(tmp_path / "baseline", tmp_path / "candidate")
