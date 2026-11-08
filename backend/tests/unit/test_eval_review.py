import json

import pytest

from app.evals.review import review_run
from app.evals.schema import CONTRACT_VERSION


def test_review_is_an_offline_pending_worksheet_with_check_provenance(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"contract_version": CONTRACT_VERSION}))
    (tmp_path / "case_results.jsonl").write_text(json.dumps({"case_id": "sample", "result": {
        "outcome": "FAIL", "check_results": [{"id": "grounding", "gold_path": "gold.facts[0]",
        "evaluator_kind": "semantic", "expected": True, "actual": "unsupported|claim",
        "evidence_ids": ["raw1"], "result": "FAIL"}]}}))
    content = review_run(tmp_path, "sample").read_text()
    assert "gold.facts[0]" in content and "raw1" in content
    assert "unsupported\\|claim" in content
    assert "Human decision: PENDING" in content


def test_old_contract_artifacts_are_not_reviewed_as_current(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"contract_version": "old"}))
    with pytest.raises(ValueError):
        review_run(tmp_path)


def test_old_retrieval_contract_is_not_reviewed_as_current_under_same_agent_version(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"contract_version": CONTRACT_VERSION}))
    (tmp_path / "case_results.jsonl").write_text(json.dumps({
        "case_id": "opaque", "suite": "retrieval", "result": {"outcome": "FAIL"}}))
    with pytest.raises(ValueError, match="Legacy retrieval"):
        review_run(tmp_path)


def test_current_retrieval_metrics_are_reviewable_without_case_pass_gate(tmp_path):
    from app.evals.retrieval_contract import RETRIEVAL_CONTRACT_VERSION
    (tmp_path / "run.json").write_text(json.dumps({
        "contract_version": CONTRACT_VERSION, "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION}))
    (tmp_path / "case_results.jsonl").write_text(json.dumps({
        "case_id": "opaque", "suite": "retrieval", "result": {
            "outcome": "SCORED", "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
            "check_results": [{"id": "retrieval_metrics", "gold_path": "facts", "result": "SCORED",
                               "actual": {"complete_support": 0.0}}]}}))
    assert "SCORED" in review_run(tmp_path).read_text()
