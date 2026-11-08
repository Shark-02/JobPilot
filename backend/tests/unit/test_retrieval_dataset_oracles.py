"""Actual migrated Gold must satisfy the same provider-free oracle contracts."""
from collections import defaultdict

import pytest

from app.evals.compiler import compile_case
from app.evals.datasets import load_dataset
from app.evals.executor import EvaluationContext, execute_plan
from app.evals.retrieval_contract import (
    RETRIEVAL_CONTRACT_VERSION, RetrievalCase, RetrievedModule, SourceSpan, evaluate_retrieval,
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def oracle_modules(dataset, ids):
    """Ideal visible spans grouped by source, independent of current chunking.

    This proves scorer behavior, not whether today's section assembly can
    retrieve this ideal collection within its actual candidate budget.
    """
    by_source = defaultdict(list)
    for unit in dataset.retrieval_evidence.evidence:
        if unit.id in ids:
            by_source[unit.document_key].append(SourceSpan(
                document_key=unit.document_key, snapshot_hash=unit.snapshot_hash,
                canonical_hash=unit.canonical_hash, span=unit.span))
    return [RetrievedModule(module_id=key, spans=spans) for key, spans in by_source.items()]


@pytest.mark.parametrize("index", range(9))
def test_every_migrated_case_has_a_correct_full_evidence_oracle(dataset, index):
    task = [case for case in dataset.by_suite("retrieval")][index]
    refs = {uid for fact in task.facts for proof in fact.proofs for uid in proof}
    score = evaluate_retrieval(task, dataset.retrieval_evidence,
                               oracle_modules(dataset, refs), dataset.retrieval_protocol)
    if task.answerability == "answerable":
        assert all(value == 1.0 for value in score.metrics.model_dump().values())
        assert not score.missing_fact_ids
    else:
        assert all(value is None for value in score.metrics.model_dump().values())
        assert score.abstention == "NOT_EVALUATED"


def test_each_real_required_fact_loses_support_when_all_its_evidence_is_removed(dataset):
    for task in dataset.by_suite("retrieval"):
        if task.answerability != "answerable":
            continue
        all_refs = {uid for fact in task.facts for proof in fact.proofs for uid in proof}
        for fact in task.facts:
            excluded = {uid for proof in fact.proofs for uid in proof}
            score = evaluate_retrieval(task, dataset.retrieval_evidence,
                                       oracle_modules(dataset, all_refs - excluded),
                                       dataset.retrieval_protocol)
            assert fact.id in score.missing_fact_ids
            assert score.metrics.fact_recall < 1.0
            assert score.metrics.complete_support == 0.0


def test_every_real_alternative_is_a_valid_graph_witness(dataset):
    for task in dataset.by_suite("retrieval"):
        if task.answerability != "answerable":
            continue
        for fact in task.facts:
            for proof in fact.proofs:
                selected = {uid for other in task.facts for uid in
                            (proof if other.id == fact.id else other.proofs[0])}
                score = evaluate_retrieval(task, dataset.retrieval_evidence,
                                           oracle_modules(dataset, selected), dataset.retrieval_protocol)
                assert score.metrics.fact_recall == score.metrics.complete_support == 1.0


def test_compiled_retrieval_consumes_metrics_without_turning_partial_support_into_a_pass(dataset):
    task = next(task for task in dataset.by_suite("retrieval") if task.answerability == "answerable")
    score = evaluate_retrieval(task, dataset.retrieval_evidence, [], dataset.retrieval_protocol)
    observation = {"outcome": "SCORED", "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
                   **score.model_dump(mode="json")}
    result = execute_plan(compile_case(task), EvaluationContext(retrieval_result=observation))
    assert result["outcome"] == "SCORED"
    assert result["metrics"]["complete_support"] == 0.0
    assert result["check_results"][0]["result"] == "SCORED"


def test_migrated_schema_does_not_accept_legacy_scoring_fields(dataset):
    from pydantic import ValidationError
    task = dataset.by_suite("retrieval")[0]
    for field, value in [("anchors", ["a"]), ("pass_policy", {"mode": "all_required_groups"})]:
        with pytest.raises(ValidationError):
            RetrievalCase.model_validate({**task.model_dump(), field: value})
