"""Provider-free oracles for the Gold/provenance/metric contract itself."""
from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.evals.retrieval_contract import (
    EvidenceFile,
    EvidenceUnit,
    RequiredFact,
    RetrievalCase,
    RetrievalProtocol,
    RetrievedModule,
    SourceSpan,
    Span,
    evaluate_retrieval,
    validate_retrieval_dataset,
)

SNAPSHOT = "1" * 64
CANONICAL = "2" * 64
SPAN_HASH = "3" * 64
OTHER_HASH = "4" * 64


def unit(identity="eu.a", document="doc.a", start=10, end=20):
    return EvidenceUnit(
        id=identity, document_key=document, snapshot_hash=SNAPSHOT,
        canonical_hash=CANONICAL, section_path=["Section"],
        span=Span(start=start, end=end), span_hash=SPAN_HASH,
    )


def fragment(eu, *, start=None, end=None, **overrides):
    value = {
        "document_key": eu.document_key, "snapshot_hash": eu.snapshot_hash,
        "canonical_hash": eu.canonical_hash,
        "span": {"start": eu.span.start if start is None else start,
                 "end": eu.span.end if end is None else end},
    }
    value.update(overrides)
    return SourceSpan.model_validate(value)


def module(*fragments, identity="runtime-chunk-name"):
    return RetrievedModule(module_id=identity, spans=list(fragments))


def fact(identity="fact.a", proofs=None):
    return RequiredFact(
        id=identity, statement=f"Task requires {identity}",
        proofs=proofs if proofs is not None else [["eu.a"]],
    )


def case(*facts, answerability="answerable", tags=None):
    return RetrievalCase(
        id="arbitrary-case", query="An arbitrary user query", answerability=answerability,
        tags=tags or [], facts=list(facts) or [fact()],
    )


def assert_metrics(score, evidence_recall, mrr, fact_recall, complete_support):
    assert score.metrics.model_dump() == pytest.approx({
        "evidence_recall": evidence_recall, "mrr": mrr,
        "fact_recall": fact_recall, "complete_support": complete_support,
    })


@pytest.mark.parametrize("shape", [
    "single_fact", "multi_fact", "cross_document", "cross_project",
    "multi_hop", "paraphrase", "hard_negative",
])
def test_oracle_full_evidence_scores_full_for_every_answerable_shape(shape):
    a, b = unit(), unit("eu.b", "doc.b" if shape in {
        "cross_document", "cross_project", "multi_hop",
    } else "doc.a", 30, 40)
    if shape in {"single_fact", "paraphrase", "hard_negative"}:
        task, evidence = case(fact(), tags=[shape]), [a]
    elif shape == "multi_hop":
        task, evidence = case(fact(proofs=[[a.id, b.id]]), tags=[shape]), [a, b]
    else:
        task, evidence = case(fact(), fact("fact.b", [[b.id]]), tags=[shape]), [a, b]
    result = evaluate_retrieval(task, evidence, [module(*(fragment(eu) for eu in evidence))])
    assert_metrics(result, 1.0, 1.0, 1.0, 1.0)
    assert not result.missing_fact_ids
    assert result.abstention == "NOT_EVALUATED"


def test_full_oracle_includes_all_alternative_evidence_for_evidence_recall():
    a, b = unit(), unit("eu.b", "doc.b")
    task = case(fact(proofs=[[a.id], [b.id]]))
    complete = evaluate_retrieval(task, [a, b], [module(fragment(a), fragment(b))])
    minimal = evaluate_retrieval(task, [a, b], [module(fragment(b))])
    assert_metrics(complete, 1.0, 1.0, 1.0, 1.0)
    assert_metrics(minimal, 0.5, 1.0, 1.0, 1.0)
    assert minimal.fact_witnesses == {"fact.a": [b.id]}


@pytest.mark.parametrize("removed_fact", ["fact.a", "fact.b"])
def test_removing_one_required_fact_lowers_fact_recall(removed_fact):
    a, b = unit(), unit("eu.b", "doc.b")
    task = case(fact(), fact("fact.b", [[b.id]]))
    retained = b if removed_fact == "fact.a" else a
    full = evaluate_retrieval(task, [a, b], [module(fragment(a)), module(fragment(b))])
    missing = evaluate_retrieval(task, [a, b], [module(fragment(retained))])
    assert_metrics(full, 1.0, 1.0, 1.0, 1.0)
    assert_metrics(missing, 0.5, 1.0, 0.5, 0.0)
    assert missing.missing_fact_ids == [removed_fact]


@pytest.mark.parametrize("chosen", [0, 1])
def test_any_proof_alternative_can_support_a_fact(chosen):
    units = [unit(), unit("eu.b", "doc.b")]
    task = case(fact(proofs=[[eu.id] for eu in units]))
    result = evaluate_retrieval(task, units, [module(fragment(units[chosen]))])
    assert result.metrics.fact_recall == result.metrics.complete_support == 1.0
    assert result.fact_witnesses == {"fact.a": [units[chosen].id]}


def test_inner_proof_requires_all_units_and_can_span_multiple_documents():
    a, b = unit(), unit("eu.b", "doc.b")
    task = case(fact(proofs=[[a.id, b.id]]))
    partial = evaluate_retrieval(task, [a, b], [module(fragment(a))])
    full = evaluate_retrieval(task, [a, b], [module(fragment(a)), module(fragment(b))])
    assert_metrics(partial, 0.5, 1.0, 0.0, 0.0)
    assert_metrics(full, 1.0, 1.0, 1.0, 1.0)
    assert full.fact_witnesses == {"fact.a": [a.id, b.id]}


def test_leading_distractor_lowers_reciprocal_rank_without_losing_facts():
    a, distractor = unit(), unit("unrelated", "other.doc")
    exact = evaluate_retrieval(case(), [a], [module(fragment(a))])
    noisy = evaluate_retrieval(case(), [a], [module(fragment(distractor)), module(fragment(a))])
    assert_metrics(exact, 1.0, 1.0, 1.0, 1.0)
    assert_metrics(noisy, 1.0, 0.5, 1.0, 1.0)
    assert noisy.first_evidence_rank == 2


def test_trailing_distractor_does_not_damage_existing_fact_coverage():
    a = unit()
    exact = evaluate_retrieval(case(), [a], [module(fragment(a))])
    noisy = evaluate_retrieval(case(), [a], [module(fragment(a)), module()])
    assert exact.metrics == noisy.metrics
    assert exact.fact_witnesses == noisy.fact_witnesses


def test_k_uses_returned_modules_and_never_scores_evidence_after_cutoff():
    a = unit()
    result = evaluate_retrieval(
        case(), [a], [module(), module(fragment(a))], RetrievalProtocol(retrieval_k=1),
    )
    assert_metrics(result, 0.0, 0.0, 0.0, 0.0)
    assert result.returned_modules == result.retrieval_k == 1


@pytest.mark.parametrize("intervals", [
    [(10, 15), (15, 20)],
    [(10, 17), (13, 20)],
    [(15, 20), (10, 15)],
    [(0, 15), (15, 30)],
])
def test_union_of_split_or_overlapping_modules_can_complete_one_eu(intervals):
    a = unit()
    result = evaluate_retrieval(case(), [a], [
        module(fragment(a, start=start, end=end)) for start, end in intervals
    ])
    assert_metrics(result, 1.0, 0.5, 1.0, 1.0)
    assert result.first_evidence_rank == 2


def test_multiple_spans_in_one_module_complete_at_that_module_rank():
    a = unit()
    result = evaluate_retrieval(case(), [a], [
        module(fragment(a, start=10, end=15), fragment(a, start=15, end=20)),
    ])
    assert_metrics(result, 1.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize("intervals", [[(10, 14), (15, 20)], [(11, 20)], [(10, 19)]])
def test_partial_or_gapped_source_coverage_never_completes_an_eu(intervals):
    a = unit()
    result = evaluate_retrieval(case(), [a], [
        module(fragment(a, start=start, end=end)) for start, end in intervals
    ])
    assert_metrics(result, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("identity_field,value", [
    ("document_key", "other.doc"),
    ("snapshot_hash", OTHER_HASH),
    ("canonical_hash", OTHER_HASH),
])
def test_different_source_identity_cannot_complete_or_join_an_eu(identity_field, value):
    a = unit()
    result = evaluate_retrieval(case(), [a], [
        module(fragment(a, start=10, end=15)),
        module(fragment(a, start=15, end=20, **{identity_field: value})),
    ])
    assert_metrics(result, 0.0, 0.0, 0.0, 0.0)


def test_duplicate_delivery_never_inflates_evidence_or_fact_recall():
    a, b = unit(), unit("eu.b", "doc.b")
    task = case(fact(), fact("fact.b", [[b.id]]))
    result = evaluate_retrieval(task, [a, b], [module(fragment(a))] * 5)
    assert_metrics(result, 0.5, 1.0, 0.5, 0.0)
    assert result.covered_evidence_ids == [a.id]


def test_no_returned_evidence_is_zero_for_answerable_case():
    result = evaluate_retrieval(case(), [unit()], [])
    assert_metrics(result, 0.0, 0.0, 0.0, 0.0)
    assert result.first_evidence_rank is None


def test_case_id_query_tags_statement_and_runtime_chunk_id_have_no_score_semantics():
    a = unit()
    task = case(tags=["single_fact"])
    before = evaluate_retrieval(task, [a], [module(fragment(a))])
    changed = task.model_copy(update={
        "id": "R9", "query": "literal anchors are absent", "tags": ["multi_hop"],
        "facts": [task.facts[0].model_copy(update={"statement": "different authoring prose"})],
    })
    after = evaluate_retrieval(changed, [a], [module(fragment(a), identity="new-chunk-layout")])
    assert before == after


def test_runtime_does_not_read_case_authoring_metadata():
    task, a = case(), unit()

    class ScoringView:
        facts = task.facts
        answerability = task.answerability

        def __getattr__(self, name):
            raise AssertionError(f"runtime evaluator read non-scoring case metadata: {name}")

    result = evaluate_retrieval(ScoringView(), [a], [module(fragment(a))])
    assert_metrics(result, 1.0, 1.0, 1.0, 1.0)


def test_gold_order_does_not_change_metrics():
    a, b = unit(), unit("eu.b", "doc.b")
    task = case(fact(), fact("fact.b", [[b.id]]))
    modules = [module(fragment(a)), module(fragment(b))]
    before = evaluate_retrieval(task, [a, b], modules)
    after = evaluate_retrieval(task.model_copy(update={"facts": list(reversed(task.facts))}),
                               [b, a], modules)
    assert before.metrics == after.metrics
    assert before.fact_witnesses == after.fact_witnesses


@pytest.mark.parametrize("return_related_evidence", [False, True])
def test_unanswerable_oracle_has_null_metrics_without_forcing_empty_top_k(return_related_evidence):
    a = unit()
    task = case(fact(), fact("missing", []), answerability="unanswerable", tags=["unanswerable"])
    modules = [module(fragment(a))] if return_related_evidence else []
    result = evaluate_retrieval(task, [a], modules)
    assert result.metrics.model_dump() == {name: None for name in (
        "evidence_recall", "mrr", "fact_recall", "complete_support",
    )}
    assert result.abstention == "NOT_EVALUATED"
    assert "missing" in result.missing_fact_ids
    assert result.fact_witnesses == ({"fact.a": [a.id]} if return_related_evidence else {})


@pytest.mark.parametrize("raw", [
    {"start": 2, "end": 2}, {"start": -1, "end": 2},
    {"start": 3, "end": 2}, {"start": False, "end": 2},
    {"start": "0", "end": 2},
])
def test_span_schema_is_strict_and_nonempty(raw):
    with pytest.raises(ValidationError):
        Span.model_validate(raw)


@pytest.mark.parametrize("field", ["anchors", "chunk_id", "relevance_grade", "required_evidence_groups"])
def test_removed_authoring_or_runtime_fields_are_rejected(field):
    raw = case().model_dump()
    raw[field] = []
    with pytest.raises(ValidationError, match="Extra inputs"):
        RetrievalCase.model_validate(raw)


@pytest.mark.parametrize("proofs", [[[]], [["eu.a", "eu.a"]], [["eu.a"], ["eu.a"]]])
def test_vacuous_or_duplicate_proofs_are_preflight_errors(proofs):
    with pytest.raises(ValidationError):
        fact(proofs=proofs)


def test_answerability_is_explicit_and_consistent_with_available_proofs():
    with pytest.raises(ValidationError, match="answerable case"):
        case(fact(proofs=[]))
    with pytest.raises(ValidationError, match="unanswerable case"):
        case(answerability="unanswerable")
    raw = case().model_dump()
    del raw["answerability"]
    with pytest.raises(ValidationError, match="answerability"):
        RetrievalCase.model_validate(raw)


def test_duplicate_fact_ids_and_duplicate_case_ids_are_preflight_errors():
    with pytest.raises(ValidationError, match="fact ID"):
        case(fact(), fact())
    with pytest.raises(ValueError, match="case ID"):
        validate_retrieval_dataset([unit()], [case(), case()])


def test_unknown_evidence_reference_is_a_preflight_error():
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_retrieval_dataset([unit()], [case(fact(proofs=[["missing.eu"]]))])


def test_empty_retrieval_dataset_is_a_preflight_error():
    with pytest.raises(ValueError, match="at least one case"):
        validate_retrieval_dataset([unit()], [])


def test_duplicate_eu_and_inconsistent_snapshot_are_preflight_errors():
    a = unit()
    with pytest.raises(ValidationError, match="duplicate evidence"):
        EvidenceFile(evidence=[a, a])
    changed = unit("eu.b").model_copy(update={"canonical_hash": OTHER_HASH})
    with pytest.raises(ValidationError, match="inconsistent canonical"):
        EvidenceFile(evidence=[a, changed])


@pytest.mark.parametrize("metrics", [
    ["evidence_recall", "mrr", "fact_recall"],
    ["evidence_recall", "mrr", "fact_recall", "fact_recall"],
    ["evidence_recall", "mrr", "fact_recall", "complete_support", "ndcg"],
])
def test_protocol_does_not_silently_enable_expanded_metrics(metrics):
    with pytest.raises(ValidationError):
        RetrievalProtocol(metrics=metrics)


def test_three_file_minimum_shapes_validate_without_extra_runtime_metadata():
    evidence_raw = {"evidence": [unit().model_dump()]}
    case_raw = case().model_dump(exclude={"suite", "task_type"})
    protocol_raw = {"retrieval_k": 5, "max_context_tokens": 8000,
                    "metrics": ["evidence_recall", "mrr", "fact_recall", "complete_support"]}
    frozen_before = deepcopy((evidence_raw, case_raw, protocol_raw))
    evidence = EvidenceFile.model_validate(evidence_raw)
    task = RetrievalCase.model_validate(case_raw)
    protocol = RetrievalProtocol.model_validate(protocol_raw)
    validate_retrieval_dataset(evidence, [task])
    evaluate_retrieval(task, evidence, [module(fragment(evidence.evidence[0]))], protocol)
    assert (evidence_raw, case_raw, protocol_raw) == frozen_before


def test_half_open_codepoint_spans_support_unicode_boundary_coordinates():
    source = "项目甲🚀校验通过"
    start, end = source.index("🚀"), source.index("验") + 1
    a = unit(start=start, end=end)
    result = evaluate_retrieval(case(), [a], [
        module(fragment(a, start=start, end=start + 1)),
        module(fragment(a, start=start + 1, end=end)),
    ])
    assert_metrics(result, 1.0, 0.5, 1.0, 1.0)
