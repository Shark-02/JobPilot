from uuid import uuid4

import pytest

from app.evals.compiler import CompiledCheck, ExecutableEvalPlan, SemanticCheck
from app.evals.evidence import EvidenceItem, EvidencePack
from app.evals.executor import EvaluationContext, execute_plan


def semantic_plan(*definitions, checks=()):
    return ExecutableEvalPlan(case_id="opaque", suite="agent_tasks", task_type="synthetic",
        semantic_checks=list(definitions), extractions=[], checks=[
            CompiledCheck(id=item.id, gold_path=item.gold_path, evaluator_kind="semantic", expected="PASS")
            for item in definitions] + list(checks))


def verdict(identifier, result="PASS", *, span=None, evidence=()):
    return {"id": identifier, "result": result, "reason": "Fixed semantic test judgment",
            "answer_span_id": span, "evidence_ids": list(evidence)}


@pytest.mark.parametrize("required,result,expected", [
    (True, "PASS", "PASS"), (True, "FAIL", "FAIL"), (True, "UNKNOWN", "FAIL"),
    (False, "PASS", "PASS"), (False, "FAIL", "FAIL"),
])
def test_direct_required_and_optional_material_results_do_not_need_an_extraction(required, result, expected):
    item = SemanticCheck(id="material.latency", gold_path="gold.material_claims[0]",
                         statement="Required latency, or if used optional latency, must be supported",
                         presence="required" if required else "conditional")
    outcome = execute_plan(semantic_plan(item), EvaluationContext(),
                           {"checks": [verdict(item.id, result)], "extractions": []})
    assert outcome["outcome"] == expected


@pytest.mark.parametrize("result,expected", [("PASS", "PASS"), ("FAIL", "FAIL"), ("UNKNOWN", "FAIL")])
def test_extra_material_grounding_verdict_is_required_without_canonical_claim_ids(result, expected):
    item = SemanticCheck(id="answer_grounding", gold_path="source_policy",
                         statement="Extra quantitative claims must have correct project and metric support",
                         presence="conditional")
    outcome = execute_plan(semantic_plan(item), EvaluationContext(),
                           {"checks": [verdict(item.id, result)], "extractions": []})
    assert outcome["outcome"] == expected


def test_ordinary_semantic_result_is_minimal_but_provided_sources_cannot_be_hidden_or_derived():
    item = SemanticCheck(id="answer_grounding", gold_path="source_policy", statement="Grounded answer")
    evidence = EvidencePack(items=[EvidenceItem(
        id="derived", origin_type="knowledge_module", source_ref=f"document://{uuid4()}/section/{uuid4()}",
        content="Generated summary", visible_to_agent=True, authoritative_for_grounding=False, turn_id="turn")])
    minimal = execute_plan(semantic_plan(item), EvaluationContext(evidence_pack=evidence),
                          {"checks": [{"id": item.id, "result": "PASS", "reason": "Direct judgment"}], "extractions": []})
    assert minimal["outcome"] == "PASS"
    bad_ref = execute_plan(semantic_plan(item), EvaluationContext(evidence_pack=evidence),
                          {"checks": [verdict(item.id, evidence=["derived"])], "extractions": []})
    assert bad_ref["outcome"] == "ERROR"


@pytest.mark.parametrize("refs,expected", [([], "FAIL"), (["wrong"], "FAIL"), (["exact"], "PASS")])
def test_conditional_fixture_pass_cannot_bypass_exact_source_by_omitting_span(refs, expected):
    exact_ref, wrong_ref = f"message://{uuid4()}", f"message://{uuid4()}"
    item = SemanticCheck(id="material.latency", gold_path="gold.material_claims[0]",
                         statement="If expressed use exact fixture", presence="conditional",
                         required_fixture_id="supplement")
    evidence = EvidencePack(items=[EvidenceItem(
        id=identifier, origin_type="user_message", source_ref=source_ref, content="Same numerical claim",
        visible_to_agent=True, authoritative_for_grounding=True, turn_id="turn")
        for identifier, source_ref in [("exact", exact_ref), ("wrong", wrong_ref)]])
    result = execute_plan(semantic_plan(item),
        EvaluationContext(evidence_pack=evidence, fixture_source_refs={"supplement": exact_ref}),
        {"checks": [verdict(item.id, evidence=refs)], "extractions": []})
    assert result["outcome"] == expected


@pytest.mark.parametrize("kind,expected,extraction", [
    ("bullet_count", {"extraction_id": "delivered", "max": 2},
     {"id": "delivered", "kind": "bullets", "items": [{"text": "one"}, {"text": "two"}, {"text": "three"}]}),
    ("question_limits", {"extraction_id": "delivered", "main_questions": 1, "max_followups_per_question": 1},
     {"id": "delivered", "kind": "questions", "items": [{"followups": ["one", "two"]}]}),
    ("local_revision", {"extraction_id": "delivered", "draft": [{"id": "keep", "text": "original"},
                                                                              {"id": "target", "text": "old"}],
                         "target_id": "target"},
     {"id": "delivered", "kind": "bullets", "items": [{"id": "keep", "text": "changed"},
                                                                {"id": "target", "text": "new"}]}),
])
def test_semantic_pass_cannot_override_real_count_followup_or_unchanged_text_failure(kind, expected, extraction):
    semantic = SemanticCheck(id="answer_grounding", gold_path="source_policy", statement="Faithful answer")
    computed = CompiledCheck(id="shape", gold_path="gold.shape", evaluator_kind=kind, expected=expected)
    outcome = execute_plan(semantic_plan(semantic, checks=[computed]), EvaluationContext(),
                           {"checks": [verdict(semantic.id)], "extractions": [extraction]})
    assert outcome["outcome"] == "FAIL"
    assert {row["id"]: row["result"] for row in outcome["check_results"]} == {
        "answer_grounding": "PASS", "shape": "FAIL"}
