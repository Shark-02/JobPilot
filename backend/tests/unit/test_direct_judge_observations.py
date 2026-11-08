"""Cross-task regressions for direct judgments and code-owned computations.

Synthetic observations are outside the authored evaluation dataset. These tests
exercise protocol/executor behavior; they do not claim model agreement.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.evals.compiler import CompiledCheck, ExecutableEvalPlan, ExtractionSpec, SemanticCheck
from app.evals.executor import EvaluationContext, execute_plan
from app.evals.judge import StructuredJudge, restore_observation_text
from app.evals.judge_tasks import JudgeTaskPlanner


def plan(*, checks=(), semantic=(), specs=()):
    return ExecutableEvalPlan(case_id="synthetic", suite="agent_tasks", task_type="synthetic",
                              checks=list(checks), semantic_checks=list(semantic), extractions=list(specs))


def observed(lines):
    return {"task": "synthetic task", "source_policy": {}, "final_answer": "\n".join(lines),
            "answer_spans": [{"id": f"answer_{i}", "text": line} for i, line in enumerate(lines, 1)],
            "evidence": [], "memories": []}


def restored_result(compiled, payload, observations, items=()):
    tasks = JudgeTaskPlanner().plan(compiled, payload)
    assert len(tasks) == 1
    result = tasks[0].validate(json.dumps({"items": list(items), "observations": observations}))
    result["extractions"] = restore_observation_text(result["extractions"], payload)
    return result


def test_resume_additional_bullets_remain_observed_and_code_applies_limit():
    spec = ExtractionSpec(id="bullets", kind="bullets", instructions="Identify every delivered bullet.")
    compiled = plan(specs=[spec], checks=[CompiledCheck(id="limit", gold_path="gold.max_bullets",
                    evaluator_kind="bullet_count", expected={"extraction_id": "bullets", "max": 2})])
    payload = observed(["- supported mechanism", "- supported implementation", "- additional delivered bullet"])
    before = deepcopy(payload)
    observations = [{"id": "bullets", "kind": "bullets", "items": [
        {"answer_span_id": f"answer_{i}"} for i in range(1, 4)]}]
    result = restored_result(compiled, payload, observations)
    assert [item["text"] for item in result["extractions"][0]["items"]] == [
        "supported mechanism", "supported implementation", "additional delivered bullet"]
    evaluated = execute_plan(compiled, EvaluationContext(), result)
    assert evaluated["check_results"][0]["actual"] == 3
    assert evaluated["outcome"] == "FAIL"
    assert payload == before


@pytest.mark.parametrize("labelled", [False, True])
def test_local_revision_uses_original_text_and_code_checks_non_target_equality(labelled):
    draft = [{"id": "keep", "text": "Keep C++ identifiers and punctuation."},
             {"id": "target", "text": "Old target text."}]
    spec = ExtractionSpec(id="revision", kind="bullets", allowed_ids=["keep", "target"],
                          instructions="Identify every delivered draft item by ID and span.")
    check = CompiledCheck(id="revision", gold_path="gold.draft", evaluator_kind="local_revision", expected={
        "extraction_id": "revision", "draft": draft, "target_id": "target"})
    compiled = plan(checks=[check], specs=[spec])
    keep_label, target_label = ("keep: ", "target： ") if labelled else ("", "")
    payload = observed([f"1. {keep_label}Keep C++ identifiers and punctuation.",
                        f"2. {target_label}New target text."])
    observations = [{"id": "revision", "kind": "bullets", "items": [
        {"id": "keep", "answer_span_id": "answer_1"}, {"id": "target", "answer_span_id": "answer_2"}]}]
    result = restored_result(compiled, payload, observations)
    assert execute_plan(compiled, EvaluationContext(), result)["outcome"] == "PASS"
    payload["answer_spans"][0]["text"] = f"1. {keep_label}Keep C identifiers and punctuation."
    changed_result = restored_result(compiled, payload, observations)
    evaluated = execute_plan(compiled, EvaluationContext(), changed_result)
    assert evaluated["outcome"] == "FAIL"
    assert evaluated["check_results"][0]["actual"]["non_target_unchanged"] == {"keep": False}


def test_questions_followups_and_extra_main_question_are_not_capped_to_gold():
    spec = ExtractionSpec(id="questions", kind="questions", allowed_ids=["causality", "runtime"],
                          instructions="Identify all main questions, follow-ups and topic memberships.")
    compiled = plan(specs=[spec], checks=[
        CompiledCheck(id="limits", gold_path="gold.main_questions", evaluator_kind="question_limits",
                      expected={"extraction_id": "questions", "main_questions": 2, "max_followups_per_question": 1}),
        CompiledCheck(id="topics", gold_path="gold.topic_groups", evaluator_kind="topic_coverage",
                      expected={"extraction_id": "questions", "topic_ids": ["causality", "runtime"], "min": 2}),
    ])
    payload = observed(["1. Why did latency change?", "- What was measured?", "- Which control was held?",
                        "2. How do retries terminate?", "3. What else did you implement?"])
    observations = [{"id": "questions", "kind": "questions", "items": [
        {"id": "q1", "answer_span_id": "answer_1", "topic_ids": ["causality"],
         "followup_span_ids": ["answer_2", "answer_3"]},
        {"id": "q2", "answer_span_id": "answer_4", "topic_ids": ["runtime"]},
        {"id": "q3", "answer_span_id": "answer_5", "topic_ids": []},
    ]}]
    result = restored_result(compiled, payload, observations)
    assert result["extractions"][0]["items"][0]["followups"] == payload["final_answer"].splitlines()[1:3]
    evaluated = execute_plan(compiled, EvaluationContext(), result)
    assert evaluated["check_results"][0]["actual"] == {"main_questions": 3, "followup_counts": [2, 0, 0]}
    assert [check["result"] for check in evaluated["check_results"]] == ["FAIL", "PASS"]


@pytest.mark.parametrize("kind,item", [
    ("bullets", {"answer_span_id": "answer_1", "text": "Invented original text"}),
    ("questions", {"id": "q1", "answer_span_id": "answer_1", "text": "Invented question"}),
    ("questions", {"id": "q1", "answer_span_id": "answer_1", "followup_span_ids": ["absent"]}),
])
def test_observation_wire_cannot_supply_copied_or_invented_text(kind, item):
    compiled = plan(specs=[ExtractionSpec(id="items", kind=kind, instructions="Identify observed item spans.")])
    task = JudgeTaskPlanner().plan(compiled, observed(["An actual delivered item."]))[0]
    with pytest.raises(ValueError):
        task.validate(json.dumps({"items": [], "observations": [{"id": "items", "kind": kind, "items": [item]}]}))


def test_each_semantic_check_gets_one_task_and_keeps_its_source_permissions():
    criteria = [SemanticCheck(id="fact", gold_path="gold.facts", statement="Factual answer"),
                SemanticCheck(id="fixture", gold_path="gold.facts", statement="Use named source",
                              required_fixture_id="supplement"),
                SemanticCheck(id="technical", gold_path="gold.technical", statement="Technical explanation",
                              support_mode="technical_rubric")]
    compiled = plan(semantic=criteria)
    payload = observed(["An answer."])
    payload.update(evidence=[{"id": "raw", "content": "Raw observed fact.", "source_ref": "message://supplement",
                             "visible_to_agent": True, "authoritative_for_grounding": True}],
                   fixture_source_refs={"supplement": "message://supplement"})
    tasks = JudgeTaskPlanner().plan(compiled, payload)
    assert [task.id for task in tasks] == ["semantic:fact", "semantic:fixture", "semantic:technical"]
    contracts = [json.loads(task.messages[0]["content"].split("\n", 1)[1]) for task in tasks]
    assert [len(contract["checks"]) for contract in contracts] == [1, 1, 1]
    assert [contract["checks"][0]["statement"] for contract in contracts] == [
        "Factual answer", "Use named source", "Technical explanation"]
    submitted = [json.loads(task.messages[-1]["content"]) for task in tasks]
    assert submitted[0]["evidence"] == [{"id": "e1", "content": "Raw observed fact."}]
    assert submitted[1]["evidence"] == [{
        "id": "e1", "content": "Raw observed fact.", "source_ref": "message://supplement"}]
    assert submitted[2]["evidence"] == []


async def test_code_only_plan_needs_no_configured_judge_or_provider(monkeypatch):
    import app.evals.judge as module

    monkeypatch.setattr(module, "TokenCounter", lambda _: SimpleNamespace(count=lambda _: 0))
    monkeypatch.setattr(module, "AsyncOpenAI", lambda **_: pytest.fail("Code-only checks must not open a provider"))
    judge = StructuredJudge(Settings(_env_file=None, llm_api_key="", eval_judge_api_key=""))
    result = await judge.judge(plan=plan(), payload=observed(["An answer."]))
    assert "execution_error" not in result
    assert result["checks"] == [] and result["extractions"] == []
    assert result["summary"]["total_subtasks"] == 0


async def test_old_frozen_contract_is_rejected_before_provider(monkeypatch):
    import app.evals.judge as module

    monkeypatch.setattr(module, "TokenCounter", lambda _: SimpleNamespace(count=lambda _: 0))
    monkeypatch.setattr(module, "AsyncOpenAI", lambda **_: pytest.fail("Old plans must fail before opening provider"))
    judge = StructuredJudge(Settings(_env_file=None, llm_api_key="synthetic"))
    old = plan(semantic=[SemanticCheck(id="fact", gold_path="gold.fact", statement="fact")])
    old.contract_version = "eval_contract_v2"
    result = await judge.judge(plan=old, payload=observed(["An answer."]))
    assert result["execution_error"]["code"] == "JUDGE_SCHEMA_ERROR"
    assert result["summary"]["total_subtasks"] == 0


@pytest.mark.parametrize("identifier", ["project-one", "研发一", "resume.item_3"])
@pytest.mark.parametrize("colon", [":", "："])
def test_validated_bullet_labels_are_removed_without_altering_body(identifier, colon):
    spec = ExtractionSpec(id="bullets", kind="bullets", allowed_ids=[identifier],
                          instructions="Select the delivered draft item and span.")
    compiled = plan(specs=[spec])
    body = f" C++: 125k / 3.0；keep {identifier}: inside正文！"
    payload = observed([f"- {identifier}{colon}{body}"])
    before = deepcopy(payload)
    result = restored_result(compiled, payload, [{"id": "bullets", "kind": "bullets", "items": [
        {"id": identifier, "answer_span_id": "answer_1"}]}])
    assert result["extractions"][0]["items"][0]["text"] == body
    assert payload == before


@pytest.mark.parametrize("body", [
    "keepExtra: text", "prefix keep: text", "keep : text", "keep/other: text",
    "Keep: text", "**keep**: text", "C++: 125k / 3.0 remains unchanged.",
])
def test_bullet_body_without_exact_leading_label_is_preserved(body):
    spec = ExtractionSpec(id="bullets", kind="bullets", allowed_ids=["keep"],
                          instructions="Select the delivered draft item and span.")
    result = restored_result(plan(specs=[spec]), observed([f"- {body}"]), [{
        "id": "bullets", "kind": "bullets", "items": [{"id": "keep", "answer_span_id": "answer_1"}]}])
    assert result["extractions"][0]["items"][0]["text"] == body


def test_unmapped_bullet_does_not_infer_or_remove_a_display_label():
    spec = ExtractionSpec(id="bullets", kind="bullets", instructions="Select all delivered bullet spans.")
    body = "keep: C++: 125k / 3.0 remains unchanged."
    result = restored_result(plan(specs=[spec]), observed([f"- {body}"]), [{
        "id": "bullets", "kind": "bullets", "items": [{"answer_span_id": "answer_1"}]}])
    assert result["extractions"][0]["items"][0]["id"] is None
    assert result["extractions"][0]["items"][0]["text"] == body
