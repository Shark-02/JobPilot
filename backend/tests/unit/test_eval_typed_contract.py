from __future__ import annotations

import shutil
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError, create_model

from app.evals import compiler
from app.evals.compiler import CompileError, compile_case
from app.evals.datasets import DatasetError, load_dataset
from app.evals.schema import (
    CASE_ADAPTER, CorpusManifest, EvidenceFusionCase, JudgeResponse,
)
from app.evals.retrieval_contract import RETRIEVAL_CONTRACT_VERSION, RetrievalCase

PACKAGE = Path(__file__).resolve().parents[3] / "jobpilot_eval_v0.1.1"


def all_cases():
    return list(load_dataset().cases.values())


def test_all_27_cases_compile_with_unique_atomic_contracts():
    cases = all_cases()
    assert len(cases) == 27
    assert len({case.id for case in cases}) == 27
    assert {case.task_type for case in cases if case.suite == "agent_tasks"} == {
        "profile_summary", "mechanism_mapping", "project_deep_dive", "jd_matching",
        "resume_generation", "local_revision", "interview_preparation", "interview_review",
        "evidence_fusion",
    }
    for case in cases:
        plan = compile_case(case)
        assert len({check.id for check in plan.checks}) == len(plan.checks)
        assert all(check.gold_path for check in plan.checks)
        assert {check.id for check in plan.semantic_checks} <= {check.id for check in plan.checks}
        assert plan.authoring_field_mappings
        assert all(mapping.check_ids and mapping.evaluator_kinds
                   for mapping in plan.authoring_field_mappings)
        assert all(set(mapping.check_ids) <= {check.id for check in plan.checks}
                   for mapping in plan.authoring_field_mappings)


def test_case_identity_cannot_change_scoring_semantics():
    for case in all_cases():
        before = compile_case(case).model_dump()
        after = compile_case(case.model_copy(update={"id": "arbitrary_identity"})).model_dump()
        before.pop("case_id")
        after.pop("case_id")
        assert before == after


@pytest.mark.parametrize("path", ["case", "gold", "setup", "source_policy"])
def test_unknown_authoring_field_rejected(path):
    value = all_cases()[0].model_dump()
    target = value if path == "case" else value[path]
    target["unmapped_scoring_requirement"] = True
    with pytest.raises(ValidationError):
        CASE_ADAPTER.validate_python(value)


def test_scoring_check_without_executor_is_preflight_error(monkeypatch):
    case = all_cases()[0]
    monkeypatch.setattr(compiler, "CHECK_REGISTRY", compiler.CHECK_REGISTRY - {"semantic"})
    with pytest.raises(CompileError, match="no executor"):
        compile_case(case)


def test_new_typed_scoring_field_without_compiler_mapping_is_error():
    case = all_cases()[0]
    changed_gold = create_model("ProfileSummaryGold", new_score=(bool, True), __base__=type(case.gold))
    changed_case = case.model_copy(update={"gold": changed_gold.model_validate(case.gold.model_dump())})
    with pytest.raises(CompileError, match="no compiler/executor mapping"):
        compile_case(changed_case)


def test_unknown_declared_check_rejected():
    case = all_cases()[0].model_copy(update={"checks": ["judge_dimensions"]})
    with pytest.raises(CompileError, match="Unknown or unbound"):
        compile_case(case)


def test_audit_annotations_are_absent_from_agent_visible_fixtures():
    interview = (PACKAGE / "fixtures/interview_alibaba_gaode_ai_native_v1.md").read_text()
    supplement = (PACKAGE / "fixtures/user_supplement_apollo_new_user_query_v1.md").read_text()
    assert "Eval Gold" not in interview
    assert "明确不稳定：" not in interview
    assert "相对稳定：" not in interview
    assert "但本轮没有系统性说出" not in interview
    assert "本轮回答偏通用" not in interview
    assert "Gold semantics" not in supplement
    assert "Provenance requirement" not in supplement
    assert "6.00s" in supplement and "0.18s" in supplement
    case = next(case for case in all_cases() if isinstance(case, EvidenceFusionCase))
    assert [item.id for item in case.setup.user_fixtures] == [case.gold.required_fixture_id]


def test_memory_evolution_is_topic_scoped_and_has_explicit_seed_inputs():
    cases = {case.id: case for case in all_cases()}
    for case_id in ["M5", "M6"]:
        case = cases[case_id]
        assert len(case.gold.target_topics) == 1
        assert case.gold.target_topics[0].topic_id == "hashmap_1_8"
        assert case.gold.mutation.target_step_id == "target"
        assert case.gold.mutation.target_existing
        assert case.gold.mutation.require_new_source
        assert set(case.gold.mutation.allowed_operations) == {"MERGE", "SUPERSEDE"}
        assert case.setup.seed_steps
    assert cases["M6"].source_policy.fact_time_policy == "user_reported_state"


def test_answerable_retrieval_requires_a_nonempty_proof_for_every_fact():
    case = next(case for case in all_cases() if case.suite == "retrieval")
    raw = case.model_dump()
    raw["facts"][0]["proofs"] = []
    with pytest.raises(ValidationError, match="answerable case requires a corpus proof"):
        CASE_ADAPTER.validate_python(raw)


@pytest.mark.parametrize("field,value", [
    ("gold", {"required_evidence_groups": []}), ("anchors", ["literal"]),
    ("pass_policy", {"mode": "clean_negative"}), ("checks", []), ("setup", {}),
    ("source_policy", {"fact_time_policy": "current_snapshot"}), ("max_modules", 5),
])
def test_retrieval_case_rejects_legacy_policy_and_execution_fields(field, value):
    case = next(case for case in all_cases() if isinstance(case, RetrievalCase))
    raw = case.model_dump()
    raw[field] = value
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RetrievalCase.model_validate(raw)


def test_judge_envelope_refuses_overall_verdict_and_old_fields_wrapper():
    with pytest.raises(ValidationError):
        JudgeResponse.model_validate({"fields": {"judge_status": "PASS"}})
    with pytest.raises(ValidationError):
        JudgeResponse.model_validate({"checks": [], "extractions": [], "judge_status": "PASS"})


def test_corpus_manifest_unknown_policy_field_rejected():
    value = yaml.safe_load((PACKAGE / "corpora/corpus_v1_manifest.yaml").read_text())
    CorpusManifest.model_validate(value)
    value["evidence_policy"]["unknown_scoring_rule"] = True
    with pytest.raises(ValidationError):
        CorpusManifest.model_validate(value)


def _preflight_package(tmp_path):
    """Schema errors must be found before opening fixture/corpus data or a provider."""
    shutil.copytree(PACKAGE / "cases", tmp_path / "cases")
    (tmp_path / "corpora").mkdir()
    shutil.copyfile(PACKAGE / "corpora/corpus_v1_manifest.yaml",
                    tmp_path / "corpora/corpus_v1_manifest.yaml")
    for name in ("retrieval_evidence.yaml", "retrieval_cases.yaml", "retrieval_protocol.yaml"):
        shutil.copyfile(PACKAGE / name, tmp_path / name)
    return tmp_path


def test_dataset_loader_unknown_scoring_yaml_is_error_before_execution(tmp_path):
    root = _preflight_package(tmp_path)
    path = root / "cases/agent_tasks.yaml"
    document = yaml.safe_load(path.read_text())
    document["cases"][0]["gold"]["new_required_score"] = True
    path.write_text(yaml.safe_dump(document, allow_unicode=True))
    with pytest.raises(DatasetError, match="Dataset preflight ERROR"):
        load_dataset(root)


def test_dataset_loader_declared_score_without_executor_is_error(tmp_path, monkeypatch):
    root = _preflight_package(tmp_path)
    monkeypatch.setattr(compiler, "CHECK_REGISTRY", compiler.CHECK_REGISTRY - {"semantic"})
    with pytest.raises(DatasetError, match="no executor"):
        load_dataset(root)


def test_dataset_loader_checks_the_actual_executor_registry(tmp_path, monkeypatch):
    from app.evals import executor
    root = _preflight_package(tmp_path)
    registry = dict(executor.EXECUTOR_REGISTRY)
    registry.pop("classification")
    monkeypatch.setattr(executor, "EXECUTOR_REGISTRY", registry)
    with pytest.raises(DatasetError, match="no executor"):
        load_dataset(root)


def test_dataset_loader_rejects_unknown_evidence_before_fixture_or_corpus_access(tmp_path):
    root = _preflight_package(tmp_path)
    path = root / "retrieval_cases.yaml"
    cases = yaml.safe_load(path.read_text())
    cases[0]["facts"][0]["proofs"] = [["eu.does.not.exist"]]
    path.write_text(yaml.safe_dump(cases, allow_unicode=True))
    with pytest.raises(DatasetError, match="unknown evidence unit references"):
        load_dataset(root)


@pytest.mark.parametrize("name", ["retrieval_evidence.yaml", "retrieval_cases.yaml", "retrieval_protocol.yaml"])
def test_dataset_loader_requires_all_three_retrieval_files(tmp_path, name):
    root = _preflight_package(tmp_path)
    (root / name).unlink()
    with pytest.raises(DatasetError, match="Dataset preflight ERROR: FileNotFoundError"):
        load_dataset(root)


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(schema_version="1"),
    lambda data: data["cases"][0].update(task_type="generic_semantic_checks"),
    lambda data: data["cases"][0].update(checks=["unsupported_custom_evaluator"]),
])
def test_dataset_rejects_invalid_version_task_dispatch_and_check_registry(tmp_path, mutation):
    root = _preflight_package(tmp_path)
    path = root / "cases/agent_tasks.yaml"
    document = yaml.safe_load(path.read_text())
    mutation(document)
    path.write_text(yaml.safe_dump(document, allow_unicode=True))
    with pytest.raises(DatasetError):
        load_dataset(root)


def test_fact_time_policy_changes_compiled_semantic_rubric():
    case = all_cases()[0]
    original = compile_case(case)
    changed = case.model_copy(update={"source_policy": case.source_policy.model_copy(
        update={"fact_time_policy": "preserve_event_time"})})
    updated = compile_case(changed)
    assert original.semantic_checks[0].statement != updated.semantic_checks[0].statement
    assert "历史事件发生时" in updated.semantic_checks[0].statement


def test_local_revision_semantic_facts_only_apply_to_the_target_output_span():
    case = next(case for case in all_cases() if case.task_type == "local_revision")
    plan = compile_case(case)
    target_checks = [check for check in plan.semantic_checks
                     if check.gold_path.startswith("gold.required_target_facts[")]
    assert len(target_checks) == len(case.gold.required_target_facts)
    assert all(check.presence == "required" for check in target_checks)
    assert all(check.required_extraction_id == "revised_bullets" for check in target_checks)
    assert all(check.required_item_id == case.gold.target_id for check in target_checks)


def test_semantic_presence_and_memory_scope_are_compiled_from_contract():
    for case in all_cases():
        plan = compile_case(case)
        checks = {check.id: check for check in plan.semantic_checks}
        for item in plan.semantic_checks:
            if item.gold_path.startswith("gold.prohibitions["):
                assert item.presence == "absence"
        if case.suite == "memory":
            for item in plan.semantic_checks:
                if item.gold_path.startswith(("gold.semantics[", "gold.prohibitions[", "gold.source_requirement")):
                    assert item.observation_scope == "target_mutation"
                    assert item.target_step_id == case.gold.mutation.target_step_id
            if case.gold.recall:
                assert checks["memory_recall_answer"].observation_scope == "memory_recall"
                assert checks["memory_recall_answer"].target_step_id == case.gold.recall.step_id
                assert checks["memory_recall_answer"].presence == "required"


def test_optional_material_claims_are_conditional_and_not_required_to_appear():
    case = next(case for case in all_cases() if case.task_type == "resume_generation")
    plan = compile_case(case)
    material_checks = [check for check in plan.semantic_checks if check.id.startswith("material.")]
    assert material_checks
    assert all(check.presence == "conditional" for check in material_checks)
    assert not any(item.kind == "material_claims" for item in plan.extractions)
    assert not any(item.evaluator_kind == "material_source_binding" for item in plan.checks)
    for item in case.gold.material_claims:
        statement = next(check.statement for check in material_checks if check.id == f"material.{item.id}")
        assert item.metric in statement
        assert item.mechanism in statement
        assert all(value in statement for value in item.values)


def test_agent_prohibitions_receive_actual_visible_sources_without_changing_memory_binding():
    from app.evals.compiler import ExecutableEvalPlan
    from app.evals.judge_tasks import JudgeTaskPlanner

    for case in all_cases():
        checks = [item for item in compile_case(case).semantic_checks
                  if item.gold_path.startswith("gold.prohibitions[")]
        if case.suite == "memory":
            assert all(item.support_mode == "bound_memory_source" for item in checks)
        if case.suite != "agent_tasks" or not checks:
            continue
        assert all(item.support_mode == "authoritative_visible" for item in checks)
        plan = ExecutableEvalPlan(case_id=case.id, suite=case.suite, task_type=case.task_type,
                                  checks=[], extractions=[], semantic_checks=checks)
        raw = {"id": "observed", "content": "Actual observed career fact.",
               "origin_type": "knowledge_module", "visible_to_agent": True,
               "authoritative_for_grounding": True}
        hidden = {**raw, "id": "hidden", "visible_to_agent": False}
        summary = {**raw, "id": "summary", "authoritative_for_grounding": False}
        observation = {"final_answer": "I relied on my records.",
                       "answer_spans": [{"id": "answer_1", "text": "I relied on my records."}],
                       "source_policy": case.source_policy.model_dump(),
                       "evidence": [raw, hidden, summary]}
        tasks = JudgeTaskPlanner().plan(plan, observation)
        for task in tasks:
            data = json.loads(task.messages[-1]["content"])
            assert data["evidence"] == [{"id": "e1", "content": raw["content"]}]


def test_mutation_contract_contains_exact_target_topics():
    for case in all_cases():
        if case.suite != "memory":
            continue
        plan = compile_case(case)
        mutation = next(check for check in plan.checks if check.evaluator_kind == "memory_mutation")
        assert mutation.expected["target_topic_ids"] == [topic.topic_id for topic in case.gold.target_topics]


@pytest.mark.parametrize("suite", ["runtime_failure", "memory"])
def test_material_source_flag_without_applicable_executor_is_preflight_error(suite):
    case = next(case for case in all_cases() if case.suite == suite
                and (suite != "memory" or not case.gold.material_claims))
    changed = case.model_copy(update={"source_policy": case.source_policy.model_copy(
        update={"allow_other_supported_material_claims": True})})
    with pytest.raises(CompileError, match="has no executor"):
        compile_case(changed)


def test_direct_material_grounding_requires_an_explicit_source_flag():
    case = next(case for case in all_cases() if case.suite == "agent_tasks")
    changed = case.model_copy(update={"source_policy": case.source_policy.model_copy(
        update={"allow_other_supported_material_claims": None})})
    with pytest.raises(CompileError, match="explicit"):
        compile_case(changed)


def test_retrieval_compiles_to_versioned_metrics_only_without_judge_or_case_specific_gate():
    case = next(case for case in all_cases() if case.suite == "retrieval")
    plan = compile_case(case)
    assert plan.contract_version == RETRIEVAL_CONTRACT_VERSION
    assert plan.pass_policy == {"mode": "metrics_only"}
    assert len(plan.checks) == 1 and plan.checks[0].evaluator_kind == "retrieval_metrics"
    assert not plan.semantic_checks and not plan.extractions
    assert plan.checks[0].expected["facts"] == [fact.model_dump(mode="json") for fact in case.facts]
    assert plan.checks[0].expected["answerability"] == case.answerability
    changed = case.model_copy(update={"query": "Different wording", "tags": ["other-slice"]})
    assert compile_case(changed) == plan


def test_runtime_recovery_checks_real_task_fulfillment_atomically():
    case = next(case for case in all_cases() if case.suite == "runtime_failure"
                and case.expected_outcome.mode == "recovery")
    plan = compile_case(case)
    completion = next(item for item in plan.semantic_checks if item.id == "runtime_task_completion")
    assert completion.presence == "required"
    assert completion.support_mode == "authoritative_visible"
    assert case.prompt in completion.statement


def test_structured_output_mapping_is_in_task_rubric_without_meta_judges():
    task_types = {"resume_generation", "local_revision", "interview_preparation", "interview_review", "jd_matching"}
    for case in all_cases():
        if case.task_type not in task_types:
            continue
        plan = compile_case(case)
        assert not any(item.id.startswith("extraction_complete.") for item in plan.semantic_checks)
        grounding = next(item for item in plan.semantic_checks if item.id == "answer_grounding")
        assert "映射必须" in grounding.statement
        assert "实际" in grounding.statement


def test_all_material_fields_map_to_direct_rubrics_and_extra_policy_is_consumed():
    for case in all_cases():
        if case.suite not in {"agent_tasks", "memory"}:
            continue
        plan = compile_case(case)
        assert not any(item.kind == "material_claims" for item in plan.extractions)
        for index, claim in enumerate(case.gold.material_claims):
            check = next(item for item in plan.semantic_checks if item.id == f"material.{claim.id}")
            assert check.presence == ("required" if claim.usage == "required" else "conditional")
            assert check.required_fixture_id == claim.required_fixture_id
            assert all(str(field) in check.statement for field in [claim.project, claim.metric,
                                                                 claim.values, claim.mechanism,
                                                                 claim.controls, claim.forbidden_attribution])
            paths = [mapping for mapping in plan.authoring_field_mappings
                     if mapping.path.startswith(f"gold.material_claims[{index}].")]
            assert paths and all(f"material.{claim.id}" in mapping.check_ids for mapping in paths)
        if case.source_policy.allow_other_supported_material_claims is not None:
            grounding = next(item for item in plan.semantic_checks
                             if item.id == ("answer_grounding" if case.suite == "agent_tasks" else "material_grounding"))
            assert "额外" in grounding.statement
            assert "预设" in grounding.statement
            changed = case.model_copy(update={"source_policy": case.source_policy.model_copy(
                update={"allow_other_supported_material_claims": False})})
            changed_check = next(item for item in compile_case(changed).semantic_checks if item.id == grounding.id)
            assert changed_check.statement != grounding.statement
            assert "不允许" in changed_check.statement


def test_historical_review_and_memory_checks_keep_their_distinct_source_time_scope():
    cases = all_cases()
    review = next(case for case in cases if case.task_type == "interview_review")
    plan = compile_case(review)
    historical = next(item for item in plan.semantic_checks if item.id == "historical_evidence_fidelity")
    technical = [item for item in plan.semantic_checks if item.id.startswith("technical.")]
    assert historical.required_fixture_id == review.gold.historical_event_fixture_id
    assert historical.support_mode == "authoritative_visible"
    assert technical and all(item.support_mode == "technical_rubric" for item in technical)
    assert "当前能力不得覆盖历史分类" in historical.statement
    for case in cases:
        if case.suite != "memory":
            continue
        for check in compile_case(case).semantic_checks:
            if check.support_mode == "bound_memory_source":
                assert check.observation_scope in {"target_mutation", "memory_recall"}
                assert check.target_step_id
