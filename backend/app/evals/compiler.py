"""Compile typed authoring Gold to a fully inspectable executable contract."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from app.evals.schema import (
    CONTRACT_VERSION, StrictModel, Metadata, AgentCaseBase, EvidenceFusionCase, Fact,
    InterviewPreparationCase, InterviewReviewCase, JDMatchingCase, LocalRevisionCase,
    MechanismMappingCase, MemoryCase, ProfileSummaryCase, ProjectDeepDiveCase,
    ResumeGenerationCase, RetrievalCase, RuntimeFailureCase,
)

CHECK_REGISTRY = frozenset({
    "terminal_completed", "final_answer_nonempty", "source_refs_valid",
    "evidence_capture_complete", "semantic", "semantic_group", "project_selection",
    "classification", "bullet_count", "local_revision", "question_limits",
    "topic_coverage", "fixture_visibility", "retrieval_metrics", "memory_mutation",
    "memory_target_state", "memory_source_binding", "memory_recall", "fault_triggered",
    "fault_recovery", "fault_attempts", "single_final_answer",
})
DECLARABLE_CHECKS = frozenset({
    "terminal_completed", "final_answer_nonempty", "source_refs_valid",
    "evidence_capture_complete", "single_final_answer",
})
# These lists are the compiler's authoring-field dispatch contract. A developer adding a
# new scoring model/field must implement its executor mapping before datasets can load.
SCORING_FIELD_REGISTRY = {
    "ProfileSummaryGold": {"facts", "project_attribution", "current_state_boundaries", "prohibitions", "material_claims"},
    "MechanismMappingGold": {"mappings", "prohibitions", "material_claims"},
    "ProjectDeepDiveGold": {"story_elements", "engineering_mechanisms", "min_engineering_mechanisms", "measurement_boundaries", "prohibitions", "material_claims"},
    "JDMatchingGold": {"requirement_classification", "project_priority", "capability_boundaries", "prohibitions", "material_claims"},
    "ResumeGenerationGold": {"project_selection", "max_bullets", "capability_coverage", "prohibitions", "material_claims"},
    "LocalRevisionGold": {"draft", "target_id", "require_target_changed", "preserve_equality", "preserve_order", "allow_additions", "required_target_facts", "prohibitions", "material_claims"},
    "InterviewPreparationGold": {"main_questions", "max_followups_per_question", "topic_groups", "min_topic_groups", "prohibitions", "material_claims"},
    "InterviewReviewGold": {"historical_event_fixture_id", "expected_classification", "priority_review", "priority_order_sensitive", "technical_answer_rubrics", "prohibitions", "material_claims"},
    "EvidenceFusionGold": {"facts", "required_fixture_id", "source_requirements", "prohibitions", "material_claims"},
    "MemoryGold": {"mutation", "target_topics", "semantics", "source_requirement", "recall", "prohibitions", "material_claims"},
    "RequiredFact": {"id", "statement", "proofs"},
    "Fact": {"id", "statement"},
    "MaterialClaim": {"id", "project", "metric", "values", "mechanism", "controls", "usage", "required_fixture_id", "forbidden_attribution"},
    "Attribution": {"project", "attribution"},
    "MechanismMapping": {"id", "project", "problem", "mechanism", "responsibility"},
    "ClassificationTarget": {"id", "description", "label"},
    "ProjectSelection": {"required", "optional", "max_selected"},
    "ProjectPriority": {"required", "any_of", "order_sensitive"},
    "CapabilityCoverage": {"all_of", "any_of_groups"},
    "DraftBullet": {"id", "text"},
    "TechnicalRubric": {"id", "topic", "reference", "statement"},
    "MemoryMutationPolicy": {"should_write", "target_step_id", "allowed_operations", "allowed_kinds", "expected_count", "target_existing", "require_new_source"},
    "MemoryTargetState": {"topic_id", "description", "state", "active_count", "preserve_history", "reject_conflicting_active"},
    "MemorySourceRequirement": {"mode", "fixture_id", "statement"},
    "MemoryRecall": {"step_id", "required_topics", "require_search_memory", "require_final_answer_use"},
    "FaultInjection": {"target_tools", "trigger", "occurrence", "error_code", "message", "retryable", "then"},
    "RecoveryExpected": {"mode", "task_completed", "tool_error_count", "max_attempts", "subsequent_real_success", "single_final_answer"},
    "GracefulStopExpected": {"mode", "task_completed", "max_attempts", "state_evidence_unavailable", "no_unsupported_project_details", "single_final_answer"},
    "SourcePolicy": {"fact_time_policy", "support", "allow_other_supported_material_claims"},
}


def _validate_scoring_mapping(value, path):
    if isinstance(value, Metadata):
        return
    if isinstance(value, BaseModel):
        allowed = SCORING_FIELD_REGISTRY.get(type(value).__name__)
        fields = set(type(value).model_fields)
        if allowed is None or fields != allowed:
            raise CompileError(f"Scoring field has no compiler/executor mapping at {path}: {fields - (allowed or set())}")
        for name in fields:
            _validate_scoring_mapping(getattr(value, name), f"{path}.{name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_scoring_mapping(item, f"{path}[{index}]")
EXTRACTION_KINDS = frozenset({
    "project_selection", "classification", "bullets", "questions", "topic_set",
    "memory_topics",
})


class CompileError(ValueError):
    pass


class CompiledCheck(StrictModel):
    id: str
    gold_path: str
    evaluator_kind: str
    expected: JsonValue
    required: bool = True


class SemanticCheck(StrictModel):
    id: str
    gold_path: str
    statement: str
    support_mode: Literal[
        "authoritative_visible", "answer_only", "technical_rubric", "bound_memory_source"
    ] = "authoritative_visible"
    required: bool = True
    required_fixture_id: str | None = None
    presence: Literal["required", "conditional", "absence", "memory"] = "required"
    required_extraction_id: str | None = None
    required_item_id: str | None = None
    observation_scope: Literal["answer", "target_mutation", "memory_recall"] = "answer"
    target_step_id: str | None = None


class ExtractionSpec(StrictModel):
    id: str
    kind: str
    allowed_ids: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    instructions: str


class AuthoringFieldMapping(StrictModel):
    path: str
    check_ids: list[str]
    evaluator_kinds: list[str]


class ExecutableEvalPlan(StrictModel):
    contract_version: str = CONTRACT_VERSION
    case_id: str
    suite: str
    task_type: str
    checks: list[CompiledCheck]
    semantic_checks: list[SemanticCheck]
    extractions: list[ExtractionSpec]
    authoring_field_mappings: list[AuthoringFieldMapping] = Field(default_factory=list)
    pass_policy: JsonValue = {"mode": "all_required_checks"}


def compile_case(case) -> ExecutableEvalPlan:
    """Every typed Gold field is consumed here; there is no legacy evaluator dispatch."""
    if isinstance(case, RetrievalCase):
        from app.evals.retrieval_contract import RETRIEVAL_CONTRACT_VERSION
        if "retrieval_metrics" not in CHECK_REGISTRY:
            raise CompileError("Retrieval metrics have no executor mapping")
        _validate_scoring_mapping(case.facts, "facts")
        check = CompiledCheck(id="retrieval_metrics", gold_path="facts+answerability",
                              evaluator_kind="retrieval_metrics",
                              expected={"contract_version": RETRIEVAL_CONTRACT_VERSION,
                                        "answerability": case.answerability,
                                        "facts": [fact.model_dump(mode="json") for fact in case.facts]})
        return ExecutableEvalPlan(
            contract_version=RETRIEVAL_CONTRACT_VERSION, case_id=case.id,
            suite=case.suite, task_type=case.task_type,
            checks=[check], semantic_checks=[], extractions=[],
            authoring_field_mappings=[AuthoringFieldMapping(
                path=path, check_ids=[check.id], evaluator_kinds=[check.evaluator_kind])
                for path in ["answerability", *[f"facts[{i}]" for i in range(len(case.facts))]]],
            pass_policy={"mode": "metrics_only"})
    _validate_scoring_mapping(case.source_policy, "source_policy")
    material_policy_applies = isinstance(case, AgentCaseBase) or (
        isinstance(case, MemoryCase) and bool(case.gold.material_claims))
    flag = case.source_policy.allow_other_supported_material_claims
    if material_policy_applies and flag is None:
        raise CompileError("Material grounding requires an explicit allow_other_supported_material_claims policy")
    if not material_policy_applies and flag is not None:
        raise CompileError("allow_other_supported_material_claims has no executor for this task")
    if isinstance(case, RuntimeFailureCase):
        _validate_scoring_mapping(case.fault_injection, "fault_injection")
        _validate_scoring_mapping(case.expected_outcome, "expected_outcome")
    else:
        _validate_scoring_mapping(case.gold, "gold")
    checks: list[CompiledCheck] = []
    semantic: list[SemanticCheck] = []
    extractions: list[ExtractionSpec] = []

    def add(check_id, path, kind, expected=True, *, required=True):
        if kind not in CHECK_REGISTRY:
            raise CompileError(f"Scoring field {path} has no executor: {kind}")
        checks.append(CompiledCheck(id=check_id, gold_path=path, evaluator_kind=kind,
                                    expected=expected, required=required))

    def sem(check_id, path, statement, *, required=True, mode="authoritative_visible", fixture=None,
            presence="required", extraction=None, item=None, scope="answer", step=None):
        time_rules = {
            "current_snapshot": "事实时点：以本次冻结corpus/实际可见用户Evidence为准，不凭外部现实变化修改事实。",
            "preserve_event_time": "事实时点：保留历史事件发生时的事实/表现，当前能力不得覆盖历史分类。",
            "user_reported_state": "事实时点：忠实保存用户自报的学习稳定状态，不评其一般技术知识真伪。",
        }
        statement = time_rules[case.source_policy.fact_time_policy] + statement
        semantic.append(SemanticCheck(id=check_id, gold_path=path, statement=statement,
                                     support_mode=mode, required=required,
                                     required_fixture_id=fixture, presence=presence,
                                     required_extraction_id=extraction, required_item_id=item,
                                     observation_scope=scope, target_step_id=step))
        add(check_id, path, "semantic", "PASS", required=required)

    def facts(items: list[Fact], path, *, required=True, mode="authoritative_visible", prefix="fact",
              presence="required", extraction=None, item=None):
        ids = []
        for index, fact in enumerate(items):
            check_id = f"{prefix}.{fact.id}"
            ids.append(check_id)
            sem(check_id, f"{path}[{index}]", fact.statement, required=required, mode=mode,
                presence=presence, extraction=extraction, item=item)
        return ids

    def extract(extraction_id, kind, allowed=(), labels=(), instructions=""):
        if kind not in EXTRACTION_KINDS:
            raise CompileError(f"Unregistered extraction {kind}")
        extractions.append(ExtractionSpec(id=extraction_id, kind=kind,
                                         allowed_ids=list(allowed), labels=list(labels),
                                         instructions=instructions))

    if len(case.checks) != len(set(case.checks)):
        raise CompileError("Duplicate declared check")
    for index, kind in enumerate(case.checks):
        if kind not in DECLARABLE_CHECKS:
            raise CompileError(f"Unknown or unbound scoring check {kind}")
        add(kind, f"checks[{index}]", kind)

    if isinstance(case, RuntimeFailureCase):
        fault = case.fault_injection.model_dump(mode="json")
        expected = case.expected_outcome.model_dump(mode="json")
        add("fault_triggered", "fault_injection+source_policy", "fault_triggered", fault)
        add("fault_attempts", "expected_outcome.max_attempts", "fault_attempts", {
            "max_attempts": expected["max_attempts"], "target_tools": fault["target_tools"],
        })
        if expected["mode"] == "recovery":
            add("fault_recovery", "expected_outcome", "fault_recovery", expected)
            sem("runtime_task_completion", "expected_outcome.task_completed",
                "最终回答实质完成当前用户prompt的主要任务，提供请求的项目介绍/分析；"
                "仅说执行完成、只报告工具成功或空泛回答不满足本原子检查。判断只能基于实际可见证据，"
                "不判断整个case最终PASS/FAIL。当前任务：" + case.prompt)
        else:
            sem("evidence_unavailable", "expected_outcome.state_evidence_unavailable",
                "最终回答明确说明本次证据不可取得，并正常停止。", mode="answer_only")
            sem("no_unsupported_project_details", "expected_outcome.no_unsupported_project_details",
                "回答没有根据文件名或模型先验编造本次未获得证据的任何关键项目细节或数字。",
                presence="absence")
            add("graceful_task_completed", "expected_outcome", "terminal_completed")
        if expected["single_final_answer"] and "single_final_answer" not in case.checks:
            add("single_final_answer", "expected_outcome.single_final_answer", "single_final_answer")
    else:
        gold = case.gold
        if isinstance(case, AgentCaseBase):
            output_mapping = {
                "jd_matching": "需求分类和优先项目映射必须对应实际表达，不凭Gold补齐未表达的分类。",
                "resume_generation": "简历条目映射必须包含实际交付的全部bullet，包括超出上限的条目；解释文字不算交付稿。",
                "local_revision": "修改稿映射必须包含实际交付的全部bullet及新增、重复、重排序条目；引用旧稿不能冒充交付稿。",
                "interview_preparation": "问题映射必须包含实际交付的全部主问题及每题全部追问，包括超出上限的条目；说明不算追问。",
                "interview_review": "历史表现分类映射必须对应实际表达，不凭Gold补齐未表达的主题分类。",
            }.get(case.task_type, "")
            material_policy = (
                "允许预设主张之外的额外量化主张，但每条仍须由正确项目、指标、数值单位/范围及实际因果关系的证据支持。"
                if flag else
                "不允许预设material主张之外的项目量化主张，即使额外主张另有证据也不满足本任务政策。"
            )
            sem("answer_grounding", "source_policy",
                "答案中的所有实质职业事实、职责、项目能力和量化主张均由本次Agent实际可见的权威事实证据支持；"
                "不得把JD要求、模型生成摘要、Evaluator Gold或一般常识变成用户经历。未知事实须说明未知。"
                "并列事实不能补造成因果，数值相同但项目/指标/口径不同不构成支持。" + material_policy +
                "预设material主张身份：" + str([
                    {"id": claim.id, "project": claim.project, "metric": claim.metric}
                    for claim in gold.material_claims]) + output_mapping,
                presence="conditional")
        for index, prohibition in enumerate(gold.prohibitions):
            sem(f"prohibition.{prohibition.id}", f"gold.prohibitions[{index}]",
                f"确认答案/Memory没有以下禁止行为（允许明确否认该行为）：{prohibition.statement}",
                mode=("bound_memory_source" if isinstance(case, MemoryCase) else
                      "authoritative_visible" if isinstance(case, AgentCaseBase) else "answer_only"),
                presence="absence", scope="target_mutation" if isinstance(case, MemoryCase) else "answer",
                step=case.gold.mutation.target_step_id if isinstance(case, MemoryCase) else None)
        if gold.material_claims:
            if isinstance(case, MemoryCase):
                sem("material_grounding", "source_policy",
                    "实际答案中的所有项目量化主张须由本次实际可见权威事实证据支持，正确保留项目、指标、数值口径、"
                    "单位/范围及明确因果关系；不能将并列描述补造成因果。" +
                    ("允许预设主张之外的额外主张，但每条仍须有上述支持。" if flag else
                     "不允许预设material主张之外的项目量化主张，即使另有支持也不满足政策。") +
                    "预设主张身份：" + str([{"id": claim.id, "project": claim.project, "metric": claim.metric}
                                               for claim in gold.material_claims]),
                    presence="conditional")
            for index, claim in enumerate(gold.material_claims):
                sem(f"material.{claim.id}", f"gold.material_claims[{index}]",
                    f"{'必须表达并由证据支持' if claim.usage == 'required' else '若表达则必须由证据支持，否则PASS'}："
                    f"项目={claim.project}; 指标={claim.metric}; 数字={claim.values}; 机制={claim.mechanism}; "
                    f"控制条件={claim.controls}; 禁止归因={claim.forbidden_attribution}。"
                    "判断实际表达及其证据支持，不凭Gold补齐未表达的内容；"
                    "允许等义改写，须忠实保留数值单位、限定范围与测量口径，不能仅因其他项目有相同数字而PASS。",
                    fixture=claim.required_fixture_id,
                    presence="required" if claim.usage == "required" else "conditional")

        if isinstance(case, ProfileSummaryCase):
            facts(gold.facts, "gold.facts")
            facts(gold.current_state_boundaries, "gold.current_state_boundaries", prefix="boundary",
                  presence="conditional")
            for index, item in enumerate(gold.project_attribution):
                sem(f"attribution.{item.project}", f"gold.project_attribution[{index}]",
                    f"准确归属：{item.project} 是 {item.attribution}。")
        elif isinstance(case, MechanismMappingCase):
            for index, item in enumerate(gold.mappings):
                sem(f"mapping.{item.id}", f"gold.mappings[{index}]",
                    f"项目={item.project}; 问题={item.problem}; 机制={item.mechanism}; 职责={item.responsibility}。"
                    "应正确解释机制与问题的对应关系，不可混为其他项目机制。")
        elif isinstance(case, ProjectDeepDiveCase):
            facts(gold.story_elements, "gold.story_elements")
            ids = facts(gold.engineering_mechanisms, "gold.engineering_mechanisms",
                        required=False, prefix="engineering")
            add("engineering_minimum", "gold.min_engineering_mechanisms", "semantic_group", {
                "mode": "min_count", "min": gold.min_engineering_mechanisms, "check_ids": ids,
            })
            facts(gold.measurement_boundaries, "gold.measurement_boundaries", prefix="boundary",
                  presence="conditional")
        elif isinstance(case, JDMatchingCase):
            extract("requirement_classification", "classification",
                    [item.id for item in gold.requirement_classification],
                    sorted({item.label for item in gold.requirement_classification}),
                    "完整映射实际表达的需求分类；没有提及的需求不填，不凭Gold补齐。"
                    "直接但有边界需映射direct_with_boundary。")
            add("requirement_classification", "gold.requirement_classification", "classification", {
                "extraction_id": "requirement_classification",
                "items": [item.model_dump(mode="json") for item in gold.requirement_classification],
            })
            extract("project_priority", "project_selection", gold.project_priority.required + gold.project_priority.any_of,
                    instructions="提取最终建议优先突出的项目集合，不是全文所有提到项目。顺序不影响。")
            add("project_priority", "gold.project_priority", "project_selection", {
                "extraction_id": "project_priority", **gold.project_priority.model_dump(mode="json"),
            })
            facts(gold.capability_boundaries, "gold.capability_boundaries", prefix="boundary")
            for item in gold.requirement_classification:
                sem(f"requirement_support.{item.id}", "gold.requirement_classification",
                    f"对需求 {item.description} 的证据对应、能力边界正确。预期类别 {item.label}。")
        elif isinstance(case, ResumeGenerationCase):
            extract("selected_projects", "project_selection",
                    gold.project_selection.required + gold.project_selection.optional,
                    instructions="提取输出简历bullet实际选择的项目集合，未知项目仍明确返回unknown。")
            add("project_selection", "gold.project_selection", "project_selection", {
                "extraction_id": "selected_projects", **gold.project_selection.model_dump(mode="json"),
            })
            extract("resume_bullets", "bullets", instructions="选择实际交付简历的全部bullet所对应的已有answer_span_id，"
                    "包括超出数量上限的条目。不将解释文字作为bullet，不重抄正文、不按Gold补齐。")
            add("resume_bullet_limit", "gold.max_bullets", "bullet_count", {
                "extraction_id": "resume_bullets", "max": gold.max_bullets,
            })
            facts(gold.capability_coverage.all_of, "gold.capability_coverage.all_of", prefix="capability")
            for index, group in enumerate(gold.capability_coverage.any_of_groups):
                ids = facts(group, f"gold.capability_coverage.any_of_groups[{index}]",
                            required=False, prefix=f"capability_group_{index}")
                add(f"capability_any_{index}", f"gold.capability_coverage.any_of_groups[{index}]",
                    "semantic_group", {"mode": "min_count", "min": 1, "check_ids": ids})
        elif isinstance(case, LocalRevisionCase):
            extract("revised_bullets", "bullets", [item.id for item in gold.draft],
                    instructions="按实际交付顺序选择最终修改稿全部bullet的已有answer_span_id和条目ID；"
                    "解释中引用旧稿不得冒充最终稿。包含新增、重复、重排序和超出上限的条目，未知条目ID用null。"
                    "不重抄正文、不按Gold补齐。")
            add("local_revision", "gold.draft+target_id+require_target_changed+preserve_equality+preserve_order+allow_additions", "local_revision", {
                "extraction_id": "revised_bullets", "draft": [item.model_dump() for item in gold.draft],
                "target_id": gold.target_id, "require_target_changed": gold.require_target_changed,
                "preserve_equality": gold.preserve_equality, "preserve_order": gold.preserve_order,
                "allow_additions": gold.allow_additions,
            })
            facts(gold.required_target_facts, "gold.required_target_facts", prefix="target_fact",
                  extraction="revised_bullets", item=gold.target_id)
        elif isinstance(case, InterviewPreparationCase):
            extract("interview_questions", "questions", [item.id for item in gold.topic_groups],
                    instructions="完整映射实际交付的全部主问题、真实项目ID和topic IDs；"
                    "主问题选已有answer_span_id，每题全部实际追问选已有followup_span_ids，包括超出上限的条目。"
                    "不要把说明误当追问，不重抄正文、不按Gold补齐。")
            add("question_limits", "gold.main_questions+max_followups_per_question", "question_limits", {
                "extraction_id": "interview_questions", "main_questions": gold.main_questions,
                "max_followups_per_question": gold.max_followups_per_question,
            })
            add("topic_coverage", "gold.topic_groups+min_topic_groups", "topic_coverage", {
                "extraction_id": "interview_questions", "topic_ids": [item.id for item in gold.topic_groups],
                "min": gold.min_topic_groups,
            })
            for index, item in enumerate(gold.topic_groups):
                sem(f"topic_grounding.{item.id}", f"gold.topic_groups[{index}]",
                    f"若问题映射到主题 {item.id}，则确实体现该主题且来自实际经历：{item.statement}；未映射则PASS。",
                    presence="conditional")
        elif isinstance(case, InterviewReviewCase):
            extract("review_classification", "classification", [item.id for item in gold.expected_classification],
                    sorted({item.label for item in gold.expected_classification}),
                    "完整映射实际表达的历史面试逐主题分类，不凭Gold补齐未表达分类。"
                    "不根据现在能力或面试官提到某技术替换历史表现。"
                    "以下仅为topic身份映射，label必须提取答案实际表达：" + str([
                        {"id": item.id, "description": item.description} for item in gold.expected_classification]))
            add("review_classification", "gold.expected_classification", "classification", {
                "extraction_id": "review_classification",
                "items": [item.model_dump() for item in gold.expected_classification],
            })
            extract("priority_review", "topic_set", gold.priority_review,
                    instructions="只提取最终下一轮优先复测的主题集合；顺序不敏感。")
            add("priority_review", "gold.priority_review+priority_order_sensitive", "topic_coverage", {
                "extraction_id": "priority_review", "topic_ids": gold.priority_review,
                "min": len(gold.priority_review), "exact_set": True,
                "order_sensitive": gold.priority_order_sensitive,
            })
            sem("historical_evidence_fidelity", "gold.historical_event_fixture_id",
                "分类忠实于这轮实际面试问答，不以当前项目状态或未检验知识替代历史事件事实。",
                fixture=gold.historical_event_fixture_id, presence="conditional")
            for index, rubric in enumerate(gold.technical_answer_rubrics):
                sem(f"technical.{rubric.id}", f"gold.technical_answer_rubrics[{index}]",
                    f"独立一般技术答案rubric：主题={rubric.topic}; 参考={rubric.reference}; {rubric.statement}",
                    mode="technical_rubric")
        elif isinstance(case, EvidenceFusionCase):
            facts(gold.facts, "gold.facts")
            for index, item in enumerate(gold.source_requirements):
                sem(f"source_requirement.{item.id}", f"gold.source_requirements[{index}]",
                    item.statement, fixture=gold.required_fixture_id)
            add("supplement_visibility", "gold.required_fixture_id", "fixture_visibility", {
                "fixture_id": gold.required_fixture_id, "require_visible_once": True,
            })
        elif isinstance(case, MemoryCase):
            add("memory_mutation", "gold.mutation+target_topics+source_policy", "memory_mutation", {
                **gold.mutation.model_dump(), "target_topic_ids": [item.topic_id for item in gold.target_topics],
            })
            if gold.target_topics:
                extract("memory_topics", "memory_topics", [item.topic_id for item in gold.target_topics],
                        instructions="映射观察中全部Memory实体属于哪些目标主题及用户自报学习状态。其他主题不要映射为目标。"
                        "同一Memory可属于多个不同主题；每memory/topic配对只输出一次。这里只判用户表达稳定/不稳定，"
                        "不判断Java技术知识真伪。主题身份映射：" + str([
                            {"id": item.topic_id, "description": item.description} for item in gold.target_topics]))
                add("memory_target_state", "gold.target_topics", "memory_target_state", {
                    "topics": [item.model_dump() for item in gold.target_topics],
                    "extraction_id": "memory_topics", "target_step_id": gold.mutation.target_step_id,
                })
            for index, item in enumerate(gold.semantics):
                sem(f"memory_semantics.{item.id}", f"gold.semantics[{index}]",
                    f"目标turn成功mutation后相关Memory忠实表达：{item.statement}；且被其实际绑定的source正文直接支持。",
                    mode="bound_memory_source", presence="memory", scope="target_mutation",
                    step=gold.mutation.target_step_id)
            if gold.source_requirement:
                add("memory_source_binding", "gold.source_requirement", "memory_source_binding", {
                        **gold.source_requirement.model_dump(), "target_step_id": gold.mutation.target_step_id,
                        "target_topic_ids": [item.topic_id for item in gold.target_topics],
                })
                sem("memory_source_support", "gold.source_requirement.statement",
                    gold.source_requirement.statement, mode="bound_memory_source",
                    fixture=gold.source_requirement.fixture_id, presence="memory", scope="target_mutation",
                    step=gold.mutation.target_step_id)
            if gold.recall:
                add("memory_recall", "gold.recall", "memory_recall", gold.recall.model_dump())
                sem("memory_recall_answer", "gold.recall.require_final_answer_use",
                    "跨会话B最终回答正确使用实际召回的目标Memory，完整保留关键事实及来源口径，不可仅靠Knowledge或先验碰巧答对。",
                    mode="bound_memory_source", presence="required", scope="memory_recall",
                    step=gold.recall.step_id)
        else:
            raise CompileError(f"No compiler for typed case {type(case).__name__}")

    check_ids = [item.id for item in checks]
    extraction_ids = [item.id for item in extractions]
    if len(set(check_ids)) != len(check_ids) or len(set(extraction_ids)) != len(extraction_ids):
        raise CompileError("Duplicate compiled check or extraction ID")
    # Expand each scoring leaf to its executable checks, retaining exact authoring paths.
    # Empty lists/None are vacuous contracts; nonempty declarations must have a route.
    def leaves(value, path):
        if isinstance(value, Metadata):
            return
        if isinstance(value, BaseModel):
            for name in type(value).model_fields:
                yield from leaves(getattr(value, name), f"{path}.{name}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from leaves(item, f"{path}[{index}]")
        elif value is not None:
            yield path

    roots = {"gold", "source_policy", "pass_policy", "fault_injection", "expected_outcome", "checks"}
    def declared_paths(check):
        segments = check.gold_path.split("+")
        parent = segments[0].split(".")[0]
        return [segment if segment.split(".")[0].split("[")[0] in roots else f"{parent}.{segment}"
                for segment in segments]

    scoring = [(case.source_policy, "source_policy"), (case.checks, "checks")]
    if isinstance(case, RuntimeFailureCase):
        scoring += [(case.fault_injection, "fault_injection"), (case.expected_outcome, "expected_outcome")]
    else:
        scoring += [(case.gold, "gold")]
    mappings = []
    for value, root in scoring:
        for path in leaves(value, root):
            related = [check for check in checks if any(
                path == declared or path.startswith(declared + ".") or path.startswith(declared + "[")
                or (declared == "gold.preserve" and path.startswith("gold.preserve_"))
                for declared in declared_paths(check)
            )]
            # Fact time and visibility are compiler policy for every semantic check.
            if root == "source_policy" and not related:
                related = [check for check in checks if check.evaluator_kind in {
                    "semantic", "memory_mutation", "memory_source_binding", "fixture_visibility"}]
            if not related:
                raise CompileError(f"Scoring field has no emitted executor check: {path}")
            mappings.append(AuthoringFieldMapping(path=path, check_ids=[check.id for check in related],
                                                  evaluator_kinds=sorted({check.evaluator_kind for check in related})))
    return ExecutableEvalPlan(case_id=case.id, suite=case.suite, task_type=case.task_type,
                              checks=checks, semantic_checks=semantic, extractions=extractions,
                              authoring_field_mappings=mappings,
                              pass_policy={"mode": "all_required_checks"})
