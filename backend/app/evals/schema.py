"""The single authoring and bounded-Judge contract for prebaseline Eval.

No unknown authoring field is accepted. Case identifiers carry no scoring semantics.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from app.evals.retrieval_contract import RetrievalCase

SCHEMA_VERSION = "2"
CONTRACT_VERSION = "eval_contract_v3"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Metadata(StrictModel):
    notes: list[str] = Field(default_factory=list)
    annotation_origin: str | None = None
    critical_identity: bool = False


class FixtureInput(StrictModel):
    id: str
    path: str


class InputStep(StrictModel):
    id: str
    conversation: str = "A"
    user: str
    user_fixtures: list[FixtureInput] = Field(default_factory=list)


class Setup(StrictModel):
    memory: Literal["empty"] = "empty"
    conversation: Literal["new"] = "new"
    user_fixtures: list[FixtureInput] = Field(default_factory=list)
    seed_steps: list[InputStep] = Field(default_factory=list)


class SourcePolicy(StrictModel):
    fact_time_policy: Literal["current_snapshot", "preserve_event_time", "user_reported_state"]
    support: Literal["actual_visible_sources"] = "actual_visible_sources"
    allow_other_supported_material_claims: bool | None = None


class Fact(StrictModel):
    id: str
    statement: str


class MaterialClaim(StrictModel):
    id: str
    project: str
    metric: str
    values: list[str]
    mechanism: str
    controls: list[str] = Field(default_factory=list)
    usage: Literal["required", "optional"]
    required_fixture_id: str | None = None
    forbidden_attribution: str | None = None


class GoldBase(StrictModel):
    prohibitions: list[Fact] = Field(default_factory=list)
    material_claims: list[MaterialClaim] = Field(default_factory=list)


class Attribution(StrictModel):
    project: str
    attribution: str


class ProfileSummaryGold(GoldBase):
    facts: list[Fact]
    project_attribution: list[Attribution]
    current_state_boundaries: list[Fact]


class MechanismMapping(StrictModel):
    id: str
    project: str
    problem: str
    mechanism: str
    responsibility: str


class MechanismMappingGold(GoldBase):
    mappings: list[MechanismMapping]


class ProjectDeepDiveGold(GoldBase):
    story_elements: list[Fact]
    engineering_mechanisms: list[Fact]
    min_engineering_mechanisms: int = Field(ge=1)
    measurement_boundaries: list[Fact]


class ClassificationTarget(StrictModel):
    id: str
    description: str
    label: Literal["direct", "direct_with_boundary", "partial", "gap", "unstable", "expression_or_abstraction", "stable"]


class ProjectSelection(StrictModel):
    required: list[str]
    optional: list[str]
    max_selected: int = Field(ge=1)


class ProjectPriority(StrictModel):
    required: list[str]
    any_of: list[str] = Field(default_factory=list)
    order_sensitive: Literal[False] = False


class JDMatchingGold(GoldBase):
    requirement_classification: list[ClassificationTarget]
    project_priority: ProjectPriority
    capability_boundaries: list[Fact]

    @model_validator(mode="after")
    def jd_labels(self):
        if any(item.label not in {"direct", "direct_with_boundary", "partial", "gap"}
               for item in self.requirement_classification):
            raise ValueError("JD classification label is not a requirement category")
        return self


class CapabilityCoverage(StrictModel):
    all_of: list[Fact]
    any_of_groups: list[list[Fact]] = Field(default_factory=list)


class ResumeGenerationGold(GoldBase):
    project_selection: ProjectSelection
    max_bullets: int = Field(ge=1)
    capability_coverage: CapabilityCoverage


class DraftBullet(StrictModel):
    id: str
    text: str


class LocalRevisionGold(GoldBase):
    draft: list[DraftBullet]
    target_id: str
    require_target_changed: Literal[True] = True
    preserve_equality: Literal["normalized_exact"] = "normalized_exact"
    preserve_order: Literal[True] = True
    allow_additions: Literal[False] = False
    required_target_facts: list[Fact]

    @model_validator(mode="after")
    def target_exists(self):
        ids = [item.id for item in self.draft]
        if len(ids) != len(set(ids)) or self.target_id not in ids:
            raise ValueError("local revision requires unique draft IDs and an existing target")
        return self


class InterviewPreparationGold(GoldBase):
    main_questions: int = Field(ge=1)
    max_followups_per_question: int = Field(ge=0)
    topic_groups: list[Fact]
    min_topic_groups: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_min(self):
        if self.min_topic_groups > len(self.topic_groups):
            raise ValueError("min_topic_groups exceeds available topic groups")
        return self


class TechnicalRubric(StrictModel):
    id: str
    topic: str
    reference: str
    statement: str


class InterviewReviewGold(GoldBase):
    historical_event_fixture_id: str
    expected_classification: list[ClassificationTarget]
    priority_review: list[str]
    priority_order_sensitive: Literal[False] = False
    technical_answer_rubrics: list[TechnicalRubric]

    @model_validator(mode="after")
    def event_labels(self):
        if any(item.label not in {"unstable", "expression_or_abstraction", "stable"}
               for item in self.expected_classification):
            raise ValueError("Interview classification label is not an event performance category")
        return self


class EvidenceFusionGold(GoldBase):
    facts: list[Fact]
    required_fixture_id: str
    source_requirements: list[Fact]


class CaseBase(StrictModel):
    id: str
    name: str
    setup: Setup = Field(default_factory=Setup)
    source_policy: SourcePolicy
    checks: list[str]
    metadata: Metadata = Field(default_factory=Metadata)
    variance_repeats: int = Field(default=0, ge=0)


class AgentCaseBase(CaseBase):
    suite: Literal["agent_tasks"] = "agent_tasks"
    prompt: str


class ProfileSummaryCase(AgentCaseBase):
    task_type: Literal["profile_summary"]
    gold: ProfileSummaryGold


class MechanismMappingCase(AgentCaseBase):
    task_type: Literal["mechanism_mapping"]
    gold: MechanismMappingGold


class ProjectDeepDiveCase(AgentCaseBase):
    task_type: Literal["project_deep_dive"]
    gold: ProjectDeepDiveGold


class JDMatchingCase(AgentCaseBase):
    task_type: Literal["jd_matching"]
    gold: JDMatchingGold


class ResumeGenerationCase(AgentCaseBase):
    task_type: Literal["resume_generation"]
    gold: ResumeGenerationGold


class LocalRevisionCase(AgentCaseBase):
    task_type: Literal["local_revision"]
    gold: LocalRevisionGold


class InterviewPreparationCase(AgentCaseBase):
    task_type: Literal["interview_preparation"]
    gold: InterviewPreparationGold


class InterviewReviewCase(AgentCaseBase):
    task_type: Literal["interview_review"]
    gold: InterviewReviewGold


class EvidenceFusionCase(AgentCaseBase):
    task_type: Literal["evidence_fusion"]
    gold: EvidenceFusionGold


AgentTaskCase = Annotated[
    ProfileSummaryCase | MechanismMappingCase | ProjectDeepDiveCase | JDMatchingCase
    | ResumeGenerationCase | LocalRevisionCase | InterviewPreparationCase
    | InterviewReviewCase | EvidenceFusionCase,
    Field(discriminator="task_type"),
]


class MemoryMutationPolicy(StrictModel):
    should_write: bool
    target_step_id: str
    allowed_operations: list[Literal["ADD", "MERGE", "SUPERSEDE", "ARCHIVE"]]
    allowed_kinds: list[Literal["CAREER_GOAL", "PREFERENCE", "EXPERIENCE", "LEARNING", "ACHIEVEMENT"]]
    expected_count: int | None = Field(default=None, ge=0)
    target_existing: bool = False
    require_new_source: bool = True

    @model_validator(mode="after")
    def no_write_contract(self):
        if not self.should_write and (self.expected_count != 0 or self.allowed_operations):
            raise ValueError("no-write policy requires exactly zero mutations and no operations")
        if self.should_write and (not self.allowed_operations or not self.allowed_kinds):
            raise ValueError("write policy requires operations and kinds")
        return self


class MemoryTargetState(StrictModel):
    topic_id: str
    description: str
    state: Literal["unstable", "stable", "present"]
    active_count: int = Field(default=1, ge=1)
    preserve_history: bool = False
    reject_conflicting_active: bool = True


class MemorySourceRequirement(StrictModel):
    mode: Literal["target_message", "bound_section_or_chunk", "fixture_message"]
    fixture_id: str | None = None
    statement: str

    @model_validator(mode="after")
    def valid_fixture(self):
        if (self.mode == "fixture_message") != (self.fixture_id is not None):
            raise ValueError("fixture_message source requirement needs a fixture ID exclusively")
        return self


class MemoryRecall(StrictModel):
    step_id: str
    required_topics: list[str]
    require_search_memory: Literal[True] = True
    require_final_answer_use: Literal[True] = True


class MemoryGold(GoldBase):
    mutation: MemoryMutationPolicy
    target_topics: list[MemoryTargetState]
    semantics: list[Fact]
    source_requirement: MemorySourceRequirement | None = None
    recall: MemoryRecall | None = None


class MemoryCase(CaseBase):
    suite: Literal["memory"] = "memory"
    task_type: Literal["memory_lifecycle"] = "memory_lifecycle"
    steps: list[InputStep] = Field(min_length=1)
    gold: MemoryGold

    @model_validator(mode="after")
    def valid_steps(self):
        ids = [step.id for step in self.steps]
        if len(set(ids)) != len(ids) or self.gold.mutation.target_step_id not in ids:
            raise ValueError("target_step_id must reference a unique case step")
        if self.gold.recall and self.gold.recall.step_id not in ids:
            raise ValueError("recall.step_id must reference a case step")
        if self.gold.mutation.target_existing and not self.setup.seed_steps:
            raise ValueError("target_existing requires seed_steps")
        return self


class FaultInjection(StrictModel):
    target_tools: list[Literal["search_knowledge", "read_source", "search_memory", "update_user_model", "list_documents", "save_source"]] = Field(min_length=1)
    trigger: Literal["occurrence", "every_call"]
    occurrence: int | None = Field(default=None, ge=1)
    error_code: Literal["VALIDATION_ERROR", "NOT_FOUND", "DOMAIN_ERROR", "TIMEOUT", "INTERNAL_ERROR", "CANCELLED", "POSSIBLE_DUPLICATE"]
    message: str
    retryable: bool
    then: Literal["real_tool", "continue_fault"]

    @model_validator(mode="after")
    def valid_trigger(self):
        if len(set(self.target_tools)) != len(self.target_tools):
            raise ValueError("duplicate fault target tool")
        if self.trigger == "occurrence" and self.occurrence is None:
            raise ValueError("occurrence trigger requires occurrence")
        if self.trigger == "every_call" and self.occurrence is not None:
            raise ValueError("every_call cannot declare occurrence")
        if (self.trigger == "every_call") != (self.then == "continue_fault"):
            raise ValueError("fault then behavior contradicts trigger")
        return self


class RecoveryExpected(StrictModel):
    mode: Literal["recovery"]
    task_completed: Literal[True] = True
    tool_error_count: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    subsequent_real_success: Literal[True] = True
    single_final_answer: Literal[True] = True


class GracefulStopExpected(StrictModel):
    mode: Literal["graceful_stop"]
    task_completed: Literal[True] = True
    max_attempts: int = Field(ge=1)
    state_evidence_unavailable: Literal[True] = True
    no_unsupported_project_details: Literal[True] = True
    single_final_answer: Literal[True] = True


RuntimeExpected = Annotated[RecoveryExpected | GracefulStopExpected, Field(discriminator="mode")]


class RuntimeFailureCase(CaseBase):
    suite: Literal["runtime_failure"] = "runtime_failure"
    task_type: Literal["runtime_failure"] = "runtime_failure"
    prompt: str
    fault_injection: FaultInjection
    expected_outcome: RuntimeExpected

    @model_validator(mode="after")
    def fault_matches_outcome(self):
        if self.expected_outcome.mode == "recovery":
            if not self.fault_injection.retryable or self.fault_injection.then != "real_tool":
                raise ValueError("recovery requires retryable fault followed by a real tool")
        elif self.fault_injection.then != "continue_fault":
            raise ValueError("graceful_stop evidence failure must persist")
        return self


EvalCase = Annotated[RetrievalCase | AgentTaskCase | MemoryCase | RuntimeFailureCase,
                    Field(discriminator="suite")]
CASE_ADAPTER = TypeAdapter(EvalCase)


class SuiteFile(StrictModel):
    schema_version: Literal["2"]
    suite: Literal["retrieval", "agent_tasks", "memory", "runtime_failure"]
    corpus: Literal["corpus_v1"]
    cases: list[EvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def same_suite(self):
        if any(case.suite != self.suite for case in self.cases):
            raise ValueError("case suite does not match file suite")
        return self


class JudgeCheckResult(StrictModel):
    id: str
    result: Literal["PASS", "FAIL", "UNKNOWN"]
    reason: str
    answer_span_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ExtractionBase(StrictModel):
    id: str


class ProjectSelectionExtraction(ExtractionBase):
    kind: Literal["project_selection"]
    items: list[str]
    answer_span_id: str | None = None


class ClassificationItem(StrictModel):
    id: str
    label: str
    answer_span_id: str | None = None


class ClassificationExtraction(ExtractionBase):
    kind: Literal["classification"]
    items: list[ClassificationItem]


class BulletItem(StrictModel):
    """An observed delivered bullet; code restores its original text."""
    answer_span_id: str
    id: str | None = None


class BulletsExtraction(ExtractionBase):
    kind: Literal["bullets"]
    items: list[BulletItem]


class QuestionItem(StrictModel):
    id: str
    answer_span_id: str
    topic_ids: list[str] = Field(default_factory=list)
    followup_span_ids: list[str] = Field(default_factory=list)


class QuestionsExtraction(ExtractionBase):
    kind: Literal["questions"]
    items: list[QuestionItem]


class TopicSetExtraction(ExtractionBase):
    kind: Literal["topic_set"]
    items: list[str]
    answer_span_id: str | None = None


class MemoryTopicItem(StrictModel):
    memory_id: str
    topic_id: str
    state: Literal["unstable", "stable", "present", "unknown"]


class MemoryTopicsExtraction(ExtractionBase):
    kind: Literal["memory_topics"]
    items: list[MemoryTopicItem]


ExtractionUnion = Annotated[
    ProjectSelectionExtraction | ClassificationExtraction | BulletsExtraction
    | QuestionsExtraction | TopicSetExtraction | MemoryTopicsExtraction,
    Field(discriminator="kind"),
]


class JudgeResponse(StrictModel):
    checks: list[JudgeCheckResult]
    extractions: list[ExtractionUnion]


class FrozenDocument(StrictModel):
    key: str
    title: str
    filename: str
    snapshot_path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_role: Literal["USER_CAREER_SOURCE"]


class CorpusEvidencePolicy(StrictModel):
    authoritative_origins: list[Literal["knowledge", "user_message", "memory", "core_profile"]]
    grounding: Literal["actual_visible_sources"]
    exact_message_provenance: Literal[True]


class CorpusManifest(StrictModel):
    schema_version: Literal["2"]
    corpus_version: Literal["corpus_v1"]
    frozen_documents: list[FrozenDocument] = Field(min_length=6, max_length=6)
    evidence_policy: CorpusEvidencePolicy
    metadata: Metadata = Field(default_factory=Metadata)

    @model_validator(mode="after")
    def unique_documents(self):
        if len({item.key for item in self.frozen_documents}) != len(self.frozen_documents):
            raise ValueError("duplicate corpus logical key")
        if len({item.sha256 for item in self.frozen_documents}) != len(self.frozen_documents):
            raise ValueError("duplicate corpus snapshot hash")
        return self
