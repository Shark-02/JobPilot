"""Small, provider-free retrieval contract over frozen source evidence.

Offsets are half-open Unicode codepoint ranges in DocumentParser v1's frozen
``normalized_text``. The adapter must verify source identities and span hashes;
this module only accepts verified provenance, never retrieved text or anchors.

All four metrics use the first ``retrieval_k`` returned *modules*. EvidenceRecall
counts unique complete Gold evidence units, including all proof alternatives.
MRR is the reciprocal module rank at which the first Gold EU becomes complete.
FactRecall counts required facts for which one complete proof is delivered;
CompleteSupport requires every fact. Spans from several modules can jointly
complete one EU, and EUs from several documents can jointly complete a proof.
Unanswerable tasks report null retrieval metrics and no abstention verdict.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SOURCE_VIEW_VERSION = "canonical_source_v1"
RETRIEVAL_CONTRACT_VERSION = "retrieval_contract_v1"
METRIC_NAMES = ("evidence_recall", "mrr", "fact_recall", "complete_support")

NonemptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MetricName = Literal["evidence_recall", "mrr", "fact_recall", "complete_support"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Span(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def nonempty_interval(self):
        if self.start >= self.end:
            raise ValueError("span must be a nonempty half-open interval")
        return self


class EvidenceUnit(StrictModel):
    id: NonemptyString
    document_key: NonemptyString
    snapshot_hash: Sha256
    canonical_hash: Sha256
    section_path: list[NonemptyString]
    span: Span
    span_hash: Sha256


class EvidenceFile(StrictModel):
    evidence: list[EvidenceUnit] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_and_consistent_sources(self):
        ids = [unit.id for unit in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evidence unit ID")
        versions: dict[tuple[str, str], str] = {}
        for unit in self.evidence:
            source = unit.document_key, unit.snapshot_hash
            if source in versions and versions[source] != unit.canonical_hash:
                raise ValueError("one document snapshot has inconsistent canonical hashes")
            versions[source] = unit.canonical_hash
        return self


class RequiredFact(StrictModel):
    id: NonemptyString
    statement: NonemptyString
    # Outer list ANY; each inner list ALL. [] means no corpus proof exists.
    proofs: list[list[NonemptyString]]

    @model_validator(mode="after")
    def nonvacuous_unique_proofs(self):
        seen: set[frozenset[str]] = set()
        for proof in self.proofs:
            if not proof:
                raise ValueError("a proof option must contain at least one evidence unit")
            if len(proof) != len(set(proof)):
                raise ValueError("duplicate evidence unit in a proof")
            key = frozenset(proof)
            if key in seen:
                raise ValueError("duplicate proof alternative")
            seen.add(key)
        return self


class RetrievalCase(StrictModel):
    id: NonemptyString
    query: NonemptyString
    answerability: Literal["answerable", "unanswerable"]
    tags: list[NonemptyString] = Field(default_factory=list)
    facts: list[RequiredFact] = Field(min_length=1)
    # Defaults allow integration with the existing suite discriminated union;
    # authors need not include either implementation field in their YAML.
    suite: Literal["retrieval"] = "retrieval"
    task_type: Literal["retrieval"] = "retrieval"

    @model_validator(mode="after")
    def coherent_requirements(self):
        ids = [fact.id for fact in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate required fact ID")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("duplicate task tag")
        has_unprovable_fact = any(not fact.proofs for fact in self.facts)
        if self.answerability == "answerable" and has_unprovable_fact:
            raise ValueError("answerable case requires a corpus proof for every fact")
        if self.answerability == "unanswerable" and not has_unprovable_fact:
            raise ValueError("unanswerable case requires at least one fact without corpus proof")
        return self


class RetrievalProtocol(StrictModel):
    retrieval_k: int = Field(default=5, ge=1)
    max_context_tokens: int = Field(default=8000, ge=1)
    metrics: list[MetricName] = Field(default_factory=lambda: list(METRIC_NAMES))

    @model_validator(mode="after")
    def exact_metric_contract(self):
        if len(self.metrics) != len(METRIC_NAMES) or set(self.metrics) != set(METRIC_NAMES):
            raise ValueError("retrieval protocol must declare exactly the four supported metrics")
        return self


class SourceSpan(StrictModel):
    """A source interval proved visible by the retrieval adapter."""

    document_key: NonemptyString
    snapshot_hash: Sha256
    canonical_hash: Sha256
    span: Span


class RetrievedModule(StrictModel):
    module_id: NonemptyString
    spans: list[SourceSpan]


class RetrievalMetrics(StrictModel):
    evidence_recall: float | None
    mrr: float | None
    fact_recall: float | None
    complete_support: float | None


class RetrievalScore(StrictModel):
    answerability: Literal["answerable", "unanswerable"]
    metrics: RetrievalMetrics
    covered_evidence_ids: list[str]
    satisfied_fact_ids: list[str]
    missing_fact_ids: list[str]
    fact_witnesses: dict[str, list[str]]
    first_evidence_rank: int | None
    retrieval_k: int
    returned_modules: int
    abstention: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"


def _units(evidence: EvidenceFile | Sequence[EvidenceUnit]) -> list[EvidenceUnit]:
    if isinstance(evidence, EvidenceFile):
        return evidence.evidence
    # The same validation also applies to programmatically constructed catalogs.
    return EvidenceFile(evidence=list(evidence)).evidence


def validate_retrieval_dataset(
    evidence: EvidenceFile | Sequence[EvidenceUnit], cases: Sequence[RetrievalCase],
) -> None:
    """Cross-file authoring validation, before any source or provider is opened."""
    known = {unit.id for unit in _units(evidence)}
    if not cases:
        raise ValueError("retrieval dataset requires at least one case")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate retrieval case ID")
    for case in cases:
        _validate_evidence_references(case, known)


def _validate_evidence_references(case: RetrievalCase, known: set[str]) -> None:
    refs = {unit_id for fact in case.facts for proof in fact.proofs for unit_id in proof}
    missing = refs - known
    if missing:
        raise ValueError(f"unknown evidence unit references: {sorted(missing)}")


def _covers(intervals: Sequence[tuple[int, int]], span: Span) -> bool:
    cursor = span.start
    for start, end in sorted(intervals):
        if end <= cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= span.end:
            return True
    return False


def evaluate_retrieval(
    case: RetrievalCase,
    evidence: EvidenceFile | Sequence[EvidenceUnit],
    modules: Sequence[RetrievedModule],
    protocol: RetrievalProtocol | None = None,
) -> RetrievalScore:
    """Score exact delivered provenance with identical logic for every case.

    Evidence and fact IDs only join explicit annotations. Query, statement,
    case identity, tags, source text, and runtime chunk identity have no scoring
    semantics. The adapter is responsible for the context token budget; only
    actually returned, visible module intervals may enter this function.
    """
    protocol = protocol or RetrievalProtocol()
    units = _units(evidence)
    _validate_evidence_references(case, {unit.id for unit in units})
    gold_ids = {unit_id for fact in case.facts for proof in fact.proofs for unit_id in proof}
    gold_units = [unit for unit in units if unit.id in gold_ids]
    prefix = modules[:protocol.retrieval_k]
    intervals: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    completed: dict[str, int] = {}
    for rank, module in enumerate(prefix, 1):
        for fragment in module.spans:
            source = fragment.document_key, fragment.snapshot_hash, fragment.canonical_hash
            intervals.setdefault(source, []).append((fragment.span.start, fragment.span.end))
        for unit in gold_units:
            if unit.id in completed:
                continue
            source = unit.document_key, unit.snapshot_hash, unit.canonical_hash
            if _covers(intervals.get(source, []), unit.span):
                completed[unit.id] = rank
    covered = set(completed)
    witnesses: dict[str, list[str]] = {}
    for fact in case.facts:
        for proof in fact.proofs:
            if set(proof) <= covered:
                # Preserve the author-declared proof order for diagnostics.
                witnesses[fact.id] = list(proof)
                break
    satisfied = [fact.id for fact in case.facts if fact.id in witnesses]
    missing = [fact.id for fact in case.facts if fact.id not in witnesses]
    first_rank = min(completed.values(), default=None)
    if case.answerability == "unanswerable":
        metrics = RetrievalMetrics(
            evidence_recall=None, mrr=None, fact_recall=None, complete_support=None,
        )
    else:
        metrics = RetrievalMetrics(
            evidence_recall=len(covered) / len(gold_ids),
            mrr=1.0 / first_rank if first_rank is not None else 0.0,
            fact_recall=len(satisfied) / len(case.facts),
            complete_support=float(not missing),
        )
    return RetrievalScore(
        answerability=case.answerability, metrics=metrics,
        covered_evidence_ids=sorted(covered), satisfied_fact_ids=satisfied,
        missing_fact_ids=missing, fact_witnesses=witnesses,
        first_evidence_rank=first_rank, retrieval_k=protocol.retrieval_k,
        returned_modules=len(prefix),
    )
