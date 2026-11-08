from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from pydantic import TypeAdapter

from app.evals.schema import CorpusManifest, EvalCase, SuiteFile
from app.evals.compiler import ExecutableEvalPlan, compile_case
from app.evals.retrieval_contract import (
    EvidenceFile, RetrievalCase, RetrievalProtocol, validate_retrieval_dataset,
)
from app.evals.retrieval_provenance import FrozenSource, frozen_sources, validate_evidence_sources

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import (
    DocumentStatus, EmbeddingProfile, EmbeddingProfileStatus, KnowledgeChunk, KnowledgeChunkEmbedding,
    KnowledgeDocument, KnowledgeSection,
)
from app.embedding.profiles import profile_definition, validate_profile
from app.embedding.service import EmbeddingProfileService


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PACKAGE_ROOT = PROJECT_ROOT / "jobpilot_eval_v0.1.1"


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvalDataset:
    root: Path
    manifest: CorpusManifest
    cases: dict[str, EvalCase]
    plans: dict[str, ExecutableEvalPlan]
    retrieval_evidence: EvidenceFile
    retrieval_protocol: RetrievalProtocol
    retrieval_sources: dict[str, FrozenSource]

    def by_suite(self, suite: str) -> list[EvalCase]:
        return [case for case in self.cases.values() if case.suite == suite]

    def fixture(self, relative_path: str) -> str:
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to((self.root / "fixtures").resolve()) or not path.is_file():
            raise DatasetError(f"Invalid or missing agent-visible fixture: {relative_path}")
        return path.read_text(encoding="utf-8")


def _load_yaml(path: Path):
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise DatasetError(f"Duplicate YAML field in {path.name}: {key}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result
    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)


def load_dataset(package_root: Path = DEFAULT_PACKAGE_ROOT) -> EvalDataset:
    """Validate and compile every scoring field before any database/provider work."""
    package_root = package_root.resolve()
    try:
        manifest = CorpusManifest.model_validate(_load_yaml(package_root / "corpora" / "corpus_v1_manifest.yaml"))
        cases: dict[str, EvalCase] = {}
        plans: dict[str, ExecutableEvalPlan] = {}
        suites: set[str] = set()
        for path in sorted((package_root / "cases").glob("*.yaml")):
            suite_file = SuiteFile.model_validate(_load_yaml(path))
            if suite_file.suite == "retrieval":
                raise DatasetError("Legacy cases/retrieval.yaml is not supported; use the three retrieval YAML files")
            if suite_file.suite in suites:
                raise DatasetError(f"Duplicate suite file: {suite_file.suite}")
            suites.add(suite_file.suite)
            for case in suite_file.cases:
                if not case.id or case.id in cases:
                    raise DatasetError(f"Invalid or duplicate case ID: {case.id}")
                cases[case.id] = case
                plans[case.id] = compile_case(case)
                from app.evals.executor import validate_executor_mapping
                validate_executor_mapping(plans[case.id])
        evidence = EvidenceFile.model_validate(_load_yaml(package_root / "retrieval_evidence.yaml"))
        protocol = RetrievalProtocol.model_validate(_load_yaml(package_root / "retrieval_protocol.yaml"))
        retrieval_cases = TypeAdapter(list[RetrievalCase]).validate_python(
            _load_yaml(package_root / "retrieval_cases.yaml"))
        validate_retrieval_dataset(evidence, retrieval_cases)
        for case in retrieval_cases:
            if case.id in cases:
                raise DatasetError("Duplicate retrieval case identity")
            cases[case.id] = case
            plans[case.id] = compile_case(case)
            validate_executor_mapping(plans[case.id])
        suites.add("retrieval")
        if suites != {"retrieval", "agent_tasks", "memory", "runtime_failure"}:
            raise DatasetError("Dataset requires all four typed suites")
        fixture_ids: set[str] = set()
        for case in cases.values():
            if isinstance(case, RetrievalCase):
                continue
            fixtures = list(case.setup.user_fixtures)
            for step in [*case.setup.seed_steps, *getattr(case, "steps", [])]:
                fixtures.extend(step.user_fixtures)
            case_fixtures = {item.id for item in fixtures}
            for fixture in fixtures:
                fixture_path = (package_root / fixture.path).resolve()
                if not fixture_path.is_relative_to((package_root / "fixtures").resolve()) or not fixture_path.is_file():
                    raise DatasetError("Invalid or missing agent-visible fixture")
            for check in plans[case.id].semantic_checks:
                if check.required_fixture_id and check.required_fixture_id not in case_fixtures:
                    raise DatasetError(f"Check {check.id} has no fixture binding")
            fixture_ids.update(case_fixtures)
        for spec in manifest.frozen_documents:
            snapshot = (package_root / spec.snapshot_path).resolve()
            if not snapshot.is_relative_to((package_root / "corpora").resolve()) or not snapshot.is_file():
                raise DatasetError(f"Invalid or missing frozen snapshot: {spec.key}")
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != spec.sha256:
                raise DatasetError(f"Frozen snapshot SHA256 mismatch: {spec.key}")
        sources = frozen_sources(package_root, manifest)
        validate_evidence_sources(evidence, sources)
        return EvalDataset(package_root, manifest, cases, plans, evidence, protocol, sources)
    except (ValueError, yaml.YAMLError, OSError) as exc:
        raise DatasetError(f"Dataset preflight ERROR: {type(exc).__name__}: {exc}") from exc


async def resolve_source_embedding_profile(
    *, source_database_url: str, settings: Settings, selector: str | None
) -> EmbeddingProfile:
    """Pin the source active (or explicitly requested) profile once per run."""

    engine = create_async_engine(source_database_url, pool_pre_ping=True,
                                 connect_args={"options": "-c default_transaction_read_only=on"})
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            service = EmbeddingProfileService(session, settings)
            profile = await service.get(selector) if selector else await service.resolve_active()
            validate_profile(profile)
            return profile
    finally:
        await engine.dispose()


async def _copy_profile(
    session: AsyncSession, source_profile: EmbeddingProfile, settings: Settings
) -> EmbeddingProfile:
    """Preserve immutable identity without activating a production profile."""

    service = EmbeddingProfileService(session, settings)
    try:
        copied = await service.register(profile_definition(source_profile))
    except ValueError as exc:
        raise DatasetError("Eval profile identity differs from the source profile configuration") from exc
    if copied.id != source_profile.id:
        raise DatasetError("Eval profile UUID differs from source; reset canonical jobpilot_eval")
    await service.ensure_indexes(copied)
    await session.commit()
    return copied


async def verify_and_copy_corpus(
    *,
    dataset: EvalDataset,
    source_database_url: str,
    target_session: AsyncSession,
    settings: Settings,
    embedding_profile: EmbeddingProfile,
    require_vectors: bool = True,
) -> dict[str, KnowledgeDocument]:
    """Set up the exact local frozen snapshot in canonical Eval only.

    Snapshot bytes are authoritative. A read-only source DB supplies cached
    sections/chunks/vectors when its SHA matches. New snapshot bytes are parsed
    locally with stable identities, never uploaded to product or embedded here.
    Normal runs require complete pinned-profile coverage; setup can report the
    pending vector gate without requesting any provider.
    """
    from app.evals.corpus import frozen_snapshot_texts, setup_local_document
    from app.evals.database import CANONICAL_EVAL_DATABASE, assert_canonical_eval_database
    from sqlalchemy import text

    assert_canonical_eval_database(str(target_session.bind.url), settings.database_url)
    if await target_session.scalar(text("SELECT current_database()")) != CANONICAL_EVAL_DATABASE:
        raise DatasetError("Corpus writes require canonical jobpilot_eval")

    manifest = dataset.manifest.model_dump(mode="json")
    frozen = manifest.get("frozen_documents", [])
    expected_by_sha = {item["sha256"]: item for item in frozen}
    if len(expected_by_sha) != 6:
        raise DatasetError("corpus_v1 must contain exactly six unique SHA256 entries")
    snapshots = frozen_snapshot_texts(dataset)

    source_engine = create_async_engine(source_database_url, pool_pre_ping=True,
                                       connect_args={"options": "-c default_transaction_read_only=on"})
    source_factory = async_sessionmaker(source_engine, expire_on_commit=False)
    try:
        async with source_factory() as source_session:
            rows = list((await source_session.scalars(
                select(KnowledgeDocument).where(KnowledgeDocument.sha256.in_(expected_by_sha))
            )).all())
            found = {item.sha256: item for item in rows}
            for sha, source in found.items():
                actual = hashlib.sha256(source.raw_text.encode("utf-8")).hexdigest()
                if actual != sha:
                    raise DatasetError(f"Source document bytes do not match stored SHA256: {source.filename}")
                if source.status != DocumentStatus.READY:
                    raise DatasetError(f"Frozen source document is not READY: {source.filename}")
                if source.source_role.value != expected_by_sha[sha]["source_role"]:
                    raise DatasetError("Frozen source role differs from the typed corpus policy")

            document_ids = [row.id for row in rows]
            source_sections = list((await source_session.scalars(
                select(KnowledgeSection).where(KnowledgeSection.document_id.in_(document_ids))
                .order_by(KnowledgeSection.ordinal)
            )).all())
            source_chunks = list((await source_session.scalars(
                select(KnowledgeChunk).where(KnowledgeChunk.document_id.in_(document_ids))
                .order_by(KnowledgeChunk.ordinal)
            )).all())
            if any(not any(chunk.document_id == document.id for chunk in source_chunks) for document in rows):
                raise DatasetError("Matched source corpus document has no indexed chunks")
            source_chunk_by_id = {row.id: row for row in source_chunks}
            source_embeddings = list((await source_session.scalars(
                select(KnowledgeChunkEmbedding).where(
                    KnowledgeChunkEmbedding.chunk_id.in_([row.id for row in source_chunks]),
                    KnowledgeChunkEmbedding.profile_id == embedding_profile.id,
                )
            )).all())
            vectors = {row.chunk_id: row for row in source_embeddings}
            if any(len(row.embedding) != embedding_profile.dimension for row in source_embeddings):
                raise DatasetError("Selected profile corpus contains a vector with an incompatible dimension")

            copied_profile = await _copy_profile(target_session, embedding_profile, settings)
            for sha, spec in expected_by_sha.items():
                if sha not in found:
                    await setup_local_document(target_session, spec, snapshots[spec["key"]], manifest["corpus_version"])
                    continue
                source = found[sha]
                existing = await target_session.scalar(
                    select(KnowledgeDocument).where(KnowledgeDocument.sha256 == sha)
                )
                if existing is not None and existing.id != source.id:
                    raise DatasetError("Existing eval corpus UUIDs differ from source; reset canonical jobpilot_eval")
                if existing is None:
                    document = KnowledgeDocument(
                        id=source.id, title=source.title, filename=source.filename, format=source.format,
                        source_role=source.source_role, raw_text=source.raw_text,
                        normalized_text=source.normalized_text, sha256=source.sha256,
                        status=source.status, parser_version=source.parser_version,
                        chunker_version=source.chunker_version,
                        created_at=source.created_at, updated_at=source.updated_at,
                        metadata_={**source.metadata_, "eval_corpus": manifest["corpus_version"], "logical_document_key": spec["key"]},
                    )
                    target_session.add(document)
                    await target_session.flush()
                    for row in source_sections:
                        if row.document_id == source.id:
                            target_session.add(KnowledgeSection(
                                id=row.id, document_id=row.document_id,
                                parent_section_id=row.parent_section_id, heading=row.heading,
                                level=row.level, section_path=row.section_path, ordinal=row.ordinal,
                                start_offset=row.start_offset, end_offset=row.end_offset,
                                created_at=row.created_at,
                            ))
                    await target_session.flush()
                    for row in source_chunks:
                        if row.document_id == source.id:
                            target_session.add(KnowledgeChunk(
                                id=row.id, document_id=row.document_id, section_id=row.section_id,
                                ordinal=row.ordinal, content=row.content, retrieval_text=row.retrieval_text,
                                start_offset=row.start_offset, end_offset=row.end_offset,
                                token_count=row.token_count, created_at=row.created_at,
                            ))
                    await target_session.flush()
                # A repeated run can add a new profile's historical vectors to
                # the same unchanged corpus without replacing business rows.
                document_chunk_ids = [row.id for row in source_chunks if row.document_id == source.id]
                target_chunks = list((await target_session.scalars(
                    select(KnowledgeChunk).where(KnowledgeChunk.document_id == source.id)
                )).all())
                if {row.id for row in target_chunks} != set(document_chunk_ids):
                    raise DatasetError("Eval corpus chunk UUIDs differ from source; reset canonical jobpilot_eval")
                for row in target_chunks:
                    source_chunk = source_chunk_by_id[row.id]
                    fields = ("section_id", "ordinal", "content", "retrieval_text", "start_offset", "end_offset", "token_count")
                    if any(getattr(row, field) != getattr(source_chunk, field) for field in fields):
                        raise DatasetError("Eval corpus chunk content differs from source; reset canonical jobpilot_eval")
                existing_vectors = list((await target_session.scalars(
                    select(KnowledgeChunkEmbedding).where(
                        KnowledgeChunkEmbedding.chunk_id.in_(document_chunk_ids),
                        KnowledgeChunkEmbedding.profile_id == embedding_profile.id,
                    )
                )).all())
                existing_vector_by_id = {row.chunk_id: row for row in existing_vectors}
                for chunk_id, vector in existing_vector_by_id.items():
                    if chunk_id in vectors and list(vector.embedding) != list(vectors[chunk_id].embedding):
                        raise DatasetError("Eval vector values differ from source for the selected profile")
                for chunk_id in document_chunk_ids:
                    if chunk_id not in existing_vector_by_id and chunk_id in vectors:
                        vector = vectors[chunk_id]
                        target_session.add(KnowledgeChunkEmbedding(
                            chunk_id=chunk_id, profile_id=embedding_profile.id,
                            embedding=vector.embedding, created_at=vector.created_at,
                        ))
                await target_session.commit()
    finally:
        await source_engine.dispose()

    target_rows = list((await target_session.scalars(
        select(KnowledgeDocument).where(KnowledgeDocument.sha256.in_(expected_by_sha))
    )).all())
    by_sha = {row.sha256: row for row in target_rows}
    if set(by_sha) != set(expected_by_sha):
        raise DatasetError("Eval corpus copy did not produce the complete frozen corpus")
    for sha, row in by_sha.items():
        if hashlib.sha256(row.raw_text.encode("utf-8")).hexdigest() != sha:
            raise DatasetError(f"Eval corpus SHA256 changed after copy: {row.filename}")
    target_chunks = list((await target_session.scalars(select(KnowledgeChunk).where(
        KnowledgeChunk.document_id.in_([row.id for row in target_rows])
    ))).all())
    vector_chunk_ids = set((await target_session.scalars(select(KnowledgeChunkEmbedding.chunk_id).where(
        KnowledgeChunkEmbedding.profile_id == embedding_profile.id,
        KnowledgeChunkEmbedding.chunk_id.in_([row.id for row in target_chunks]),
    ))).all())
    missing_vectors = [chunk for chunk in target_chunks if chunk.id not in vector_chunk_ids]
    copied_profile.status = EmbeddingProfileStatus.REGISTERED if missing_vectors else EmbeddingProfileStatus.READY
    if missing_vectors:
        copied_profile.validation_passed_at = None
        copied_profile.validation_note = None
    await target_session.commit()
    if require_vectors and missing_vectors:
        raise DatasetError(
            f"Frozen corpus is set up but profile {embedding_profile.name} has incomplete vectors: "
            f"{len(vector_chunk_ids)}/{len(target_chunks)}. Explicitly backfill canonical Eval only "
            "after provider-cost authorization; product corpus is never changed by setup."
        )
    return {expected_by_sha[sha]["key"]: row for sha, row in by_sha.items()}
