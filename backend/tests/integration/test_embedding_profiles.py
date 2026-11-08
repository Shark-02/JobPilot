from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text

from app.config import get_settings
from app.db.models import (
    ActiveEmbeddingProfile,
    DocumentFormat,
    DocumentRole,
    DocumentStatus,
    EmbeddingProfile,
    EmbeddingProfileStatus,
    KnowledgeChunk,
    KnowledgeChunkEmbedding,
    KnowledgeDocument,
    KnowledgeSection,
    MemoryImportance,
    MemoryItem,
    MemoryItemEmbedding,
    MemoryKind,
    MemoryStatus,
)
from app.embedding.profiles import ProfileDefinition, index_name, profile_definition
from app.evals.scope import EvalScope, reset_eval_scope, set_eval_scope
from app.infra.embedding import EmbeddingClient
from app.knowledge.ingest import IngestRequest, KnowledgeIngestService
from app.knowledge.retrieval import KnowledgeRetriever
from app.knowledge.sources import SourceRef, SourceResolver
from app.memory.retrieval import MemoryRetriever
from app.memory.service import MemoryService, MutationCommand


class TestCounter:
    __test__ = False

    def count(self, value: str) -> int:
        return len(value)

    def truncate(self, value: str, limit: int) -> str:
        return value[:limit]

    def split(self, value: str, limit: int) -> list[str]:
        return [value[start:start + limit] for start in range(0, len(value), limit)]


def unit_vector(dimension: int, axis: int = 0) -> list[float]:
    values = [0.0] * dimension
    values[axis] = 1.0
    return values


class DeterministicEmbedding(EmbeddingClient):
    """No client construction, network, credentials, or provider calls in integration tests."""

    def __init__(self, profile: EmbeddingProfile, *, fail_on: str | None = None) -> None:
        self.profile = profile
        self.calls = []
        self.fail_on = fail_on

    async def embed_many(self, texts, *, purpose="document"):
        self.calls.append((list(texts), purpose))
        if self.fail_on and any(self.fail_on in value for value in texts):
            raise RuntimeError("deterministic provider batch failure")
        return [unit_vector(self.profile.dimension) for _ in texts]

    async def embed_one(self, value, *, purpose="document"):
        return (await self.embed_many([value], purpose=purpose))[0]


def profile_service(session):
    from app.embedding.service import EmbeddingProfileService

    return EmbeddingProfileService(session, get_settings())


async def register(session, name: str, dimension: int, **options):
    profile = await profile_service(session).register(ProfileDefinition(
        name=name, base_url="https://embedding.invalid/v1", model=f"model-{name}",
        dimension=dimension, **options,
    ))
    await session.commit()
    return profile


async def seed_entities(session, count: int = 2):
    document = KnowledgeDocument(
        title="Synthetic profile corpus", filename=None, format=DocumentFormat.TEXT,
        source_role=DocumentRole.USER_CAREER_SOURCE, raw_text="synthetic corpus",
        normalized_text="synthetic corpus", sha256=uuid4().hex.ljust(64, "0"),
        status=DocumentStatus.READY, parser_version="test", chunker_version="test", metadata_={},
    )
    session.add(document)
    await session.flush()
    chunks, memories = [], []
    for ordinal in range(count):
        section = KnowledgeSection(
            document_id=document.id, parent_section_id=None, heading=f"section {ordinal}", level=1,
            section_path=[f"section {ordinal}"], ordinal=ordinal, start_offset=0, end_offset=16,
        )
        session.add(section)
        await session.flush()
        chunk = KnowledgeChunk(
            document_id=document.id, section_id=section.id, ordinal=ordinal,
            content=f"synthetic evidence {ordinal}", retrieval_text=f"synthetic evidence {ordinal}",
            start_offset=0, end_offset=16, token_count=16,
        )
        memory = MemoryItem(
            kind=MemoryKind.SKILL, content=f"synthetic memory {ordinal}", attributes={},
            importance=MemoryImportance.NORMAL, status=MemoryStatus.ACTIVE,
        )
        session.add_all([chunk, memory])
        chunks.append(chunk)
        memories.append(memory)
    await session.commit()
    return document, chunks, memories


async def store_vectors(session, profile, chunks, memories, *, reverse=False):
    for ordinal, chunk in enumerate(chunks):
        axis = (1 - ordinal) if reverse else ordinal
        session.add(KnowledgeChunkEmbedding(
            chunk_id=chunk.id, profile_id=profile.id, embedding=unit_vector(profile.dimension, axis),
        ))
    for ordinal, memory in enumerate(memories):
        axis = (1 - ordinal) if reverse else ordinal
        session.add(MemoryItemEmbedding(
            memory_item_id=memory.id, profile_id=profile.id, embedding=unit_vector(profile.dimension, axis),
        ))
    await session.commit()


async def ready_and_activate(session, profile):
    service = profile_service(session)
    await service.ensure_indexes(profile)
    await service.validate(profile, note="Synthetic deterministic retrieval smoke passed")
    await service.activate(profile)
    await session.commit()


async def test_existing_1536_migration_preserves_vectors_ids_sources_and_exact_cosine_results(profile_database_factory, monkeypatch):
    def forbidden_provider(*args, **kwargs):
        raise AssertionError("migration must not construct an embedding provider client")

    monkeypatch.setattr("app.infra.embedding.AsyncOpenAI", forbidden_provider)
    database = profile_database_factory("0004_context_compaction")
    document_id, section_id, chunk_id, memory_id, source_id = [uuid4() for _ in range(5)]
    vector = str([1.0, 0.25, -0.5] + [0.0] * 1533)
    with database.engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO knowledge_document
            (id,title,format,source_role,raw_text,normalized_text,sha256,status,parser_version,chunker_version)
            VALUES (:id,'synthetic original','TEXT','USER_CAREER_SOURCE','original','original',
                    :sha,'READY','v2','v2')
        """), {"id": document_id, "sha": "1" * 64})
        connection.execute(text("""
            INSERT INTO knowledge_section
            (id,document_id,level,ordinal,start_offset,end_offset)
            VALUES (:id,:document,1,0,0,8)
        """), {"id": section_id, "document": document_id})
        connection.execute(text("""
            INSERT INTO knowledge_chunk
            (id,document_id,section_id,ordinal,content,retrieval_text,start_offset,end_offset,token_count,embedding)
            VALUES (:id,:document,:section,0,'original','original',0,8,1,CAST(:vector AS vector))
        """), {"id": chunk_id, "document": document_id, "section": section_id, "vector": vector})
        connection.execute(text("""
            INSERT INTO memory_item (id,kind,content,importance,status,embedding)
            VALUES (:id,'SKILL','original memory','NORMAL','ACTIVE',CAST(:vector AS vector))
        """), {"id": memory_id, "vector": vector})
        connection.execute(text("""
            INSERT INTO memory_source (id,memory_id,source_type,document_chunk_id,relation)
            VALUES (:id,:memory,'DOCUMENT_CHUNK',:chunk,'SUPPORTS')
        """), {"id": source_id, "memory": memory_id, "chunk": chunk_id})
        before = {
            table: connection.execute(text(f"SELECT id,embedding::text,created_at,embedding <=> CAST(:query AS vector) FROM {table}"),
                                      {"query": str(unit_vector(1536))}).one()
            for table in ["knowledge_chunk", "memory_item"]
        }

    database.upgrade("head")

    async with database.sessions() as session:
        active = await profile_service(session).resolve_active()
        assert active.dimension == 1536
        assert active.name == "qwen-current-v1"
        assert active.status == EmbeddingProfileStatus.ACTIVE
        for entity_table, vector_table, id_column in [
            ("knowledge_chunk", "knowledge_chunk_embedding", "chunk_id"),
            ("memory_item", "memory_item_embedding", "memory_item_id"),
        ]:
            after = (await session.execute(text(
                f"SELECT e.{id_column},e.embedding::text,b.created_at,e.embedding <=> CAST(:query AS vector) "
                f"FROM {vector_table} e JOIN {entity_table} b ON b.id=e.{id_column} WHERE profile_id=:profile"
            ), {"query": str(unit_vector(1536)), "profile": active.id})).one()
            assert after == before[entity_table]
            columns = set((await session.scalars(text(
                "SELECT column_name FROM information_schema.columns WHERE table_schema=:schema AND table_name=:table"
            ), {"schema": database.schema, "table": entity_table})).all())
            assert "embedding" not in columns
        assert (await session.execute(text(
            "SELECT id,memory_id,document_chunk_id FROM memory_source"
        ))).one() == (source_id, memory_id, chunk_id)
        assert await session.scalar(select(KnowledgeChunk.id)) == chunk_id
        assert SourceRef.chunk(document_id, chunk_id).uri == f"document://{document_id}/chunk/{chunk_id}"


async def test_different_dimensions_coexist_and_ann_indexes_are_profile_scoped(profile_database_factory):
    database = profile_database_factory()
    async with database.sessions() as session:
        _, chunks, memories = await seed_entities(session)
        first = await register(session, "space-1536-v1", 1536)
        second = await register(session, "space-1024-v1", 1024, send_dimensions=False)
        await store_vectors(session, first, chunks, memories)
        await store_vectors(session, second, chunks, memories)
        for profile in [first, second]:
            await profile_service(session).ensure_indexes(profile)
        await session.commit()

        assert set((await session.scalars(text(
            "SELECT DISTINCT vector_dims(embedding) FROM knowledge_chunk_embedding"
        ))).all()) == {1536, 1024}
        for profile in [first, second]:
            for entity in ["knowledge", "memory"]:
                definition = await session.scalar(text(
                    "SELECT indexdef FROM pg_indexes WHERE schemaname=:schema AND indexname=:index"
                ), {"schema": database.schema, "index": index_name(entity, profile.id)})
                assert f"vector({profile.dimension})" in definition
                assert str(profile.id) in definition
                assert "WHERE" in definition


async def test_identity_v2_metadata_updates_preserve_legacy_and_new_vectors(profile_database_factory):
    database = profile_database_factory()
    async with database.sessions() as session:
        service = profile_service(session)
        original = await service.resolve_active()
        _, chunks, memories = await seed_entities(session)
        await store_vectors(session, original, chunks, memories)
        original_id = original.id
        original_validation = original.validation_passed_at
        updated = await service.register(replace(
            profile_definition(original), api_key_env="SCHOOL_EMBEDDING_API_KEY",
            base_url="https://equivalent-gateway.invalid/v1", batch_limit=5,
            max_input_tokens=1024,
        ))
        await session.commit()
        assert updated.id == original_id
        assert updated.identity_version == 1
        assert updated.validation_passed_at == original_validation
        assert (await service.coverage(updated)).complete
        assert (await service.resolve_active()).id == original_id
        assert await service.indexes_ready(updated)

        target = await register(session, "metadata-stable-new-space", 1024)
        await store_vectors(session, target, chunks, memories)
        target_id = target.id
        updated = await service.register(replace(profile_definition(target), batch_limit=3,
                                                 api_key_env="RENAMED_EMBEDDING_KEY"))
        await session.commit()
        assert updated.id == target_id
        assert updated.identity_version == 2
        assert (await service.coverage(updated)).complete
        with pytest.raises(ValueError, match="different immutable embedding space"):
            await service.register(replace(profile_definition(updated), normalize=True))
        with pytest.raises(ValueError, match="already registered as"):
            await service.register(replace(profile_definition(updated), name="alias-of-existing-space"))


async def test_identity_v2_database_guard_allows_metadata_but_rejects_semantic_rewrite(profile_database_factory):
    database = profile_database_factory()
    async with database.sessions() as session:
        target = await register(session, "guarded-space", 1024)
        target_id = target.id
        await session.execute(text("UPDATE embedding_profile SET batch_limit=3, api_key_env='RENAMED_EMBEDDING_KEY' WHERE id=:id"), {"id": target_id})
        await session.commit()
        assert (await profile_service(session).get(target_id)).batch_limit == 3
        with pytest.raises(Exception, match="identity and encoding rules are immutable"):
            await session.execute(text("UPDATE embedding_profile SET space_revision='different-weights' WHERE id=:id"), {"id": target_id})
        await session.rollback()


async def test_identity_v2_migration_retains_legacy_uuid_vectors_indexes_and_pointer(profile_database_factory):
    database = profile_database_factory("0005_embedding_profiles")
    with database.engine.begin() as connection:
        before = connection.execute(text("SELECT id,status,validation_passed_at FROM embedding_profile")).one()
        pointer = connection.scalar(text("SELECT profile_id FROM active_embedding_profile"))
        indexes = connection.scalars(text("SELECT indexname FROM pg_indexes WHERE schemaname=:schema AND indexname LIKE '%embedding%hnsw' ORDER BY indexname"), {"schema": database.schema}).all()
    database.upgrade("head")
    async with database.sessions() as session:
        profile = await profile_service(session).resolve_active()
        assert (profile.id, profile.status.value, profile.validation_passed_at) == tuple(before)
        assert profile.id == pointer
        assert profile.identity_version == 1
        assert profile.legacy_identity_json is not None
        after_indexes = (await session.scalars(text("SELECT indexname FROM pg_indexes WHERE schemaname=:schema AND indexname LIKE '%embedding%hnsw' ORDER BY indexname"), {"schema": database.schema})).all()
        assert after_indexes == indexes
        assert await profile_service(session).indexes_ready(profile)


@pytest.mark.parametrize("target_dimension", [1024, 1536])
async def test_retrieval_uses_only_resolved_profile_even_for_same_dimension_other_model(
    profile_database_factory, monkeypatch, target_dimension,
):
    database = profile_database_factory()
    async with database.sessions() as session:
        _, chunks, memories = await seed_entities(session)
        first = await register(session, "first-space-v1", 1536)
        second = await register(session, "second-model-space-v1", target_dimension)
        await store_vectors(session, first, chunks, memories)
        await store_vectors(session, second, chunks, memories, reverse=True)

        async def no_lexical(*args):
            return []

        for profile, expected in [(first, 0), (second, 1)]:
            await ready_and_activate(session, profile)
            client = DeterministicEmbedding(profile)
            knowledge = KnowledgeRetriever(session, client, TestCounter())
            memory = MemoryRetriever(session, client)
            monkeypatch.setattr(knowledge, "_lexical", no_lexical)
            monkeypatch.setattr(memory, "_lexical", no_lexical)
            modules, trace = await knowledge.search_with_debug(
                "query", document_ids=None, max_modules=1, max_tokens=1000,
            )
            matches = await memory.search("query", kinds=None, limit=1)
            assert modules[0].source_refs[1] == SourceRef.chunk(chunks[expected].document_id, chunks[expected].id)
            assert matches[0].id == memories[expected].id
            assert trace["metadata"]["embedding_profile_id"] == str(profile.id)
            assert trace["metadata"]["embedding_dimension"] == profile.dimension
            assert all(purpose == "query" for _, purpose in client.calls)

        wrong_client = DeterministicEmbedding(first)
        with pytest.raises(ValueError, match="profile"):
            await KnowledgeRetriever(session, wrong_client, TestCounter()).search(
                "query", document_ids=None, max_modules=1, max_tokens=1000,
            )


async def test_online_knowledge_memory_write_only_active_profile_and_merge_invalidates_history(profile_database_factory):
    database = profile_database_factory()
    async with database.sessions() as session:
        original = await profile_service(session).resolve_active()
        target = await register(session, "single-write-space-v1", 1024, send_dimensions=False)
        await ready_and_activate(session, target)
        client = DeterministicEmbedding(target)
        counter = TestCounter()
        result = await KnowledgeIngestService(session, client, counter).ingest(IngestRequest(
            title="Synthetic write", content="# Evidence\n\nSynthetic new knowledge.",
            format=DocumentFormat.MARKDOWN, source_role=DocumentRole.USER_CAREER_SOURCE,
            filename=None, metadata={},
        ))
        chunk = await session.scalar(select(KnowledgeChunk).where(KnowledgeChunk.document_id == result.document.id))
        refs = [SourceRef.chunk(result.document.id, chunk.id)]
        service = MemoryService(session, client, SourceResolver(session, counter))
        [memory] = await service.mutate([MutationCommand(
            operation="ADD", target_memory_id=None, kind=MemoryKind.SKILL, content="Synthetic new memory",
            attributes={}, importance=MemoryImportance.NORMAL, source_refs=refs,
        )])
        assert list((await session.scalars(select(KnowledgeChunkEmbedding.profile_id))).all()) == [target.id]
        assert list((await session.scalars(select(MemoryItemEmbedding.profile_id))).all()) == [target.id]
        assert not any(profile == original.id for profile in await session.scalars(select(MemoryItemEmbedding.profile_id)))

        # Historical vectors for editable memory become stale after MERGE and
        # must be removed, so returning to the historical profile requires backfill.
        session.add(MemoryItemEmbedding(
            memory_item_id=memory.id, profile_id=original.id, embedding=unit_vector(original.dimension),
        ))
        await session.commit()
        await service.mutate([MutationCommand(
            operation="MERGE", target_memory_id=memory.id, kind=None, content="Updated synthetic memory",
            attributes={}, importance=MemoryImportance.NORMAL, source_refs=refs,
        )])
        assert list((await session.scalars(select(MemoryItemEmbedding.profile_id))).all()) == [target.id]
        assert await session.scalar(select(KnowledgeChunk.id)) == chunk.id
        [replacement] = await service.mutate([MutationCommand(
            operation="SUPERSEDE", target_memory_id=memory.id, kind=MemoryKind.SKILL,
            content="Replacement synthetic memory", attributes={}, importance=MemoryImportance.NORMAL,
            source_refs=refs,
        )])
        assert replacement.id != memory.id
        assert memory.status == MemoryStatus.SUPERSEDED
        assert set(await session.scalars(select(MemoryItemEmbedding.profile_id))) == {target.id}
        assert await session.scalar(select(func.count()).select_from(MemoryItemEmbedding)) == 2


async def test_backfill_resumes_missing_vectors_is_idempotent_and_never_activates(profile_database_factory):
    database = profile_database_factory()
    async with database.sessions() as session:
        original = await profile_service(session).resolve_active()
        _, chunks, memories = await seed_entities(session)
        target = await register(session, "resumable-space-v1", 1024, batch_limit=1)
        await store_vectors(session, target, chunks[:1], memories[:1])
        client = DeterministicEmbedding(target)
        progress = []
        report = await profile_service(session).backfill(target, client=client, progress=progress.append)
        assert (report.total, report.existing, report.pending, report.completed, report.failed) == (4, 2, 2, 2, 0)
        assert report.coverage.complete
        assert report.as_dict()["status"] == "READY"
        assert "total=4 existing=2 pending=2 completed=0 failed=0" in progress
        assert progress[0].startswith("compatibility=")
        assert report.knowledge.as_dict() == {"total": 2, "existing": 1, "pending": 1, "completed": 1, "failed": 0}
        assert report.memory.as_dict() == {"total": 2, "existing": 1, "pending": 1, "completed": 1, "failed": 0}
        assert progress[-1] == "READY"
        texts = [value for batch, _ in client.calls for value in batch]
        assert set(texts) == {chunks[1].retrieval_text, memories[1].content}
        assert await session.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding)) == 2
        assert await session.scalar(select(func.count()).select_from(MemoryItemEmbedding)) == 2
        assert (await profile_service(session).resolve_active()).id == original.id
        client.calls.clear()
        report = await profile_service(session).backfill(target, client=client, progress=lambda value: None)
        assert (report.existing, report.pending, report.completed, report.failed) == (4, 0, 0, 0)
        assert client.calls == []
        assert await session.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding)) == 2
        assert await session.scalar(select(func.count()).select_from(MemoryItemEmbedding)) == 2


async def test_backfill_batch_failure_preserves_success_and_can_resume(profile_database_factory):
    database = profile_database_factory()
    async with database.sessions() as session:
        _, chunks, memories = await seed_entities(session)
        missing_text = chunks[1].retrieval_text
        target = await register(session, "failure-isolated-space-v1", 1024, batch_limit=1)
        client = DeterministicEmbedding(target, fail_on="evidence 1")
        report = await profile_service(session).backfill(target, client=client, progress=lambda value: None)
        assert (report.completed, report.failed) == (3, 1)
        assert report.as_dict()["status"] == "INCOMPLETE"
        assert await session.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding)) == 1
        assert await session.scalar(select(func.count()).select_from(MemoryItemEmbedding)) == 2
        remaining = DeterministicEmbedding(target)
        report = await profile_service(session).backfill(target, client=remaining, progress=lambda value: None)
        assert (report.existing, report.pending, report.completed, report.failed) == (3, 1, 1, 0)
        assert [value for batch, _ in remaining.calls for value in batch] == [missing_text]
        assert await session.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding)) == 2
        assert await session.scalar(select(func.count()).select_from(MemoryItemEmbedding)) == 2


async def test_global_backfill_scan_blocks_all_provider_calls_even_for_existing_incompatible_rows(profile_database_factory):
    from app.embedding.compatibility import CompatibilityError

    database = profile_database_factory()
    async with database.sessions() as session:
        service = profile_service(session)
        original_id = (await service.resolve_active()).id
        _, chunks, memories = await seed_entities(session)
        chunks[1].retrieval_text = " x" * 20
        memories[1].content = " x" * 25
        await session.commit()
        target = await register(session, "global-input-gate-v1", 1024, batch_limit=1, max_input_tokens=10)
        # An existing row does not exempt that eligible entity from the global
        # admission audit. Missing compatible rows must not be sent first.
        await store_vectors(session, target, chunks[1:], [])
        client = DeterministicEmbedding(target)
        progress = []
        with pytest.raises(CompatibilityError) as caught:
            await service.backfill(target, client=client, progress=progress.append)
        report = caught.value.report
        assert report.knowledge.total == report.memory.total == 2
        assert report.knowledge.compatible == report.memory.compatible == 1
        assert report.knowledge.incompatible[0].entity_id == chunks[1].id
        assert report.knowledge.incompatible[0].tokens == 20
        assert report.memory.incompatible[0].entity_id == memories[1].id
        assert report.memory.incompatible[0].as_dict()["excess"] == 15
        assert client.calls == []
        assert target.status == EmbeddingProfileStatus.REGISTERED
        assert await session.scalar(select(ActiveEmbeddingProfile.profile_id)) == original_id
        assert await session.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding)) == 1
        assert await session.scalar(select(func.count()).select_from(MemoryItemEmbedding)) == 0
        assert " x" not in "\n".join(progress)


async def test_validation_rechecks_existing_stored_vectors_and_rejects_zero_cosine_vectors(profile_database_factory):
    database = profile_database_factory()
    async with database.sessions() as session:
        _, chunks, memories = await seed_entities(session)
        target = await register(session, "stored-value-validation-v1", 1024)
        await store_vectors(session, target, chunks, memories)
        row = await session.get(KnowledgeChunkEmbedding, (chunks[0].id, target.id))
        row.embedding = [0.0] * target.dimension
        await session.commit()
        service = profile_service(session)
        await service.ensure_indexes(target)
        with pytest.raises(ValueError, match="stored vector validation.*zero_vector"):
            await service.validate(target, note="Synthetic validation evidence")
        assert target.validation_passed_at is None
        assert target.status == EmbeddingProfileStatus.REGISTERED


async def test_coverage_activation_switch_and_switch_back_require_complete_coverage(profile_database_factory):
    database = profile_database_factory()
    async with database.sessions() as session:
        original = await profile_service(session).resolve_active()
        original_id = original.id
        _, chunks, memories = await seed_entities(session)
        target = await register(session, "switch-space-v1", 1024)
        target_id = target.id
        # Original profile was migrated before this synthetic corpus was added.
        await profile_service(session).backfill(original, client=DeterministicEmbedding(original))
        await ready_and_activate(session, original)
        await store_vectors(session, target, chunks, memories[:1])
        await profile_service(session).ensure_indexes(target)
        coverage = await profile_service(session).coverage(target)
        assert (coverage.knowledge.total, coverage.knowledge.existing, coverage.knowledge.missing) == (2, 2, 0)
        assert (coverage.memory.total, coverage.memory.existing, coverage.memory.missing) == (2, 1, 1)
        assert coverage.as_dict()["status"] == "INCOMPLETE"
        with pytest.raises(ValueError, match="coverage|missing|incomplete"):
            await profile_service(session).activate(target)
        await session.rollback()
        assert (await profile_service(session).resolve_active()).id == original_id
        target = await session.get(EmbeddingProfile, target_id)
        await profile_service(session).backfill(target, client=DeterministicEmbedding(target))
        with pytest.raises(ValueError, match="validat"):
            await profile_service(session).activate(target)
        await session.rollback()
        target = await session.get(EmbeddingProfile, target_id)
        await ready_and_activate(session, target)
        assert (await profile_service(session).resolve_active()).id == target.id

        counter = TestCounter()
        await KnowledgeIngestService(session, DeterministicEmbedding(target), counter).ingest(IngestRequest(
            title="Added while target active", content="Synthetic later evidence", format=DocumentFormat.TEXT,
            source_role=DocumentRole.USER_CAREER_SOURCE, filename=None, metadata={},
        ))
        original = await session.get(EmbeddingProfile, original_id)
        with pytest.raises(ValueError, match="coverage|missing|incomplete"):
            await profile_service(session).activate(original)
        await session.rollback()
        original = await session.get(EmbeddingProfile, original_id)
        await profile_service(session).backfill(original, client=DeterministicEmbedding(original))
        await ready_and_activate(session, original)
        assert (await profile_service(session).resolve_active()).id == original.id
        active_rows = list((await session.scalars(select(EmbeddingProfile).where(
            EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE,
        ))).all())
        assert [profile.id for profile in active_rows] == [original.id]
        assert await session.scalar(select(func.count()).select_from(ActiveEmbeddingProfile)) == 1


async def test_activation_requires_ready_index_and_explicit_validation_and_generates_no_vectors(
    profile_database_factory, monkeypatch,
):
    database = profile_database_factory()
    async with database.sessions() as session:
        target = await register(session, "activation-gates-space-v1", 1024)
        target_id = target.id
        service = profile_service(session)
        assert (await service.coverage(target)).complete
        with pytest.raises(ValueError, match="index"):
            await service.validate(target, note="Synthetic smoke")
        await session.rollback()
        target = await session.get(EmbeddingProfile, target_id)
        await service.ensure_indexes(target)
        await service.validate(target, note="Synthetic smoke")
        await session.commit()
        await session.execute(text(f"DROP INDEX {index_name('memory', target_id)}"))
        await session.commit()
        with pytest.raises(ValueError, match="index"):
            await service.activate(target)
        await session.rollback()
        target = await session.get(EmbeddingProfile, target_id)
        await service.ensure_indexes(target)

        async def forbidden_generation(*args, **kwargs):
            raise AssertionError("activate must never call an embedding provider")

        monkeypatch.setattr(EmbeddingClient, "embed_many", forbidden_generation)
        monkeypatch.setattr(EmbeddingClient, "embed_one", forbidden_generation)
        await service.activate(target)
        await session.commit()
        assert (await service.resolve_active()).id == target_id
        assert await session.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding)) == 0
        assert await session.scalar(select(func.count()).select_from(MemoryItemEmbedding)) == 0


async def test_exact_99_percent_coverage_cannot_activate(profile_database_factory):
    database = profile_database_factory()
    async with database.sessions() as session:
        original_id = (await profile_service(session).resolve_active()).id
        _, chunks, memories = await seed_entities(session, count=100)
        target = await register(session, "ninety-nine-percent-space-v1", 1024)
        await store_vectors(session, target, chunks[:99], memories[:99])
        service = profile_service(session)
        coverage = await service.coverage(target)
        assert (coverage.knowledge.total, coverage.knowledge.existing, coverage.knowledge.missing) == (100, 99, 1)
        assert (coverage.memory.total, coverage.memory.existing, coverage.memory.missing) == (100, 99, 1)
        with pytest.raises(ValueError, match="coverage.*incomplete"):
            await service.activate(target)
        await session.rollback()
        assert (await service.resolve_active()).id == original_id


async def test_switch_to_lower_uuid_profile_respects_single_active_unique_constraint(profile_database_factory):
    database = profile_database_factory()
    async with database.sessions() as session:
        first = await register(session, "uuid-order-a-space-v1", 1024)
        second = await register(session, "uuid-order-b-space-v1", 1024)
        lower, higher = sorted([first, second], key=lambda profile: profile.id.int)
        await ready_and_activate(session, higher)
        await ready_and_activate(session, lower)
        assert (await profile_service(session).resolve_active()).id == lower.id
        assert higher.status == EmbeddingProfileStatus.INACTIVE
        assert lower.status == EmbeddingProfileStatus.ACTIVE
        assert await session.scalar(select(func.count()).select_from(EmbeddingProfile).where(
            EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE,
        )) == 1


async def test_coverage_and_backfill_use_ready_knowledge_and_active_production_memory_only(profile_database_factory):
    database = profile_database_factory()
    async with database.sessions() as session:
        _, chunks, memories = await seed_entities(session)
        ignored_document, ignored_chunks, ignored_memories = await seed_entities(session)
        ignored_document.status = DocumentStatus.FAILED
        ignored_memories[0].status = MemoryStatus.ARCHIVED
        ignored_memories[1].eval_run_id = uuid4()
        ignored_memories[1].eval_case_id = "synthetic-case"
        await session.commit()
        target = await register(session, "eligible-space-v1", 1024)
        report = await profile_service(session).backfill(
            target, client=DeterministicEmbedding(target), progress=lambda value: None,
        )
        assert (report.coverage.knowledge.total, report.coverage.memory.total) == (2, 2)
        assert (report.total, report.completed) == (4, 4)
        assert set(await session.scalars(select(KnowledgeChunkEmbedding.chunk_id))) == {chunk.id for chunk in chunks}
        assert set(await session.scalars(select(MemoryItemEmbedding.memory_item_id))) == {memory.id for memory in memories}


async def test_eval_binds_nonactive_ready_profile_without_changing_active_pointer(profile_database_factory, monkeypatch):
    database = profile_database_factory()
    async with database.sessions() as session:
        original = await profile_service(session).resolve_active()
        original_id = original.id
        document, chunks, memories = await seed_entities(session)
        candidate = await register(session, "eval-candidate-space-v1", 1024)
        await store_vectors(session, candidate, chunks, memories, reverse=True)
        service = profile_service(session)
        await service.ensure_indexes(candidate)
        await service.validate(candidate, note="Synthetic candidate retrieval smoke")
        await session.commit()
        candidate_id = candidate.id
        token = set_eval_scope(EvalScope(
            run_id=uuid4(), case_id="synthetic-eval", corpus_document_ids=frozenset({document.id}),
            embedding_profile_id=candidate_id,
        ))
        try:
            resolved = await service.resolve_active()
            assert resolved.id == candidate_id
            retriever = KnowledgeRetriever(session, DeterministicEmbedding(resolved), TestCounter())

            async def no_lexical(*args):
                return []

            monkeypatch.setattr(retriever, "_lexical", no_lexical)
            modules, trace = await retriever.search_with_debug(
                "query", document_ids=None, max_modules=1, max_tokens=1000,
            )
            assert modules[0].source_refs[1] == SourceRef.chunk(document.id, chunks[1].id)
            assert trace["metadata"]["embedding_profile_id"] == str(candidate_id)
            assert await session.scalar(select(ActiveEmbeddingProfile.profile_id)) == original_id
            with pytest.raises(ValueError, match="evaluation|operator"):
                await service.activate(candidate)
        finally:
            reset_eval_scope(token)
        assert (await service.resolve_active()).id == original_id
