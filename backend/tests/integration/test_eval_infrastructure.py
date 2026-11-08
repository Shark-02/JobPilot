import hashlib

from sqlalchemy import func, select

from app.config import get_settings
from app.db.models import (
    ActiveEmbeddingProfile, EmbeddingProfileStatus, KnowledgeChunk, KnowledgeChunkEmbedding,
    KnowledgeDocument, KnowledgeSection,
)
from app.evals.corpus import setup_local_document
from app.evals.database import CanonicalEvalDatabase


async def test_local_frozen_snapshot_setup_is_idempotent_and_generates_no_provider_vectors(profile_database_factory, monkeypatch):
    def forbidden_provider(*args, **kwargs):
        raise AssertionError("local corpus setup must never construct a provider")

    monkeypatch.setattr("app.infra.embedding.AsyncOpenAI", forbidden_provider)
    database = profile_database_factory()
    raw = "# Frozen current project\n\nImplemented Eval; formal baseline is not frozen.\n"
    spec = {"key": "synthetic_current", "title": "Current project", "filename": "synthetic.md",
            "sha256": hashlib.sha256(raw.encode()).hexdigest(), "source_role": "USER_CAREER_SOURCE",
            "snapshot_path": "corpora/corpus_v1/synthetic.md"}
    async with database.sessions() as session:
        await setup_local_document(session, spec, raw, "synthetic_corpus")
        document = await session.scalar(select(KnowledgeDocument))
        document_id = document.id
        before_sections = list(await session.scalars(select(KnowledgeSection.id)))
        before_chunks = list(await session.scalars(select(KnowledgeChunk.id)))
        await setup_local_document(session, spec, raw, "synthetic_corpus")
        assert await session.scalar(select(func.count()).select_from(KnowledgeDocument)) == 1
        assert (await session.scalar(select(KnowledgeDocument))).id == document_id
        assert list(await session.scalars(select(KnowledgeSection.id))) == before_sections
        assert list(await session.scalars(select(KnowledgeChunk.id))) == before_chunks
        assert await session.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding)) == 0
        chunk = await session.scalar(select(KnowledgeChunk))
        assert document.normalized_text[chunk.start_offset:chunk.end_offset] == chunk.content


async def test_canonical_database_setup_migrates_idempotently_without_reset(profile_database_factory):
    # The fixture opts in and verifies canonical-only configuration before access.
    profile_database_factory()
    lifecycle = CanonicalEvalDatabase(get_settings())
    before = lifecycle.smoke()
    first = lifecycle.setup()
    second = lifecycle.setup()
    assert first["dropped"] == second["dropped"] == []
    assert lifecycle.smoke() == before
    assert second["after"]["eval_databases"] == ["jobpilot_eval"]


async def test_explicit_eval_profile_does_not_depend_on_active_pointer_and_rejects_incomplete_profile(profile_database_factory):
    from sqlalchemy import delete
    from app.embedding.service import EmbeddingProfileService
    from app.evals.scope import EvalScope, reset_eval_scope, set_eval_scope
    from uuid import uuid4
    import pytest

    database = profile_database_factory()
    async with database.sessions() as session:
        service = EmbeddingProfileService(session, get_settings())
        profile = await service.resolve_active()
        await session.execute(delete(ActiveEmbeddingProfile))
        profile.status = EmbeddingProfileStatus.REGISTERED
        await session.commit()
        token = set_eval_scope(EvalScope(run_id=uuid4(), case_id="synthetic-explicit-profile", embedding_profile_id=profile.id))
        try:
            with pytest.raises(RuntimeError, match="not ready"):
                await service.resolve_active()
            profile.status = EmbeddingProfileStatus.READY
            await session.commit()
            assert (await service.resolve_active()).id == profile.id
            assert await session.scalar(select(func.count()).select_from(ActiveEmbeddingProfile)) == 0
        finally:
            reset_eval_scope(token)
        with pytest.raises(RuntimeError, match="Active embedding profile is missing"):
            await service.resolve_active()
