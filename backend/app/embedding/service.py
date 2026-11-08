from __future__ import annotations

import os
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from uuid import UUID

from dotenv import dotenv_values
from sqlalchemy import exists, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import ENV_FILE, Settings
from app.db.models import (
    ActiveEmbeddingProfile, DocumentStatus, EmbeddingProfile, EmbeddingProfileStatus,
    KnowledgeChunk, KnowledgeChunkEmbedding, KnowledgeDocument, MemoryItem,
    MemoryItemEmbedding, MemoryStatus,
)
from app.embedding.profiles import (
    ProfileDefinition, RUNTIME_METADATA_FIELDS, index_ddl, index_name,
    profile_definition, validate_profile,
)
from app.embedding.compatibility import (
    CompatibilityError, CompatibilityReport, EntityBackfillReport, EntityCompatibility, IncompatibleEntity,
)
from app.embedding.tokenizers import load_token_counter
from app.evals.scope import apply_eval_scope, get_eval_scope


@dataclass(frozen=True)
class EntityCoverage:
    total: int
    existing: int

    @property
    def missing(self) -> int:
        return self.total - self.existing

    @property
    def complete(self) -> bool:
        return self.missing == 0

    def as_dict(self) -> dict:
        return {"total": self.total, "existing": self.existing, "missing": self.missing}


@dataclass(frozen=True)
class CoverageReport:
    profile_id: UUID
    profile_name: str
    knowledge: EntityCoverage
    memory: EntityCoverage

    @property
    def complete(self) -> bool:
        return self.knowledge.complete and self.memory.complete

    def as_dict(self) -> dict:
        return {"profile_id": str(self.profile_id), "profile_name": self.profile_name,
                "knowledge": self.knowledge.as_dict(), "memory": self.memory.as_dict(),
                "status": "READY" if self.complete else "INCOMPLETE"}


@dataclass(frozen=True)
class BackfillReport:
    total: int
    existing: int
    pending: int
    completed: int
    failed: int
    coverage: CoverageReport
    knowledge: EntityBackfillReport | None = None
    memory: EntityBackfillReport | None = None
    embedding_batches: int = 0
    http_attempts: int = 0

    @property
    def status(self) -> str:
        return "READY" if self.coverage.complete and not self.failed else "INCOMPLETE"

    def as_dict(self) -> dict:
        result = {"total": self.total, "existing": self.existing, "pending": self.pending,
                "completed": self.completed, "failed": self.failed,
                "coverage": self.coverage.as_dict(),
                "embedding_batches": self.embedding_batches, "http_attempts": self.http_attempts,
                "status": self.status}
        if self.knowledge is not None:
            result["knowledge"] = self.knowledge.as_dict()
        if self.memory is not None:
            result["memory"] = self.memory.as_dict()
        return result


class EmbeddingProfileService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    async def get(self, identity: str | UUID) -> EmbeddingProfile:
        try:
            profile_id = UUID(str(identity))
        except ValueError:
            statement = select(EmbeddingProfile).where(EmbeddingProfile.name == str(identity))
        else:
            statement = select(EmbeddingProfile).where(EmbeddingProfile.id == profile_id)
        profile = await self.session.scalar(statement.execution_options(populate_existing=True))
        if profile is None:
            raise ValueError(f"Embedding profile does not exist: {identity}")
        validate_profile(profile)
        return profile

    async def register(self, definition: ProfileDefinition) -> EmbeddingProfile:
        existing = await self.session.scalar(select(EmbeddingProfile).where(EmbeddingProfile.name == definition.name))
        if existing:
            current = validate_profile(existing)
            if current.space_id != definition.space_id:
                raise ValueError("Profile name already refers to a different immutable embedding space; register a new name")
            # Credential, endpoint and admission policy changes leave stored
            # vectors, validation and the active pointer attached to this ID.
            # Endpoint equivalence is an operator assertion about space_revision.
            for field in RUNTIME_METADATA_FIELDS:
                setattr(existing, field, getattr(definition, field))
            await self.session.flush()
            return existing
        if definition.identity_version == 1 and definition.legacy_identity_json is None:
            raise ValueError("A legacy profile registration requires its frozen identity snapshot")
        # A different label is not a new vector space. Report an alias collision
        # before inserting a duplicate primary key (also covers grandfathered IDs).
        candidates = (await self.session.scalars(select(EmbeddingProfile))).all()
        for candidate in candidates:
            if validate_profile(candidate).space_id == definition.space_id:
                raise ValueError(f"Embedding space is already registered as {candidate.name}; reuse that profile name")
        profile = EmbeddingProfile(id=definition.id, **asdict(definition), status=EmbeddingProfileStatus.REGISTERED)
        self.session.add(profile)
        await self.session.flush()
        return profile

    async def resolve_active(self, *, for_write: bool = False) -> EmbeddingProfile:
        scope = get_eval_scope()
        override = getattr(scope, "embedding_profile_id", None) if scope else None
        if override is not None:
            profile = await self.get(override)
            if profile.status not in {EmbeddingProfileStatus.READY, EmbeddingProfileStatus.INACTIVE, EmbeddingProfileStatus.ACTIVE}:
                raise RuntimeError("Specified evaluation embedding profile is not ready")
            return profile
        # Held until the caller commits/rolls back. Activation takes FOR UPDATE,
        # so an in-flight ingest or memory mutation cannot straddle the switch.
        pointer = await self.session.scalar(
            select(ActiveEmbeddingProfile).where(ActiveEmbeddingProfile.singleton_id == 1)
            .with_for_update(read=True).execution_options(populate_existing=True)
        )
        if pointer is None:
            raise RuntimeError("Active embedding profile is missing; apply embedding profile migration")
        profile = await self.get(pointer.profile_id)
        if profile.status != EmbeddingProfileStatus.ACTIVE:
            raise RuntimeError("Active profile pointer and lifecycle status disagree")
        return profile

    def client_for_profile(self, profile: EmbeddingProfile):
        from app.infra.embedding import EmbeddingClient

        validate_profile(profile)
        key = os.environ.get(profile.api_key_env)
        if key is None and profile.api_key_env == "EMBEDDING_API_KEY":
            key = self.settings.embedding_api_key
        if key is None:
            key = dotenv_values(ENV_FILE).get(profile.api_key_env)
        if not key:
            raise RuntimeError(f"Embedding credential is missing: {profile.api_key_env}")
        return EmbeddingClient(profile_definition(profile), key)

    async def client_for_active(self, *, for_write: bool = False):
        return self.client_for_profile(await self.resolve_active(for_write=for_write))

    def _knowledge(self):
        statement = select(KnowledgeChunk).join(KnowledgeDocument).where(KnowledgeDocument.status == DocumentStatus.READY)
        scope = get_eval_scope()
        if scope is not None and scope.corpus_document_ids:
            statement = statement.where(KnowledgeChunk.document_id.in_(scope.corpus_document_ids))
        return statement

    def _memory(self):
        return apply_eval_scope(select(MemoryItem).where(MemoryItem.status == MemoryStatus.ACTIVE), MemoryItem)

    async def coverage(self, profile: EmbeddingProfile | str | UUID) -> CoverageReport:
        if not isinstance(profile, EmbeddingProfile):
            profile = await self.get(profile)
        validate_profile(profile)
        knowledge = self._knowledge().with_only_columns(KnowledgeChunk.id).subquery()
        memory = self._memory().with_only_columns(MemoryItem.id).subquery()
        async def counts(eligible, vector_model, entity_column):
            total = await self.session.scalar(select(func.count()).select_from(eligible))
            available = await self.session.scalar(
                select(func.count()).select_from(eligible).join(vector_model, entity_column == eligible.c.id)
                .where(vector_model.profile_id == profile.id)
            )
            return EntityCoverage(total or 0, available or 0)
        return CoverageReport(profile.id, profile.name,
                              await counts(knowledge, KnowledgeChunkEmbedding, KnowledgeChunkEmbedding.chunk_id),
                              await counts(memory, MemoryItemEmbedding, MemoryItemEmbedding.memory_item_id))

    async def compatibility(self, profile: EmbeddingProfile | str | UUID) -> CompatibilityReport:
        """Read every eligible input locally before any provider request or lifecycle change."""
        if not isinstance(profile, EmbeddingProfile):
            profile = await self.get(profile)
        definition = validate_profile(profile)
        counter = load_token_counter(definition.tokenizer_encoding)
        reports = []
        for model, content_column, eligible in (
            (KnowledgeChunk, KnowledgeChunk.retrieval_text, self._knowledge),
            (MemoryItem, MemoryItem.content, self._memory),
        ):
            total = max_tokens = max_characters = 0
            incompatible = []
            rows = await self.session.stream(eligible().with_only_columns(model.id, content_column).order_by(model.id))
            async for entity_id, content in rows:
                value = definition.document_prefix + content
                tokens = counter.count(value)
                total += 1
                max_tokens = max(max_tokens, tokens)
                max_characters = max(max_characters, len(value))
                if tokens > definition.max_input_tokens:
                    incompatible.append(IncompatibleEntity(entity_id, tokens, definition.max_input_tokens))
            reports.append(EntityCompatibility(total, total - len(incompatible), max_tokens,
                                               max_characters, tuple(incompatible)))
        return CompatibilityReport(profile.id, profile.name, definition.tokenizer_encoding,
                                   definition.max_input_tokens, *reports)

    async def vector_health(self, profile: EmbeddingProfile | str | UUID) -> dict:
        """Inspect stored values, including existing rows, rather than trusting provider success."""
        if not isinstance(profile, EmbeddingProfile):
            profile = await self.get(profile)
        validate_profile(profile)
        report = {}
        for entity, model, id_column in (
            ("knowledge", KnowledgeChunkEmbedding, KnowledgeChunkEmbedding.chunk_id),
            ("memory", MemoryItemEmbedding, MemoryItemEmbedding.memory_item_id),
        ):
            total = 0
            invalid = []
            rows = await self.session.stream(select(id_column, model.embedding).where(model.profile_id == profile.id))
            async for entity_id, vector in rows:
                total += 1
                values = [float(value) for value in vector]
                reason = None
                if len(values) != profile.dimension:
                    reason = "dimension"
                elif not all(math.isfinite(value) for value in values):
                    reason = "non_finite"
                elif math.hypot(*values) == 0:
                    reason = "zero_vector"
                if reason:
                    invalid.append({"entity_id": str(entity_id), "reason": reason})
            report[entity] = {"total": total, "invalid": invalid, "valid": total - len(invalid)}
        return report

    async def ensure_indexes(self, profile: EmbeddingProfile) -> None:
        validate_profile(profile)
        for entity in ("knowledge", "memory"):
            await self.session.execute(text(index_ddl(entity, profile.id, profile.dimension)))
        if not await self.indexes_ready(profile):
            raise RuntimeError("Profile ANN indexes do not match the required dimension, predicate or operator class")

    async def indexes_ready(self, profile: EmbeddingProfile) -> bool:
        validate_profile(profile)
        for entity, table in (("knowledge", "knowledge_chunk_embedding"), ("memory", "memory_item_embedding")):
            row = (await self.session.execute(text("""
                SELECT i.indisvalid, i.indisready, am.amname,
                       pg_get_expr(i.indexprs, i.indrelid) AS expression,
                       pg_get_expr(i.indpred, i.indrelid) AS predicate,
                       ARRAY(SELECT o.opcname FROM unnest(i.indclass) c
                             JOIN pg_opclass o ON o.oid = c) AS opclasses
                FROM pg_index i JOIN pg_class idx ON idx.oid = i.indexrelid
                JOIN pg_class tbl ON tbl.oid = i.indrelid
                JOIN pg_namespace ns ON ns.oid = tbl.relnamespace
                JOIN pg_am am ON am.oid = idx.relam
                WHERE idx.relname = :index_name AND tbl.relname = :table
                  AND ns.nspname = current_schema()
            """), {"index_name": index_name(entity, profile.id), "table": table})).mappings().first()
            if row is None or not row["indisvalid"] or not row["indisready"] or row["amname"] != "hnsw":
                return False
            expression = "".join((row["expression"] or "").split())
            predicate = "".join((row["predicate"] or "").split()).strip("()")
            expected = f"profile_id='{profile.id}'::uuid"
            if expression not in {f"(embedding)::vector({profile.dimension})", f"embedding::vector({profile.dimension})"}:
                return False
            if predicate != expected or list(row["opclasses"]) != ["vector_cosine_ops"]:
                return False
        return True

    async def validate(self, profile: EmbeddingProfile | str | UUID, *, note: str) -> CoverageReport:
        """Record an explicit operator retrieval/eval attestation plus DB checks."""
        if not note.strip():
            raise ValueError("Validation requires a retrieval smoke/evaluation evidence note")
        profile = await self.get(profile.id if isinstance(profile, EmbeddingProfile) else profile)
        validate_profile(profile)
        if profile.status == EmbeddingProfileStatus.BACKFILLING:
            raise ValueError("Cannot validate while profile backfill is running")
        report = await self.coverage(profile)
        if not report.complete:
            raise ValueError(f"Profile coverage is incomplete: {report.as_dict()}")
        health = await self.vector_health(profile)
        if any(entity["invalid"] for entity in health.values()):
            raise ValueError(f"Profile stored vector validation failed: {health}")
        if not await self.indexes_ready(profile):
            raise ValueError("Profile ANN indexes are missing or not ready")
        profile.validation_passed_at = datetime.now(timezone.utc)
        profile.validation_note = note.strip()
        if profile.status != EmbeddingProfileStatus.ACTIVE:
            profile.status = EmbeddingProfileStatus.READY
        await self.session.flush()
        return report

    async def activate(self, profile: EmbeddingProfile | str | UUID) -> EmbeddingProfile:
        if get_eval_scope() is not None:
            raise ValueError("Profile activation is an operator action outside evaluation cases")
        pointer = await self.session.scalar(
            select(ActiveEmbeddingProfile).where(ActiveEmbeddingProfile.singleton_id == 1)
            .with_for_update().execution_options(populate_existing=True)
        )
        if pointer is None:
            raise RuntimeError("Active embedding profile pointer is missing")
        profile = await self.get(profile.id if isinstance(profile, EmbeddingProfile) else profile)
        validate_profile(profile)
        report = await self.coverage(profile)
        if not report.complete:
            raise ValueError(f"Cannot activate: profile coverage is incomplete: {report.as_dict()}")
        if profile.status not in {EmbeddingProfileStatus.READY, EmbeddingProfileStatus.INACTIVE, EmbeddingProfileStatus.ACTIVE}:
            raise ValueError("Cannot activate: profile lifecycle is not ready")
        if not await self.indexes_ready(profile):
            raise ValueError("Cannot activate: profile ANN indexes are missing or not ready")
        if profile.validation_passed_at is None or not profile.validation_note:
            raise ValueError("Cannot activate: explicit retrieval/evaluation validation has not passed")
        # Resolving credentials is local only, and proves the target is usable;
        # no embedding request occurs during activation.
        self.client_for_profile(profile)
        previous = await self.get(pointer.profile_id)
        if previous.id != profile.id:
            previous.status = EmbeddingProfileStatus.INACTIVE
            await self.session.flush()
        profile.status = EmbeddingProfileStatus.ACTIVE
        pointer.profile_id = profile.id
        pointer.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return profile

    async def backfill(self, profile: EmbeddingProfile | str | UUID, *, client=None, progress=print) -> BackfillReport:
        """Missing-only offline generation; successful batches commit independently."""
        if not isinstance(profile, EmbeddingProfile):
            profile = await self.get(profile)
        validate_profile(profile)
        profile_id = profile.id
        definition = profile_definition(profile)
        compatibility = await self.compatibility(profile)
        progress("compatibility=" + json.dumps(compatibility.as_dict(), ensure_ascii=False))
        if not compatibility.compatible:
            raise CompatibilityError(compatibility)
        initial = await self.coverage(profile)
        total = initial.knowledge.total + initial.memory.total
        existing = initial.knowledge.existing + initial.memory.existing
        pending = total - existing
        completed = failed = 0
        entity_reports = {
            "knowledge": {"total": initial.knowledge.total, "existing": initial.knowledge.existing,
                          "pending": initial.knowledge.missing, "completed": 0, "failed": 0},
            "memory": {"total": initial.memory.total, "existing": initial.memory.existing,
                       "pending": initial.memory.missing, "completed": 0, "failed": 0},
        }
        client = client or self.client_for_profile(profile)
        initial_batches = getattr(client, "embedding_batches", 0)
        initial_attempts = getattr(client, "http_attempts", 0)
        counter = load_token_counter(definition.tokenizer_encoding)
        if getattr(getattr(client, "profile", None), "id", profile_id) != profile_id:
            raise ValueError("Backfill client belongs to a different embedding profile")
        active_id = await self.session.scalar(select(ActiveEmbeddingProfile.profile_id).where(ActiveEmbeddingProfile.singleton_id == 1))
        if profile_id != active_id:
            profile.status = EmbeddingProfileStatus.BACKFILLING
        await self.session.commit()
        progress(f"total={total} existing={existing} pending={pending} completed=0 failed=0")
        for entity, state in entity_reports.items():
            progress(entity + ": " + " ".join(f"{key}={value}" for key, value in state.items()))
        for entity, model, vector_model, id_column, content_column, eligible in (
            ("knowledge", KnowledgeChunk, KnowledgeChunkEmbedding, KnowledgeChunkEmbedding.chunk_id, KnowledgeChunk.retrieval_text, self._knowledge),
            ("memory", MemoryItem, MemoryItemEmbedding, MemoryItemEmbedding.memory_item_id, MemoryItem.content, self._memory),
        ):
            missing = ~exists(select(vector_model.profile_id).where(id_column == model.id, vector_model.profile_id == profile_id))
            # Snapshot only IDs. Read and lock current text again for each batch,
            # preventing a memory MERGE from racing a stale corpus embedding.
            ids = list((await self.session.scalars(eligible().with_only_columns(model.id).where(missing).order_by(model.id))).all())
            await self.session.commit()
            for start in range(0, len(ids), definition.batch_limit):
                batch_ids = ids[start:start + definition.batch_limit]
                try:
                    await self.session.scalar(select(ActiveEmbeddingProfile).where(ActiveEmbeddingProfile.singleton_id == 1).with_for_update(read=True))
                    rows = list((await self.session.execute(
                        eligible().with_only_columns(model.id, content_column).where(model.id.in_(batch_ids), missing)
                        .order_by(model.id).with_for_update(of=model)
                    )).all())
                    if not rows:
                        await self.session.commit()
                        continue
                    # rollback expires mapped state, including an injected
                    # client's profile; reload before the next provider batch.
                    profile = await self.get(profile_id)
                    for entity_id, content in rows:
                        tokens = counter.count(definition.document_prefix + content)
                        if tokens > definition.max_input_tokens:
                            raise ValueError(f"Embedding input incompatible with profile {definition.name}: "
                                             f"entity_id={entity_id} tokens={tokens} limit={definition.max_input_tokens}")
                    vectors = await client.embed_many([row[1] for row in rows])
                    if len(vectors) != len(rows):
                        raise ValueError("Backfill provider vector count mismatch")
                    inserted = 0
                    for row, vector in zip(rows, vectors, strict=True):
                        if len(vector) != definition.dimension:
                            raise ValueError("Backfill provider vector dimension mismatch")
                        if not all(math.isfinite(float(value)) for value in vector) or math.hypot(*vector) == 0:
                            raise ValueError("Backfill provider returned a non-finite or zero vector")
                        result = await self.session.execute(
                            insert(vector_model).values(**{id_column.key: row[0]}, profile_id=profile_id, embedding=vector)
                            .on_conflict_do_nothing(index_elements=[id_column, vector_model.profile_id])
                            .returning(id_column)
                        )
                        inserted += int(result.scalar_one_or_none() is not None)
                    profile = await self.get(profile_id)
                    if inserted:
                        profile.validation_passed_at = None
                        profile.validation_note = None
                    await self.session.commit()
                    completed += inserted
                    entity_reports[entity]["completed"] += inserted
                except Exception as exc:
                    await self.session.rollback()
                    failed += len(batch_ids)
                    entity_reports[entity]["failed"] += len(batch_ids)
                    # Provider exception text can contain private input or URLs.
                    if isinstance(exc, ValueError) and str(exc).startswith("Embedding input incompatible with profile "):
                        progress(f"batch_incompatible entity={model.__tablename__} count={len(batch_ids)} "
                                 f"profile={definition.name} max_input_tokens={definition.max_input_tokens} "
                                 f"tokenizer={definition.tokenizer_encoding}; no truncation or rechunking performed")
                    else:
                        progress(f"batch_failed entity={model.__tablename__} count={len(batch_ids)} error={type(exc).__name__}")
                progress(f"total={total} existing={existing} pending={pending} completed={completed} failed={failed}")
                progress(entity + ": " + " ".join(f"{key}={value}" for key, value in entity_reports[entity].items()))
        profile = await self.get(profile_id)
        final = await self.coverage(profile)
        try:
            if final.complete and not failed:
                await self.ensure_indexes(profile)
        except Exception as exc:
            await self.session.rollback()
            failed += 1
            progress(f"index_failed error={type(exc).__name__}")
            profile = await self.get(profile_id)
        active_id = await self.session.scalar(select(ActiveEmbeddingProfile.profile_id).where(ActiveEmbeddingProfile.singleton_id == 1))
        if profile_id != active_id:
            profile.status = EmbeddingProfileStatus.READY if final.complete and not failed else EmbeddingProfileStatus.FAILED
        await self.session.commit()
        report = BackfillReport(total, existing, pending, completed, failed, final,
                                EntityBackfillReport(**entity_reports["knowledge"]),
                                EntityBackfillReport(**entity_reports["memory"]),
                                getattr(client, "embedding_batches", 0) - initial_batches,
                                getattr(client, "http_attempts", 0) - initial_attempts)
        progress("READY" if final.complete and not failed else "INCOMPLETE")
        return report
