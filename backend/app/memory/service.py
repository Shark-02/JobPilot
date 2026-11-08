from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentRole, MemoryItem, MemoryItemEmbedding, MemorySource, MemoryStatus, MessageRole, SourceRelation, SourceType
from app.evals.scope import apply_eval_scope, scoped_values
from app.evals.evidence import get_observation_collector
from app.infra.embedding import EmbeddingClient
from app.knowledge.sources import SourceRef, SourceResolver
from app.memory.repository import MemoryRepository


class PossibleDuplicateError(ValueError):
    pass


@dataclass(frozen=True)
class MutationCommand:
    operation: str
    target_memory_id: UUID | None
    kind: object | None
    content: str | None
    attributes: dict
    importance: object
    source_refs: list[SourceRef]


class MemoryService:
    def __init__(self, session: AsyncSession, embedding: EmbeddingClient, resolver: SourceResolver) -> None:
        self.session = session
        self.embedding = embedding
        self.resolver = resolver
        self.repo = MemoryRepository(session)

    async def mutate(self, mutations: list[MutationCommand]) -> list[MemoryItem]:
        collector = get_observation_collector()
        events: list[dict] = []
        try:
            await self.embedding.assert_resolved_profile(self.session, for_write=True)
            changed: list[MemoryItem] = []
            for mutation in mutations:
                event = {"operation": mutation.operation,
                         "target_memory_id": str(mutation.target_memory_id) if mutation.target_memory_id else None,
                         "new_source_refs": [ref.uri for ref in mutation.source_refs],
                         "before": [], "after": [], "sources": [], "success": False}
                if collector is not None:
                    if mutation.target_memory_id:
                        event["before"] = await self._eval_snapshots([mutation.target_memory_id])
                    events.append(event)
                if mutation.operation == "ADD":
                    changed.append(await self._add(mutation))
                elif mutation.operation == "MERGE":
                    changed.append(await self._merge(mutation))
                elif mutation.operation == "SUPERSEDE":
                    changed.append(await self._supersede(mutation))
                elif mutation.operation == "ARCHIVE":
                    changed.append(await self._archive(mutation))
                else:
                    raise ValueError(f"Unknown memory mutation: {mutation.operation}")
                if collector is not None:
                    event["result_memory_id"] = str(changed[-1].id)
                    snapshot_ids = list({item for item in (mutation.target_memory_id, changed[-1].id) if item})
                    event["after"] = await self._eval_snapshots(snapshot_ids)
                    # This is evaluator-only source annotation. It never becomes
                    # a main-Agent message or a substitute for visible evidence.
                    for ref in mutation.source_refs:
                        sources = await self.resolver.read([ref], max_tokens=None)
                        event["sources"].extend({"source_ref": source.source_ref.uri, "content": source.content,
                                                 "metadata": source.metadata} for source in sources)
            await self.session.commit()
            if collector is not None:
                for event in events:
                    collector.record_mutation({**event, "success": True})
            return changed
        except Exception as exc:
            await self.session.rollback()
            if collector is not None:
                if not events:
                    events = [{"operation": command.operation, "target_memory_id": str(command.target_memory_id) if command.target_memory_id else None,
                               "new_source_refs": [ref.uri for ref in command.source_refs], "before": [], "after": [], "sources": []}
                              for command in mutations]
                for event in events:
                    collector.record_mutation({**event, "success": False, "after": [], "error": type(exc).__name__})
            raise

    async def _eval_snapshots(self, memory_ids: list[UUID]) -> list[dict]:
        """Capture actual transactional state, including durable source bindings."""
        from app.db.models import KnowledgeChunk, KnowledgeSection
        rows = list((await self.session.scalars(apply_eval_scope(
            select(MemoryItem).where(MemoryItem.id.in_(memory_ids)), MemoryItem
        ))).all())
        sources = list((await self.session.execute(
            select(MemorySource, KnowledgeSection.document_id, KnowledgeChunk.document_id)
            .outerjoin(KnowledgeSection, KnowledgeSection.id == MemorySource.document_section_id)
            .outerjoin(KnowledgeChunk, KnowledgeChunk.id == MemorySource.document_chunk_id)
            .where(MemorySource.memory_id.in_(memory_ids))
        )).all())
        bindings: dict[UUID, list[str]] = {}
        for source, section_document_id, chunk_document_id in sources:
            ref = SourceRef.message(source.message_id) if source.message_id else (
                SourceRef.section(section_document_id, source.document_section_id) if section_document_id else
                SourceRef.chunk(chunk_document_id, source.document_chunk_id) if chunk_document_id else None)
            if ref:
                bindings.setdefault(source.memory_id, []).append(ref.uri)
        return [{"id": str(item.id), "kind": item.kind.value, "content": item.content,
                 "attributes": dict(item.attributes), "status": item.status.value,
                 "source_refs": sorted(set(bindings.get(item.id, [])))} for item in rows]

    async def _validate_sources(self, refs: list[SourceRef]) -> None:
        if not refs:
            raise ValueError("A non-ARCHIVE memory mutation requires at least one source_ref")
        # Validate every pointer independently. Content budgeting must never make a later
        # source evade existence, ownership, or authority checks.
        resolved = []
        for ref in refs:
            items = await self.resolver.read([ref], max_tokens=1)
            if len(items) != 1:
                raise LookupError(f"Source not found: {ref.uri}")
            resolved.append(items[0])
        for source in resolved:
            if source.metadata.get("role") and source.metadata["role"] != MessageRole.USER.value:
                raise ValueError("Assistant messages cannot be used as evidence for user memory")
            if source.metadata.get("source_role") and source.metadata["source_role"] != DocumentRole.USER_CAREER_SOURCE.value:
                raise ValueError("REFERENCE_SOURCE documents cannot establish user memory")

    async def _add(self, mutation: MutationCommand) -> MemoryItem:
        if mutation.target_memory_id is not None or mutation.kind is None or not mutation.content:
            raise ValueError("ADD requires kind and content, and cannot have target_memory_id")
        await self._validate_sources(mutation.source_refs)
        duplicate = await self.session.scalar(
            apply_eval_scope(
                select(MemoryItem).where(MemoryItem.status == MemoryStatus.ACTIVE, MemoryItem.content == mutation.content),
                MemoryItem,
            )
        )
        if duplicate:
            raise PossibleDuplicateError(f"An active memory already has this content: {duplicate.id}")
        item = MemoryItem(
            kind=mutation.kind, content=mutation.content.strip(), attributes=mutation.attributes,
            importance=mutation.importance, status=MemoryStatus.ACTIVE,
            **scoped_values(),
        )
        self.session.add(item)
        await self.session.flush()
        self.session.add(MemoryItemEmbedding(
            memory_item_id=item.id, profile_id=self.embedding.profile.id,
            embedding=await self.embedding.embed_one(mutation.content),
        ))
        await self._add_sources(item.id, mutation.source_refs)
        return item

    async def _merge(self, mutation: MutationCommand) -> MemoryItem:
        item = await self._require_active(mutation.target_memory_id)
        if not mutation.content:
            raise ValueError("MERGE requires content")
        await self._validate_sources(mutation.source_refs)
        item.content = mutation.content.strip()
        item.attributes = mutation.attributes
        item.importance = mutation.importance
        if mutation.kind is not None:
            item.kind = mutation.kind
        # The entity ID and evidence stay stable, but old content vectors are invalid.
        # Historical profiles are backfilled explicitly if the operator switches back.
        await self.session.execute(delete(MemoryItemEmbedding).where(MemoryItemEmbedding.memory_item_id == item.id))
        self.session.add(MemoryItemEmbedding(
            memory_item_id=item.id, profile_id=self.embedding.profile.id,
            embedding=await self.embedding.embed_one(item.content),
        ))
        await self._add_sources(item.id, mutation.source_refs)
        return item

    async def _supersede(self, mutation: MutationCommand) -> MemoryItem:
        old = await self._require_active(mutation.target_memory_id)
        if mutation.kind is None or not mutation.content:
            raise ValueError("SUPERSEDE requires target_memory_id, kind, and content")
        await self._validate_sources(mutation.source_refs)
        old.status = MemoryStatus.SUPERSEDED
        old.valid_to = datetime.now(timezone.utc)
        item = MemoryItem(
            kind=mutation.kind, content=mutation.content.strip(), attributes=mutation.attributes,
            importance=mutation.importance, status=MemoryStatus.ACTIVE,
            valid_from=datetime.now(timezone.utc),
            **scoped_values(),
        )
        self.session.add(item)
        await self.session.flush()
        self.session.add(MemoryItemEmbedding(
            memory_item_id=item.id, profile_id=self.embedding.profile.id,
            embedding=await self.embedding.embed_one(mutation.content),
        ))
        await self._add_sources(item.id, mutation.source_refs)
        return item

    async def _archive(self, mutation: MutationCommand) -> MemoryItem:
        if mutation.target_memory_id is None:
            raise ValueError("ARCHIVE requires target_memory_id")
        item = await self.repo.get(mutation.target_memory_id)
        if item is None:
            raise LookupError("Memory not found")
        item.status = MemoryStatus.ARCHIVED
        return item

    async def _require_active(self, target_memory_id: UUID | None) -> MemoryItem:
        if target_memory_id is None:
            raise ValueError("target_memory_id is required")
        item = await self.repo.get(target_memory_id)
        if item is None:
            raise LookupError("Memory not found")
        if item.status is not MemoryStatus.ACTIVE:
            raise ValueError("Only ACTIVE memories can be changed")
        return item

    async def _add_sources(self, memory_id: UUID, refs: list[SourceRef]) -> None:
        for ref in refs:
            scheme, first_id, kind, source_id = ref.parts()
            if scheme == "message":
                row = MemorySource(
                    memory_id=memory_id, source_type=SourceType.MESSAGE, message_id=first_id,
                    relation=SourceRelation.SUPPORTS,
                )
            elif kind == "section":
                row = MemorySource(
                    memory_id=memory_id, source_type=SourceType.DOCUMENT_SECTION, document_section_id=source_id,
                    relation=SourceRelation.SUPPORTS,
                )
            else:
                row = MemorySource(
                    memory_id=memory_id, source_type=SourceType.DOCUMENT_CHUNK, document_chunk_id=source_id,
                    relation=SourceRelation.SUPPORTS,
                )
            self.session.add(row)
        await self.session.flush()
