from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import case, cast, func, literal, select
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryItem, MemoryItemEmbedding, MemoryKind, MemorySource, MemoryStatus
from app.evals.scope import apply_eval_scope
from app.infra.embedding import EmbeddingClient
from app.knowledge.sources import SourceRef

RRF_K = 60


@dataclass(frozen=True)
class MemoryMatch:
    id: UUID
    kind: MemoryKind
    content: str
    attributes: dict
    source_refs: list[SourceRef]

    def agent_payload(self) -> dict:
        return {
            "id": str(self.id), "kind": self.kind.value, "content": self.content,
            "attributes": self.attributes, "source_refs": [item.uri for item in self.source_refs],
        }


def source_ref_from_row(row: MemorySource) -> SourceRef:
    if row.message_id:
        return SourceRef.message(row.message_id)
    if row.document_section_id:
        document_id = row.document_section_id  # Replaced by source resolver-free join in service when necessary.
        # This branch cannot reconstruct the document UUID without an extra join; search query returns source refs below.
        raise ValueError(f"Cannot reconstruct section source ref without document id: {document_id}")
    if row.document_chunk_id:
        raise ValueError(f"Cannot reconstruct chunk source ref without document id: {row.document_chunk_id}")
    raise ValueError("Memory source has no target")


class MemoryRetriever:
    def __init__(self, session: AsyncSession, embedding: EmbeddingClient) -> None:
        self.session = session
        self.embedding = embedding

    async def search(self, query: str, kinds: list[MemoryKind] | None, limit: int) -> list[MemoryMatch]:
        await self.embedding.assert_resolved_profile(self.session)
        base = apply_eval_scope(
            select(MemoryItem).where(MemoryItem.status == MemoryStatus.ACTIVE), MemoryItem
        )
        if kinds:
            base = base.where(MemoryItem.kind.in_(kinds))
        lexical = await self._lexical(base, query)
        semantic = await self._semantic(base, query)
        score: dict[UUID, float] = defaultdict(float)
        items: dict[UUID, MemoryItem] = {}
        for rank, item in enumerate(lexical, start=1):
            score[item.id] += 1 / (RRF_K + rank)
            items[item.id] = item
        for rank, item in enumerate(semantic, start=1):
            score[item.id] += 1 / (RRF_K + rank)
            items[item.id] = item
        ordered = sorted(items.values(), key=lambda item: score[item.id], reverse=True)[:limit]
        return await self._matches(ordered)

    async def _lexical(self, base: object, query: str) -> list[MemoryItem]:
        terms = [term for term in re.findall(r"[\w.-]+", query.lower()) if len(term) > 1]
        boost = sum(
            (case((func.lower(MemoryItem.content).contains(term), 1.0), else_=0.0) for term in terms), start=0.0
        )
        score = func.similarity(MemoryItem.content, query) + boost
        statement = base.where(MemoryItem.content.ilike(f"%{query[:160]}%") | (score > 0.05)).order_by(score.desc()).limit(30)
        return list((await self.session.scalars(statement)).all())

    async def _semantic(self, base: object, query: str) -> list[MemoryItem]:
        vector = await self.embedding.embed_one(query, purpose="query")
        if vector is None:
            return []
        profile = self.embedding.profile
        distance = cast(MemoryItemEmbedding.embedding, Vector(profile.dimension)).cosine_distance(vector)
        statement = (
            base.join(MemoryItemEmbedding, MemoryItemEmbedding.memory_item_id == MemoryItem.id)
            .where(MemoryItemEmbedding.profile_id == literal(profile.id, type_=PGUUID(as_uuid=True), literal_execute=True))
            .order_by(distance).limit(30)
        )
        return list((await self.session.scalars(statement)).all())

    async def _matches(self, items: list[MemoryItem]) -> list[MemoryMatch]:
        # Resolve document IDs in SQL so returned refs are independently readable.
        from app.db.models import KnowledgeChunk, KnowledgeSection

        source_rows = list(
            (await self.session.execute(
                select(MemorySource, KnowledgeSection.document_id, KnowledgeChunk.document_id)
                .outerjoin(KnowledgeSection, KnowledgeSection.id == MemorySource.document_section_id)
                .outerjoin(KnowledgeChunk, KnowledgeChunk.id == MemorySource.document_chunk_id)
                .where(MemorySource.memory_id.in_([item.id for item in items]))
            )).all()
        ) if items else []
        refs: dict[UUID, list[SourceRef]] = defaultdict(list)
        for source, section_document_id, chunk_document_id in source_rows:
            if source.message_id:
                refs[source.memory_id].append(SourceRef.message(source.message_id))
            elif source.document_section_id and section_document_id:
                refs[source.memory_id].append(SourceRef.section(section_document_id, source.document_section_id))
            elif source.document_chunk_id and chunk_document_id:
                refs[source.memory_id].append(SourceRef.chunk(chunk_document_id, source.document_chunk_id))
        return [MemoryMatch(item.id, item.kind, item.content, item.attributes, refs[item.id]) for item in items]
