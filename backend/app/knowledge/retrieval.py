from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import case, cast, func, literal, select
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentStatus, KnowledgeChunk, KnowledgeChunkEmbedding, KnowledgeDocument, KnowledgeSection
from app.evals.scope import get_eval_scope
from app.infra.embedding import EmbeddingClient
from app.infra.token_count import TokenCounter
from app.knowledge.sources import SourceRef

RRF_K = 60

SearchRow = tuple[KnowledgeChunk, KnowledgeDocument, KnowledgeSection, float]


@dataclass(frozen=True)
class KnowledgeModule:
    document_id: UUID
    document_title: str
    section_id: UUID
    section_path: list[str]
    content: str
    source_refs: list[SourceRef]
    token_count: int

    def agent_payload(self) -> dict:
        return {
            "document_id": str(self.document_id), "document_title": self.document_title,
            "section_path": self.section_path, "content": self.content,
            "source_refs": [item.uri for item in self.source_refs], "token_count": self.token_count,
        }


class KnowledgeRetriever:
    def __init__(self, session: AsyncSession, embedding: EmbeddingClient, counter: TokenCounter) -> None:
        self.session = session
        self.embedding = embedding
        self.counter = counter

    async def search(
        self, query: str, *, document_ids: list[UUID] | None, max_modules: int, max_tokens: int
    ) -> list[KnowledgeModule]:
        modules, _ = await self.search_with_debug(query, document_ids=document_ids, max_modules=max_modules, max_tokens=max_tokens)
        return modules

    async def search_with_debug(
        self, query: str, *, document_ids: list[UUID] | None, max_modules: int, max_tokens: int
    ) -> tuple[list[KnowledgeModule], dict]:
        await self.embedding.assert_resolved_profile(self.session)
        base = (
            select(KnowledgeChunk, KnowledgeDocument, KnowledgeSection)
            .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
            .join(KnowledgeSection, KnowledgeSection.id == KnowledgeChunk.section_id)
            .where(KnowledgeDocument.status == DocumentStatus.READY)
        )
        if document_ids:
            base = base.where(KnowledgeChunk.document_id.in_(document_ids))
        scope = get_eval_scope()
        if scope is not None:
            base = base.where(KnowledgeChunk.document_id.in_(scope.corpus_document_ids))

        lexical = await self._lexical(base, query)
        semantic = await self._semantic(base, query)
        fused: dict[UUID, float] = defaultdict(float)
        rows: dict[UUID, SearchRow] = {}
        lexical_rank: dict[UUID, int] = {}
        semantic_rank: dict[UUID, int] = {}
        for rank, row in enumerate(lexical, start=1):
            chunk = row[0]
            lexical_rank[chunk.id] = rank
            fused[chunk.id] += 1 / (RRF_K + rank)
            rows[chunk.id] = row
        for rank, row in enumerate(semantic, start=1):
            chunk = row[0]
            semantic_rank[chunk.id] = rank
            fused[chunk.id] += 1 / (RRF_K + rank)
            rows[chunk.id] = row
        ranked = sorted(rows.values(), key=lambda row: fused[row[0].id], reverse=True)
        modules = self._to_modules(ranked, max_modules=max_modules, max_tokens=max_tokens)
        preview = lambda row: row[0].content[:500]
        debug = {
            "input": {"query": query, "document_ids": [str(item) for item in document_ids or []], "max_modules": max_modules, "max_tokens": max_tokens},
            "output": {
                "lexical": [{"chunk_id": str(row[0].id), "document_id": str(row[1].id), "section_id": str(row[2].id), "score": float(row[3]), "rank": rank, "text_preview": preview(row)} for rank, row in enumerate(lexical, 1)],
                "vector": [{"chunk_id": str(row[0].id), "document_id": str(row[1].id), "section_id": str(row[2].id), "distance": float(row[3]), "rank": rank, "text_preview": preview(row)} for rank, row in enumerate(semantic, 1)],
                "fusion": [{"chunk_id": str(row[0].id), "lexical_rank": lexical_rank.get(row[0].id), "vector_rank": semantic_rank.get(row[0].id), "fusion_score": fused[row[0].id], "final_rank": rank} for rank, row in enumerate(ranked, 1)],
                "modules": [{**module.agent_payload(), "included_chunk_ids": [ref.uri.rsplit("/", 1)[-1] for ref in module.source_refs if ref.uri.startswith("document://") and "/chunk/" in ref.uri]} for module in modules],
            },
            "metadata": {"rrf_k": RRF_K, "algorithm": "lexical + cosine vector + reciprocal-rank fusion + section assembly", **self.embedding.profile_metadata()},
            "source_refs": [ref.uri for module in modules for ref in module.source_refs],
        }
        return modules, debug

    async def _lexical(self, base: object, query: str) -> list[SearchRow]:
        terms = [term for term in re.findall(r"[\w.-]+", query.lower()) if len(term) > 1]
        exact_boost = sum(
            (case((func.lower(KnowledgeChunk.retrieval_text).contains(term), 1.0), else_=0.0) for term in terms),
            start=0.0,
        )
        score = func.similarity(KnowledgeChunk.retrieval_text, query) + exact_boost
        statement = base.add_columns(score.label("lexical_score")).where(KnowledgeChunk.retrieval_text.ilike(f"%{query[:160]}%") | (score > 0.05)).order_by(score.desc()).limit(30)
        return list((await self.session.execute(statement)).all())

    async def _semantic(self, base: object, query: str) -> list[SearchRow]:
        vector = await self.embedding.embed_one(query, purpose="query")
        if vector is None:
            return []
        profile = self.embedding.profile
        distance = cast(KnowledgeChunkEmbedding.embedding, Vector(profile.dimension)).cosine_distance(vector)
        statement = (
            base.join(KnowledgeChunkEmbedding, KnowledgeChunkEmbedding.chunk_id == KnowledgeChunk.id)
            .add_columns(distance.label("vector_distance"))
            .where(KnowledgeChunkEmbedding.profile_id == literal(profile.id, type_=PGUUID(as_uuid=True), literal_execute=True))
            .order_by(distance).limit(30)
        )
        return list((await self.session.execute(statement)).all())

    def _to_modules(
        self,
        ranked: list[SearchRow],
        *,
        max_modules: int,
        max_tokens: int,
    ) -> list[KnowledgeModule]:
        grouped: dict[UUID, list[SearchRow]] = {}
        for row in ranked:
            grouped.setdefault(row[2].id, []).append(row)
        remaining = max_tokens
        modules: list[KnowledgeModule] = []
        for section_rows in grouped.values():
            if len(modules) >= max_modules or remaining <= 0:
                break
            _, document, section, _ = section_rows[0]
            included: list[str] = []
            refs = [SourceRef.section(document.id, section.id)]
            for chunk, _, _, _ in sorted(section_rows, key=lambda row: row[0].ordinal):
                if self.counter.count("\n\n".join([*included, chunk.content])) > remaining:
                    break
                included.append(chunk.content)
                refs.append(SourceRef.chunk(document.id, chunk.id))
            content = "\n\n".join(included)
            if not content:
                continue
            tokens = self.counter.count(content)
            modules.append(KnowledgeModule(document.id, document.title, section.id, section.section_path, content, refs, tokens))
            remaining -= tokens
        return modules
