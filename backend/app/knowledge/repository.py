from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentStatus, KnowledgeChunk, KnowledgeDocument, KnowledgeSection
from app.evals.scope import get_eval_scope


class KnowledgeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def by_sha256(self, sha256: str) -> KnowledgeDocument | None:
        return await self.session.scalar(
            select(KnowledgeDocument).where(KnowledgeDocument.sha256 == sha256).order_by(KnowledgeDocument.created_at)
        )

    async def get(self, document_id: UUID) -> KnowledgeDocument | None:
        return await self.session.scalar(select(KnowledgeDocument).where(KnowledgeDocument.id == document_id))

    async def list(self, query: str | None, limit: int) -> list[KnowledgeDocument]:
        statement = select(KnowledgeDocument).order_by(KnowledgeDocument.updated_at.desc()).limit(limit)
        scope = get_eval_scope()
        if scope is not None:
            statement = statement.where(KnowledgeDocument.id.in_(scope.corpus_document_ids))
        if query:
            statement = statement.where(KnowledgeDocument.title.ilike(f"%{query}%"))
        return list((await self.session.scalars(statement)).all())

    async def sections(self, document_id: UUID) -> list[KnowledgeSection]:
        return list(
            (await self.session.scalars(
                select(KnowledgeSection).where(KnowledgeSection.document_id == document_id).order_by(KnowledgeSection.ordinal)
            )).all()
        )

    async def counts(self, document_id: UUID) -> tuple[int, int]:
        sections = await self.session.scalar(
            select(func.count()).select_from(KnowledgeSection).where(KnowledgeSection.document_id == document_id)
        )
        chunks = await self.session.scalar(
            select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id)
        )
        return sections or 0, chunks or 0

    async def archive(self, document_id: UUID) -> KnowledgeDocument | None:
        item = await self.get(document_id)
        if item:
            item.status = DocumentStatus.ARCHIVED
            await self.session.flush()
        return item

    async def delete(self, document_id: UUID) -> bool:
        result = await self.session.execute(delete(KnowledgeDocument).where(KnowledgeDocument.id == document_id))
        return bool(result.rowcount)
