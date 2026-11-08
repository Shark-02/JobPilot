from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryItem, MemorySource, MemoryStatus
from app.evals.scope import apply_eval_scope


class MemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, memory_id: UUID) -> MemoryItem | None:
        return await self.session.scalar(
            apply_eval_scope(select(MemoryItem).where(MemoryItem.id == memory_id), MemoryItem)
        )

    async def list(self, *, status: MemoryStatus | None = None, limit: int = 100) -> list[MemoryItem]:
        statement = apply_eval_scope(
            select(MemoryItem).order_by(MemoryItem.updated_at.desc()).limit(limit), MemoryItem
        )
        if status:
            statement = statement.where(MemoryItem.status == status)
        return list((await self.session.scalars(statement)).all())

    async def source_rows(self, memory_ids: list[UUID]) -> list[MemorySource]:
        if not memory_ids:
            return []
        return list((await self.session.scalars(select(MemorySource).where(MemorySource.memory_id.in_(memory_ids)))).all())

    async def archive(self, memory_id: UUID) -> MemoryItem | None:
        item = await self.get(memory_id)
        if item:
            item.status = MemoryStatus.ARCHIVED
            await self.session.flush()
        return item

    async def delete(self, memory_id: UUID) -> bool:
        result = await self.session.execute(
            apply_eval_scope(delete(MemoryItem).where(MemoryItem.id == memory_id), MemoryItem)
        )
        return bool(result.rowcount)

    async def cleanup_orphans(self) -> int:
        result = await self.session.execute(
            apply_eval_scope(
                delete(MemoryItem).where(~exists(select(MemorySource.id).where(MemorySource.memory_id == MemoryItem.id))),
                MemoryItem,
            )
        )
        return result.rowcount or 0
