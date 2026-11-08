from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import MemoryOut
from app.memory.repository import MemoryRepository
from app.infra.db import get_session

router = APIRouter(prefix="/api/memories", tags=["memories"])


@router.get("", response_model=list[MemoryOut])
async def list_memories(session: AsyncSession = Depends(get_session)) -> list[MemoryOut]:
    return await MemoryRepository(session).list()


@router.get("/{memory_id}", response_model=MemoryOut)
async def get_memory(memory_id: UUID, session: AsyncSession = Depends(get_session)) -> MemoryOut:
    item = await MemoryRepository(session).get(memory_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return item


@router.post("/{memory_id}/archive", response_model=MemoryOut)
async def archive_memory(memory_id: UUID, session: AsyncSession = Depends(get_session)) -> MemoryOut:
    item = await MemoryRepository(session).archive(memory_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    await session.commit()
    await session.refresh(item)
    return item


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: UUID, session: AsyncSession = Depends(get_session)) -> Response:
    if not await MemoryRepository(session).delete(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
