from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ConversationCreate, ConversationDetail, ConversationOut, MessageOut
from app.conversation.repository import ConversationRepository
from app.conversation.service import ConversationService
from app.infra.db import get_session
from app.memory.repository import MemoryRepository

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationOut])
async def list_conversations(session: AsyncSession = Depends(get_session)) -> list[ConversationOut]:
    return await ConversationRepository(session).list()


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate, session: AsyncSession = Depends(get_session)
) -> ConversationOut:
    return await ConversationService(session).create(payload.title)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: UUID, session: AsyncSession = Depends(get_session)) -> ConversationDetail:
    return await ConversationService(session).require(conversation_id, with_messages=True)


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(conversation_id: UUID, session: AsyncSession = Depends(get_session)) -> list[MessageOut]:
    service = ConversationService(session)
    await service.require(conversation_id)
    return await ConversationRepository(session).messages(conversation_id)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: UUID, session: AsyncSession = Depends(get_session)) -> Response:
    service = ConversationService(session)
    await service.require(conversation_id)
    await ConversationRepository(session).delete(conversation_id)
    await MemoryRepository(session).cleanup_orphans()
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
