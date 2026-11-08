from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversation.repository import ConversationRepository
from app.db.models import Conversation, Message, MessageRole


class ConversationService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = ConversationRepository(session)

    async def create(self, title: str | None = None) -> Conversation:
        item = await self.repo.create(title.strip() if title else None)
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def require(self, conversation_id: UUID, *, with_messages: bool = False) -> Conversation:
        item = await self.repo.get(conversation_id, with_messages=with_messages)
        if item is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        return item

    async def append_user_message(
        self, conversation_id: UUID, content: str, attachment_document_ids: list[UUID] | None = None
    ) -> Message:
        await self.require(conversation_id)
        message = await self.repo.add_message(
            conversation_id,
            MessageRole.USER,
            content.strip(),
            {"document_ids": [str(item) for item in attachment_document_ids or []]},
        )
        await self.session.commit()
        await self.session.refresh(message)
        return message

    async def append_assistant_message(self, conversation_id: UUID, content: str, turn_id: UUID) -> Message:
        message = await self.repo.add_message(
            conversation_id, MessageRole.ASSISTANT, content, {"agent_turn_id": str(turn_id)}
        )
        await self.session.commit()
        await self.session.refresh(message)
        return message

