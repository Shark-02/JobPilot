from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Conversation, Message, MessageRole
from app.evals.scope import apply_eval_scope, scoped_values


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, title: str | None = None) -> Conversation:
        item = Conversation(title=title, **scoped_values())
        self.session.add(item)
        await self.session.flush()
        return item

    async def get(self, conversation_id: UUID, *, with_messages: bool = False) -> Conversation | None:
        statement = apply_eval_scope(
            select(Conversation).where(Conversation.id == conversation_id), Conversation
        )
        if with_messages:
            statement = statement.options(selectinload(Conversation.messages))
        return await self.session.scalar(statement)

    async def list(self, *, limit: int = 50, offset: int = 0) -> list[Conversation]:
        statement = apply_eval_scope(
            select(Conversation).order_by(Conversation.updated_at.desc()).offset(offset).limit(limit),
            Conversation,
        )
        return list((await self.session.scalars(statement)).all())

    async def add_message(
        self,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata_=metadata or {},
            **scoped_values(),
        )
        self.session.add(message)
        statement = apply_eval_scope(
            update(Conversation).where(Conversation.id == conversation_id), Conversation
        )
        await self.session.execute(statement.values(updated_at=func.now()))
        await self.session.flush()
        return message

    async def messages(self, conversation_id: UUID) -> list[Message]:
        statement = apply_eval_scope(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at), Message
        )
        return list((await self.session.scalars(statement)).all())

    async def delete(self, conversation_id: UUID) -> bool:
        result = await self.session.execute(
            apply_eval_scope(delete(Conversation).where(Conversation.id == conversation_id), Conversation)
        )
        return bool(result.rowcount)
