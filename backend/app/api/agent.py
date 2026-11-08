from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.events import cancellation_registry, event_broker
from app.agent.loop import start_turn_background
from app.api.schemas import RunCreate, RunOut
from app.config import get_settings
from app.conversation.repository import ConversationRepository
from app.db.models import AgentTurn, Conversation, DocumentStatus, KnowledgeDocument, MessageRole, TurnStatus
from app.infra.db import get_session

router = APIRouter(tags=["agent"])


async def _create_turn(conversation_id: UUID, content: str, attachment_document_ids: list[UUID], session: AsyncSession) -> AgentTurn:
    """Persist the user turn once; both REST and AG-UI transports share it."""
    try:
        async with session.begin():
            conversation = await session.scalar(select(Conversation).where(Conversation.id == conversation_id).with_for_update())
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            existing = await session.scalar(select(AgentTurn.id).where(AgentTurn.conversation_id == conversation_id, AgentTurn.status == TurnStatus.RUNNING))
            if existing is not None:
                raise HTTPException(status_code=409, detail="This conversation already has a running agent turn")
            if attachment_document_ids:
                ready_ids = set((await session.scalars(select(KnowledgeDocument.id).where(KnowledgeDocument.id.in_(attachment_document_ids), KnowledgeDocument.status == DocumentStatus.READY))).all())
                if ready_ids != set(attachment_document_ids):
                    raise HTTPException(status_code=400, detail="Every attachment must reference a READY document")
            message = await ConversationRepository(session).add_message(conversation_id, MessageRole.USER, content.strip(), {"document_ids": [str(item) for item in attachment_document_ids]})
            turn = AgentTurn(conversation_id=conversation_id, user_message_id=message.id, status=TurnStatus.RUNNING)
            session.add(turn)
            await session.flush()
        return turn
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="This conversation already has a running agent turn") from exc


@router.post("/api/conversations/{conversation_id}/runs", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    conversation_id: UUID, payload: RunCreate, session: AsyncSession = Depends(get_session)
) -> RunOut:
    turn = await _create_turn(conversation_id, payload.content, payload.attachment_document_ids, session)
    start_turn_background(turn.id, get_settings())
    return RunOut(turn_id=turn.id, status=turn.status, events_url=f"/api/turns/{turn.id}/events")


@router.post("/api/ag-ui/conversations/{conversation_id}")
async def run_ag_ui(conversation_id: UUID, request: Request, session: AsyncSession = Depends(get_session)) -> StreamingResponse:
    """The only browser chat transport: standard AG-UI over one POST stream."""
    payload = await request.json()
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    user = next((item for item in reversed(messages) if isinstance(item, dict) and item.get("role") == "user"), None)
    content = user.get("content") if isinstance(user, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=422, detail="AG-UI request must end with a user text message")
    turn = await _create_turn(conversation_id, content, [], session)
    start_turn_background(turn.id, get_settings())

    async def stream() -> object:
        async for event in event_broker.subscribe(turn.id):
            # AG-UI's HTTP transport parses JSON data frames; event names are
            # intentionally not duplicated as SSE event fields.
            yield f"data: {event.json()}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.get("/api/turns/{turn_id}/events")
async def stream_events(turn_id: UUID, session: AsyncSession = Depends(get_session)) -> StreamingResponse:
    if await session.scalar(select(AgentTurn.id).where(AgentTurn.id == turn_id)) is None:
        raise HTTPException(status_code=404, detail="Agent turn not found")

    async def stream() -> object:
        async for event in event_broker.subscribe(turn_id):
            yield event.sse()
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.post("/api/turns/{turn_id}/cancel", response_model=RunOut)
async def cancel_turn(turn_id: UUID, session: AsyncSession = Depends(get_session)) -> RunOut:
    turn = await session.scalar(select(AgentTurn).where(AgentTurn.id == turn_id))
    if turn is None:
        raise HTTPException(status_code=404, detail="Agent turn not found")
    if turn.status == TurnStatus.RUNNING:
        cancellation_registry.cancel(turn_id)
    return RunOut(turn_id=turn.id, status=turn.status, events_url=f"/api/turns/{turn.id}/events")
