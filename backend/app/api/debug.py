from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.trace import TraceRecorder
from app.api.schemas import AgentTurnOut, ModelCallOut, ToolCallOut
from app.config import Settings, get_settings
from app.db.models import AgentObservation, AgentTurn, Message, ModelCallTrace, ToolCallTrace
from app.infra.db import get_session

router = APIRouter(prefix="/api/debug", tags=["debug"])
observability_router = APIRouter(prefix="/api/observability", tags=["observability"])


def require_debug(settings: Settings = Depends(get_settings)) -> Settings:
    if settings.app_env != "development" or not settings.model_egress_preview_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    return settings


def require_observability(settings: Settings = Depends(get_settings)) -> Settings:
    if not settings.observability_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    return settings


def observation_payload(row: AgentObservation) -> dict:
    return {"id": str(row.id), "turn_id": str(row.turn_id), "parent_observation_id": str(row.parent_observation_id) if row.parent_observation_id else None,
        "sequence": row.sequence, "kind": row.kind, "name": row.name, "status": row.status,
        "input_payload": row.input_payload, "output_payload": row.output_payload, "metadata": row.metadata_, "source_refs": row.source_refs,
        "started_at": row.started_at, "finished_at": row.finished_at, "latency_ms": row.latency_ms, "created_at": row.created_at}


def duration_ms(turn: AgentTurn) -> int:
    """Use wall-clock duration for a live turn; completed turns stay immutable."""
    end = turn.finished_at or datetime.now(timezone.utc)
    return max(0, int((end - turn.started_at).total_seconds() * 1000))


@observability_router.get("/turns")
async def list_observability_turns(limit: int = 50, session: AsyncSession = Depends(get_session), _: Settings = Depends(require_observability)) -> list[dict]:
    turns = list((await session.scalars(select(AgentTurn).order_by(AgentTurn.started_at.desc()).limit(max(1, min(limit, 200))))).all())
    output: list[dict] = []
    for turn in turns:
        message = await session.scalar(select(Message.content).where(Message.id == turn.user_message_id))
        model_calls = await session.scalar(select(func.count()).select_from(ModelCallTrace).where(
            ModelCallTrace.turn_id == turn.id, ModelCallTrace.call_type == "MAIN_AGENT"
        ))
        compaction_calls = await session.scalar(select(func.count()).select_from(ModelCallTrace).where(
            ModelCallTrace.turn_id == turn.id, ModelCallTrace.call_type == "CONTEXT_COMPACTION"
        ))
        tool_calls = await session.scalar(select(func.count()).select_from(ToolCallTrace).where(ToolCallTrace.turn_id == turn.id))
        output.append({"turn_id": str(turn.id), "conversation_id": str(turn.conversation_id), "user_message_preview": (message or "")[:240],
            "status": turn.status.value, "model_calls": model_calls or 0,
            "context_compaction_calls": compaction_calls or 0, "tool_calls": tool_calls or 0,
            "duration_ms": duration_ms(turn),
            "input_tokens": await session.scalar(select(func.coalesce(func.sum(ModelCallTrace.input_tokens), 0)).where(ModelCallTrace.turn_id == turn.id)),
            "output_tokens": await session.scalar(select(func.coalesce(func.sum(ModelCallTrace.output_tokens), 0)).where(ModelCallTrace.turn_id == turn.id)), "started_at": turn.started_at})
    return output


@observability_router.get("/turns/{turn_id}")
async def observability_turn(turn_id: UUID, session: AsyncSession = Depends(get_session), _: Settings = Depends(require_observability)) -> dict:
    turn = await session.scalar(select(AgentTurn).where(AgentTurn.id == turn_id))
    if turn is None:
        raise HTTPException(status_code=404, detail="Turn not found")
    input_tokens = await session.scalar(select(func.coalesce(func.sum(ModelCallTrace.input_tokens), 0)).where(ModelCallTrace.turn_id == turn.id))
    output_tokens = await session.scalar(select(func.coalesce(func.sum(ModelCallTrace.output_tokens), 0)).where(ModelCallTrace.turn_id == turn.id))
    compaction_calls = await session.scalar(select(func.count()).select_from(ModelCallTrace).where(
        ModelCallTrace.turn_id == turn.id, ModelCallTrace.call_type == "CONTEXT_COMPACTION"
    ))
    return {"turn_id": str(turn.id), "conversation_id": str(turn.conversation_id), "user_message_id": str(turn.user_message_id),
        "user_message": await session.scalar(select(Message.content).where(Message.id == turn.user_message_id)),
        "assistant_message_id": str(turn.assistant_message_id) if turn.assistant_message_id else None, "status": turn.status.value,
        "model_calls": turn.model_turn_count, "context_compaction_calls": compaction_calls or 0,
        "tool_calls": turn.tool_execution_count, "error_code": turn.error_code,
        "started_at": turn.started_at, "finished_at": turn.finished_at,
        "duration_ms": duration_ms(turn),
        "input_tokens": input_tokens or 0, "output_tokens": output_tokens or 0, "total_tokens": (input_tokens or 0) + (output_tokens or 0)}


@observability_router.get("/turns/{turn_id}/observations")
async def turn_observations(turn_id: UUID, session: AsyncSession = Depends(get_session), _: Settings = Depends(require_observability)) -> list[dict]:
    return [observation_payload(row) for row in (await session.scalars(select(AgentObservation).where(AgentObservation.turn_id == turn_id).order_by(AgentObservation.sequence))).all()]


@observability_router.get("/observations/{observation_id}")
async def observation(observation_id: UUID, session: AsyncSession = Depends(get_session), _: Settings = Depends(require_observability)) -> dict:
    row = await session.scalar(select(AgentObservation).where(AgentObservation.id == observation_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return observation_payload(row)


@router.get("/turns/{turn_id}", response_model=AgentTurnOut)
async def get_turn(turn_id: UUID, session: AsyncSession = Depends(get_session), _: Settings = Depends(require_debug)) -> AgentTurnOut:
    item = await session.scalar(select(AgentTurn).where(AgentTurn.id == turn_id))
    if item is None:
        raise HTTPException(status_code=404, detail="Agent turn not found")
    return item


@router.get("/turns/{turn_id}/model-calls", response_model=list[ModelCallOut])
async def model_calls(turn_id: UUID, session: AsyncSession = Depends(get_session), settings: Settings = Depends(require_debug)) -> list[ModelCallOut]:
    return await TraceRecorder(session, settings).turn_model_calls(turn_id)


@router.get("/model-calls/{call_id}/egress")
async def model_egress(call_id: UUID, session: AsyncSession = Depends(get_session), settings: Settings = Depends(require_debug)) -> dict:
    item = await TraceRecorder(session, settings).model_egress(call_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Model call not found")
    return item


@router.get("/turns/{turn_id}/tool-calls", response_model=list[ToolCallOut])
async def tool_calls(turn_id: UUID, session: AsyncSession = Depends(get_session), settings: Settings = Depends(require_debug)) -> list[ToolCallOut]:
    return await TraceRecorder(session, settings).turn_tool_calls(turn_id)
