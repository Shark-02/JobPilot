from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, update

from app.agent.trace import TraceRecorder
from app.api import agent, conversations, debug, documents, memories
from app.config import get_settings
from app.db.models import AgentTurn, TurnStatus
from app.infra.db import SessionLocal


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    async with SessionLocal() as session:
        await session.execute(
            update(AgentTurn)
            .where(AgentTurn.status == TurnStatus.RUNNING)
            .values(status=TurnStatus.ABORTED, error_code="PROCESS_RESTART", finished_at=func.now())
        )
        await session.commit()
        if settings.trace_enabled:
            await TraceRecorder(session, settings).cleanup_expired()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="JobPilot API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(conversations.router)
    app.include_router(agent.router)
    app.include_router(documents.router)
    app.include_router(memories.router)
    app.include_router(debug.router)
    app.include_router(debug.observability_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
