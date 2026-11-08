from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

settings = get_settings()
engine = create_async_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


def configure_database(database_url: str) -> None:
    """Point runtime-owned short sessions at an isolated database.

    This is used only by the offline evaluator before it invokes the real
    agent runtime.  The API process keeps using its configured production URL.
    """

    global engine, SessionLocal
    engine = create_async_engine(database_url, pool_pre_ping=True)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


def session_factory() -> AsyncSession:
    """Open one session from the currently configured runtime factory."""
    from app.evals.scope import get_eval_scope
    if get_eval_scope() is not None:
        from app.evals.database import assert_canonical_eval_database
        assert_canonical_eval_database(engine.url, settings.database_url)
    return SessionLocal()


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
