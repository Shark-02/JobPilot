from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class EvalScope:
    """The only mutable state visible to one evaluation case.

    A scope is intentionally process-local.  It is set by the offline runner
    before it enters the real runtime, and is inherited by the short-lived
    observability tasks the runtime creates.
    """

    run_id: UUID
    case_id: str
    corpus_document_ids: frozenset[UUID] = frozenset()
    embedding_profile_id: UUID | None = None


_scope: ContextVar[EvalScope | None] = ContextVar("jobpilot_eval_scope", default=None)


def get_eval_scope() -> EvalScope | None:
    return _scope.get()


def set_eval_scope(scope: EvalScope | None) -> Token[EvalScope | None]:
    return _scope.set(scope)


def reset_eval_scope(token: Token[EvalScope | None]) -> None:
    _scope.reset(token)


def scoped_values() -> dict[str, Any]:
    scope = get_eval_scope()
    if scope is None:
        return {}
    return {"eval_run_id": scope.run_id, "eval_case_id": scope.case_id}


def apply_eval_scope(statement: Any, model: Any, *, include_retracted: bool = False) -> Any:
    """Add the centralized case predicate to a SQLAlchemy statement.

    Production queries exclude all evaluation rows. Eval queries see only rows
    belonging to the current run/case, and mutable rows additionally exclude
    logical retractions by default. This default-deny behaviour prevents a
    future repository method from accidentally surfacing eval state in product
    memory or conversations.
    """

    scope = get_eval_scope()
    if scope is None:
        return statement.where(model.eval_run_id.is_(None))
    statement = statement.where(
        model.eval_run_id == scope.run_id,
        model.eval_case_id == scope.case_id,
    )
    if hasattr(model, "retracted_at") and not include_retracted:
        statement = statement.where(model.retracted_at.is_(None))
    return statement


async def retract_current_case_state(session: AsyncSession) -> None:
    """Logically withdraw mutable eval state without deleting audit traces."""

    scope = get_eval_scope()
    if scope is None:
        return
    # Import locally to keep the generic scope module free of model import
    # cycles during application startup.
    from app.db.models import Conversation, MemoryItem, Message

    at = datetime.now(timezone.utc)
    for model in (Conversation, Message, MemoryItem):
        await session.execute(
            update(model)
            .where(model.eval_run_id == scope.run_id, model.eval_case_id == scope.case_id)
            .values(retracted_at=at)
        )
    await session.commit()
