from __future__ import annotations

import logging
import re
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context import AgentContext
from app.agent.context_recovery import recover_interaction
from app.agent.model import CheckpointTaskStateArgs, ReadWorkingContextArgs, ToolCall, ToolError, ToolResult
from app.infra.db import session_factory
from app.knowledge.sources import SourceRef, SourceResolver


INTERNAL_CONTEXT_TOOLS = frozenset({"read_working_context", "checkpoint_task_state"})
logger = logging.getLogger(__name__)


def context_tool_error(reason: str, *, code: str = "DOMAIN_ERROR", **details: object) -> ToolResult:
    return ToolResult(
        ok=False, agent_payload={"reason": reason, **details},
        error=ToolError(code=code, message=reason, retryable=False),  # type: ignore[arg-type]
    )


class WorkingContextTools:
    """Turn-local tools. Their results are admitted as one complete tool batch."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def execute(self, call: ToolCall, context: AgentContext) -> ToolResult:
        try:
            if call.name == "checkpoint_task_state":
                args = CheckpointTaskStateArgs.model_validate(call.arguments)
                resolver = SourceResolver(self.session, context.counter)
                for entries in args.model_dump().values():
                    for entry in entries:
                        for ref in entry["refs"]:
                            if not ref.startswith("ctx://"):
                                await resolver.validate(SourceRef(ref))
                context.apply_checkpoint(args.model_dump())
                return ToolResult(ok=True, agent_payload={"checkpoint_applied": True})
            args = ReadWorkingContextArgs.model_validate(call.arguments)
            return await self._read(args, context)
        except (ValidationError, ValueError) as exc:
            # Schema/ref/budget rejection is a recoverable tool result, not a
            # reason to misclassify a malformed delta as a provider failure.
            return context_tool_error("WORKING_CONTEXT_VALIDATION_ERROR", code="VALIDATION_ERROR", detail=str(exc))
        except LookupError:
            return context_tool_error("WORKING_CONTEXT_NOT_FOUND", code="NOT_FOUND")

    async def _read(self, args: ReadWorkingContextArgs, context: AgentContext) -> ToolResult:
        # Validate the entire request before attempting any best-effort reads.
        known = set(context.known_refs())
        for ref in args.refs:
            match = re.fullmatch(r"ctx://([0-9a-f-]{36})/i/([1-9][0-9]*)", ref)
            if not match or str(UUID(match[1])) != match[1]:
                return context_tool_error("WORKING_CONTEXT_INVALID_REF", code="VALIDATION_ERROR")
            if UUID(match[1]) != context.turn_id:
                return context_tool_error("WORKING_CONTEXT_TURN_MISMATCH", code="VALIDATION_ERROR")
            if ref not in known:
                return context_tool_error("WORKING_CONTEXT_NOT_FOUND", code="NOT_FOUND", refs=[ref])

        interactions = []
        unrecoverable_refs: list[str] = []
        source_refs: dict[str, SourceRef] = {}
        for ref in args.refs:
            interaction = context.get_interaction(ref)
            if interaction is None:
                # An observability read must never poison the business Session
                # or become a new correctness-critical persistence dependency.
                try:
                    async with session_factory() as recovery_session:
                        interaction = await recover_interaction(recovery_session, context.turn_id, ref)
                except Exception as exc:
                    logger.warning("Working context fallback unavailable", extra={
                        "turn_id": str(context.turn_id), "error_type": type(exc).__name__,
                    })
                    interaction = None
                if interaction is None:
                    context.mark_not_recoverable(ref)
                    unrecoverable_refs.append(ref)
                    continue
                try:
                    context.register_recovered(ref, interaction)
                except ValueError:
                    context.mark_not_recoverable(ref)
                    unrecoverable_refs.append(ref)
                    continue
            interactions.append({
                "ref": ref,
                "assistant_message": interaction.assistant,
                "tool_messages": list(interaction.tool_messages),
                "source_refs": list(interaction.source_refs),
            })
            for source in interaction.source_refs:
                source_refs[source] = SourceRef(source)
        if unrecoverable_refs:
            return context_tool_error("WORKING_CONTEXT_NOT_RECOVERABLE", refs=unrecoverable_refs)
        return ToolResult(
            ok=True, agent_payload={"interactions": interactions},
            source_refs=list(source_refs.values()),
        )
