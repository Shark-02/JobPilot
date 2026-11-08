from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context import AgentContext, ContextBuilder, ContextWindowError
from app.agent.context_tools import INTERNAL_CONTEXT_TOOLS, WorkingContextTools, context_tool_error
from app.agent.events import cancellation_registry, event_broker
from app.agent.executor import ToolExecutionState, ToolExecutor
from app.agent.model import ModelCompleted, ModelRequest, TextDelta, ToolCall, ToolResult
from app.agent.model_client import ConfigurationError, OpenAICompatibleModelClient
from app.agent.policy import LoopPolicy, PolicyError, TurnRuntimeState
from app.agent.registry import build_default_registry
from app.agent.trace import TraceHandle, TraceRecorder
from app.config import Settings
from app.db.models import AgentTurn, Message, MessageRole, TurnStatus
from app.evals.scope import apply_eval_scope, scoped_values
from app.evals.evidence import get_observation_collector
from app.infra.db import session_factory
from app.infra.provider_errors import normalize_provider_error
from app.infra.token_count import TokenCounter

logger = logging.getLogger(__name__)


class TurnCancelled(Exception):
    pass


@dataclass
class _ContextExecution:
    call: ToolCall
    sequence: int
    result: ToolResult
    trace: TraceHandle | None
    started: float


class ThinAgentLoop:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.counter = TokenCounter(settings.llm_model)
        self.policy = LoopPolicy(settings)
        self.trace = TraceRecorder(session, settings)
        self.registry = build_default_registry(session, settings)
        self.executor = ToolExecutor(self.registry, self.policy, self.trace, settings)
        self.model_client = OpenAICompatibleModelClient(settings)

    async def run(self, turn_id: UUID) -> None:
        cancel_event = cancellation_registry.event(turn_id)
        turn = await self.session.scalar(
            apply_eval_scope(select(AgentTurn).where(AgentTurn.id == turn_id), AgentTurn)
        )
        if turn is None:
            await event_broker.finish(turn_id)
            return
        state = TurnRuntimeState()
        tools = self.registry.schemas()
        tool_state = ToolExecutionState()
        text_block_open = False
        try:
            context = await ContextBuilder(self.session, self.settings, self.counter).build(turn_id)
            await event_broker.publish(turn_id, "RUN_STARTED", threadId=str(turn.conversation_id), runId=str(turn_id))
            while True:
                self._check_cancelled(cancel_event)
                self.policy.check_model_budget(state)
                messages, component_tokens = await self._prepare_context_messages(
                    context, turn_id, state, cancel_event
                )
                request = ModelRequest(
                    model=self.settings.llm_model or "", messages=messages, tools=tools,
                    max_tokens=self.settings.llm_max_output_tokens, temperature=self.settings.llm_temperature,
                )
                # Each provider call has one stream message identity.  The
                # terminal call persists that exact identity for history.
                stream_message_id = str(uuid4())
                started = perf_counter()
                trace = await self.trace.model_started(turn_id, state.model_turns + 1, request, component_tokens)
                response = None
                await event_broker.publish(turn_id, "TEXT_MESSAGE_START", messageId=stream_message_id, role="assistant")
                text_block_open = True
                try:
                    async with asyncio.timeout(
                            self.policy.remaining_seconds(state, self.settings.agent_model_timeout_seconds)):
                        async for event in self.model_client.stream(request):
                            self._check_cancelled(cancel_event)
                            if isinstance(event, TextDelta):
                                await event_broker.publish(turn_id, "TEXT_MESSAGE_CONTENT", messageId=stream_message_id,
                                                           delta=event.text)
                            elif isinstance(event, ModelCompleted):
                                response = event.response
                    if response is None:
                        raise RuntimeError("Model provider ended without a response")
                except Exception as exc:
                    await self.trace.model_failed(trace, self._error_code(exc), int((perf_counter() - started) * 1000))
                    raise
                await self.trace.model_finished(trace, response, int((perf_counter() - started) * 1000))
                state.model_turns += 1
                turn.model_turn_count = state.model_turns
                await self.session.commit()
                context.append_assistant(response.assistant_message)
                if not response.tool_calls:
                    assistant = Message(
                        id=UUID(stream_message_id),
                        conversation_id=turn.conversation_id, role=MessageRole.ASSISTANT,
                        content=response.final_text, metadata_={"agent_turn_id": str(turn_id)},
                        **scoped_values(),
                    )
                    self.session.add(assistant)
                    await self.session.flush()
                    turn.assistant_message_id = assistant.id
                    turn.status = TurnStatus.COMPLETED
                    turn.finished_at = datetime.now(timezone.utc)
                    await self.session.commit()
                    await event_broker.publish(turn_id, "TEXT_MESSAGE_END", messageId=str(assistant.id))
                    text_block_open = False
                    await event_broker.publish(turn_id, "RUN_FINISHED", runId=str(turn_id), outcome={"type": "success"})
                    return
                await event_broker.publish(turn_id, "TEXT_MESSAGE_END", messageId=stream_message_id)
                text_block_open = False
                context = await self._execute_tools(
                    turn_id, response.tool_calls, stream_message_id, context, tool_state, state, cancel_event,
                )
                turn.tool_execution_count = state.tool_calls
        except TurnCancelled:
            if text_block_open:
                await event_broker.publish(turn_id, "TEXT_MESSAGE_END")
            await self._safe_finish_error(turn_id, TurnStatus.CANCELLED, "CANCELLED", state)
            await event_broker.publish(turn_id, "RUN_ERROR", code="CANCELLED", message="Run cancelled")
        except (PolicyError, ContextWindowError, ConfigurationError, asyncio.TimeoutError) as exc:
            if text_block_open:
                await event_broker.publish(turn_id, "TEXT_MESSAGE_END")
            await self._safe_finish_error(turn_id, TurnStatus.FAILED, self._error_code(exc), state)
            await event_broker.publish(turn_id, "RUN_ERROR", code=self._error_code(exc), message=str(exc))
        except Exception as exc:
            logger.exception("Agent run failed", extra={"turn_id": str(turn_id)})
            if text_block_open:
                await event_broker.publish(turn_id, "TEXT_MESSAGE_END")
            error_code = self._error_code(exc)
            await self._safe_finish_error(turn_id, TurnStatus.FAILED, error_code, state)
            await event_broker.publish(turn_id, "RUN_ERROR", code=error_code, message="Agent run failed")
        finally:
            await event_broker.finish(turn_id)
            cancellation_registry.remove(turn_id)

    async def _execute_tools(
        self, turn_id: UUID, calls: list[ToolCall], message_id: str,
        context: AgentContext, tool_state: ToolExecutionState,
        state: TurnRuntimeState, cancel_event: asyncio.Event,
    ) -> AgentContext:
        """Stage internal tools until the complete native group can be admitted.

        Business tools run once in their original order. Only internal reads may
        have their provisional payload replaced by an explicit capacity error.
        """
        has_internal = any(call.name in INTERNAL_CONTEXT_TOOLS for call in calls)
        staged = context.fork() if has_internal else context
        context_tools = WorkingContextTools(self.session)
        entries: list[tuple[ToolCall, ToolResult]] = []
        internal: list[_ContextExecution] = []
        try:
            for call in calls:
                self._check_cancelled(cancel_event)
                self.policy.check_tool_budget(state)
                state.tool_calls += 1
                if call.name in INTERNAL_CONTEXT_TOOLS:
                    started = perf_counter()
                    handle = await self.trace.tool_started(
                        turn_id, state.tool_calls, call.id, call.name, call.arguments,
                        self.policy.fingerprint(call.name, call.arguments),
                    )
                    execution = _ContextExecution(
                        call, state.tool_calls, context_tool_error("WORKING_CONTEXT_BATCH_INCOMPLETE"), handle, started,
                    )
                    internal.append(execution)
                    try:
                        async with asyncio.timeout(self.policy.remaining_seconds(
                            state, self.settings.agent_tool_timeout_seconds,
                        )):
                            result = await context_tools.execute(call, staged)
                    except asyncio.TimeoutError:
                        result = context_tool_error(
                            "WORKING_CONTEXT_NOT_RECOVERABLE" if call.name == "read_working_context" else "WORKING_CONTEXT_TIMEOUT",
                            code="TIMEOUT",
                        )
                    execution.result = result
                else:
                    await event_broker.publish(
                        turn_id, "TOOL_CALL_START", toolCallId=call.id,
                        toolCallName=call.name, parentMessageId=message_id,
                    )
                    await event_broker.publish(
                        turn_id, "TOOL_CALL_ARGS", toolCallId=call.id,
                        delta=json.dumps(self._safe_arguments(call.arguments), ensure_ascii=False),
                    )
                    result = await self.executor.execute(turn_id, state.tool_calls, call, tool_state, state)
                    await event_broker.publish(turn_id, "TOOL_CALL_END", toolCallId=call.id)
                    await event_broker.publish(
                        turn_id, "TOOL_CALL_RESULT", messageId=f"{turn_id}:{call.id}:result", toolCallId=call.id,
                        role="tool", content=json.dumps({"summary": result.ui_summary,
                                                         "result": result.provider_payload()}, ensure_ascii=False),
                    )
                entries.append((call, result))
                staged.append_tool_result(call.id, call.name, result)
                self.policy.check_deadline(state)

            reads = [entry for entry in internal if entry.call.name == "read_working_context" and entry.result.ok]
            if reads:
                try:
                    # Includes every ordinary tool result, post-checkpoint
                    # state, JSON envelopes, schemas and the newest raw group.
                    staged.prepare_messages(self.registry.schemas())
                except ContextWindowError:
                    _, tokens = staged.build_messages(self.registry.schemas())
                    replacements = {
                        entry.call.id: context_tool_error(
                            "WORKING_CONTEXT_CAPACITY_EXCEEDED",
                            refs=entry.call.arguments["refs"], required_tokens=tokens["total"],
                            available_tokens=self.settings.hard_input_limit,
                        ) for entry in reads
                    }
                    admitted = context.fork()
                    # Optional successful reconstruction only repairs an
                    # already-known backing entry; it never creates new refs.
                    for ref in context.known_refs():
                        if context.get_interaction(ref) is None:
                            recovered = staged.get_interaction(ref)
                            if recovered is not None:
                                admitted.register_recovered(ref, recovered)
                            elif ref in context.active_refs and ref not in staged.active_refs:
                                admitted.mark_not_recoverable(ref)
                    for call, result in entries:
                        result = replacements.get(call.id, result)
                        if call.name == "checkpoint_task_state" and result.ok:
                            admitted.apply_checkpoint(call.arguments)
                        admitted.append_tool_result(call.id, call.name, result)
                    for entry in reads:
                        entry.result = replacements[entry.call.id]
                    staged = admitted
                    # The next real model call still fails closed if the error
                    # envelopes and ordinary evidence themselves cannot fit.
        except BaseException:
            # Preserve the actual tool-execution ledger even if a later tool
            # hits a deadline/budget or cancellation before the group completes.
            for entry in internal:
                if entry.result.ok:
                    entry.result = context_tool_error("WORKING_CONTEXT_BATCH_INCOMPLETE")
            await self._finish_context_tools(turn_id, internal)
            raise
        await self._finish_context_tools(turn_id, internal)
        return staged

    async def _finish_context_tools(self, turn_id: UUID, entries: list[_ContextExecution]) -> None:
        for entry in entries:
            collector = get_observation_collector()
            if collector is not None:
                collector.record_tool(
                    turn_id=turn_id, sequence=entry.sequence, call_id=entry.call.id,
                    tool_name=entry.call.name, arguments=entry.call.arguments,
                    result=entry.result.provider_payload(), execution_kind="real",
                    fingerprint=self.policy.fingerprint(entry.call.name, entry.call.arguments),
                )
            await self.trace.tool_finished(
                entry.trace, entry.result, int((perf_counter() - entry.started) * 1000),
            )
            if entry.call.name == "read_working_context":
                await self.trace.context_event(turn_id, "page_in", {
                    "refs": entry.call.arguments.get("refs", []), "ok": entry.result.ok,
                    "tokens_restored": self.counter.count(json.dumps(entry.result.agent_payload, ensure_ascii=False))
                    if entry.result.ok else 0,
                    "reason": None if entry.result.ok else (entry.result.agent_payload or {}).get("reason"),
                })
            else:
                await self.trace.context_event(turn_id, "checkpoint", {"ok": entry.result.ok})

    async def _prepare_context_messages(
        self,
        context: AgentContext,
        turn_id: UUID,
        state: TurnRuntimeState,
        cancel_event: asyncio.Event,
    ) -> tuple[list[dict], dict[str, int]]:
        """Evict complete old interactions deterministically; never call an LLM."""
        self._check_cancelled(cancel_event)
        self.policy.check_deadline(state)
        messages, component_tokens = context.prepare_messages(self.registry.schemas())
        for event in context.drain_events():
            await self.trace.context_event(turn_id, event["type"], event)
        return messages, component_tokens

    async def _safe_finish_error(self, turn_id: UUID, status: TurnStatus, error_code: str,
                                 state: TurnRuntimeState | None = None) -> None:
        """Persist terminal state without allowing a secondary DB error to mask
        the original Agent failure.
        """
        try:
            await self._finish_error(turn_id, status, error_code, state)
        except Exception:
            logger.exception(
                "Failed to persist terminal agent-turn state",
                extra={"turn_id": str(turn_id), "error_code": error_code},
            )
            try:
                await self.session.rollback()
            except Exception:
                logger.exception(
                    "Failed to rollback after terminal-state persistence error",
                    extra={"turn_id": str(turn_id)},
                )

    async def _finish_error(self, turn_id: UUID, status: TurnStatus, error_code: str,
                            state: TurnRuntimeState | None = None) -> None:
        # Any database error earlier in the agent loop may leave the AsyncSession
        # in a failed transaction and may expire ORM instances.  Always reset the
        # transaction first, then re-load by the plain UUID instead of touching
        # attributes on a possibly-expired AgentTurn object.
        await self.session.rollback()

        fresh = await self.session.scalar(
            apply_eval_scope(select(AgentTurn).where(AgentTurn.id == turn_id), AgentTurn)
        )
        if fresh is None:
            return

        fresh.status = status
        fresh.error_code = error_code
        fresh.finished_at = datetime.now(timezone.utc)
        # Tool results are observed outside this transaction. Rollback must not
        # erase their counters and make complete capture appear incomplete.
        if state is not None:
            fresh.model_turn_count = state.model_turns
            fresh.tool_execution_count = state.tool_calls
        await self.session.commit()

    @staticmethod
    def _safe_arguments(arguments: dict) -> dict:
        encoded = json.dumps(arguments, ensure_ascii=False, default=str)
        if len(encoded) <= 4096:
            return arguments
        return {"truncated": True, "preview": encoded[:4096]}

    @staticmethod
    def _check_cancelled(event: asyncio.Event) -> None:
        if event.is_set():
            raise TurnCancelled()

    @staticmethod
    def _error_code(error: Exception) -> str:
        if isinstance(error, PolicyError):
            return error.code
        if isinstance(error, ContextWindowError):
            return "CONTEXT_OVERFLOW"
        if isinstance(error, ConfigurationError):
            return "PROVIDER_NOT_CONFIGURED"
        if isinstance(error, TurnCancelled):
            return "CANCELLED"
        provider_error = normalize_provider_error(error, timeout_code="MODEL_TIMEOUT")
        if provider_error:
            return provider_error
        return "INTERNAL_ERROR"


_running_tasks: dict[UUID, asyncio.Task[None]] = {}


def start_turn_background(turn_id: UUID, settings: Settings) -> None:
    async def runner() -> None:
        try:
            async with session_factory() as session:
                try:
                    await ThinAgentLoop(session, settings).run(turn_id)
                except Exception:
                    # Constructor/startup failures happen before ThinAgentLoop
                    # can enter its own error boundary. A failed flush/commit may
                    # leave the session unusable until rollback, so reset it first.
                    logger.exception("Agent runtime failed before execution", extra={"turn_id": str(turn_id)})
                    try:
                        await session.rollback()
                        turn = await session.scalar(select(AgentTurn).where(AgentTurn.id == turn_id))
                        if turn and turn.status == TurnStatus.RUNNING:
                            turn.status = TurnStatus.FAILED
                            turn.error_code = "RUNTIME_START_FAILED"
                            turn.finished_at = datetime.now(timezone.utc)
                            await session.commit()
                    except Exception:
                        # Error reporting must not crash the background task.
                        logger.exception(
                            "Failed to persist runtime-start failure",
                            extra={"turn_id": str(turn_id)},
                        )
                        try:
                            await session.rollback()
                        except Exception:
                            logger.exception(
                                "Failed to rollback runtime-start recovery transaction",
                                extra={"turn_id": str(turn_id)},
                            )

                    try:
                        await event_broker.publish(
                            turn_id,
                            "RUN_ERROR",
                            code="RUNTIME_START_FAILED",
                            message="Agent runtime failed before execution",
                        )
                    except Exception:
                        logger.exception(
                            "Failed to publish runtime-start error",
                            extra={"turn_id": str(turn_id)},
                        )
        except Exception:
            # Never let a detached asyncio task finish with an un-retrieved
            # exception; log it here and let the task terminate normally.
            logger.exception("Unhandled background agent task failure", extra={"turn_id": str(turn_id)})
        finally:
            # ThinAgentLoop closes this on its normal path. Keeping the wrapper
            # idempotent also closes the browser stream when setup fails before
            # ThinAgentLoop.run() installs its own finally block.
            try:
                await event_broker.finish(turn_id)
            except Exception:
                logger.exception("Failed to finish event stream", extra={"turn_id": str(turn_id)})
            cancellation_registry.remove(turn_id)
            _running_tasks.pop(turn_id, None)

    _running_tasks[turn_id] = asyncio.create_task(runner(), name=f"agent-turn-{turn_id}")
