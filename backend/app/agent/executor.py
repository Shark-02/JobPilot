from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter
from uuid import UUID

from pydantic import ValidationError

from app.agent.model import ToolCall, ToolError, ToolResult
from app.agent.policy import LoopPolicy, TurnRuntimeState
from app.agent.registry import ToolRegistry
from app.agent.trace import TraceRecorder
from app.config import Settings
from app.evals.faults import injected_result, mark_real_result
from app.evals.evidence import get_observation_collector
from app.infra.provider_errors import normalize_provider_error
from app.memory.service import PossibleDuplicateError


@dataclass
class ToolExecutionState:
    fingerprints: Counter[str] = field(default_factory=Counter)
    cache: dict[str, ToolResult] = field(default_factory=dict)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, policy: LoopPolicy, trace: TraceRecorder, settings: Settings) -> None:
        self.registry = registry
        self.policy = policy
        self.trace = trace
        self.settings = settings

    async def execute(
        self, turn_id: UUID, sequence: int, call: ToolCall, state: ToolExecutionState, runtime_state: TurnRuntimeState
    ) -> ToolResult:
        fingerprint = self.policy.fingerprint(call.name, call.arguments)
        started = perf_counter()
        trace = await self.trace.tool_started(turn_id, sequence, call.id, call.name, call.arguments, fingerprint)
        def capture(result: ToolResult, execution_kind: str, provider_error_code: str | None = None) -> None:
            collector = get_observation_collector()
            if collector is not None:
                collector.record_tool(turn_id=turn_id, sequence=sequence, call_id=call.id, tool_name=call.name,
                                      arguments=call.arguments, result=result.provider_payload(),
                                      execution_kind=execution_kind, fingerprint=fingerprint,
                                      provider_error_code=provider_error_code)
        state.fingerprints[fingerprint] += 1
        if state.fingerprints[fingerprint] == 2:
            cached = state.cache.get(fingerprint)
            if cached is not None:
                result = cached.model_copy(update={"ui_summary": f"Reused cached result: {cached.ui_summary}"})
                capture(result, "cache")
                await self.trace.tool_finished(trace, result, int((perf_counter() - started) * 1000))
                return result
        if state.fingerprints[fingerprint] > 2:
            result = ToolResult(
                ok=False, ui_summary="Stopped duplicate tool call",
                error=ToolError(code="POSSIBLE_DUPLICATE", message="No progress: this tool call already ran twice in this turn", retryable=False),
            )
            await self.trace.tool_finished(trace, result, int((perf_counter() - started) * 1000))
            capture(result, "duplicate_guard")
            return result
        injected = injected_result(call.name, turn_id=turn_id, call_id=call.id)
        if injected is not None:
            capture(injected, "injected")
            await self.trace.tool_finished(trace, injected, int((perf_counter() - started) * 1000))
            return injected
        definition = self.registry.get(call.name)
        if definition is None:
            result = ToolResult(
                ok=False, ui_summary="Unknown tool",
                error=ToolError(code="VALIDATION_ERROR", message=f"Unknown tool: {call.name}", retryable=True),
            )
            await self.trace.tool_finished(trace, result, int((perf_counter() - started) * 1000))
            capture(result, "validation")
            return result
        provider_error_code = None
        tool_timeout = None
        try:
            args = definition.args_model.model_validate(call.arguments)
            tool_timeout = asyncio.timeout(self.policy.remaining_seconds(runtime_state, self.settings.agent_tool_timeout_seconds))
            async with tool_timeout:
                result = await definition.executor(args)
        except ValidationError as exc:
            result = ToolResult(ok=False, ui_summary="Invalid tool arguments", error=ToolError(
                code="VALIDATION_ERROR", message=exc.errors(include_url=False).__str__()[:1000], retryable=True
            ))
        except PossibleDuplicateError as exc:
            result = ToolResult(ok=False, ui_summary="Possible duplicate memory", error=ToolError(
                code="POSSIBLE_DUPLICATE", message=str(exc), retryable=True
            ))
        except LookupError as exc:
            result = ToolResult(ok=False, ui_summary="Source not found", error=ToolError(
                code="NOT_FOUND", message=str(exc), retryable=True
            ))
        except asyncio.TimeoutError as exc:
            # A provider timeout raised inside the tool invalidates the Eval
            # opportunity. Expiry of our own tool budget remains a task failure.
            if tool_timeout is None or not tool_timeout.expired():
                provider_error_code = normalize_provider_error(exc, timeout_code="MODEL_TIMEOUT")
            result = ToolResult(ok=False, ui_summary="Tool timed out", error=ToolError(
                code="TIMEOUT", message="The tool timed out", retryable=True
            ))
        except ValueError as exc:
            result = ToolResult(ok=False, ui_summary="Tool request rejected", error=ToolError(
                code="DOMAIN_ERROR", message=str(exc), retryable=True
            ))
        except Exception as exc:
            provider_error_code = normalize_provider_error(exc, timeout_code="MODEL_TIMEOUT")
            result = ToolResult(ok=False, ui_summary="Tool failed", error=ToolError(
                code="INTERNAL_ERROR", message="The tool could not complete", retryable=True
            ))
        result = result.model_copy(update={"ui_summary": result.ui_summary[:2048]})
        mark_real_result(call.name, result, turn_id=turn_id, call_id=call.id)
        capture(result, "real", provider_error_code)
        if result.observation_payload:
            await self.trace.retrieval_finished(trace, result.observation_payload, int((perf_counter() - started) * 1000))
        await self.trace.tool_finished(trace, result, int((perf_counter() - started) * 1000))
        if result.ok:
            state.cache[fingerprint] = result
        return result
