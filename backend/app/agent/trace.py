from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.model import ModelRequest, ModelResponse, ToolResult
from app.config import Settings
from app.db.models import AgentObservation, ModelCallTrace, ToolCallTrace
from app.evals.scope import scoped_values
from app.evals.evidence import get_observation_collector
from app.infra.db import session_factory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceHandle:
    """IDs only: never retain ORM rows across observability transactions."""

    trace_id: UUID | None
    observation_id: UUID | None
    turn_id: UUID
    name: str | None = None
    metadata: dict | None = None
    egress_preview: dict | None = None


class TraceRecorder:
    """Best-effort Agent observability.

    Writes use their own short AsyncSession/transaction so a trace failure can
    never poison the Agent's business session. Read methods still use the
    caller-provided session because debug APIs are ordinary read paths.
    """

    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self._sequence: dict[UUID, int] = {}

    def _next_observation_sequence(self, turn_id: UUID) -> int:
        sequence = self._sequence.get(turn_id, 0) + 1
        self._sequence[turn_id] = sequence
        return sequence

    async def model_started(
        self,
        turn_id: UUID,
        sequence: int,
        request: ModelRequest,
        component_tokens: dict[str, int],
        *,
        call_type: str = "MAIN_AGENT",
        metadata: dict | None = None,
    ) -> TraceHandle | None:
        metadata = dict(metadata or {})
        if call_type == "MAIN_AGENT":
            # Observation.sequence also counts tool/lifecycle events. Persist
            # the actual interaction identity explicitly for optional recovery.
            metadata.setdefault("interaction_seq", sequence)
            metadata.setdefault("model_sequence", sequence)
        collector = get_observation_collector()
        if collector is not None:
            collector.record_model_input(turn_id, sequence, request.messages, call_type)
        if not self.settings.trace_enabled and not self.settings.observability_enabled:
            return None

        source_refs = sorted(
            set(re.findall(r"(?:message|document)://[^\s\"']+", str(request.messages)))
        )
        preview: dict = {
            "call_type": call_type,
            "model": request.model,
            "input_tokens_estimate": component_tokens["total"],
            "components": component_tokens,
            "source_refs": source_refs,
            "message_count": len(request.messages),
            "tool_schema_names": [item["function"]["name"] for item in request.tools],
        }
        if metadata:
            preview["context_compaction" if call_type == "CONTEXT_COMPACTION" else "context_lifecycle"] = metadata
        if (
            self.settings.app_env == "development"
            and self.settings.model_egress_preview_store_content
        ):
            preview["outgoing_messages"] = request.messages
            preview["tool_schemas"] = request.tools

        trace_id = uuid4() if self.settings.trace_enabled else None
        observation_id = uuid4() if self.settings.observability_enabled else None
        started_at = datetime.now(timezone.utc)

        try:
            async with session_factory() as session:
                if trace_id is not None:
                    session.add(
                        ModelCallTrace(
                            id=trace_id,
                            turn_id=turn_id,
                            sequence=sequence,
                            call_type=call_type,
                            provider=self.settings.llm_base_url,
                            model=request.model,
                            input_tokens=component_tokens["total"],
                            status="RUNNING",
                            egress_preview=preview,
                            started_at=started_at,
                            **scoped_values(),
                        )
                    )

                if observation_id is not None:
                    session.add(
                        AgentObservation(
                            id=observation_id,
                            turn_id=turn_id,
                            sequence=self._next_observation_sequence(turn_id),
                            kind="MODEL",
                            name="context_compaction" if call_type == "CONTEXT_COMPACTION" else request.model,
                            status="RUNNING",
                            metadata_={"call_type": call_type, **(metadata or {})},
                            input_payload=(
                                {
                                    "provider": self.settings.llm_base_url,
                                    "model": request.model,
                                    "messages": request.messages,
                                    "tools": request.tools,
                                    "max_tokens": request.max_tokens,
                                    "temperature": request.temperature,
                                }
                                if self.settings.observability_capture_payload
                                else None
                            ),
                            started_at=started_at,
                            **scoped_values(),
                        )
                    )

                await session.commit()
        except Exception:
            # Observability is strictly best-effort. Never let it alter Agent
            # execution semantics or the caller's transaction state.
            logger.exception(
                "Failed to persist model-start observation",
                extra={"turn_id": str(turn_id), "sequence": sequence},
            )
            return None

        return TraceHandle(
            trace_id,
            observation_id,
            turn_id,
            "context_compaction" if call_type == "CONTEXT_COMPACTION" else request.model,
            {"call_type": call_type, **(metadata or {})},
            preview,
        )

    async def model_finished(
        self,
        handle: TraceHandle | None,
        response: ModelResponse,
        latency_ms: int,
        *,
        metadata: dict | None = None,
    ) -> None:
        if handle is None:
            return

        finished_at = datetime.now(timezone.utc)
        try:
            async with session_factory() as session:
                if handle.trace_id is not None:
                    values: dict = {
                        "status": "COMPLETED",
                        "output_tokens": response.output_tokens,
                        "finish_reason": response.finish_reason,
                        "latency_ms": latency_ms,
                        "finished_at": finished_at,
                    }
                    if response.input_tokens is not None:
                        values["input_tokens"] = response.input_tokens
                    if metadata and handle.egress_preview is not None:
                        preview = {**handle.egress_preview}
                        key = "context_compaction" if (handle.metadata or {}).get("call_type") == "CONTEXT_COMPACTION" else "context_lifecycle"
                        preview[key] = {
                            **preview.get(key, {}),
                            **metadata,
                        }
                        values["egress_preview"] = preview
                    await session.execute(
                        update(ModelCallTrace)
                        .where(ModelCallTrace.id == handle.trace_id)
                        .values(**values)
                    )

                if handle.observation_id is not None:
                    await session.execute(
                        update(AgentObservation)
                        .where(AgentObservation.id == handle.observation_id)
                        .values(
                            status="COMPLETED",
                            output_payload=(
                                {
                                    "assistant_message": response.assistant_message,
                                    "final_text": response.final_text,
                                    "tool_calls": [
                                        item.model_dump(mode="json")
                                        for item in response.tool_calls
                                    ],
                                    "finish_reason": response.finish_reason,
                                    "usage": {
                                        "input_tokens": response.input_tokens,
                                        "output_tokens": response.output_tokens,
                                    },
                                }
                                if self.settings.observability_capture_payload
                                else None
                            ),
                            metadata_={**(handle.metadata or {}), **(metadata or {})},
                            latency_ms=latency_ms,
                            finished_at=finished_at,
                        )
                    )

                await session.commit()
        except Exception:
            logger.exception(
                "Failed to persist model-finished observation",
                extra={"turn_id": str(handle.turn_id)},
            )

    async def model_failed(
        self,
        handle: TraceHandle | None,
        error_code: str,
        latency_ms: int,
    ) -> None:
        if handle is None:
            return

        finished_at = datetime.now(timezone.utc)
        try:
            async with session_factory() as session:
                if handle.trace_id is not None:
                    await session.execute(
                        update(ModelCallTrace)
                        .where(ModelCallTrace.id == handle.trace_id)
                        .values(
                            status="FAILED",
                            error_code=error_code,
                            latency_ms=latency_ms,
                            finished_at=finished_at,
                        )
                    )

                if handle.observation_id is not None:
                    await session.execute(
                        update(AgentObservation)
                        .where(AgentObservation.id == handle.observation_id)
                        .values(
                            status="FAILED",
                            output_payload=(
                                {"error_code": error_code}
                                if self.settings.observability_capture_payload
                                else None
                            ),
                            metadata_={**(handle.metadata or {}), "error_code": error_code},
                            latency_ms=latency_ms,
                            finished_at=finished_at,
                        )
                    )

                await session.commit()
        except Exception:
            logger.exception(
                "Failed to persist model-failed observation",
                extra={"turn_id": str(handle.turn_id)},
            )

    async def context_event(self, turn_id: UUID, name: str, metadata: dict) -> None:
        """Lifecycle telemetry remains optional and cannot affect execution."""
        if not self.settings.observability_enabled:
            return
        try:
            async with session_factory() as session:
                session.add(AgentObservation(
                    turn_id=turn_id,
                    sequence=self._next_observation_sequence(turn_id),
                    kind="CONTEXT", name=name, status="COMPLETED",
                    metadata_=metadata,
                    finished_at=datetime.now(timezone.utc),
                    **scoped_values(),
                ))
                await session.commit()
        except Exception:
            logger.exception("Failed to persist context lifecycle observation", extra={"turn_id": str(turn_id)})

    async def tool_started(
        self,
        turn_id: UUID,
        sequence: int,
        provider_call_id: str,
        name: str,
        arguments: dict,
        fingerprint: str,
    ) -> TraceHandle | None:
        if not self.settings.trace_enabled and not self.settings.observability_enabled:
            return None

        trace_id = uuid4() if self.settings.trace_enabled else None
        observation_id = uuid4() if self.settings.observability_enabled else None
        started_at = datetime.now(timezone.utc)

        try:
            async with session_factory() as session:
                if trace_id is not None:
                    session.add(
                        ToolCallTrace(
                            id=trace_id,
                            turn_id=turn_id,
                            provider_call_id=provider_call_id,
                            sequence=sequence,
                            tool_name=name,
                            arguments=arguments,
                            fingerprint=fingerprint,
                            status="RUNNING",
                            started_at=started_at,
                            **scoped_values(),
                        )
                    )

                if observation_id is not None:
                    session.add(
                        AgentObservation(
                            id=observation_id,
                            turn_id=turn_id,
                            sequence=self._next_observation_sequence(turn_id),
                            kind="TOOL",
                            name=name,
                            status="RUNNING",
                            input_payload=(
                                {
                                    "tool_call_id": provider_call_id,
                                    "arguments": arguments,
                                }
                                if self.settings.observability_capture_payload
                                else None
                            ),
                            started_at=started_at,
                            **scoped_values(),
                        )
                    )

                await session.commit()
        except Exception:
            logger.exception(
                "Failed to persist tool-start observation",
                extra={
                    "turn_id": str(turn_id),
                    "sequence": sequence,
                    "tool_name": name,
                },
            )
            return None

        return TraceHandle(trace_id, observation_id, turn_id, name)

    async def tool_finished(
        self,
        handle: TraceHandle | None,
        result: ToolResult,
        latency_ms: int,
    ) -> None:
        if handle is None:
            return

        status = "COMPLETED" if result.ok else "FAILED"
        source_refs = [item.uri for item in result.source_refs]
        finished_at = datetime.now(timezone.utc)

        try:
            async with session_factory() as session:
                if handle.trace_id is not None:
                    await session.execute(
                        update(ToolCallTrace)
                        .where(ToolCallTrace.id == handle.trace_id)
                        .values(
                            status=status,
                            ui_summary=result.ui_summary[:2048],
                            source_refs=source_refs,
                            latency_ms=latency_ms,
                            error_code=result.error.code if result.error else None,
                            finished_at=finished_at,
                        )
                    )

                if handle.observation_id is not None:
                    await session.execute(
                        update(AgentObservation)
                        .where(AgentObservation.id == handle.observation_id)
                        .values(
                            status=status,
                            output_payload=(
                                result.provider_payload()
                                if self.settings.observability_capture_payload
                                else None
                            ),
                            source_refs=source_refs,
                            latency_ms=latency_ms,
                            finished_at=finished_at,
                        )
                    )

                await session.commit()
        except Exception:
            logger.exception(
                "Failed to persist tool-finished observation",
                extra={"turn_id": str(handle.turn_id)},
            )

    async def retrieval_finished(
        self,
        handle: TraceHandle | None,
        payload: dict,
        latency_ms: int,
    ) -> None:
        if not self.settings.observability_enabled:
            return

        # A retrieval span is useful even if the parent Tool trace failed to
        # persist, so only parent_observation_id is conditional.
        turn_id = handle.turn_id if handle is not None else None
        if turn_id is None:
            return

        observation_id = uuid4()
        now = datetime.now(timezone.utc)
        try:
            async with session_factory() as session:
                session.add(
                    AgentObservation(
                        id=observation_id,
                        turn_id=turn_id,
                        parent_observation_id=(
                            handle.observation_id if handle is not None else None
                        ),
                        sequence=self._next_observation_sequence(turn_id),
                        kind="RETRIEVAL",
                        name=handle.name or "retrieval",
                        status="COMPLETED",
                        input_payload=(
                            payload.get("input")
                            if self.settings.observability_capture_payload
                            else None
                        ),
                        output_payload=(
                            payload.get("output")
                            if self.settings.observability_capture_payload
                            else None
                        ),
                        metadata_=payload.get("metadata") or {},
                        source_refs=payload.get("source_refs", []),
                        started_at=now,
                        finished_at=now,
                        latency_ms=latency_ms,
                        **scoped_values(),
                    )
                )
                await session.commit()
        except Exception:
            logger.exception(
                "Failed to persist retrieval observation",
                extra={"turn_id": str(turn_id)},
            )

    async def cleanup_expired(self) -> None:
        """Best-effort cleanup; observability maintenance must not block startup."""

        threshold = datetime.now(timezone.utc) - timedelta(
            days=self.settings.trace_retention_days
        )
        try:
            async with session_factory() as session:
                await session.execute(
                    delete(ModelCallTrace).where(ModelCallTrace.created_at < threshold)
                )
                await session.execute(
                    delete(ToolCallTrace).where(ToolCallTrace.created_at < threshold)
                )
                await session.execute(
                    delete(AgentObservation).where(
                        AgentObservation.created_at < threshold
                    )
                )
                await session.commit()
        except Exception:
            logger.exception("Failed to clean up expired agent observations")

    async def turn_model_calls(self, turn_id: UUID) -> list[ModelCallTrace]:
        return list(
            (
                await self.session.scalars(
                    select(ModelCallTrace)
                    .where(ModelCallTrace.turn_id == turn_id)
                    .order_by(ModelCallTrace.sequence, ModelCallTrace.call_type)
                )
            ).all()
        )

    async def turn_tool_calls(self, turn_id: UUID) -> list[ToolCallTrace]:
        return list(
            (
                await self.session.scalars(
                    select(ToolCallTrace)
                    .where(ToolCallTrace.turn_id == turn_id)
                    .order_by(ToolCallTrace.sequence)
                )
            ).all()
        )

    async def model_egress(self, call_id: UUID) -> dict | None:
        trace = await self.session.scalar(
            select(ModelCallTrace).where(ModelCallTrace.id == call_id)
        )
        return trace.egress_preview if trace else None
