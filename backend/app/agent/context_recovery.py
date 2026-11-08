"""Optional reconstruction of one known, current-turn context interaction.

Observability is incomplete by design. This reader therefore fails closed on
missing or ambiguous capture and never writes, repairs, or resumes a turn.
The caller must enforce its in-memory ref allowlist and use a separate short
database session so an observability read failure cannot affect business work.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context import RuntimeInteraction
from app.agent.model import ToolResult
from app.db.models import AgentObservation, ModelCallTrace
from app.evals.scope import apply_eval_scope

logger = logging.getLogger(__name__)


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _ref_sequence(turn_id: UUID, ref: str) -> int | None:
    prefix = f"ctx://{turn_id}/i/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        return None
    suffix = ref[len(prefix):]
    if not suffix.isascii() or not suffix.isdecimal() or suffix.startswith("0"):
        return None
    # Bound parsing independently of Python's integer-string conversion limit.
    if len(suffix) > 20:
        return None
    return int(suffix)


def _main_agent(row: AgentObservation) -> bool:
    return row.kind == "MODEL" and isinstance(row.metadata_, dict) and row.metadata_.get("call_type") == "MAIN_AGENT"


def _metadata_matches(row: AgentObservation, sequence: int, model_sequence: int) -> bool:
    metadata = row.metadata_
    interaction = metadata.get("interaction_seq")
    model = metadata.get("model_sequence")
    if "interaction_seq" in metadata:
        if not _positive_int(interaction) or interaction != sequence:
            return False
        return "model_sequence" not in metadata or (_positive_int(model) and model == model_sequence)
    return _positive_int(model) and model == model_sequence


def _assistant_calls(output: dict[str, Any]) -> tuple[dict, list[dict]] | None:
    assistant = output.get("assistant_message")
    normalized = output.get("tool_calls")
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant" or not isinstance(normalized, list):
        return None
    calls = assistant.get("tool_calls", [])
    if not isinstance(calls, list) or len(calls) != len(normalized):
        return None
    seen: set[str] = set()
    expected = []
    for call, recorded in zip(calls, normalized, strict=True):
        if not isinstance(call, dict) or not isinstance(recorded, dict) or call.get("type") != "function":
            return None
        call_id, function = call.get("id"), call.get("function")
        if not isinstance(call_id, str) or not call_id or call_id in seen or not isinstance(function, dict):
            return None
        name, raw_arguments = function.get("name"), function.get("arguments")
        if not isinstance(name, str) or not name or not isinstance(raw_arguments, str):
            return None
        try:
            # Match the existing model-client normalization without replacing
            # the original native string in the recovered assistant message.
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"__raw_arguments__": raw_arguments}
        if not isinstance(arguments, dict):
            return None
        if recorded.get("id") != call_id or recorded.get("name") != name or recorded.get("arguments") != arguments:
            return None
        seen.add(call_id)
        expected.append({"id": call_id, "name": name, "arguments": arguments})
    return assistant, expected


async def recover_interaction(
    session: AsyncSession,
    turn_id: UUID,
    ref: str,
    *,
    model_sequence: int | None = None,
) -> RuntimeInteraction | None:
    """Return an independent complete interaction, or None for any uncertainty.

    Ref sequence is an assistant-response sequence, never an observation
    sequence. V1 has no main-call retries, so an absent explicit model sequence
    can use the ref sequence to locate a MAIN_AGENT model call. Observation
    sequence is used only to bound that model's subsequent tool observations.
    """

    sequence = _ref_sequence(turn_id, ref)
    if sequence is None or (model_sequence is not None and not _positive_int(model_sequence)):
        return None
    expected_model_sequence = model_sequence if model_sequence is not None else sequence
    try:
        observations = list(await session.scalars(apply_eval_scope(
            select(AgentObservation)
            .where(AgentObservation.turn_id == turn_id)
            .order_by(AgentObservation.sequence),
            AgentObservation,
        )))
        # The SQL predicate is authoritative; this extra check also prevents
        # accidental cross-turn association in custom session adapters.
        observations = [item for item in observations if item.turn_id == turn_id]
        main_models = [item for item in observations if _main_agent(item)]
        candidates = [item for item in main_models if _metadata_matches(item, sequence, expected_model_sequence)]
        if not candidates:
            # Older captures lack explicit sequence metadata. Their model trace
            # and observation share the same exact started_at value. A trace
            # sequence alone, or the ordinal position of an observation, is not
            # enough to establish identity.
            traces = list(await session.scalars(apply_eval_scope(
                select(ModelCallTrace).where(
                    ModelCallTrace.turn_id == turn_id,
                    ModelCallTrace.call_type == "MAIN_AGENT",
                    ModelCallTrace.sequence == expected_model_sequence,
                    ModelCallTrace.status == "COMPLETED",
                ),
                ModelCallTrace,
            )))
            if len(traces) != 1 or traces[0].started_at is None:
                return None
            trace = traces[0]
            candidates = [item for item in main_models
                          if "interaction_seq" not in item.metadata_ and "model_sequence" not in item.metadata_
                          and item.started_at == trace.started_at and item.name == trace.model]
        if len(candidates) != 1:
            return None
        model = candidates[0]
        if model.status != "COMPLETED" or not isinstance(model.input_payload, dict) or not isinstance(model.output_payload, dict):
            return None
        # Require the complete provider input capture as well as the output;
        # previews, token counts, and final_text are insufficient for recovery.
        if not isinstance(model.input_payload.get("messages"), list) or not isinstance(model.input_payload.get("tools"), list):
            return None
        parsed = _assistant_calls(model.output_payload)
        if parsed is None:
            return None
        assistant, calls = parsed
        # Provider IDs are normally unique. Reuse anywhere in the turn makes
        # association unsafe when an intermediate best-effort MODEL record is
        # absent, even if the visible observation boundaries appear sufficient.
        target_ids = {call["id"] for call in calls}
        seen_target_ids: set[str] = set()
        for row in observations:
            if row.kind != "TOOL" or not isinstance(row.input_payload, dict):
                continue
            call_id = row.input_payload.get("tool_call_id")
            if isinstance(call_id, str) and call_id in target_ids:
                if call_id in seen_target_ids:
                    return None
                seen_target_ids.add(call_id)
        next_model_sequence = min((item.sequence for item in main_models if item.sequence > model.sequence), default=None)
        tool_rows = [item for item in observations if item.kind == "TOOL" and item.sequence > model.sequence
                     and (next_model_sequence is None or item.sequence < next_model_sequence)]
        if len(tool_rows) != len(calls):
            return None
        messages: list[dict] = []
        source_refs: set[str] = set()
        by_call_id: dict[str, AgentObservation] = {}
        for row in tool_rows:
            if not isinstance(row.input_payload, dict):
                return None
            call_id = row.input_payload.get("tool_call_id")
            if not isinstance(call_id, str) or call_id in by_call_id:
                return None
            by_call_id[call_id] = row
        for call in calls:
            row = by_call_id.get(call["id"])
            if row is None or row.name != call["name"] or row.input_payload.get("arguments") != call["arguments"]:
                return None
            payload = row.output_payload
            if row.status not in {"COMPLETED", "FAILED"} or not isinstance(payload, dict):
                return None
            if not {"ok", "agent_payload", "source_refs", "error"}.issubset(payload):
                return None
            if type(payload["ok"]) is not bool:
                return None
            result = ToolResult.model_validate(payload)
            if result.provider_payload() != payload or (row.status == "COMPLETED") != result.ok:
                return None
            source_refs.update(item.uri for item in result.source_refs)
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(payload, ensure_ascii=False)})
        return RuntimeInteraction(deepcopy(assistant), tuple(deepcopy(messages)), seq=sequence, ref=ref,
                                  source_refs=tuple(sorted(source_refs)))
    except Exception:
        # Do not log captured content or refs. The caller reports the explicit
        # WORKING_CONTEXT_NOT_RECOVERABLE result while execution continues.
        logger.warning("Working-context observability recovery unavailable", extra={"turn_id": str(turn_id)})
        return None
