"""Lossless eval-only observations and factual evidence identities.

The collector is installed by the offline harness. Production runs have no
collector and retain the ordinary best-effort observability behaviour.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    origin_type: str
    source_ref: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    content: str
    visible_to_agent: bool
    authoritative_for_grounding: bool
    turn_id: str
    call_id: str | None = None
    conversation: str = "A"
    phase: str = "target"


class EvidencePack(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[EvidenceItem] = Field(default_factory=list)
    capture_complete: bool = True
    errors: list[str] = Field(default_factory=list)

    def judge_items(self) -> list[dict[str, Any]]:
        # Keep the complete A/B/C inventory in the artifact, but send only
        # authoritative raw excerpts to the Judge's ordinary factual channel.
        return [item.model_dump() for item in self.items if item.visible_to_agent and item.authoritative_for_grounding]


@dataclass(frozen=True)
class ObservationTurn:
    turn_id: str
    conversation: str = "A"
    phase: str = "target"


@dataclass
class EvalObservationCollector:
    model_inputs: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    memory_events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def validate_capture(self, data: dict[str, Any]) -> None:
        """Verify capture against runtime counters, independently of DB traces."""
        for turn in data.get("turns", []):
            turn_id = str(turn.id)
            inputs = [item for item in self.model_inputs if item["turn_id"] == turn_id and item["call_type"] == "MAIN_AGENT"]
            tool_events = [item for item in self.tool_events if item["turn_id"] == turn_id]
            expected_models = getattr(turn, "model_turn_count", 0) or 0
            expected_tools = getattr(turn, "tool_execution_count", 0) or 0
            if len(inputs) < expected_models:
                self.errors.append(f"MODEL_INPUT_CAPTURE_MISSING:{turn_id}")
            if len(tool_events) != expected_tools:
                self.errors.append(f"TOOL_RESULT_CAPTURE_MISSING:{turn_id}")
        for model in data.get("model_calls", []):
            if getattr(model, "call_type", "MAIN_AGENT") == "MAIN_AGENT" and not any(
                item["turn_id"] == str(model.turn_id) and item["sequence"] == model.sequence
                and item["call_type"] == "MAIN_AGENT" for item in self.model_inputs
            ):
                self.errors.append(f"MODEL_INPUT_CAPTURE_MISSING:{model.turn_id}:{model.sequence}")
        mutation_turns = {event["turn_id"] for event in self.tool_events if event["tool_name"] == "update_user_model"}
        for turn_id in mutation_turns:
            expected_mutations = sum(len(event["arguments"].get("mutations") or []) for event in self.tool_events
                                     if event["turn_id"] == turn_id and event["tool_name"] == "update_user_model"
                                     and event["execution_kind"] == "real" and event["result"].get("ok"))
            captured_mutations = sum(event["turn_id"] == turn_id and event["success"] for event in self.memory_events)
            if captured_mutations != expected_mutations:
                self.errors.append(f"MUTATION_CAPTURE_MISMATCH:{turn_id}")

    def record_model_input(self, turn_id: object, sequence: int, messages: list[dict], call_type: str) -> None:
        origin = observation_origin(turn_id)
        self.model_inputs.append({**origin, "sequence": sequence, "call_type": call_type, "messages": deepcopy(messages)})

    def record_tool(self, *, turn_id: object, sequence: int, call_id: str, tool_name: str,
                    arguments: dict, result: dict, execution_kind: str, fingerprint: str,
                    provider_error_code: str | None = None) -> None:
        self.tool_events.append({**observation_origin(turn_id), "sequence": sequence, "call_id": call_id,
                                 "tool_name": tool_name, "arguments": deepcopy(arguments), "result": deepcopy(result),
                                 "execution_kind": execution_kind, "fingerprint": fingerprint,
                                 "provider_error_code": provider_error_code})

    def record_mutation(self, event: dict[str, Any]) -> None:
        self.memory_events.append({**observation_origin(), **deepcopy(event)})

    def evidence_pack(self) -> EvidencePack:
        items: list[EvidenceItem] = []
        seen: set[tuple[str | None, str, str]] = set()
        # Only requests sent to the main Agent establish actual visibility.
        for request in self.model_inputs:
            if request["call_type"] != "MAIN_AGENT":
                continue
            for message in request["messages"]:
                role, content = message.get("role"), message.get("content", "")
                if not isinstance(content, str) or not content:
                    continue
                base = {key: request[key] for key in ("turn_id", "conversation", "phase")}
                call_id = str(message.get("tool_call_id") or f"model:{request['sequence']}")
                if role == "user" and not content.startswith("Compacted Task State"):
                    ref = re.search(r"message_source_ref: (message://[0-9a-fA-F-]{36})", content)
                    self._append(items, seen, content=content, source_ref=ref.group(1) if ref else None,
                                 origin_type="user_message", authoritative=True, call_id=call_id, **base)
                elif role == "tool":
                    try:
                        payload = json.loads(content)
                    except (TypeError, ValueError):
                        self.errors.append(f"INVALID_VISIBLE_TOOL_PAYLOAD:{request['turn_id']}:{call_id}")
                        continue
                    agent_payload = payload.get("agent_payload") or {}
                    if payload.get("ok") and isinstance(agent_payload, dict):
                        for raw in agent_payload.get("modules", []):
                            refs = raw.get("source_refs") or []
                            # A retrieved module is a stitched raw source excerpt, not a model summary.
                            identity = refs[0] if refs else None
                            self._append(items, seen, content=str(raw.get("content", "")), source_ref=identity,
                                         source_refs=refs, origin_type="knowledge_module", authoritative=True, call_id=call_id, **base)
                        for raw in agent_payload.get("sources", []):
                            message_role = (raw.get("metadata") or {}).get("role")
                            factual = message_role is None or message_role == "USER"
                            self._append(items, seen, content=str(raw.get("content", "")), source_ref=raw.get("source_ref"),
                                         origin_type="resolved_source" if factual else "assistant_source", authoritative=factual, call_id=call_id, **base)
                        for raw in agent_payload.get("memories", []):
                            self._append(items, seen, content=json.dumps(raw, ensure_ascii=False, sort_keys=True),
                                         source_ref=None, source_refs=raw.get("source_refs") or [],
                                         origin_type="memory_derived", authoritative=False, call_id=call_id, **base)
                    self._append(items, seen, content=content, source_ref=None, origin_type="tool_execution",
                                 authoritative=False, call_id=call_id, **base)
                else:
                    origin = ("assistant_derived" if role == "assistant" else "compacted_state" if content.startswith("Compacted Task State")
                              else "core_profile_derived" if content.startswith("Core Profile") else "execution_context")
                    self._append(items, seen, content=content, source_ref=None, origin_type=origin,
                                 authoritative=False, call_id=call_id, **base)
        return EvidencePack(items=items, capture_complete=not self.errors, errors=list(dict.fromkeys(self.errors)))

    @staticmethod
    def _append(items: list[EvidenceItem], seen: set, *, content: str, source_ref: str | None,
                origin_type: str, authoritative: bool, turn_id: str, call_id: str,
                conversation: str, phase: str, source_refs: list[str] | None = None) -> None:
        if not content:
            return
        # Preserve distinct excerpts from the same source. Identical content with
        # identical source identity is de-duplicated across repeated retrievals.
        key = (source_ref, content, origin_type if not authoritative else "raw")
        if key in seen:
            return
        seen.add(key)
        identity = sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:20]
        items.append(EvidenceItem(id=f"evidence_{identity}", origin_type=origin_type, source_ref=source_ref,
                                  source_refs=source_refs or ([source_ref] if source_ref else []),
                                  content=content, visible_to_agent=True, authoritative_for_grounding=authoritative,
                                  turn_id=turn_id, call_id=call_id, conversation=conversation, phase=phase))


_collector: ContextVar[EvalObservationCollector | None] = ContextVar("jobpilot_eval_observation_collector", default=None)
_turn: ContextVar[ObservationTurn | None] = ContextVar("jobpilot_eval_observation_turn", default=None)


def get_observation_collector() -> EvalObservationCollector | None:
    return _collector.get()


def set_observation_collector(collector: EvalObservationCollector | None) -> Token:
    return _collector.set(collector)


def reset_observation_collector(token: Token) -> None:
    _collector.reset(token)


def set_observation_turn(turn_id: object, conversation: str = "A", phase: str = "target") -> Token:
    return _turn.set(ObservationTurn(str(turn_id), conversation, phase))


def reset_observation_turn(token: Token) -> None:
    _turn.reset(token)


def observation_origin(turn_id: object | None = None) -> dict[str, str]:
    origin = _turn.get()
    return {"turn_id": str(turn_id) if turn_id is not None else origin.turn_id if origin else "",
            "conversation": origin.conversation if origin else "A", "phase": origin.phase if origin else "target"}
