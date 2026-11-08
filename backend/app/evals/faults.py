"""Configuration-driven fault injection around actual tool execution."""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from app.agent.model import ToolError, ToolResult


@dataclass
class ToolFaultPlan:
    targets: frozenset[str]
    trigger_mode: str
    occurrence: int
    then: str
    code: str
    message: str
    retryable: bool
    seen: dict[str, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Any) -> "ToolFaultPlan":
        return cls(targets=frozenset(config.target_tools), trigger_mode=config.trigger,
                   occurrence=config.occurrence or 1, then=config.then,
                   code=config.error_code, message=config.message, retryable=config.retryable)

    def maybe_result(self, tool_name: str, *, turn_id: object | None = None, call_id: str | None = None) -> ToolResult | None:
        if tool_name not in self.targets:
            return None
        count = self.seen.get(tool_name, 0) + 1
        self.seen[tool_name] = count
        triggered = self.trigger_mode == "every_call" or count == self.occurrence or (self.then == "continue_fault" and count >= self.occurrence)
        if not triggered:
            return None
        self.events.append({"tool_name": tool_name, "occurrence": count, "turn_id": str(turn_id) if turn_id else None,
                            "call_id": call_id, "injected": True, "ok": False, "error_code": self.code})
        return ToolResult(ok=False, ui_summary="Injected evaluation fault",
                          error=ToolError(code=self.code, message=self.message, retryable=self.retryable))

    def mark_real_result(self, tool_name: str, result: ToolResult, *, turn_id: object, call_id: str) -> None:
        if tool_name in self.targets:
            self.events.append({"tool_name": tool_name, "occurrence": self.seen.get(tool_name, 0),
                                "turn_id": str(turn_id), "call_id": call_id, "injected": False,
                                "ok": result.ok, "error_code": result.error.code if result.error else None,
                                "agent_payload": result.agent_payload})


_fault_plan: ContextVar[ToolFaultPlan | None] = ContextVar("jobpilot_eval_fault_plan", default=None)


def set_fault_plan(plan: ToolFaultPlan | None) -> Token:
    return _fault_plan.set(plan)


def reset_fault_plan(token: Token) -> None:
    _fault_plan.reset(token)


def injected_result(tool_name: str, *, turn_id: object | None = None, call_id: str | None = None) -> ToolResult | None:
    plan = _fault_plan.get()
    return plan.maybe_result(tool_name, turn_id=turn_id, call_id=call_id) if plan else None


def mark_real_result(tool_name: str, result: ToolResult, *, turn_id: object, call_id: str) -> None:
    plan = _fault_plan.get()
    if plan is not None:
        plan.mark_real_result(tool_name, result, turn_id=turn_id, call_id=call_id)
