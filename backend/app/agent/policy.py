from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

from app.config import Settings


class PolicyError(RuntimeError):
    code = "DOMAIN_ERROR"


class TurnLimitError(PolicyError):
    code = "TURN_LIMIT"


class ToolLimitError(PolicyError):
    code = "TOOL_LIMIT"


class DeadlineExceededError(PolicyError):
    code = "DEADLINE_EXCEEDED"


@dataclass
class TurnRuntimeState:
    started_monotonic: float = field(default_factory=time.monotonic)
    model_turns: int = 0
    tool_calls: int = 0


class LoopPolicy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check_model_budget(self, state: TurnRuntimeState) -> None:
        if state.model_turns >= self.settings.agent_max_model_turns:
            raise TurnLimitError("Model-turn budget exceeded")
        self.check_deadline(state)

    def check_tool_budget(self, state: TurnRuntimeState) -> None:
        if state.tool_calls >= self.settings.agent_max_tool_executions:
            raise ToolLimitError("Tool-execution budget exceeded")
        self.check_deadline(state)

    def check_deadline(self, state: TurnRuntimeState) -> None:
        if time.monotonic() - state.started_monotonic >= self.settings.agent_total_deadline_seconds:
            raise DeadlineExceededError("Total turn deadline exceeded")

    def remaining_seconds(self, state: TurnRuntimeState, requested_timeout: int) -> float:
        remaining = self.settings.agent_total_deadline_seconds - (time.monotonic() - state.started_monotonic)
        if remaining <= 0:
            raise DeadlineExceededError("Total turn deadline exceeded")
        return min(remaining, requested_timeout)

    @staticmethod
    def fingerprint(tool_name: str, arguments: dict) -> str:
        canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()
