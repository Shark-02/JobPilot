from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agent.loop import ThinAgentLoop
from app.agent.model import ModelCompleted, ModelResponse, ToolCall, ToolResult
from app.agent.policy import LoopPolicy
from app.config import Settings
from app.db.models import TurnStatus
from app.evals.evidence import EvalObservationCollector


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_tool", [True, False])
async def test_failed_next_model_preserves_completed_tool_counter_after_rollback(monkeypatch, capture_tool):
    """One successful model/tool followed by timeout must remain auditable."""
    import app.agent.loop as module

    turn_id = uuid4()
    turn = SimpleNamespace(id=turn_id, conversation_id=uuid4(), model_turn_count=0,
                           tool_execution_count=0, status=TurnStatus.RUNNING)
    collector = EvalObservationCollector()

    class Session:
        persisted_models = 0
        persisted_tools = 0

        async def scalar(self, statement):
            return turn

        async def commit(self):
            self.persisted_models = turn.model_turn_count
            self.persisted_tools = turn.tool_execution_count

        async def rollback(self):
            turn.model_turn_count = self.persisted_models
            turn.tool_execution_count = self.persisted_tools

    session = Session()
    context = SimpleNamespace(append_assistant=lambda value: None,
                              append_tool_result=lambda *args: None)
    monkeypatch.setattr(module, "ContextBuilder", lambda *args: SimpleNamespace(build=AsyncMock(return_value=context)))
    monkeypatch.setattr(module, "event_broker", SimpleNamespace(publish=AsyncMock(), finish=AsyncMock()))

    class Model:
        calls = 0

        async def stream(self, request):
            self.calls += 1
            if self.calls == 2:
                raise asyncio.TimeoutError("synthetic provider timeout")
            yield ModelCompleted(response=ModelResponse(
                assistant_message={"role": "assistant", "content": ""}, final_text="",
                tool_calls=[ToolCall(id="lookup", name="search_memory", arguments={"query": "topic"})]))

    async def model_started(identifier, sequence, request, tokens):
        collector.record_model_input(identifier, sequence, request.messages, "MAIN_AGENT")
        return SimpleNamespace()

    async def execute(identifier, sequence, call, tool_state, runtime_state):
        result = ToolResult(ok=True, agent_payload={"memories": []})
        if capture_tool:
            collector.record_tool(turn_id=identifier, sequence=sequence, call_id=call.id,
                                  tool_name=call.name, arguments=call.arguments,
                                  result=result.provider_payload(), execution_kind="real", fingerprint="lookup")
        return result

    loop = ThinAgentLoop.__new__(ThinAgentLoop)
    loop.session = session
    loop.settings = Settings(_env_file=None, llm_model="synthetic")
    loop.counter = None
    loop.policy = LoopPolicy(loop.settings)
    loop.registry = SimpleNamespace(schemas=lambda: [])
    loop.model_client = Model()
    loop.trace = SimpleNamespace(model_started=model_started, model_finished=AsyncMock(), model_failed=AsyncMock())
    loop.executor = SimpleNamespace(execute=execute)
    loop._prepare_context_messages = AsyncMock(return_value=([], {"total": 0}))

    await loop.run(turn_id)

    assert turn.status == TurnStatus.FAILED
    assert turn.error_code == "MODEL_TIMEOUT"
    assert session.persisted_models == 1
    assert session.persisted_tools == 1
    collector.validate_capture({"turns": [turn]})
    assert collector.evidence_pack().capture_complete is capture_tool
    assert (f"TOOL_RESULT_CAPTURE_MISSING:{turn_id}" in collector.errors) is (not capture_tool)
