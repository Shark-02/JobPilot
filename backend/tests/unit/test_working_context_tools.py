from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from app.agent.context import AgentContext
from app.agent.context_tools import WorkingContextTools
from app.agent.executor import ToolExecutionState
from app.agent.loop import ThinAgentLoop
from app.agent.model import ModelCompleted, ModelResponse, ToolCall, ToolResult
from app.agent.policy import LoopPolicy, TurnRuntimeState
from app.agent.registry import build_default_registry
from app.config import Settings
from app.db.models import TurnStatus
from app.evals.evaluators import EvalFailureClassifier
from app.infra.token_count import TokenCounter
from app.knowledge.sources import SourceRef


def context() -> AgentContext:
    settings = Settings(
        _env_file=None, llm_model="test", llm_context_window=12_000,
        llm_max_output_tokens=1024, context_safety_margin_tokens=512,
        context_compaction_max_tokens=512, context_eviction_trigger_tokens=4000,
    )
    return AgentContext(settings, TokenCounter(), [{"role": "user", "content": "Continue the task"}], "", uuid4())


def native(calls: list[ToolCall]) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
        for call in calls
    ]}


def append(ctx: AgentContext, number: int, text: str) -> str:
    call = ToolCall(id=f"original-{number}", name="read_source", arguments={})
    ref = ctx.append_assistant(native([call]))
    ctx.append_tool_result(call.id, call.name, ToolResult(
        ok=True, agent_payload={"text": text}, source_refs=[SourceRef.message(UUID(int=number))],
    ))
    return ref


def loop_for(ctx: AgentContext) -> ThinAgentLoop:
    loop = ThinAgentLoop.__new__(ThinAgentLoop)
    loop.settings = ctx.settings
    loop.session = None
    loop.counter = ctx.counter
    loop.registry = build_default_registry(None, ctx.settings)
    loop.policy = LoopPolicy(ctx.settings)
    loop.trace = SimpleNamespace(tool_started=AsyncMock(return_value=None), tool_finished=AsyncMock(), context_event=AsyncMock())
    loop.executor = SimpleNamespace(execute=AsyncMock(return_value=ToolResult(ok=True, agent_payload={"business": "done"})))
    return loop


async def batch(loop: ThinAgentLoop, ctx: AgentContext, calls: list[ToolCall]) -> AgentContext:
    ctx.append_assistant(native(calls))
    return await loop._execute_tools(
        ctx.turn_id, calls, "message", ctx, ToolExecutionState(), TurnRuntimeState(), asyncio.Event(),
    )


def last_result(ctx: AgentContext, call_id: str) -> dict:
    item = ctx.get_interaction(ctx.known_refs()[-1])
    return json.loads(next(message["content"] for message in item.tool_messages if message["tool_call_id"] == call_id))


@pytest.mark.asyncio
async def test_exact_memory_read_never_opens_observability_session(monkeypatch):
    ctx = context()
    ref = append(ctx, 1, "原文 30s → 6.5s, quote: \\\"exact\\\"")
    original = ctx.get_interaction(ref)
    fallback = AsyncMock(side_effect=AssertionError("memory hit must not read trace"))
    monkeypatch.setattr("app.agent.context_tools.recover_interaction", fallback)
    result = await WorkingContextTools(None).execute(ToolCall(id="r", name="read_working_context", arguments={"refs": [ref]}), ctx)
    assert result.ok
    restored = result.agent_payload["interactions"][0]
    assert restored["assistant_message"] == original.assistant
    assert restored["tool_messages"] == list(original.tool_messages)
    assert ctx.get_interaction(ref) == original
    fallback.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_ref,reason", [
    ("ctx://bad/i/1", "WORKING_CONTEXT_INVALID_REF"),
    (f"ctx://{UUID(int=99)}/i/1", "WORKING_CONTEXT_TURN_MISMATCH"),
])
async def test_bad_and_cross_turn_refs_rejected_without_fallback(bad_ref, reason, monkeypatch):
    fallback = AsyncMock(side_effect=AssertionError("invalid ref must not query trace"))
    monkeypatch.setattr("app.agent.context_tools.recover_interaction", fallback)
    result = await WorkingContextTools(None).execute(ToolCall(id="r", name="read_working_context", arguments={"refs": [bad_ref]}), context())
    assert result.agent_payload["reason"] == reason
    fallback.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("session_failure", [False, True])
async def test_missing_memory_and_unavailable_capture_returns_not_recoverable(monkeypatch, session_failure):
    ctx = context()
    ref = append(ctx, 1, "retained")
    del ctx._interactions[ref]  # Simulate missing memory while identity remains known.

    @asynccontextmanager
    async def recovery_session():
        if session_failure:
            raise RuntimeError("optional observability unavailable")
        yield object()

    monkeypatch.setattr("app.agent.context_tools.session_factory", recovery_session)
    monkeypatch.setattr("app.agent.context_tools.recover_interaction", AsyncMock(return_value=None))
    result = await WorkingContextTools(None).execute(ToolCall(id="r", name="read_working_context", arguments={"refs": [ref]}), ctx)
    assert not result.ok
    assert result.agent_payload["reason"] == "WORKING_CONTEXT_NOT_RECOVERABLE"
    assert ctx.known_refs() == (ref,)
    assert ref not in ctx.active_refs
    assert '"recoverable": false' in str(ctx.build_messages()[0])


@pytest.mark.asyncio
async def test_successful_fallback_repairs_only_known_record(monkeypatch):
    ctx = context()
    ref = append(ctx, 1, "complete original")
    original = ctx.get_interaction(ref)
    del ctx._interactions[ref]

    @asynccontextmanager
    async def recovery_session():
        yield object()

    monkeypatch.setattr("app.agent.context_tools.session_factory", recovery_session)
    monkeypatch.setattr("app.agent.context_tools.recover_interaction", AsyncMock(return_value=original))
    result = await WorkingContextTools(None).execute(ToolCall(id="r", name="read_working_context", arguments={"refs": [ref]}), ctx)
    assert result.ok
    assert ctx.get_interaction(ref) == original
    assert ctx.known_refs() == (ref,)


@pytest.mark.asyncio
async def test_invalid_recovery_candidate_returns_not_recoverable_and_batch_can_continue(monkeypatch):
    ctx = context()
    ref = append(ctx, 1, "known original")
    wrong = ctx.get_interaction(ref)
    wrong.assistant["tool_calls"][0]["function"]["name"] = "wrong-tool"
    del ctx._interactions[ref]

    @asynccontextmanager
    async def recovery_session():
        yield object()

    monkeypatch.setattr("app.agent.context_tools.session_factory", recovery_session)
    monkeypatch.setattr("app.agent.context_tools.recover_interaction", AsyncMock(return_value=wrong))
    monkeypatch.setattr("app.agent.loop.event_broker", SimpleNamespace(publish=AsyncMock()))
    loop = loop_for(ctx)
    ctx = await batch(loop, ctx, [ToolCall(id="read", name="read_working_context", arguments={"refs": [ref]})])
    assert last_result(ctx, "read")["agent_payload"]["reason"] == "WORKING_CONTEXT_NOT_RECOVERABLE"
    messages, _ = ctx.prepare_messages(loop.registry.schemas())
    assert "WORKING_CONTEXT_NOT_RECOVERABLE" in str(messages)
    assert ref in ctx.known_refs()


@pytest.mark.asyncio
async def test_checkpoint_and_repeated_exact_reads_bypass_cache_and_hide_internal_payload(monkeypatch):
    ctx = context()
    ref = append(ctx, 1, "read precisely")
    loop = loop_for(ctx)
    published = AsyncMock()
    monkeypatch.setattr("app.agent.loop.event_broker", SimpleNamespace(publish=published))
    delta = {"established_facts_add": [{"text": "confirmed", "refs": [ref]}]}
    for number in range(3):
        calls = [
            ToolCall(id=f"c-{number}", name="checkpoint_task_state", arguments=delta),
            ToolCall(id=f"r-{number}", name="read_working_context", arguments={"refs": [ref]}),
        ]
        ctx = await batch(loop, ctx, calls)
        assert last_result(ctx, f"r-{number}")["ok"]
    assert len(ctx.working_task_state["established_facts"]) == 1
    published.assert_not_awaited()
    loop.executor.execute.assert_not_awaited()
    assert loop.trace.tool_finished.await_count == 6
    messages, tokens = ctx.prepare_messages(loop.registry.schemas())
    assert tokens["tool_schema_tokens"] > 0
    assert "confirmed" in str(messages)


@pytest.mark.asyncio
async def test_oversized_page_in_is_explicit_error_and_preserves_original_and_checkpoint(monkeypatch):
    ctx = context()
    ref = append(ctx, 1, "original evidence " * 20_000)
    append(ctx, 2, "newest evidence")
    loop = loop_for(ctx)
    ctx.prepare_messages(loop.registry.schemas())
    assert ref not in ctx.active_refs
    original = ctx.get_interaction(ref)
    monkeypatch.setattr("app.agent.loop.event_broker", SimpleNamespace(publish=AsyncMock()))
    calls = [
        ToolCall(id="checkpoint", name="checkpoint_task_state", arguments={
            "decisions_add": [{"text": "retained decision", "refs": [ref]}],
        }),
        ToolCall(id="read", name="read_working_context", arguments={"refs": [ref]}),
        ToolCall(id="business", name="list_documents", arguments={}),
    ]
    ctx = await batch(loop, ctx, calls)
    result = last_result(ctx, "read")
    assert not result["ok"]
    assert result["agent_payload"]["reason"] == "WORKING_CONTEXT_CAPACITY_EXCEEDED"
    assert last_result(ctx, "checkpoint")["ok"]
    assert ctx.working_task_state["decisions"][0]["text"] == "retained decision"
    assert ctx.get_interaction(ref) == original
    assert last_result(ctx, "business")["ok"]
    loop.executor.execute.assert_awaited_once()
    ctx.prepare_messages(loop.registry.schemas())


@pytest.mark.asyncio
async def test_page_in_admission_counts_later_business_results_and_schemas(monkeypatch):
    ctx = context()
    ref = append(ctx, 1, "small restored evidence")
    append(ctx, 2, "current evidence")
    loop = loop_for(ctx)
    loop.executor.execute = AsyncMock(return_value=ToolResult(ok=True, agent_payload={"large": "evidence " * 20_000}))
    monkeypatch.setattr("app.agent.loop.event_broker", SimpleNamespace(publish=AsyncMock()))
    ctx = await batch(loop, ctx, [
        ToolCall(id="read", name="read_working_context", arguments={"refs": [ref]}),
        ToolCall(id="business", name="read_source", arguments={}),
    ])
    assert last_result(ctx, "read")["agent_payload"]["reason"] == "WORKING_CONTEXT_CAPACITY_EXCEEDED"
    assert len(last_result(ctx, "business")["agent_payload"]["large"]) > 100_000
    with pytest.raises(ValueError, match="hard input limit"):
        ctx.prepare_messages(loop.registry.schemas())


@pytest.mark.asyncio
async def test_capacity_rejection_does_not_reactivate_an_unrecoverable_record(monkeypatch):
    ctx = context()
    large_ref = append(ctx, 1, "large original " * 20_000)
    append(ctx, 2, "small current evidence")
    loop = loop_for(ctx)
    ctx.prepare_messages(loop.registry.schemas())
    lost_ref = append(ctx, 3, "lost current evidence")
    del ctx._interactions[lost_ref]

    @asynccontextmanager
    async def recovery_session():
        yield object()

    monkeypatch.setattr("app.agent.context_tools.session_factory", recovery_session)
    monkeypatch.setattr("app.agent.context_tools.recover_interaction", AsyncMock(return_value=None))
    monkeypatch.setattr("app.agent.loop.event_broker", SimpleNamespace(publish=AsyncMock()))
    ctx = await batch(loop, ctx, [
        ToolCall(id="lost", name="read_working_context", arguments={"refs": [lost_ref]}),
        ToolCall(id="large", name="read_working_context", arguments={"refs": [large_ref]}),
    ])
    assert last_result(ctx, "lost")["agent_payload"]["reason"] == "WORKING_CONTEXT_NOT_RECOVERABLE"
    assert last_result(ctx, "large")["agent_payload"]["reason"] == "WORKING_CONTEXT_CAPACITY_EXCEEDED"
    assert lost_ref not in ctx.active_refs
    assert lost_ref in ctx.known_refs()
    ctx.prepare_messages(loop.registry.schemas())


@pytest.mark.asyncio
async def test_all_missing_requested_refs_are_marked_without_partial_success(monkeypatch):
    ctx = context()
    refs = [append(ctx, number, "missing raw") for number in (1, 2)]
    for ref in refs:
        del ctx._interactions[ref]

    @asynccontextmanager
    async def recovery_session():
        yield object()

    monkeypatch.setattr("app.agent.context_tools.session_factory", recovery_session)
    monkeypatch.setattr("app.agent.context_tools.recover_interaction", AsyncMock(return_value=None))
    loop = loop_for(ctx)
    ctx = await batch(loop, ctx, [ToolCall(id="read", name="read_working_context", arguments={"refs": refs})])
    result = last_result(ctx, "read")
    assert result["agent_payload"] == {"reason": "WORKING_CONTEXT_NOT_RECOVERABLE", "refs": refs}
    assert ctx.known_refs()[:-1] == tuple(refs)
    assert not set(refs).intersection(ctx.active_refs)
    messages, _ = ctx.prepare_messages(loop.registry.schemas())
    assert "WORKING_CONTEXT_NOT_RECOVERABLE" in str(messages)


@pytest.mark.asyncio
async def test_pressure_path_has_no_compactor_or_provider_calls():
    ctx = context()
    for number in range(3):
        append(ctx, number + 1, "evidence " * 3000)
    loop = loop_for(ctx)
    loop.compactor = SimpleNamespace(compact=AsyncMock(side_effect=AssertionError("no pressure LLM")))
    messages, tokens = await loop._prepare_context_messages(ctx, ctx.turn_id, TurnRuntimeState(), asyncio.Event())
    loop.compactor.compact.assert_not_awaited()
    assert len(ctx.active_refs) < len(ctx.known_refs())
    assert tokens["total"] <= ctx.settings.hard_input_limit
    assert "evidence" in str(messages)


@pytest.mark.asyncio
async def test_top_level_provider_failure_preserves_trace_turn_sse_and_unchanged_eval_classification(monkeypatch):
    import app.agent.loop as module

    class ProviderFailure(Exception):
        status_code = 500

    ctx = context()
    loop = loop_for(ctx)
    turn = SimpleNamespace(id=ctx.turn_id, conversation_id=uuid4(), model_turn_count=0,
                           tool_execution_count=0, status=TurnStatus.RUNNING)
    loop.session = SimpleNamespace(scalar=AsyncMock(return_value=turn), commit=AsyncMock(), rollback=AsyncMock())
    loop.trace.model_started = AsyncMock(return_value=object())
    loop.trace.model_finished = AsyncMock()
    loop.trace.model_failed = AsyncMock()

    class FailedModel:
        async def stream(self, request):
            raise ProviderFailure("provider unavailable")
            yield ModelCompleted(response=ModelResponse(assistant_message={}, final_text=""))

    loop.model_client = FailedModel()
    monkeypatch.setattr(module, "ContextBuilder", lambda *args: SimpleNamespace(build=AsyncMock(return_value=ctx)))
    broker = SimpleNamespace(publish=AsyncMock(), finish=AsyncMock())
    monkeypatch.setattr(module, "event_broker", broker)
    await loop.run(ctx.turn_id)
    assert turn.status == TurnStatus.FAILED
    assert turn.error_code == "PROVIDER_HTTP_500"
    assert loop.trace.model_failed.await_args.args[1] == turn.error_code
    assert any(call.kwargs.get("code") == turn.error_code for call in broker.publish.await_args_list)
    outcome = EvalFailureClassifier.agent_interruption({
        "turns": [turn], "model_calls": [SimpleNamespace(status="FAILED", error_code=turn.error_code)],
    })
    assert outcome.outcome.value == "BLOCKED"
    assert outcome.reason == "MODEL_PROVIDER_UNAVAILABLE"
