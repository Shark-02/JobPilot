from uuid import UUID
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent.context import AgentContext, ContextBuilder, ContextWindowError
from app.agent.model import MemoryMutation, ToolError, ToolResult
from app.agent.policy import LoopPolicy
from app.config import Settings
from app.db.models import MemoryKind, MessageRole
from app.infra.token_count import TokenCounter
from app.knowledge.sources import SourceRef


def test_memory_mutation_requires_evidence_except_archive() -> None:
    with pytest.raises(ValidationError):
        MemoryMutation(operation="ADD", kind=MemoryKind.SKILL, content="Python")
    archive = MemoryMutation(operation="ARCHIVE", target_memory_id=UUID(int=1))
    assert archive.operation == "ARCHIVE"


def test_memory_mutation_accepts_user_message_evidence() -> None:
    mutation = MemoryMutation(
        operation="ADD", kind=MemoryKind.SKILL, content="Python",
        source_refs=[SourceRef.message(UUID(int=1))],
    )
    assert mutation.source_refs[0].uri.startswith("message://")


def test_fingerprint_is_canonical() -> None:
    assert LoopPolicy.fingerprint("search_memory", {"query": "a", "limit": 8}) == LoopPolicy.fingerprint(
        "search_memory", {"limit": 8, "query": "a"}
    )


def _context_settings() -> Settings:
    return Settings(
        llm_model="test-model",
        llm_context_window=10_000,
        llm_max_output_tokens=1_000,
        context_safety_margin_tokens=1_000,
        recent_conversation_target_tokens=1_000,
        recent_conversation_max_tokens=1_500,
        context_compaction_max_tokens=512,
    )


def _agent_context(settings: Settings | None = None, user_message: str = "继续当前任务") -> AgentContext:
    settings = settings or _context_settings()
    counter = TokenCounter()
    return AgentContext(settings, counter, [{"role": "user", "content": user_message}], "", UUID(int=1))


def _append_tool_interaction(context: AgentContext, number: int, payload: str) -> None:
    call_id = f"call-{number}"
    context.append_assistant({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "search_knowledge", "arguments": "{}"},
        }],
    })
    context.append_tool_result(call_id, "search_knowledge", ToolResult(
        ok=True,
        agent_payload={"evidence": payload},
        ui_summary=f"UI-only summary {number}",
        source_refs=[SourceRef.message(UUID(int=number))],
    ))


def test_tool_results_remain_raw_across_rounds_below_soft_limit() -> None:
    context = _agent_context(Settings(llm_context_window=80_000))
    for number in range(1, 4):
        _append_tool_interaction(context, number, f"complete evidence-{number}")

    messages, tokens = context.build_messages()

    assert tokens["total"] < context.settings.soft_input_limit
    assert context.prepare_messages()[0] == messages
    rendered = str(messages)
    for number in range(1, 4):
        assert f"complete evidence-{number}" in rendered
    assert "UI-only summary" not in rendered


def test_native_tool_pairing_remains_complete_after_eviction() -> None:
    context = _agent_context()
    for number in range(1, 4):
        _append_tool_interaction(context, number, "detail " * 2_000)
    messages, _ = context.prepare_messages()

    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            call_ids = [call["id"] for call in message.get("tool_calls", [])]
            following = messages[index + 1 : index + 1 + len(call_ids)]
            assert [item["role"] for item in following] == ["tool"] * len(call_ids)
            assert [item["tool_call_id"] for item in following] == call_ids
        elif message["role"] == "tool":
            assert any(
                item["role"] == "assistant"
                and message["tool_call_id"] in [call["id"] for call in item.get("tool_calls", [])]
                for item in messages[:index]
            )


def test_hard_limit_raises_without_silently_dropping_current_evidence() -> None:
    context = _agent_context(user_message="current evidence " * 10_000)
    _, tokens = context.build_messages()

    with pytest.raises(ContextWindowError, match="preserved"):
        context.assert_within_hard_limit(tokens)


def test_tool_error_contract_has_no_business_specific_error() -> None:
    error = ToolError(code="POSSIBLE_DUPLICATE", message="duplicate")
    assert error.retryable is False


def test_user_message_context_exposes_evidence_and_attachment_ids_but_not_attachment_text() -> None:
    message = SimpleNamespace(
        id=UUID(int=1), role=MessageRole.USER, content="请使用附件", metadata_={"document_ids": [str(UUID(int=2))]}
    )
    model_message = ContextBuilder._message_to_model(message)  # type: ignore[arg-type]
    assert "message://00000000-0000-0000-0000-000000000001" in model_message["content"]
    assert "00000000-0000-0000-0000-000000000002" in model_message["content"]
    assert "not included here" in model_message["content"]
