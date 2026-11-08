from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

from app.agent.context_recovery import recover_interaction
from app.db.models import AgentObservation, ModelCallTrace
from app.evals.scope import EvalScope, reset_eval_scope, set_eval_scope


def captured_interaction(turn_id, *, seq=2, observation_sequence=7, calls=2):
    raw_arguments = [' { "query" : "甲", "limit": 2 } ', '{"query":"乙","limit":1}'][:calls]
    native_calls = [{"id": f"call-{index}", "type": "function", "function": {
        "name": "search_knowledge", "arguments": arguments,
    }} for index, arguments in enumerate(raw_arguments)]
    assistant = {"role": "assistant", "content": None, "tool_calls": native_calls}
    model = AgentObservation(
        id=uuid4(), turn_id=turn_id, sequence=observation_sequence, kind="MODEL", name="test-model",
        status="COMPLETED", metadata_={"call_type": "MAIN_AGENT", "interaction_seq": seq, "model_sequence": seq},
        input_payload={"messages": [{"role": "user", "content": "question"}], "tools": []},
        output_payload={"assistant_message": assistant, "tool_calls": [
            {"id": call["id"], "name": call["function"]["name"],
             "arguments": json.loads(call["function"]["arguments"])} for call in native_calls
        ]}, started_at=datetime(2026, 10, 17, 1, 0, tzinfo=timezone.utc),
    )
    tools = [AgentObservation(
        id=uuid4(), turn_id=turn_id, sequence=observation_sequence + index + 1,
        kind="TOOL", name="search_knowledge", status="COMPLETED", metadata_={},
        input_payload={"tool_call_id": call["id"], "arguments": json.loads(call["function"]["arguments"])},
        output_payload={"ok": True, "agent_payload": {"content": f"result-{index}"},
                        "source_refs": [f"message://{uuid4()}"], "error": None},
    ) for index, call in enumerate(native_calls)]
    return [model, *tools]


def fake_session(rows, traces=()):
    return SimpleNamespace(scalars=AsyncMock(side_effect=[rows, list(traces)]))


async def test_recovers_complete_multi_tool_interaction_with_exact_native_args_and_independent_copy():
    turn_id = uuid4()
    rows = captured_interaction(turn_id)
    before = deepcopy(rows[0].output_payload["assistant_message"])
    recovered = await recover_interaction(fake_session(rows), turn_id, f"ctx://{turn_id}/i/2")
    assert recovered is not None
    assert recovered.seq == 2 and recovered.ref == f"ctx://{turn_id}/i/2"
    assert recovered.assistant == before
    assert [item["tool_call_id"] for item in recovered.tool_messages] == ["call-0", "call-1"]
    assert json.loads(recovered.tool_messages[1]["content"]) == rows[2].output_payload
    assert set(recovered.source_refs) == {ref for row in rows[1:] for ref in row.output_payload["source_refs"]}
    recovered.assistant["tool_calls"][0]["function"]["arguments"] = "changed"
    recovered.tool_messages[0]["content"] = "changed"
    assert rows[0].output_payload["assistant_message"] == before
    assert rows[1].output_payload["agent_payload"]["content"] == "result-0"


async def test_failed_tool_result_is_complete_native_history():
    turn_id = uuid4()
    rows = captured_interaction(turn_id, calls=1)
    rows[1].status = "FAILED"
    rows[1].output_payload.update(ok=False, agent_payload=None,
                                 error={"code": "TIMEOUT", "message": "timed out", "retryable": True})
    recovered = await recover_interaction(fake_session(rows), turn_id, f"ctx://{turn_id}/i/2")
    assert recovered is not None
    assert json.loads(recovered.tool_messages[0]["content"])["error"]["code"] == "TIMEOUT"


@pytest.mark.parametrize("raw_arguments,normalized", [("", {}), ("{partial", {"__raw_arguments__": "{partial"})])
async def test_existing_model_client_argument_normalization_preserves_native_string(raw_arguments, normalized):
    turn_id = uuid4()
    rows = captured_interaction(turn_id, calls=1)
    rows[0].output_payload["assistant_message"]["tool_calls"][0]["function"]["arguments"] = raw_arguments
    rows[0].output_payload["tool_calls"][0]["arguments"] = normalized
    rows[1].input_payload["arguments"] = normalized
    recovered = await recover_interaction(fake_session(rows), turn_id, f"ctx://{turn_id}/i/2")
    assert recovered is not None
    assert recovered.assistant["tool_calls"][0]["function"]["arguments"] == raw_arguments


async def test_final_assistant_without_tools_is_recovered_without_synthesizing_from_final_text():
    turn_id = uuid4()
    rows = captured_interaction(turn_id, calls=0)
    rows[0].output_payload["assistant_message"] = {"role": "assistant", "content": "the complete final response"}
    rows[0].output_payload["final_text"] = "a different redundant representation"
    recovered = await recover_interaction(fake_session(rows), turn_id, f"ctx://{turn_id}/i/2")
    assert recovered is not None
    assert recovered.assistant["content"] == "the complete final response"
    assert recovered.tool_messages == ()


@pytest.mark.parametrize("damage", [
    "model_running", "input_capture_off", "input_messages_missing", "input_tools_missing",
    "output_capture_off", "assistant_missing", "assistant_role_wrong", "normalized_calls_missing",
    "native_calls_missing", "argument_changed", "argument_not_json", "argument_not_object",
    "normalized_call_changed", "tool_missing", "tool_running", "tool_capture_off", "tool_input_missing",
    "tool_call_id_wrong", "tool_name_wrong", "tool_arguments_wrong", "tool_output_incomplete",
    "tool_status_disagrees", "tool_payload_invalid", "duplicate_native_id", "duplicate_tool_id",
    "extra_tool", "duplicate_model_mapping",
])
async def test_incomplete_or_ambiguous_observability_capture_is_not_recoverable(damage):
    turn_id = uuid4()
    rows = captured_interaction(turn_id)
    model, first, second = rows
    if damage == "model_running":
        model.status = "RUNNING"
    elif damage == "input_capture_off":
        model.input_payload = None
    elif damage == "input_messages_missing":
        model.input_payload.pop("messages")
    elif damage == "input_tools_missing":
        model.input_payload.pop("tools")
    elif damage == "output_capture_off":
        model.output_payload = None
    elif damage == "assistant_missing":
        model.output_payload.pop("assistant_message")
    elif damage == "assistant_role_wrong":
        model.output_payload["assistant_message"]["role"] = "user"
    elif damage == "normalized_calls_missing":
        model.output_payload.pop("tool_calls")
    elif damage == "native_calls_missing":
        model.output_payload["assistant_message"].pop("tool_calls")
    elif damage == "argument_changed":
        model.output_payload["assistant_message"]["tool_calls"][0]["function"]["arguments"] = '{"query":"different"}'
    elif damage == "argument_not_json":
        model.output_payload["assistant_message"]["tool_calls"][0]["function"]["arguments"] = "{partial"
    elif damage == "argument_not_object":
        model.output_payload["assistant_message"]["tool_calls"][0]["function"]["arguments"] = "[]"
    elif damage == "normalized_call_changed":
        model.output_payload["tool_calls"][0]["id"] = "different"
    elif damage == "tool_missing":
        rows.pop()
    elif damage == "tool_running":
        first.status = "RUNNING"
    elif damage == "tool_capture_off":
        first.output_payload = None
    elif damage == "tool_input_missing":
        first.input_payload = None
    elif damage == "tool_call_id_wrong":
        first.input_payload["tool_call_id"] = "different"
    elif damage == "tool_name_wrong":
        first.name = "read_source"
    elif damage == "tool_arguments_wrong":
        first.input_payload["arguments"] = {"query": "different"}
    elif damage == "tool_output_incomplete":
        first.output_payload.pop("agent_payload")
    elif damage == "tool_status_disagrees":
        first.status = "FAILED"
    elif damage == "tool_payload_invalid":
        first.output_payload["source_refs"] = ["ctx://not-evidence"]
    elif damage == "duplicate_native_id":
        model.output_payload["assistant_message"]["tool_calls"][1]["id"] = "call-0"
        model.output_payload["tool_calls"][1]["id"] = "call-0"
    elif damage == "duplicate_tool_id":
        second.input_payload["tool_call_id"] = "call-0"
    elif damage == "extra_tool":
        extra = deepcopy(second)
        extra.sequence += 1
        rows.append(extra)
    elif damage == "duplicate_model_mapping":
        duplicate = deepcopy(model)
        duplicate.sequence += 10
        rows.append(duplicate)
    assert await recover_interaction(fake_session(rows), turn_id, f"ctx://{turn_id}/i/2") is None


async def test_observation_sequence_cannot_be_used_as_ref_sequence():
    turn_id = uuid4()
    rows = captured_interaction(turn_id, seq=1, observation_sequence=2)
    assert await recover_interaction(fake_session(rows), turn_id, f"ctx://{turn_id}/i/2") is None


async def test_model_sequence_metadata_identifies_interaction_without_observation_ordinal():
    turn_id = uuid4()
    rows = captured_interaction(turn_id, seq=2, observation_sequence=23)
    rows[0].metadata_.pop("interaction_seq")
    assert await recover_interaction(fake_session(rows), turn_id, f"ctx://{turn_id}/i/2", model_sequence=2) is not None


async def test_model_trace_sequence_requires_exact_unambiguous_started_at_association():
    turn_id = uuid4()
    rows = captured_interaction(turn_id, observation_sequence=23)
    rows[0].metadata_ = {"call_type": "MAIN_AGENT"}
    trace = ModelCallTrace(turn_id=turn_id, sequence=2, call_type="MAIN_AGENT", model="test-model",
                           status="COMPLETED", started_at=rows[0].started_at)
    assert await recover_interaction(fake_session(rows, [trace]), turn_id, f"ctx://{turn_id}/i/2") is not None
    trace.started_at = datetime(2026, 10, 17, 2, 0, tzinfo=timezone.utc)
    assert await recover_interaction(fake_session(rows, [trace]), turn_id, f"ctx://{turn_id}/i/2") is None


async def test_tools_from_later_model_and_duplicate_ids_in_another_interaction_cannot_complete_partial_capture():
    turn_id = uuid4()
    rows = captured_interaction(turn_id, calls=1)
    following = captured_interaction(turn_id, seq=3, observation_sequence=10, calls=1)
    # Even apparently separated duplicate IDs are ambiguous under partial
    # best-effort capture; recovery is deliberately conservative.
    assert await recover_interaction(fake_session(rows + following), turn_id, f"ctx://{turn_id}/i/2") is None
    assert await recover_interaction(fake_session(rows[:1] + following), turn_id, f"ctx://{turn_id}/i/2") is None
    following[0].output_payload["assistant_message"]["tool_calls"][0]["id"] = "next-call"
    following[0].output_payload["tool_calls"][0]["id"] = "next-call"
    following[1].input_payload["tool_call_id"] = "next-call"
    assert await recover_interaction(fake_session(rows + following), turn_id, f"ctx://{turn_id}/i/2") is not None


async def test_compactor_model_is_never_main_agent_recovery_candidate_or_tool_boundary():
    turn_id = uuid4()
    rows = captured_interaction(turn_id, calls=1)
    compactor = captured_interaction(turn_id, observation_sequence=8, calls=0)[0]
    compactor.metadata_["call_type"] = "CONTEXT_COMPACTION"
    rows[1].sequence = 9
    assert await recover_interaction(fake_session([rows[0], compactor, rows[1]]), turn_id, f"ctx://{turn_id}/i/2") is not None
    assert await recover_interaction(fake_session([compactor]), turn_id, f"ctx://{turn_id}/i/2") is None


@pytest.mark.parametrize("suffix", ["0", "02", "-1", "1/extra", "2?query", "２", "2" * 5000])
async def test_malformed_ref_is_rejected_before_database_access(suffix):
    turn_id = uuid4()
    session = fake_session([])
    assert await recover_interaction(session, turn_id, f"ctx://{turn_id}/i/{suffix}") is None
    session.scalars.assert_not_awaited()


async def test_other_turn_ref_is_rejected_before_database_access():
    turn_id = uuid4()
    session = fake_session(captured_interaction(uuid4()))
    assert await recover_interaction(session, turn_id, f"ctx://{uuid4()}/i/2") is None
    session.scalars.assert_not_awaited()


async def test_database_failure_is_best_effort_and_does_not_write_or_rollback_caller_session():
    turn_id = uuid4()
    session = SimpleNamespace(scalars=AsyncMock(side_effect=RuntimeError("database unavailable")),
                              commit=AsyncMock(), rollback=AsyncMock())
    assert await recover_interaction(session, turn_id, f"ctx://{turn_id}/i/2") is None
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


@pytest.mark.parametrize("evaluation", [False, True])
async def test_observation_and_trace_queries_are_exact_turn_and_eval_scoped(evaluation):
    turn_id, run_id = uuid4(), uuid4()
    session = fake_session([])
    token = set_eval_scope(EvalScope(run_id=run_id, case_id="scope_case") if evaluation else None)
    try:
        assert await recover_interaction(session, turn_id, f"ctx://{turn_id}/i/2") is None
    finally:
        reset_eval_scope(token)
    assert session.scalars.await_count == 2
    for call in session.scalars.await_args_list:
        compiled = call.args[0].compile(dialect=postgresql.dialect())
        assert turn_id in compiled.params.values()
        assert "turn_id =" in str(compiled)
        if evaluation:
            assert run_id in compiled.params.values() and "scope_case" in compiled.params.values()
            assert "eval_run_id =" in str(compiled) and "eval_case_id =" in str(compiled)
        else:
            assert "eval_run_id IS NULL" in str(compiled)
