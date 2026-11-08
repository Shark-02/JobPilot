from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.evals.compiler import SemanticCheck
from app.evals.judge import JudgeSchemaError, StructuredJudge, validate_response


def plan():
    check = SemanticCheck(id="fact", gold_path="gold.fact", statement="one fact")
    return SimpleNamespace(semantic_checks=[check], extractions=[])


def payload():
    return {"answer_spans": [{"id": "answer_1", "text": "answer"}],
            "evidence": [{"id": "e1", "content": "raw", "visible_to_agent": True,
                          "authoritative_for_grounding": True}], "memories": []}


def valid():
    return {"checks": [{"id": "fact", "result": "PASS", "answer_span_id": "answer_1",
                        "evidence_ids": ["e1"], "reason": "supported"}], "extractions": []}


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(checks=[]),
    lambda value: value["checks"].append({**value["checks"][0], "id": "unexpected"}),
    lambda value: value["checks"].append(value["checks"][0]),
    lambda value: value["checks"][0].update(result="MAYBE"),
    lambda value: value["checks"][0].update(evidence_ids=["missing"]),
    lambda value: value["checks"][0].update(answer_span_id="missing"),
    lambda value: value.update(judge_status="PASS"),
    lambda value: value["checks"][0].update(reason=1),
])
def test_strict_request_specific_judge_contract(mutation):
    value = valid()
    mutation(value)
    with pytest.raises(JudgeSchemaError):
        validate_response(json.dumps(value), plan=plan(), payload=payload())


def test_atomic_fail_and_unknown_are_valid_responses_without_an_overall_verdict():
    for status in ("PASS", "FAIL", "UNKNOWN"):
        value = valid()
        value["checks"][0]["result"] = status
        parsed = validate_response(json.dumps(value), plan=plan(), payload=payload())
        assert parsed.checks[0].result == status
        assert "judge_status" not in parsed.model_dump()


@pytest.mark.parametrize("text", ['{"checks":[],"checks":[],"extractions":[]}', '```json\n{}\n```', '{}garbage'])
def test_no_json_parser_compatibility_or_duplicate_fields(text):
    with pytest.raises(JudgeSchemaError):
        validate_response(text, plan=plan(), payload=payload())


def test_bound_memory_sources_are_scoped_to_memory_semantics():
    data = payload()
    data["memories"] = [{"id": "m1", "bound_sources": [{"id": "bound1", "content": "user fact"}]}]
    value = valid()
    value["checks"][0]["evidence_ids"] = ["bound1"]
    request = plan()
    request.semantic_checks[0].support_mode = "bound_memory_source"
    assert validate_response(json.dumps(value), plan=request, payload=data).checks[0].result == "PASS"
    request.semantic_checks[0].support_mode = "authoritative_visible"
    with pytest.raises(JudgeSchemaError):
        validate_response(json.dumps(value), plan=request, payload=data)


def test_memory_mapping_cannot_omit_an_observed_conflicting_memory():
    request = SimpleNamespace(semantic_checks=[], extractions=[SimpleNamespace(
        id="topics", kind="memory_topics", allowed_ids=["hashmap"], labels=[])])
    data = {"evidence": [], "answer_spans": [], "memories": [{"id": "old"}, {"id": "new"}]}
    value = {"checks": [], "extractions": [{"id": "topics", "kind": "memory_topics", "items": [
        {"memory_id": "new", "topic_id": "hashmap", "state": "stable"}]}]}
    with pytest.raises(JudgeSchemaError):
        validate_response(json.dumps(value), plan=request, payload=data)


def test_bullet_extraction_cannot_rewrite_observed_answer():
    request = SimpleNamespace(semantic_checks=[], extractions=[SimpleNamespace(
        id="bullets", kind="bullets", allowed_ids=[], labels=[])])
    value = {"checks": [], "extractions": [{"id": "bullets", "kind": "bullets", "items": [
        {"id": None, "text": "invented revision", "answer_span_id": "answer_1"}]}]}
    with pytest.raises(JudgeSchemaError):
        validate_response(json.dumps(value), plan=request, payload=payload())


def test_questions_require_observed_span_ids():
    request = SimpleNamespace(semantic_checks=[], extractions=[SimpleNamespace(
        id="extract", kind="questions", allowed_ids=["topic"], labels=[])])
    value = {"checks": [], "extractions": [{"id": "extract", "kind": "questions", "items": [
        {"id": "q1", "topic_ids": [], "followup_span_ids": [], "answer_span_id": None}]}]}
    with pytest.raises(JudgeSchemaError):
        validate_response(json.dumps(value), plan=request, payload=payload())


def test_direct_semantic_wire_does_not_require_model_generated_positions_or_citations():
    parsed = validate_response(json.dumps({"checks": [
        {"id": "fact", "result": "FAIL", "reason": "Required fact is missing"}], "extractions": []}),
        plan=plan(), payload=payload())
    assert parsed.checks[0].answer_span_id is None
    assert parsed.checks[0].evidence_ids == []


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def close(self):
        self.closed = True


def chunk(content=None, reasoning=None, finish=None):
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(
        delta=SimpleNamespace(content=content, reasoning_content=reasoning), finish_reason=finish)])


def mock_client(monkeypatch, stream, captures):
    import app.evals.judge as module

    class Client:
        def __init__(self, **kwargs):
            captures["constructor"] = kwargs
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            captures["request"] = kwargs
            return stream

        async def close(self):
            captures["client_closed"] = True

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    return StructuredJudge(Settings(_env_file=None, llm_base_url="https://synthetic.example/v1",
                                    llm_api_key="synthetic", llm_model="synthetic"))


async def test_streaming_counts_reasoning_but_only_final_content_is_parsed(monkeypatch):
    content = json.dumps({"items": [{**valid()["checks"][0], "id": "c1"}]})
    captures = {}
    stream = FakeStream([chunk(reasoning='private reasoning {"invalid":true}'),
                         chunk(content=content[:12]), chunk(reasoning="more hidden"),
                         chunk(content=content[12:], finish="stop")])
    judge = mock_client(monkeypatch, stream, captures)
    result = await judge.judge(plan=plan(), payload=payload())
    assert result["checks"][0]["result"] == "PASS"
    assert "private reasoning" not in json.dumps(result)
    assert result["telemetry"]["reasoning_chunks"] == 2
    assert result["telemetry"]["reasoning_chars"] > 0
    assert result["telemetry"]["content_chars"] == len(content)
    assert result["telemetry"]["first_content_latency_seconds"] is not None
    assert captures["constructor"]["max_retries"] == 0
    assert captures["request"]["stream"] is True
    assert "thinking" not in captures["request"]
    assert "reasoning_effort" not in captures["request"]
    assert stream.closed and captures["client_closed"]


async def test_streaming_provider_504_is_returned_after_bounded_retry(monkeypatch):
    class ProviderError(Exception):
        status_code = 504
    captures = {}
    judge = mock_client(monkeypatch, FakeStream([ProviderError()]), captures)
    result = await judge.judge(plan=plan(), payload=payload())
    assert result["execution_error"]["code"] == "PROVIDER_HTTP_504"
    assert result["telemetry"]["http_status"] == 504
    assert captures["constructor"]["max_retries"] == 0
    assert result["subtasks"][0]["telemetry"]["retry_count"] == 1


async def test_judge_parser_error_is_evaluator_error(monkeypatch):
    judge = mock_client(monkeypatch, FakeStream([chunk(content='{"fields":{}}', finish="stop")]), {})
    result = await judge.judge(plan=plan(), payload=payload())
    assert result["execution_error"]["code"] == "JUDGE_SCHEMA_ERROR"


async def test_incomplete_stream_is_contract_error(monkeypatch):
    judge = mock_client(monkeypatch, FakeStream([chunk(content=json.dumps(valid()), finish="length")]), {})
    result = await judge.judge(plan=plan(), payload=payload())
    assert result["execution_error"]["code"] == "JUDGE_OUTPUT_TRUNCATED"
