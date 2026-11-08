from __future__ import annotations

import asyncio
import json
import stat
from collections import defaultdict, deque
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from app.evals.judge_scheduler import JudgeScheduler, JudgeTask
from app.evals.evaluators import EvalClassification, EvalFailureClassifier, EvalOutcome


class Counter:
    def count(self, text):
        return len(text)


def settings(**changes):
    return SimpleNamespace(**{
        "eval_judge_concurrency": 2, "eval_judge_timeout_seconds": 1,
        "eval_judge_max_output_tokens": 6144, "eval_judge_temperature": 0.0,
        "eval_judge_reasoning_effort": None,
        "eval_judge_transport_retries": 1, "eval_judge_retry_backoff_seconds": 0,
        "eval_judge_cache_dir": None, **changes,
    })


def parse(raw):
    body = json.loads(raw)
    if set(body) != {"id", "result"} or body["result"] not in {"PASS", "FAIL", "UNKNOWN"}:
        raise ValueError("Invalid atomic result")
    return body


def task(identifier="t1", *, validate=parse):
    return JudgeTask(identifier, "semantic", "semantic", [{"role": "user", "content": identifier}],
                     validate, "fixture_schema_1")


def response(identifier, *, result="PASS", delay=0, finish="stop", usage=None, raw=None):
    return {"raw": raw if raw is not None else json.dumps({"id": identifier, "result": result}),
            "delay": delay, "finish": finish, "usage": usage}


class ProviderError(Exception):
    def __init__(self, status):
        self.status_code = status


class Backend:
    def __init__(self, fixtures):
        self.fixtures = {identifier: deque(items) for identifier, items in fixtures.items()}
        self.calls = defaultdict(int)
        self.active = self.max_active = self.clients_closed = self.streams_closed = 0
        self.requests = []
        self.constructors = []
        self.streams = []
        backend = self

        class Client:
            def __init__(self, **kwargs):
                backend.constructors.append(kwargs)
                self.chat = SimpleNamespace(completions=self)

            async def create(self, **kwargs):
                backend.requests.append(kwargs)
                identifier = kwargs["messages"][-1]["content"]
                backend.calls[identifier] += 1
                fixture = backend.fixtures[identifier].popleft()
                if isinstance(fixture, Exception):
                    raise fixture
                stream = Stream(backend, fixture)
                backend.streams.append(stream)
                return stream

            async def close(self):
                backend.clients_closed += 1

        self.client_factory = Client


class Stream:
    def __init__(self, backend, fixture):
        self.backend = backend
        self.fixture = fixture
        self.closed = False
        backend.active += 1
        backend.max_active = max(backend.max_active, backend.active)

    async def __aiter__(self):
        fixture = self.fixture
        yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(reasoning_content="private hidden reasoning", content=None),
            finish_reason=None)])
        await asyncio.sleep(fixture.get("delay", 0))
        if fixture.get("partial_raw") is not None:
            yield SimpleNamespace(usage=fixture.get("partial_usage"), choices=[SimpleNamespace(
                delta=SimpleNamespace(content=fixture["partial_raw"]), finish_reason=None)])
        if fixture.get("error") is not None:
            raise fixture["error"]
        yield SimpleNamespace(usage=fixture.get("usage"), choices=[SimpleNamespace(
            delta=SimpleNamespace(content=fixture.get("raw")), finish_reason=fixture.get("finish"))])

    async def close(self):
        if not self.closed:
            self.closed = True
            self.backend.active -= 1
            self.backend.streams_closed += 1


def scheduler(backend, options=None, **changes):
    return JudgeScheduler(options or settings(), model=changes.get("model", "mock"),
                          base_url=changes.get("base_url", "https://mock.invalid/v1"),
                          api_key="mock-key", counter=Counter(), client_factory=backend.client_factory)


async def test_worker_pool_preserves_input_order_and_shares_limit_between_run_calls():
    fixtures = {f"t{i}": [response(f"t{i}", delay=(0.03 if i == 0 else 0.005))] for i in range(8)}
    backend = Backend(fixtures)
    judge = scheduler(backend)
    left, right = await asyncio.gather(
        judge.run([task(f"t{i}") for i in range(4)], identity={"case": "one"}),
        judge.run([task(f"t{i}") for i in range(4, 8)], identity={"case": "two"}),
    )
    assert [item["task_id"] for item in left + right] == [f"t{i}" for i in range(8)]
    assert all(item["body"]["result"] == "PASS" for item in left + right)
    assert backend.max_active == 2
    assert backend.clients_closed == backend.streams_closed == 8
    assert sum(item["telemetry"]["queue_wait_seconds"] for item in left + right) > 0.005
    assert all(item["max_retries"] == 0 for item in backend.constructors)
    assert all(item["max_tokens"] == 6144 and item["stream"] for item in backend.requests)
    assert "private hidden reasoning" not in json.dumps(left + right)


async def test_failed_attempt_usage_and_content_are_included_in_total_costs():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    failed = {"error": ProviderError(503)}
    backend = Backend({"t1": [failed, response("t1", result="FAIL", usage=usage)]})
    result = (await scheduler(backend).run([task()], identity={}))[0]
    telemetry = result["telemetry"]
    assert result["body"]["result"] == "FAIL"
    assert telemetry["retry_count"] == 1 and telemetry["attempt_count"] == 2
    assert telemetry["input_token_estimate"] == 2 * telemetry["planned_input_token_estimate"]
    assert telemetry["reasoning_chars"] == 2 * len("private hidden reasoning")
    assert telemetry["provider_input_tokens"] is None and telemetry["provider_output_tokens"] is None
    assert telemetry["provider_reported_input_tokens"] == 10 and telemetry["provider_reported_output_tokens"] == 5
    assert telemetry["provider_usage_complete"] is False
    assert telemetry["attempts"][0]["error_type"] == "provider"


async def test_stream_disconnect_retries_without_reusing_partial_content(tmp_path):
    partial = '{"id":"t1","result":"PASS'
    interrupted = {
        "partial_raw": partial,
        "partial_usage": SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        "error": httpx.RemoteProtocolError("peer closed connection during chunked read"),
    }
    backend = Backend({"t1": [interrupted, response(
        "t1", result="FAIL", usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7))]})
    parsed = []

    def record_parse(raw):
        parsed.append(raw)
        return parse(raw)

    result = (await scheduler(backend, settings(eval_judge_cache_dir=tmp_path)).run(
        [task(validate=record_parse)], identity={}))[0]
    assert result["body"]["result"] == "FAIL"
    assert parsed == [json.dumps({"id": "t1", "result": "FAIL"})]
    telemetry = result["telemetry"]
    assert telemetry["attempt_count"] == 2 and telemetry["retry_count"] == 1
    assert telemetry["provider_input_tokens"] == 22 and telemetry["provider_output_tokens"] == 12
    assert telemetry["attempts"][0]["content_chars"] == len(partial)
    assert telemetry["attempts"][0]["provider_error"] == "PROVIDER_CONNECTION_ERROR"
    assert telemetry["attempts"][0]["exception_type"] == "RemoteProtocolError"
    assert telemetry["attempts"][0]["finish_reason"] is None
    assert telemetry["attempts"][0]["status"] == "failed"
    assert telemetry["attempts"][1]["status"] == "success"
    assert backend.clients_closed == backend.streams_closed == 2
    assert [request["max_tokens"] for request in backend.requests] == [6144, 6144]
    cached = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert cached["raw"] == parsed[0]


async def test_stream_disconnect_exhausts_bounded_retries_and_blocks_case():
    failed = {"partial_raw": '{"id":"t1",',
              "error": httpx.RemoteProtocolError("incomplete chunked read")}
    backend = Backend({"t1": [failed] * 4})
    result = (await scheduler(backend, settings(eval_judge_transport_retries=2)).run(
        [task()], identity={}))[0]
    assert result["execution_error"] == {
        "code": "PROVIDER_CONNECTION_ERROR", "exception_type": "RemoteProtocolError"}
    assert "body" not in result
    assert backend.calls["t1"] == 3
    assert backend.clients_closed == backend.streams_closed == 3
    assert result["telemetry"]["attempt_count"] == 3
    assert result["telemetry"]["retry_count"] == 2
    assert all(attempt["exception_type"] == "RemoteProtocolError"
               for attempt in result["telemetry"]["attempts"])
    assert EvalFailureClassifier.judge_interruption(result) == EvalClassification(
        EvalOutcome.BLOCKED, "JUDGE_PROVIDER_UNAVAILABLE", "JUDGE")


@pytest.mark.parametrize("error", [OSError("network"), ProviderError(429), ProviderError(500),
                                        ProviderError(502), ProviderError(503), ProviderError(504)])
async def test_only_safe_transport_errors_are_retried(error):
    backend = Backend({"t1": [error, response("t1")]})
    result = (await scheduler(backend).run([task()], identity={}))[0]
    assert result["body"]["result"] == "PASS"
    assert backend.calls["t1"] == 2


@pytest.mark.parametrize("status", [400, 401, 402, 403])
async def test_billing_authentication_and_bad_request_are_not_retried(status):
    backend = Backend({"t1": [ProviderError(status), response("t1")]})
    result = (await scheduler(backend).run([task()], identity={}))[0]
    assert result["execution_error"]["code"] == f"PROVIDER_HTTP_{status}"
    assert "body" not in result and backend.calls["t1"] == 1


async def test_transient_failure_stops_after_retry_limit():
    backend = Backend({"t1": [ProviderError(503)] * 4})
    result = (await scheduler(backend, settings(eval_judge_transport_retries=2)).run([task()], identity={}))[0]
    assert result["execution_error"]["code"] == "PROVIDER_HTTP_503"
    assert backend.calls["t1"] == 3
    assert result["telemetry"]["retry_count"] == 2


async def test_retry_backoff_is_exponential_and_does_not_raise_output_budget(monkeypatch):
    waits = []

    async def sleep(duration):
        waits.append(duration)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    backend = Backend({"t1": [ProviderError(503), ProviderError(503), response("t1")]})
    result = (await scheduler(backend, settings(eval_judge_transport_retries=2,
                                                eval_judge_retry_backoff_seconds=0.5)).run([task()], identity={}))[0]
    assert result["body"]["result"] == "PASS"
    assert [duration for duration in waits if duration > 0] == [0.5, 1.0]
    assert [request["max_tokens"] for request in backend.requests] == [6144] * 3


async def test_timeout_closes_stream_and_never_produces_partial_success():
    backend = Backend({"t1": [response("t1", delay=0.03), response("t1")]})
    result = (await scheduler(backend, settings(eval_judge_timeout_seconds=0.005)).run([task()], identity={}))[0]
    assert result["execution_error"]["code"] == "JUDGE_TIMEOUT"
    assert result["telemetry"]["error_type"] == "timeout"
    assert "body" not in result and backend.calls["t1"] == 1
    assert backend.clients_closed == backend.streams_closed == 1


@pytest.mark.parametrize("finish,code", [("length", "JUDGE_OUTPUT_TRUNCATED"),
                                              (None, "JUDGE_INCOMPLETE_OUTPUT"),
                                              ("content_filter", "JUDGE_INCOMPLETE_OUTPUT")])
async def test_finish_reason_requires_complete_stop_even_with_valid_json(finish, code):
    backend = Backend({"t1": [response("t1", finish=finish), response("t1")]})
    result = (await scheduler(backend).run([task()], identity={}))[0]
    assert result["execution_error"]["code"] == code
    assert "body" not in result and backend.calls["t1"] == 1


async def test_schema_and_coverage_validation_failures_are_nonretryable():
    class CoverageError(ValueError):
        code = "JUDGE_COVERAGE_ERROR"

    def uncovered(_):
        raise CoverageError("Missing structural count")

    backend = Backend({"t1": [response("t1", raw='{"result":"MAYBE"}')],
                       "t2": [response("t2")]})
    results = await scheduler(backend).run([task(), task("t2", validate=uncovered)], identity={})
    assert [item["execution_error"]["code"] for item in results] == ["JUDGE_SCHEMA_ERROR", "JUDGE_COVERAGE_ERROR"]
    assert [item["telemetry"]["error_type"] for item in results] == ["schema", "coverage"]
    assert all("body" not in item for item in results)


async def test_success_cache_resume_keeps_historical_costs_separate_and_retries_failed_task(tmp_path):
    backend = Backend({"t1": [response("t1")], "t2": [response("t2", finish="length"), response("t2")]})
    judge = scheduler(backend, settings(eval_judge_cache_dir=tmp_path))
    initial = await judge.run([task(), task("t2")], identity={"evidence": ["source"]})
    resumed = await judge.run([task(), task("t2")], identity={"evidence": ["source"]})
    assert "execution_error" in initial[1] and "body" in resumed[1]
    assert backend.calls == {"t1": 1, "t2": 2}
    telemetry = resumed[0]["telemetry"]
    assert telemetry["cache_hit"] and telemetry["recovered_result"]
    assert telemetry["input_token_estimate"] == telemetry["content_token_estimate"] == 0
    assert telemetry["attempt_count"] == 0
    assert telemetry["cached_historical_telemetry"]["input_token_estimate"] > 0
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in tmp_path.glob("*.json"))
    assert "private hidden reasoning" not in "".join(path.read_text() for path in tmp_path.glob("*.json"))


@pytest.mark.parametrize("change", ["model", "base_url", "schema", "messages", "tokens", "evidence", "plan"])
async def test_cache_identity_invalidates_changed_input_or_execution_contract(tmp_path, change):
    backend = Backend({"t1": [response("t1"), response("t1")], "changed": [response("changed")]})
    options = settings(eval_judge_cache_dir=tmp_path)
    identity = {"plan": {"checks": ["one"]}, "evidence": ["actual source content"]}
    original = task()
    await scheduler(backend, options).run([original], identity=identity)
    changes = {}
    if change in {"model", "base_url"}:
        changes[change] = "different"
    elif change == "schema":
        original = replace(original, schema_version="fixture_schema_2")
    elif change == "messages":
        original = replace(original, messages=[{"role": "user", "content": "changed"}])
    elif change == "tokens":
        options.eval_judge_max_output_tokens = 5000
    else:
        identity[change] = ["changed"]
    result = (await scheduler(backend, options, **changes).run([original], identity=identity))[0]
    assert "body" in result and not result["telemetry"]["cache_hit"]
    assert sum(backend.calls.values()) == 2


async def test_corrupt_or_contract_invalid_cache_is_recomputed(tmp_path):
    backend = Backend({"t1": [response("t1")] * 3})
    judge = scheduler(backend, settings(eval_judge_cache_dir=tmp_path))
    await judge.run([task()], identity={})
    path = next(tmp_path.glob("*.json"))
    path.write_text("invalid json")
    await judge.run([task()], identity={})
    cached = json.loads(path.read_text())
    cached["raw"] = '{"result":"MAYBE"}'
    path.write_text(json.dumps(cached))
    result = (await judge.run([task()], identity={}))[0]
    assert "body" in result and backend.calls["t1"] == 3


@pytest.mark.parametrize("field,value", [("identity", "mismatch"), ("finish_reason", "length"),
                                         ("status", "failed")])
async def test_mismatched_or_incomplete_cache_metadata_is_recomputed(tmp_path, field, value):
    backend = Backend({"t1": [response("t1")] * 2})
    judge = scheduler(backend, settings(eval_judge_cache_dir=tmp_path))
    await judge.run([task()], identity={})
    path = next(tmp_path.glob("*.json"))
    cached = json.loads(path.read_text())
    (cached if field == "identity" else cached["telemetry"])[field] = value
    path.write_text(json.dumps(cached))
    result = (await judge.run([task()], identity={}))[0]
    assert "body" in result and not result["telemetry"]["cache_hit"]
    assert backend.calls["t1"] == 2


async def test_cache_write_error_does_not_discard_valid_provider_result(tmp_path):
    directory = tmp_path / "file"
    directory.write_text("not a directory")
    backend = Backend({"t1": [response("t1")]})
    result = (await scheduler(backend, settings(eval_judge_cache_dir=directory)).run([task()], identity={}))[0]
    assert result["body"]["result"] == "PASS"
    assert result["telemetry"]["cache_write_error"] in {"FileExistsError", "NotADirectoryError"}


async def test_cancelled_run_closes_clients_streams_and_all_workers():
    backend = Backend({"t1": [response("t1", delay=1)], "t2": [response("t2", delay=1)]})
    judge = scheduler(backend)
    pending = asyncio.create_task(judge.run([task(), task("t2")], identity={}))
    await asyncio.sleep(0.01)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert backend.active == 0
    assert backend.clients_closed == backend.streams_closed == 2


async def test_duplicate_task_ids_rejected_before_requests_and_empty_stage_is_valid():
    backend = Backend({})
    judge = scheduler(backend)
    with pytest.raises(ValueError, match="unique"):
        await judge.run([task(), task()], identity={})
    assert await judge.run([], identity={}) == []
    assert not backend.requests
