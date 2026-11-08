"""Bounded, recoverable streaming execution for independent Judge subtasks.

The scheduler has no evaluation policy: each task supplies its own strict parser.
Only a complete, validated response is successful or eligible for recovery.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, cast

from openai import AsyncOpenAI

from app.infra.provider_errors import normalize_provider_error

SCHEDULER_VERSION = "parallel_judge_scheduler_1"
_RETRYABLE_CODES = frozenset({
    "PROVIDER_CONNECTION_ERROR", "PROVIDER_HTTP_429", "PROVIDER_HTTP_500",
    "PROVIDER_HTTP_502", "PROVIDER_HTTP_503", "PROVIDER_HTTP_504",
})


@dataclass(frozen=True)
class JudgeTask:
    id: str
    kind: str
    stage: str
    messages: list[dict[str, Any]]
    validate: Callable[[str], dict[str, Any]]
    schema_version: str = "parallel_judge_task_1"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


class JudgeScheduler:
    def __init__(self, settings: Any, *, model: str | None, base_url: str | None,
                 api_key: str | None, counter: Any, client_factory: Any = None) -> None:
        self.settings = settings
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.counter = counter
        self.client_factory = client_factory or AsyncOpenAI
        self.concurrency = settings.eval_judge_concurrency
        # Shared across run() calls, so dependent pipeline branches share one limit.
        self._permits = asyncio.Semaphore(self.concurrency)

    def _identity(self, task: JudgeTask, identity: dict[str, Any]) -> str:
        value = {
            "scheduler_version": SCHEDULER_VERSION, "identity": identity,
            "task_id": task.id, "kind": task.kind, "stage": task.stage,
            "schema_version": task.schema_version, "messages": task.messages,
            "model": self.model, "base_url": self.base_url,
            "settings": {
                "temperature": self.settings.eval_judge_temperature,
                "max_output_tokens": self.settings.eval_judge_max_output_tokens,
                "timeout_seconds": self.settings.eval_judge_timeout_seconds,
            },
        }
        return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()

    def _telemetry(self, task: JudgeTask) -> dict[str, Any]:
        return {
            "task_id": task.id, "kind": task.kind, "stage": task.stage,
            "status": "pending", "input_token_estimate": 0,
            "planned_input_token_estimate": self.counter.count(_canonical(task.messages)),
            "first_chunk_latency_seconds": None, "first_reasoning_latency_seconds": None,
            "first_content_latency_seconds": None, "queue_wait_seconds": 0.0,
            "reasoning_chunks": 0, "reasoning_chars": 0, "content_chunks": 0,
            "content_chars": 0, "content_token_estimate": 0,
            "provider_input_tokens": None, "provider_output_tokens": None,
            "provider_usage_complete": False, "total_latency_seconds": 0.0,
            "http_status": None, "provider_error": None, "finish_reason": None,
            "max_retries": self.settings.eval_judge_transport_retries,
            "max_output_tokens": self.settings.eval_judge_max_output_tokens,
            "stream": True, "retry_count": 0, "attempt_count": 0, "attempts": [],
            "cache_hit": False, "recovered_result": False, "error_type": None,
        }

    def _cache_path(self, digest: str) -> Path | None:
        directory = self.settings.eval_judge_cache_dir
        return Path(directory) / f"{digest}.json" if directory else None

    def _recover(self, task: JudgeTask, digest: str, telemetry: dict[str, Any]) -> dict | None:
        path = self._cache_path(digest)
        if path is None:
            return None
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if (cached["identity"] != digest or cached["task_id"] != task.id
                    or cached["scheduler_version"] != SCHEDULER_VERSION
                    or cached["telemetry"]["finish_reason"] != "stop"
                    or cached["telemetry"]["status"] != "success"):
                return None
            # Reapply today's task-specific contract, never trust serialized body alone.
            body = task.validate(cached["raw"])
            if not isinstance(body, dict):
                return None
            _canonical(body)
            telemetry.update(status="success", cache_hit=True, recovered_result=True,
                             finish_reason="stop", cached_historical_telemetry=cached["telemetry"])
            return body
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _save(self, task: JudgeTask, digest: str, raw: str, telemetry: dict[str, Any]) -> None:
        path = self._cache_path(digest)
        if path is None:
            return
        temporary: str | None = None
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            # The cache can contain career claims and is intentionally user-private.
            os.chmod(path.parent, 0o700)
            descriptor, temporary = tempfile.mkstemp(prefix=".judge-", suffix=".tmp", dir=path.parent)
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(_canonical({"scheduler_version": SCHEDULER_VERSION,
                                         "identity": digest, "task_id": task.id,
                                         "raw": raw, "telemetry": telemetry}))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            temporary = None
        except OSError as exc:
            # Recovery storage is optional; an already validated result stays usable.
            telemetry["cache_write_error"] = type(exc).__name__
        finally:
            if temporary is not None:
                with suppress(OSError):
                    os.unlink(temporary)

    async def run(self, tasks: list[JudgeTask], *, identity: dict[str, Any]) -> list[dict[str, Any]]:
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("Judge subtask IDs must be unique within a run")
        if not tasks:
            return []
        results: list[dict[str, Any] | None] = [None] * len(tasks)
        queue: asyncio.Queue[tuple[int, JudgeTask]] = asyncio.Queue()
        for index, task in enumerate(tasks):
            queue.put_nowait((index, task))

        async def worker() -> None:
            while not queue.empty():
                index, task = queue.get_nowait()
                try:
                    results[index] = await self._run_one(task, identity)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(min(self.concurrency, len(tasks)))]
        try:
            await asyncio.gather(*workers)
        finally:
            for worker_task in workers:
                if not worker_task.done():
                    worker_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        # Input order is deterministic even when streamed requests finish out of order.
        if any(item is None for item in results):
            raise RuntimeError("Judge scheduler left a subtask without an execution result")
        return cast(list[dict[str, Any]], results)

    async def _attempt(self, task: JudgeTask, sequence: int) -> tuple[str, dict, dict | None]:
        telemetry = self._telemetry(task)
        # Attempt telemetry deliberately has no nested attempt list.
        for key in ("attempts", "planned_input_token_estimate"):
            telemetry.pop(key)
        telemetry["sequence"] = sequence
        raw_parts: list[str] = []
        client = stream = None
        error = None
        queued = perf_counter()
        async with self._permits:
            telemetry["queue_wait_seconds"] = perf_counter() - queued
            started = perf_counter()
            telemetry["input_token_estimate"] = self.counter.count(_canonical(task.messages))
            try:
                client = self.client_factory(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    max_retries=0,
                    timeout=self.settings.eval_judge_timeout_seconds,
                )

                async with asyncio.timeout(self.settings.eval_judge_timeout_seconds):
                    request_kwargs = {
                        "model": self.model,
                        "temperature": self.settings.eval_judge_temperature,
                        "messages": task.messages,
                        "stream": True,
                        "max_tokens": self.settings.eval_judge_max_output_tokens,
                    }

                    if self.settings.eval_judge_reasoning_effort is not None:
                        request_kwargs["reasoning_effort"] = (
                            self.settings.eval_judge_reasoning_effort
                        )

                    stream = await client.chat.completions.create(**request_kwargs)

                    async for chunk in stream:
                        elapsed = perf_counter() - started
                        if telemetry["first_chunk_latency_seconds"] is None:
                            telemetry["first_chunk_latency_seconds"] = elapsed

                        usage = getattr(chunk, "usage", None)
                        if usage is not None:
                            telemetry["provider_input_tokens"] = getattr(
                                usage, "prompt_tokens", None
                            )
                            telemetry["provider_output_tokens"] = getattr(
                                usage, "completion_tokens", None
                            )

                        choices = getattr(chunk, "choices", None)
                        if not choices:
                            continue
                        choice = choices[0]
                        if choice.finish_reason is not None:
                            telemetry["finish_reason"] = choice.finish_reason
                        delta = choice.delta
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            telemetry["reasoning_chunks"] += 1
                            telemetry["reasoning_chars"] += len(reasoning)
                            if telemetry["first_reasoning_latency_seconds"] is None:
                                telemetry["first_reasoning_latency_seconds"] = elapsed
                        content = getattr(delta, "content", None)
                        if content:
                            telemetry["content_chunks"] += 1
                            telemetry["content_chars"] += len(content)
                            if telemetry["first_content_latency_seconds"] is None:
                                telemetry["first_content_latency_seconds"] = elapsed
                            raw_parts.append(content)
            except Exception as exc:
                code = normalize_provider_error(exc, timeout_code="JUDGE_TIMEOUT") or "JUDGE_EVALUATOR_ERROR"
                telemetry.update(http_status=getattr(exc, "status_code", None), provider_error=code,
                                 exception_type=type(exc).__name__)
                error = {"code": code, "exception_type": type(exc).__name__}
            finally:
                if stream is not None:
                    await self._close(stream, telemetry)
                if client is not None:
                    await self._close(client, telemetry)
                telemetry["total_latency_seconds"] = perf_counter() - started
                telemetry["content_token_estimate"] = self.counter.count("".join(raw_parts))
        if error is None and telemetry["finish_reason"] != "stop":
            error = {"code": ("JUDGE_OUTPUT_TRUNCATED" if telemetry["finish_reason"] == "length"
                              else "JUDGE_INCOMPLETE_OUTPUT")}
        telemetry["provider_usage_complete"] = all(telemetry[key] is not None for key in
                                                    ("provider_input_tokens", "provider_output_tokens"))
        telemetry["status"] = "failed" if error else "complete"
        telemetry["error_type"] = self._error_type(error)
        return "".join(raw_parts), telemetry, error

    async def _close(self, resource: Any, telemetry: dict[str, Any]) -> None:
        try:
            # A hung network close must not defeat the attempt's bounded lifetime.
            async with asyncio.timeout(min(5.0, self.settings.eval_judge_timeout_seconds)):
                await resource.close()
        except Exception as exc:
            telemetry.setdefault("cleanup_errors", []).append(type(exc).__name__)

    @staticmethod
    def _error_type(error: dict | None) -> str | None:
        if error is None:
            return None
        code = error["code"]
        if code == "JUDGE_TIMEOUT":
            return "timeout"
        if code == "JUDGE_OUTPUT_TRUNCATED":
            return "output_truncated"
        if code == "JUDGE_INCOMPLETE_OUTPUT":
            return "incomplete_output"
        if code == "JUDGE_SCHEMA_ERROR":
            return "schema"
        if code == "JUDGE_COVERAGE_ERROR":
            return "coverage"
        if code.startswith("PROVIDER_"):
            return "provider"
        return "internal"

    async def _run_one(self, task: JudgeTask, identity: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        telemetry = self._telemetry(task)
        digest = self._identity(task, identity)
        telemetry["input_identity"] = digest
        result: dict[str, Any] = {"task_id": task.id, "telemetry": telemetry}
        recovered = self._recover(task, digest, telemetry)
        if recovered is not None:
            result["body"] = recovered
            telemetry["total_latency_seconds"] = perf_counter() - started
            return result
        if not self.model or not self.base_url or not self.api_key:
            result["execution_error"] = {"code": "JUDGE_NOT_CONFIGURED"}
            telemetry.update(status="failed", error_type="configuration",
                             total_latency_seconds=perf_counter() - started)
            return result
        last_raw = ""
        for sequence in range(1, self.settings.eval_judge_transport_retries + 2):
            last_raw, attempt, error = await self._attempt(task, sequence)
            if error is None:
                try:
                    body = task.validate(last_raw)
                    if not isinstance(body, dict):
                        raise ValueError("Subtask parser must return a JSON object")
                    _canonical(body)
                    result["body"] = body
                    attempt["status"] = "success"
                except (ValueError, TypeError) as exc:
                    error = {"code": getattr(exc, "code", "JUDGE_SCHEMA_ERROR"), "detail": str(exc)}
                    attempt.update(status="failed", error_type=self._error_type(error))
                except Exception as exc:
                    error = {"code": "JUDGE_EVALUATOR_ERROR", "exception_type": type(exc).__name__}
                    attempt.update(status="failed", error_type="internal")
            telemetry["attempts"].append(attempt)
            if error is None:
                break
            if error["code"] not in _RETRYABLE_CODES or sequence > self.settings.eval_judge_transport_retries:
                result["execution_error"] = error
                break
            await asyncio.sleep(self.settings.eval_judge_retry_backoff_seconds * 2 ** (sequence - 1))
        attempts = telemetry["attempts"]
        telemetry.update(status="failed" if "execution_error" in result else "success",
                         attempt_count=len(attempts), retry_count=max(0, len(attempts) - 1),
                         total_latency_seconds=perf_counter() - started,
                         error_type=self._error_type(result.get("execution_error")))
        for key in ("input_token_estimate", "content_token_estimate", "reasoning_chunks",
                    "reasoning_chars", "content_chunks", "content_chars", "queue_wait_seconds"):
            telemetry[key] = sum(item[key] for item in attempts)
        for key in ("first_chunk_latency_seconds", "first_reasoning_latency_seconds", "first_content_latency_seconds"):
            telemetry[key] = next((item[key] for item in attempts if item[key] is not None), None)
        for key in ("provider_input_tokens", "provider_output_tokens"):
            values = [item[key] for item in attempts if item[key] is not None]
            telemetry[key] = sum(values) if len(values) == len(attempts) else None
            telemetry[key.replace("provider_", "provider_reported_")] = sum(values) if values else None
        telemetry["provider_usage_complete"] = all(item["provider_usage_complete"] for item in attempts)
        for key in ("finish_reason", "http_status", "provider_error"):
            telemetry[key] = attempts[-1][key]
        if "body" in result:
            self._save(task, digest, last_raw, telemetry)
        return result


# Kept short for callers that describe this dependency as the task Scheduler.
Scheduler = JudgeScheduler
