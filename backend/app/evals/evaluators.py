from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any, Iterable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentObservation, AgentTurn, Message, ModelCallTrace, ToolCallTrace, TurnStatus
from app.evals.scope import apply_eval_scope
from app.infra.provider_errors import ProviderFailureKind, classify_provider_error_code
from app.infra.token_count import TokenCounter
from app.knowledge.sources import SourceRef, SourceResolver


class EvalOutcome(str, Enum):
    PASS = "PASS"
    SCORED = "SCORED"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class EvalClassification:
    outcome: EvalOutcome
    reason: str | None = None
    source: str | None = None


def case_result(classification: EvalClassification, **details: Any) -> dict[str, Any]:
    """Create the one case-result representation used by every eval suite."""

    result: dict[str, Any] = {"outcome": classification.outcome.value}
    if classification.reason:
        result["reason"] = classification.reason
    if classification.source:
        result["source"] = classification.source
    result.update(details)
    return result


class EvalFailureClassifier:
    """Attribute every non-PASS result without mixing legacy result fields."""

    _TURN_FAILURE_REASONS = {
        "TURN_LIMIT": "AGENT_TURN_LIMIT",
        "TOOL_LIMIT": "AGENT_TOOL_LIMIT",
        "CONTEXT_OVERFLOW": "AGENT_CONTEXT_OVERFLOW",
        "DEADLINE_EXCEEDED": "AGENT_TOTAL_DEADLINE",
    }
    _TOOL_FAILURE_REASONS = {
        "VALIDATION_ERROR": "AGENT_TOOL_ARGUMENT_ERROR",
        "POSSIBLE_DUPLICATE": "AGENT_DUPLICATE_TOOL_CALL",
        "NOT_FOUND": "AGENT_INVALID_SOURCE_REF",
    }

    @classmethod
    def agent_interruption(cls, data: dict[str, Any]) -> EvalClassification | None:
        """Return non-capability interruptions recorded by the Agent run."""

        model_codes = [
            getattr(item, "error_code", None)
            for item in data.get("model_calls", [])
            if getattr(item, "status", None) == "FAILED"
        ]
        turn_codes = [getattr(item, "error_code", None) for item in data.get("turns", [])]
        tool_codes = [item.get("provider_error_code") for item in data.get("tool_events", [])
                      if item.get("execution_kind") == "real" and not item.get("result", {}).get("ok")]
        for error_code in [*model_codes, *turn_codes, *tool_codes]:
            provider_kind = classify_provider_error_code(error_code)
            if provider_kind:
                return cls._provider_interruption(provider_kind, source="MAIN_AGENT")
        if "PROVIDER_NOT_CONFIGURED" in turn_codes:
            return EvalClassification(EvalOutcome.ERROR, "EVAL_PROVIDER_CONFIGURATION_ERROR", "EVAL_HARNESS")
        return None

    @classmethod
    def judge_interruption(cls, result: dict[str, Any]) -> EvalClassification | None:
        """Separate external Judge interruption from evaluator contract errors."""

        error = result.get("execution_error")
        if not error:
            return None
        error_code = error.get("code")
        provider_kind = classify_provider_error_code(error_code)
        if provider_kind:
            return cls._provider_interruption(provider_kind, source="JUDGE")
        if error_code == "JUDGE_NOT_CONFIGURED":
            return EvalClassification(EvalOutcome.ERROR, "EVAL_JUDGE_CONFIGURATION_ERROR", "EVAL_HARNESS")
        if error_code == "JUDGE_SCHEMA_ERROR":
            return EvalClassification(EvalOutcome.ERROR, "EVAL_JUDGE_SCHEMA_ERROR", "EVAL_HARNESS")
        if error_code in {"JUDGE_OUTPUT_TRUNCATED", "JUDGE_INCOMPLETE_OUTPUT", "JUDGE_COVERAGE_ERROR"}:
            return EvalClassification(EvalOutcome.ERROR, f"EVAL_{error_code}", "EVAL_HARNESS")
        return EvalClassification(EvalOutcome.ERROR, "EVAL_JUDGE_ERROR", "EVAL_HARNESS")

    @classmethod
    def agent_failure(cls, data: dict[str, Any], *, fallback: str = "AGENT_TASK_FAILED") -> EvalClassification:
        """Classify a completed evaluation opportunity that did not pass."""

        turn_codes = [getattr(item, "error_code", None) for item in data.get("turns", [])]
        for error_code in turn_codes:
            reason = cls._TURN_FAILURE_REASONS.get(error_code)
            if reason:
                return EvalClassification(EvalOutcome.FAIL, reason, "MAIN_AGENT")
        tool_codes = [
            getattr(item, "error_code", None)
            for item in data.get("tool_calls", [])
            if getattr(item, "status", None) == "FAILED"
        ]
        for error_code in tool_codes:
            reason = cls._TOOL_FAILURE_REASONS.get(error_code)
            if reason:
                return EvalClassification(EvalOutcome.FAIL, reason, "MAIN_AGENT")
        return EvalClassification(EvalOutcome.FAIL, fallback, "MAIN_AGENT")

    @staticmethod
    def pass_or_fail(passed: bool, *, failure_reason: str, source: str = "MAIN_AGENT") -> EvalClassification:
        if passed:
            return EvalClassification(EvalOutcome.PASS)
        return EvalClassification(EvalOutcome.FAIL, failure_reason, source)

    @staticmethod
    def harness_error(reason: str = "EVAL_RUNNER_ERROR") -> EvalClassification:
        return EvalClassification(EvalOutcome.ERROR, reason, "EVAL_HARNESS")

    @staticmethod
    def _provider_interruption(kind: ProviderFailureKind, *, source: str) -> EvalClassification:
        prefix = "MODEL_PROVIDER" if source == "MAIN_AGENT" else "JUDGE_PROVIDER"
        reason_suffix = {
            ProviderFailureKind.TIMEOUT: "TIMEOUT",
            ProviderFailureKind.BILLING: "BILLING",
            ProviderFailureKind.RATE_LIMITED: "RATE_LIMITED",
            ProviderFailureKind.UNAVAILABLE: "UNAVAILABLE",
        }[kind]
        return EvalClassification(EvalOutcome.BLOCKED, f"{prefix}_{reason_suffix}", source)


async def collect_turn_data(session: AsyncSession, turn_ids: Iterable[UUID]) -> dict[str, Any]:
    ids = list(turn_ids)
    if not ids:
        return {"turns": [], "tool_calls": [], "model_calls": [], "observations": [], "final_answers": []}
    turns = list((await session.scalars(
        apply_eval_scope(select(AgentTurn).where(AgentTurn.id.in_(ids)).order_by(AgentTurn.started_at, AgentTurn.id), AgentTurn)
    )).all())
    tool_calls = list((await session.scalars(
        apply_eval_scope(select(ToolCallTrace).where(ToolCallTrace.turn_id.in_(ids)).order_by(ToolCallTrace.started_at, ToolCallTrace.sequence), ToolCallTrace)
    )).all())
    model_calls = list((await session.scalars(
        apply_eval_scope(select(ModelCallTrace).where(ModelCallTrace.turn_id.in_(ids)).order_by(ModelCallTrace.started_at, ModelCallTrace.sequence), ModelCallTrace)
    )).all())
    observations = list((await session.scalars(
        apply_eval_scope(select(AgentObservation).where(AgentObservation.turn_id.in_(ids)).order_by(AgentObservation.started_at, AgentObservation.sequence), AgentObservation)
    )).all())
    answer_ids = [turn.assistant_message_id for turn in turns if turn.assistant_message_id]
    answers = list((await session.scalars(
        apply_eval_scope(select(Message).where(Message.id.in_(answer_ids)), Message)
    )).all()) if answer_ids else []
    answer_by_id = {answer.id: answer.content for answer in answers}
    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "model_calls": model_calls,
        "observations": observations,
        "final_answers": [answer_by_id.get(turn.assistant_message_id, "") for turn in turns],
    }


async def source_refs_valid(session: AsyncSession, traces: list[ToolCallTrace], counter: TokenCounter,
                            *, visible_refs: Iterable[str] = ()) -> tuple[bool, list[str]]:
    refs = sorted({ref for trace in traces for ref in trace.source_refs} | set(visible_refs))
    if not refs:
        return True, []
    resolver = SourceResolver(session, counter)
    bad: list[str] = []
    for value in refs:
        try:
            await resolver.read([SourceRef(value)], max_tokens=1)
        except Exception:
            bad.append(value)
    return not bad, bad


def runtime_metrics(data: dict[str, Any]) -> dict[str, Any]:
    turns: list[AgentTurn] = data["turns"]
    tools: list[ToolCallTrace] = data["tool_calls"]
    models: list[ModelCallTrace] = data["model_calls"]
    observations: list[AgentObservation] = data["observations"]
    main_models = [item for item in models if item.call_type == "MAIN_AGENT"]
    compactions = [item for item in models if item.call_type == "CONTEXT_COMPACTION"]
    fingerprints = Counter((str(item.turn_id), item.fingerprint) for item in tools)
    duplicate = sum(count - 1 for count in fingerprints.values() if count > 1)
    durations = [
        int((turn.finished_at - turn.started_at).total_seconds() * 1000)
        for turn in turns if turn.finished_at and turn.started_at
    ]
    executions = data.get("tool_events")
    recovery_pairs = []
    if executions is not None:
        for index, event in enumerate(executions):
            if event["result"].get("ok"):
                continue
            success = next((later for later in executions[index + 1:]
                            if later["turn_id"] == event["turn_id"] and later["tool_name"] == event["tool_name"]
                            and later["execution_kind"] == "real" and later["result"].get("ok")), None)
            if success:
                recovery_pairs.append({"failed_call_id": event["call_id"], "successful_call_id": success["call_id"]})
    return {
        "model_call_count": len(main_models),
        "context_compaction_call_count": len(compactions),
        "provider_model_call_count": len(models),
        "tool_call_count": len(tools),
        "retrieval_call_count": sum(item.kind == "RETRIEVAL" for item in observations),
        "model_error_count": sum(item.status == "FAILED" for item in models),
        "tool_error_count": sum(item.status == "FAILED" for item in tools),
        "recovered_error_count": len(recovery_pairs) if executions is not None else None,
        "recovery_lineage": recovery_pairs,
        "exact_duplicate_tool_calls": duplicate,
        "input_tokens": sum(item.input_tokens or 0 for item in models),
        "output_tokens": sum(item.output_tokens or 0 for item in models),
        "context_compaction_input_tokens": sum(item.input_tokens or 0 for item in compactions),
        "context_compaction_output_tokens": sum(item.output_tokens or 0 for item in compactions),
        "total_latency_ms": sum(durations),
        "model_latency_ms": sum(item.latency_ms or 0 for item in models),
        "context_compaction_latency_ms": sum(item.latency_ms or 0 for item in compactions),
        "tool_latency_ms": sum(item.latency_ms or 0 for item in tools),
        "retrieval_latency_ms": sum(item.latency_ms or 0 for item in observations if item.kind == "RETRIEVAL"),
        "run_failed": any(item.status == TurnStatus.FAILED for item in turns),
        "unrecovered_tool_failure": (sum(not event["result"].get("ok") for event in executions) > len(recovery_pairs)) if executions is not None else None,
        "turns": [str(item.id) for item in turns],
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval = [result for result in results if result["suite"] == "retrieval"]
    agent = [result for result in results if result["suite"] == "agent_tasks"]
    memory = [result for result in results if result["suite"] == "memory"]
    runtime = [result for result in results if result["suite"] == "runtime_failure"]
    scored = [item["result"] for item in retrieval if item["result"]["outcome"] == "SCORED"]
    answerable = [item for item in scored if item["answerability"] == "answerable"]
    from app.evals.retrieval_contract import METRIC_NAMES, RETRIEVAL_CONTRACT_VERSION
    latencies = [item["runtime"]["total_latency_ms"] for item in results if item.get("runtime")]
    retrieval_summary = _outcome_summary(retrieval)
    retrieval_summary.update({
        "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
        "answerable_case_count": len(answerable),
        "unanswerable_case_count": len(scored) - len(answerable),
        "metrics": {name: _mean([float(item["metrics"][name]) for item in answerable])
                    for name in METRIC_NAMES},
        "abstention": "NOT_EVALUATED",
    })
    overall = _outcome_summary(results)
    return {
        "case_count": len(results),
        "run_status": _run_status(overall),
        "outcomes": overall,
        "suites": {
            "retrieval": retrieval_summary,
            "agent_tasks": _outcome_summary(agent),
            "memory": _outcome_summary(memory),
            "runtime_failure": _outcome_summary(runtime),
        },
        "runtime": {"median_total_latency_ms": median(latencies) if latencies else None, "p90_total_latency_ms": percentile(latencies, 0.9)},
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _outcome_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [EvalOutcome(item["result"]["outcome"]) for item in items]
    passed = outcomes.count(EvalOutcome.PASS)
    failed = outcomes.count(EvalOutcome.FAIL)
    blocked = outcomes.count(EvalOutcome.BLOCKED)
    errors = outcomes.count(EvalOutcome.ERROR)
    scored = outcomes.count(EvalOutcome.SCORED)
    scorable = passed + failed + scored
    summary = {
        "case_count": len(items), "scorable_case_count": scorable,
        "passed": passed, "failed": failed,
        "blocked_case_count": blocked, "error_case_count": errors,
        "pass_rate": passed / (passed + failed) if passed + failed else None,
    }
    if scored:
        summary["scored"] = scored
    return summary


def _run_status(outcomes: dict[str, Any]) -> str:
    if outcomes["error_case_count"]:
        return "ERROR"
    if outcomes["blocked_case_count"]:
        return "PARTIAL"
    return "COMPLETED"
