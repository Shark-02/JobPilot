#!/usr/bin/env python3
"""Read-only A1 parallel Judge proof-of-concept; no Agent, DB, or repo mutation."""
from __future__ import annotations

import asyncio
import contextvars
import os
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import app.evals.judge as judge_module
from app.config import get_settings
from app.evals.compiler import compile_case
from app.evals.datasets import DEFAULT_PACKAGE_ROOT, load_dataset
from app.evals.evidence import EvidencePack
from app.evals.judge import StructuredJudge, answer_spans, validate_response

RUN = Path("../jobpilot_eval_v0.1.1/runs/20261009T070204Z_e428692_6ff74de7")
OUT = Path("../artifacts/judge_replays/A1_parallel_poc") / datetime.now().strftime("%Y%m%dT%H%M%S_%f")
MAX_OUTPUT = int(os.environ.get("JUDGE_PROBE_MAX_OUTPUT", "6144"))           # Prove completion with less than planned 8192 budget
PRODUCTION_OUTPUT = 8192
TIMEOUT = int(os.environ.get("JUDGE_PROBE_TIMEOUT", "600"))
MAX_CONCURRENT = int(os.environ.get("JUDGE_PROBE_CONCURRENCY", "3"))
MAX_SHARD_CHARS = 440
MAX_SHARD_SPANS = 6

# Keep original judge instructions for extraction calls. For check-only calls,
# remove the unconditional extraction instruction in the *outgoing request only*.
# This makes the split execution contract coherent without editing app/evals.
EXTRACTION_SENTENCE = (
    "Extract ALL material quantitative claims, including claims not listed in Gold, "
    "with project, metric, mechanism and supporting evidence. "
    "Use claim_id=null for other claims. "
)
OriginalClient = judge_module.AsyncOpenAI
SELECTED = set(filter(None, (s.strip() for s in os.environ.get("JUDGE_PROBE_ONLY", "").split(","))))
ACTIVE_JOB = contextvars.ContextVar("judge_probe_job", default="unknown")


def record_transport_issue(stage: str, exc: Exception, **extra: Any) -> None:
    chain = []
    cur: BaseException | None = exc
    seen = set()
    while cur is not None and id(cur) not in seen and len(chain) < 5:
        seen.add(id(cur))
        chain.append({"exception_type": type(cur).__name__, "message": str(cur)[:1200]})
        cur = cur.__cause__ or cur.__context__
    event = {"job": ACTIVE_JOB.get(), "stage": stage, "exceptions": chain, **extra}
    with (OUT / "transport_diagnostics.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


class DiagnosedStream:
    def __init__(self, real: Any):
        self.real = real

    def __aiter__(self):
        return self.iterate()

    async def iterate(self):
        chunks = 0
        start = perf_counter()
        try:
            async for chunk in self.real:
                chunks += 1
                yield chunk
        except Exception as exc:
            record_transport_issue("stream_read", exc, chunks=chunks,
                                   elapsed_seconds=round(perf_counter() - start, 3))
            raise

    async def close(self):
        await self.real.close()


class ScopedClient:
    def __init__(self, *args: Any, **kwargs: Any):
        self.real = OriginalClient(*args, **kwargs)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs: Any):
        messages = [dict(message) for message in kwargs["messages"]]
        prompt = messages[0]["content"]
        if '"expected_extractions": []' in prompt:
            if EXTRACTION_SENTENCE not in prompt:
                raise RuntimeError("Prompt changed: cannot safely scope check-only request")
            messages[0]["content"] = prompt.replace(EXTRACTION_SENTENCE, "", 1)
        kwargs["messages"] = messages
        try:
            stream = await self.real.chat.completions.create(**kwargs)
        except Exception as exc:
            record_transport_issue("create", exc)
            raise
        return DiagnosedStream(stream) if kwargs.get("stream") else stream

    async def close(self):
        await self.real.close()


def shard_spans(spans: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    shards: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    chars = 0
    for span in spans:
        length = len(span["text"])
        if current and (chars + length > MAX_SHARD_CHARS or len(current) >= MAX_SHARD_SPANS):
            shards.append(current)
            current = []
            chars = 0
        current.append(span)
        chars += length
    if current:
        shards.append(current)
    assert len({s["id"] for shard in shards for s in shard}) == len(spans)
    assert [s["id"] for shard in shards for s in shard] == [s["id"] for s in spans]
    return shards


@dataclass
class Job:
    name: str
    plan: Any
    payload: dict[str, Any]


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def make_payload(case: Any, evidence: list[dict[str, Any]], answer: str,
                 spans: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "task": case.prompt,
        "source_policy": case.source_policy.model_dump(mode="json"),
        "final_answer": answer,
        "answer_spans": spans,
        "final_answers_by_step": {},
        "evidence": evidence,
        "memories": [],
        "memory_before": [],
        "memory_events": [],
        "tool_events": [],
        "fixture_source_refs": {},
        "step_turn_ids": {},
        "step_source_refs": {},
    }


async def main() -> int:
    assert RUN.is_dir(), f"Missing artifact: {RUN}"
    case = load_dataset(DEFAULT_PACKAGE_ROOT).cases["A1"]
    plan = compile_case(case)
    rows = [json.loads(line) for line in (RUN / "case_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    original = next(row for row in rows if row["case_id"] == "A1")
    assert original["runtime"]["run_failed"] is False
    observed = original["observable"]
    answer = observed["final_answer"]
    evidence = EvidencePack.model_validate(observed["evidence"]).judge_items()
    spans = answer_spans(answer)
    assert all(x["visible_to_agent"] and x["authoritative_for_grounding"] for x in evidence)
    assert len(plan.semantic_checks) == 10 and len(plan.extractions) == 1
    assert plan.extractions[0].kind == "material_claims"

    groups = {
        "grounding": [c for c in plan.semantic_checks if c.id == "answer_grounding"],
        "facts": [c for c in plan.semantic_checks if c.id.startswith("fact.") or c.id == "attribution.jobpilot"],
        "policy": [c for c in plan.semantic_checks if c.id.startswith("prohibition.") or c.id == "boundary.eval_state"],
    }
    all_ids = [x.id for group in groups.values() for x in group]
    assert len(all_ids) == 10 and len(set(all_ids)) == 10
    assert set(all_ids) == {c.id for c in plan.semantic_checks}
    full_payload = make_payload(case, evidence, answer, spans)
    jobs: list[Job] = []
    for name, checks in groups.items():
        subplan = plan.model_copy(update={"semantic_checks": checks, "extractions": []})
        jobs.append(Job(name, subplan, full_payload))

    shards = shard_spans(spans)
    for index, shard in enumerate(shards, 1):
        subplan = plan.model_copy(update={"semantic_checks": []})
        # Use the original gold and authoritative evidence, but score only observed shard text.
        fragment = "\n".join(item["text"] for item in shard)
        payload = make_payload(case, evidence, fragment, shard)
        jobs.append(Job(f"claims_{index:02d}", subplan, payload))

    available = {j.name for j in jobs}
    if SELECTED:
        missing = SELECTED - available
        assert not missing, f"Unknown jobs: {sorted(missing)}; available: {sorted(available)}"
        jobs = [j for j in jobs if j.name in SELECTED]
    full_suite = len(jobs) == len(groups) + len(shards)
    assert 1 <= MAX_CONCURRENT <= 3, "For the diagnostic, use concurrency 1-3"
    settings = get_settings().model_copy(update={
        "eval_judge_max_output_tokens": MAX_OUTPUT,
        "eval_judge_timeout_seconds": TIMEOUT,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / "manifest.json", {
        "run": str(RUN), "case": "A1", "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "answer_chars": len(answer), "evidence_count": len(evidence), "span_count": len(spans),
        "output_limit_per_request": MAX_OUTPUT, "planned_production_limit": PRODUCTION_OUTPUT,
        "concurrency": MAX_CONCURRENT, "full_suite": full_suite, "jobs": [
            {"name": job.name, "check_ids": [c.id for c in job.plan.semantic_checks],
             "extraction_ids": [e.id for e in job.plan.extractions],
             "span_ids": [s["id"] for s in job.payload["answer_spans"]]}
            for job in jobs
        ]
    })

    print(f"A1 answer={len(answer)} chars, evidence={len(evidence)}, spans={len(spans)}", flush=True)
    print(f"Jobs={len(jobs)}, claim shards={len(shards)}, concurrent={MAX_CONCURRENT}", flush=True)
    print(f"Per-job max_tokens={MAX_OUTPUT}, target production={PRODUCTION_OUTPUT} (+{PRODUCTION_OUTPUT-MAX_OUTPUT} nominal)", flush=True)
    for job in jobs:
        print(f"  {job.name:12s} checks={len(job.plan.semantic_checks)} extractions={len(job.plan.extractions)} spans={len(job.payload['answer_spans'])} answer_chars={len(job.payload['final_answer'])}", flush=True)

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    judge_module.AsyncOpenAI = ScopedClient

    async def invoke(job: Job) -> tuple[Job, dict[str, Any]]:
        async with sem:
            started = perf_counter()
            print(f"START {job.name}", flush=True)
            token = ACTIVE_JOB.set(job.name)
            try:
                result = await StructuredJudge(settings).judge(plan=job.plan, payload=job.payload)
            except Exception as exc:
                record_transport_issue("invoke_unhandled", exc)
                result = {"execution_error": {"code": "PROBE_EXCEPTION", "detail": repr(exc)}}
            finally:
                ACTIVE_JOB.reset(token)
            dump(OUT / f"{job.name}.json", result)
            telemetry = result.get("telemetry", {})
            print(f"DONE  {job.name:12s} elapsed={perf_counter()-started:.1f}s "
                  f"finish={telemetry.get('finish_reason')} input={telemetry.get('input_token_estimate')} "
                  f"reason_chars={telemetry.get('reasoning_chars')} content_chars={telemetry.get('content_chars')} "
                  f"error={result.get('execution_error')}", flush=True)
            return job, result

    try:
        start = perf_counter()
        results = await asyncio.gather(*(invoke(job) for job in jobs))
        elapsed = perf_counter() - start
    finally:
        judge_module.AsyncOpenAI = OriginalClient

    failures: list[str] = []
    returned_checks: list[dict[str, Any]] = []
    all_claims: list[dict[str, Any]] = []
    for job, result in results:
        t = result.get("telemetry", {})
        if result.get("execution_error") or t.get("finish_reason") != "stop":
            failures.append(job.name + ": execution did not finish cleanly")
            continue
        if len(result.get("checks", [])) != len(job.plan.semantic_checks):
            failures.append(job.name + ": missing checks")
            continue
        if len(result.get("extractions", [])) != len(job.plan.extractions):
            failures.append(job.name + ": missing extractions")
            continue
        returned_checks.extend(result["checks"])
        for ext in result["extractions"]:
            if ext["id"] != "material_claims":
                failures.append(job.name + ": unexpected extraction ID")
            else:
                all_claims.extend(ext["items"])

    # Deduplicate only identical claims emitted more than once, never remove conflicting claims.
    unique: dict[str, dict[str, Any]] = {}
    for item in all_claims:
        key = json.dumps([item.get("answer_span_id"), item.get("project_id"), item.get("metric"),
                          item.get("values"), item.get("mechanism")], ensure_ascii=False, sort_keys=True)
        if key in unique and item != unique[key]:
            failures.append("Conflicting duplicate material claim: " + key[:160])
        unique[key] = item
    all_claims = list(unique.values())

    assembled = {"checks": returned_checks, "extractions": [{
        "id": "material_claims", "kind": "material_claims", "items": all_claims,
    }]}
    dump(OUT / "assembled.json", assembled)
    if not failures and full_suite:
        try:
            validate_response(json.dumps(assembled, ensure_ascii=False), plan=plan, payload=full_payload)
        except Exception as exc:
            failures.append("Full original typed contract validation failed: " + repr(exc))

    numeric_spans = [s for s in spans if re.search(r"\d|[一二三四五六七八九十][倍态层]", s["text"])]
    claim_span_ids = {item.get("answer_span_id") for item in all_claims}
    to_review = [s for s in numeric_spans if s["id"] not in claim_span_ids]
    dump(OUT / "quality_review.json", {
        "numeric_spans_without_extracted_claim": to_review,
        "claims": all_claims,
        "note": "Numeric spans may contain nonmaterial dates/ordinals; manual semantic coverage review is required."
    })
    per_job = [{"name": job.name, "finish_reason": result.get("telemetry", {}).get("finish_reason"),
                "input_tokens": result.get("telemetry", {}).get("input_token_estimate"),
                "reasoning_chars": result.get("telemetry", {}).get("reasoning_chars"),
                "content_chars": result.get("telemetry", {}).get("content_chars"),
                "elapsed_seconds": result.get("telemetry", {}).get("total_latency_seconds"),
                "error": result.get("execution_error")} for job, result in results]
    summary = {"success": not failures, "full_suite": full_suite,
               "full_contract_validated": full_suite and not failures,
               "transport_diagnostics": str(OUT / "transport_diagnostics.jsonl"),
               "elapsed_seconds": round(elapsed, 2),
               "jobs": per_job,
               "subtasks": len(jobs), "claim_shards": len(shards), "claims": len(all_claims),
               "checks": len(returned_checks), "numeric_spans_to_review": len(to_review),
               "nominal_budget_margin_tokens": PRODUCTION_OUTPUT - MAX_OUTPUT,
               "failures": failures}
    dump(OUT / "summary.json", summary)
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Artifacts:", OUT.resolve())
    if full_suite:
        print("IMPORTANT: Full typed contract checked if success. Audit quality_review.json for missed/unsupported claims.")
    else:
        print("IMPORTANT: Partial probe only; success does NOT mean full typed contract or claim coverage was checked.")
    print("Transport diagnostics (if any):", OUT / "transport_diagnostics.jsonl")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
