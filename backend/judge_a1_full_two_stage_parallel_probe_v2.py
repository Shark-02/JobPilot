#!/usr/bin/env python3
"""Read-only, resumable full-A1 Judge proof: parallel claim extraction, verification, and semantic checks.

Uses the saved successful A1 Agent artifact; no Agent runs, database writes, or app code changes.
Run from JobPilot/backend with PYTHONPATH=. and the eval dependencies installed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from openai import AsyncOpenAI

import app.evals.judge as judge_module
from app.config import get_settings
from app.evals.compiler import compile_case
from app.evals.datasets import DEFAULT_PACKAGE_ROOT, load_dataset
from app.evals.evidence import EvidencePack
from app.evals.judge import StructuredJudge, answer_spans, validate_response
from app.infra.token_count import TokenCounter

RUN = Path("../jobpilot_eval_v0.1.1/runs/20261009T070204Z_e428692_6ff74de7")
OUT = Path(os.getenv("A1_POC_DIR") or (
    "../artifacts/judge_replays/A1_full_parallel_two_stage/" + datetime.now().strftime("%Y%m%dT%H%M%S_%f")
))
CONCURRENCY = int(os.getenv("A1_POC_CONCURRENCY", "2"))
MAX_OUTPUT = int(os.getenv("A1_POC_MAX_OUTPUT", "6144"))
TIMEOUT = int(os.getenv("A1_POC_TIMEOUT", "600"))
VERIFY_BATCH = int(os.getenv("A1_POC_VERIFY_BATCH", "2"))
MAX_SHARD_CHARS = 440
MAX_SHARD_SPANS = 6
assert 1 <= CONCURRENCY <= 3
assert 1 <= MAX_OUTPUT <= 8192
assert 1 <= VERIFY_BATCH <= 3

EXTRACTION_SENTENCE = (
    "Extract ALL material quantitative claims, including claims not listed in Gold, "
    "with project, metric, mechanism and supporting evidence. "
    "Use claim_id=null for other claims. "
)
ORIGINAL_CLIENT = judge_module.AsyncOpenAI

class ScopedJudgeClient:
    """Only for check-only calls: remove unconditional extraction instruction in outgoing request."""
    def __init__(self, *args: Any, **kwargs: Any):
        self.real = ORIGINAL_CLIENT(*args, **kwargs)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs: Any):
        messages = [dict(m) for m in kwargs["messages"]]
        system = messages[0]["content"]
        if '"expected_extractions": []' not in system:
            raise RuntimeError("Expected check-only plan; cannot safely scope system prompt")
        if EXTRACTION_SENTENCE not in system:
            raise RuntimeError("Judge prompt changed: extraction sentence not found")
        messages[0]["content"] = system.replace(EXTRACTION_SENTENCE, "", 1)
        kwargs["messages"] = messages
        return await self.real.chat.completions.create(**kwargs)

    async def close(self):
        await self.real.close()


def dump(path: Path, value: Any):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def strict_json(raw: str) -> Any:
    def pairs(seq):
        obj = {}
        for k, v in seq:
            if k in obj:
                raise ValueError(f"Duplicate JSON key {k}")
            obj[k] = v
        return obj
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid JSON {x}")))


def shards_of(spans):
    shards, current, chars = [], [], 0
    for span in spans:
        if current and (chars + len(span["text"]) > MAX_SHARD_CHARS or len(current) >= MAX_SHARD_SPANS):
            shards.append(current)
            current, chars = [], 0
        current.append(span)
        chars += len(span["text"])
    if current:
        shards.append(current)
    assert [s["id"] for shard in shards for s in shard] == [s["id"] for s in spans]
    return shards


EXTRACT_SYSTEM = (
    "你是职业事实量化主张识别器。本次只负责第一阶段：对传入每条答案 Span 识别全部实质性职业/项目量化主张。"
    "不要核实真伪、不要寻找证据；不要对没有给出的答案内容做推测。"
    "每项记录实际项目或所属事项、指标、原文数值和实际陈述的机制（未陈述则空串）。"
    "与个人职业档案有关的成绩、时间、奖项年份等实际量化事实也应提取；"
    "不要把普通编号、没有数值的指标名称、一般技术常量当成成绩。"
    "一个 Span 里的不同指标必须分别列出，不得遗漏；没有实质量化主张的 Span 不应创造条目。"
    "必须沿用输入中的原始 answer_span_id，不得自创 ID。"
    "只输出严格 JSON：{\"items\":[{\"claim_id\":null,\"project_id\":\"实际项目或领域\","
    "\"metric\":\"指标\",\"values\":[\"原文数值\"],\"mechanism\":\"实际机制或空串\","
    "\"answer_span_id\":\"原始Span ID\"}]}。不需要 evidence_ids 或支持判断。"
)
VERIFY_SYSTEM = (
    "你是职业事实来源支持核验器。只核验传入的少量 Claim：其 project_id、metric、values、mechanism "
    "组成的完整事实关系是否被 Agent 实际可见的权威证据原文支持。"
    "不得把其他项目的数字借用到当前项目；不得因数字单独出现就判 PASS。"
    "只能引用传入的真实 evidence ID；Gold、摘要和背景常识不是证据。"
    "完整关系有直接证据为 PASS；明确冲突为 FAIL；无法核实为 UNKNOWN。"
    "不得增删改写 Claim。只输出严格 JSON：{\"results\":[{\"index\":0,"
    "\"support_result\":\"PASS\",\"evidence_ids\":[\"真实来源ID\"],"
    "\"support_reason\":\"简短理由\"}]}。每个从0开始的 index 恰好出现一次。"
)


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    case = load_dataset(DEFAULT_PACKAGE_ROOT).cases["A1"]
    plan = compile_case(case)
    assert len(plan.semantic_checks) == 10
    assert len(plan.extractions) == 1 and plan.extractions[0].kind == "material_claims"
    assert plan.extractions[0].allowed_ids == [], "PoC supports open-world A1 claim IDs only"
    rows = [json.loads(s) for s in (RUN / "case_results.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    row = next(r for r in rows if r["case_id"] == "A1")
    assert row["runtime"]["run_failed"] is False
    obs = row["observable"]
    answer = obs["final_answer"]
    spans = answer_spans(answer)
    evidence = EvidencePack.model_validate(obs["evidence"]).judge_items()
    assert evidence and all(e["visible_to_agent"] and e["authoritative_for_grounding"] for e in evidence)
    sources = []
    for e in evidence:
        if not isinstance(e.get("content"), str) or not e["content"].strip():
            raise ValueError(f"Evidence missing authoritative original content: {e['id']}")
        sources.append({"id": e["id"], "content": e["content"]})
    source_ids = {e["id"] for e in sources}
    groups = {
        "grounding": [c for c in plan.semantic_checks if c.id == "answer_grounding"],
        "facts": [c for c in plan.semantic_checks if c.id.startswith("fact.") or c.id == "attribution.jobpilot"],
        # Policy checks are isolated here after the combined 4-check task hit the
        # model's output budget. Keep the original Gold statements and Evidence.
        "policy_forbidden_1": [c for c in plan.semantic_checks if c.id == "prohibition.forbidden_1"],
        "policy_forbidden_2": [c for c in plan.semantic_checks if c.id == "prohibition.forbidden_2"],
        "policy_forbidden_3": [c for c in plan.semantic_checks if c.id == "prohibition.forbidden_3"],
        "policy_boundary": [c for c in plan.semantic_checks if c.id == "boundary.eval_state"],
    }
    assert {c.id for g in groups.values() for c in g} == {c.id for c in plan.semantic_checks}
    settings = get_settings().model_copy(update={
        "eval_judge_max_output_tokens": MAX_OUTPUT,
        "eval_judge_timeout_seconds": TIMEOUT,
    })
    base_url = settings.eval_judge_base_url or settings.llm_base_url
    api_key = settings.eval_judge_api_key or settings.llm_api_key
    model = settings.eval_judge_model or settings.llm_model
    assert base_url and api_key and model
    counter = TokenCounter(model)
    sem = asyncio.Semaphore(CONCURRENCY)
    shards = shards_of(spans)
    # Distinguish a resumed run from a new run; forbid reuse if configuration/input changed.
    manifest = {
        "case": "A1", "run": str(RUN), "model": model,
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "evidence_sha256": hashlib.sha256(json.dumps(sources, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "checks": sorted(c.id for c in plan.semantic_checks),
        "output_limit": MAX_OUTPUT, "timeout": TIMEOUT, "verify_batch": VERIFY_BATCH,
        "shard_span_ids": [[s["id"] for s in shard] for shard in shards],
        "protocol": "two_stage_claims_full_A1_v1", "concurrency": CONCURRENCY,
    }
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert {k: v for k, v in old.items() if k != "concurrency"} == {k: v for k, v in manifest.items() if k != "concurrency"}, "Cannot resume changed input/configuration"
    else:
        dump(manifest_path, manifest)
    print(f"A1 all spans={len(spans)} shards={len(shards)} authoritative_evidence={len(evidence)} check_jobs={len(groups)}", flush=True)
    print(f"OUTPUT_LIMIT={MAX_OUTPUT} (nominal margin vs 8192: {8192-MAX_OUTPUT}), concurrency={CONCURRENCY}, verify_batch={VERIFY_BATCH}", flush=True)
    print(f"Artifact directory: {OUT.resolve()}", flush=True)
    telemetry = []
    async with AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=TIMEOUT) as client:
        async def ask(name: str, system: str, data: dict[str, Any]):
            messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]
            signature = hashlib.sha256(json.dumps({"model": model, "messages": messages, "max_tokens": MAX_OUTPUT}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            saved_path = OUT / f"{name}.json"
            if saved_path.exists():
                cached = json.loads(saved_path.read_text(encoding="utf-8"))
                if cached.get("signature") == signature and cached.get("telemetry", {}).get("finish_reason") == "stop":
                    print(f"RESUME {name}", flush=True)
                    telemetry.append(cached["telemetry"])
                    return cached["body"]
            async with sem:
                start = perf_counter()
                input_tokens = counter.count(json.dumps(messages, ensure_ascii=False))
                print(f"START  {name:16s} input~{input_tokens}", flush=True)
                parts, reason_chars, reason_count, first_content, finish = [], 0, 0, None, None
                try:
                    with (OUT / f"{name}_reasoning.txt").open("w", encoding="utf-8") as rf, (OUT / f"{name}_content.txt").open("w", encoding="utf-8") as cf:
                        async with asyncio.timeout(TIMEOUT):
                            stream = await client.chat.completions.create(
                                model=model, temperature=0, messages=messages, stream=True, max_tokens=MAX_OUTPUT)
                            try:
                                async for chunk in stream:
                                    for choice in chunk.choices:
                                        finish = choice.finish_reason or finish
                                        delta = choice.delta
                                        r = getattr(delta, "reasoning_content", None) or ""
                                        if r:
                                            rf.write(r)
                                            reason_chars += len(r)
                                            reason_count += 1
                                        c = delta.content or ""
                                        if c:
                                            if first_content is None:
                                                first_content = perf_counter() - start
                                            cf.write(c)
                                            parts.append(c)
                            finally:
                                await stream.close()
                    raw = "".join(parts)
                    t = {"name": name, "elapsed_s": round(perf_counter() - start, 2),
                         "input_tokens_estimate": input_tokens, "finish_reason": finish,
                         "reasoning_chars": reason_chars, "reasoning_chunks": reason_count,
                         "content_chars": len(raw), "content_tokens_estimate": counter.count(raw),
                         "first_content_s": first_content, "max_output_tokens": MAX_OUTPUT}
                    if finish != "stop":
                        raise RuntimeError(f"finish={finish}; content_chars={len(raw)}")
                    body = strict_json(raw)
                    dump(saved_path, {"signature": signature, "telemetry": t, "body": body})
                    telemetry.append(t)
                    print(f"DONE   {name:16s} elapsed={t['elapsed_s']}s finish=stop reason_chars={reason_chars} content_chars={len(raw)}", flush=True)
                    return body
                except Exception as exc:
                    dump(OUT / f"{name}_error.json", {"type": type(exc).__name__, "message": str(exc)[:1000],
                                                             "elapsed_s": round(perf_counter() - start, 2),
                                                             "finish_reason": finish, "input_tokens_estimate": input_tokens,
                                                             "reasoning_chars": reason_chars, "content_chars": sum(map(len, parts))})
                    print(f"FAILED {name:16s}: {type(exc).__name__}: {exc}", flush=True)
                    raise

        async def extract_job(i: int, shard):
            name = f"extract_{i:02d}"
            obj = await ask(name, EXTRACT_SYSTEM, {"spans": shard})
            if not isinstance(obj, dict) or set(obj) != {"items"} or not isinstance(obj["items"], list):
                raise ValueError(f"{name}: wrong extraction response envelope")
            allowed = {s["id"] for s in shard}
            candidates = obj["items"]
            for c in candidates:
                if not isinstance(c, dict) or set(c) != {"claim_id", "project_id", "metric", "values", "mechanism", "answer_span_id"}:
                    raise ValueError(f"{name}: wrong candidate fields")
                if c["claim_id"] is not None or c["answer_span_id"] not in allowed:
                    raise ValueError(f"{name}: unexpected claim or span id")
                for k in ("project_id", "metric", "mechanism"):
                    if not isinstance(c[k], str):
                        raise ValueError(f"{name}: {k} must be string")
                if not isinstance(c["values"], list) or not c["values"] or not all(isinstance(v, str) and v.strip() for v in c["values"]):
                    raise ValueError(f"{name}: empty/invalid claim values")
            return candidates

        async def verify_job(i: int, batch):
            indexed = [{"index": j, **claim} for j, claim in enumerate(batch)]
            name = f"verify_{i:02d}"
            obj = await ask(name, VERIFY_SYSTEM, {"claims": indexed, "evidence": sources})
            if not isinstance(obj, dict) or set(obj) != {"results"} or not isinstance(obj["results"], list):
                raise ValueError(f"{name}: invalid verifier envelope")
            results = obj["results"]
            if len(results) != len(batch):
                raise ValueError(f"{name}: missing result")
            by_idx = {}
            for r in results:
                if not isinstance(r, dict) or set(r) != {"index", "support_result", "evidence_ids", "support_reason"}:
                    raise ValueError(f"{name}: unexpected result keys")
                idx = r["index"]
                if type(idx) is not int or idx not in range(len(batch)) or idx in by_idx:
                    raise ValueError(f"{name}: unexpected/duplicate claim index")
                if r["support_result"] not in ("PASS", "FAIL", "UNKNOWN"):
                    raise ValueError(f"{name}: invalid support_result")
                if not isinstance(r["evidence_ids"], list) or any(eid not in source_ids for eid in r["evidence_ids"]):
                    raise ValueError(f"{name}: unknown evidence IDs")
                if r["support_result"] == "PASS" and not r["evidence_ids"]:
                    raise ValueError(f"{name}: PASS has no cited source")
                if not isinstance(r["support_reason"], str):
                    raise ValueError(f"{name}: invalid support reason")
                by_idx[idx] = r
            return [{**claim, "evidence_ids": by_idx[j]["evidence_ids"],
                     "support_result": by_idx[j]["support_result"],
                     "support_reason": by_idx[j]["support_reason"]} for j, claim in enumerate(batch)]

        def check_payload():
            return {"task": case.prompt, "source_policy": case.source_policy.model_dump(mode="json"),
                    "final_answer": answer, "answer_spans": spans, "final_answers_by_step": {},
                    "evidence": evidence, "memories": [], "memory_before": [], "memory_events": [],
                    "tool_events": [], "fixture_source_refs": {}, "step_turn_ids": {}, "step_source_refs": {}}

        async def check_job(name: str, checks):
            # Avoid monkeypatching global module while invoking extraction: check jobs run after verification.
            child = plan.model_copy(update={"semantic_checks": checks, "extractions": []})
            payload = check_payload()
            saved = OUT / f"check_{name}.json"
            if saved.exists():
                cached = json.loads(saved.read_text(encoding="utf-8"))
                if cached.get("ok") and cached.get("check_ids") == [c.id for c in checks]:
                    print(f"RESUME check_{name}", flush=True)
                    telemetry.append(cached["telemetry"])
                    return cached["checks"]
            async with sem:
                print(f"START  check_{name:11s} expected={len(checks)}", flush=True)
                start = perf_counter()
                result = await StructuredJudge(settings).judge(plan=child, payload=payload)
                t = result.get("telemetry", {})
                stat = {"name": "check_" + name, "elapsed_s": round(perf_counter() - start, 2),
                        "input_tokens_estimate": t.get("input_token_estimate"),
                        "finish_reason": t.get("finish_reason"), "reasoning_chars": t.get("reasoning_chars"),
                        "content_chars": t.get("content_chars"), "content_tokens_estimate": t.get("content_token_estimate"),
                        "max_output_tokens": MAX_OUTPUT}
                dump(OUT / f"check_{name}_attempt.json", result)
                if result.get("execution_error") or t.get("finish_reason") != "stop" or len(result.get("checks", [])) != len(checks):
                    print(f"FAILED check_{name}: {result.get('execution_error')}, finish={t.get('finish_reason')}", flush=True)
                    raise RuntimeError(f"check_{name} failed: {result.get('execution_error')}, finish={t.get('finish_reason')}")
                dump(saved, {"ok": True, "telemetry": stat, "check_ids": [c.id for c in checks], "checks": result["checks"]})
                telemetry.append(stat)
                print(f"DONE   check_{name:11s} elapsed={stat['elapsed_s']}s finish=stop", flush=True)
                return result["checks"]

        total_start = perf_counter()
        print("STAGE 1: independent extraction shards", flush=True)
        shard_results = await asyncio.gather(*(extract_job(i, shard) for i, shard in enumerate(shards, 1)), return_exceptions=True)
        errors = [str(v) for v in shard_results if isinstance(v, BaseException)]
        if errors:
            raise RuntimeError("Extraction shard failures: " + "; ".join(errors))
        candidates = [x for part in shard_results for x in part]
        # Do not silently discard duplicates; flag conflicting or exact duplicates for review.
        seen = set()
        dupes = []
        for c in candidates:
            sig = json.dumps([c["answer_span_id"], c["project_id"], c["metric"], c["values"], c["mechanism"]], ensure_ascii=False, sort_keys=True)
            if sig in seen:
                dupes.append(c)
            seen.add(sig)
        dump(OUT / "candidates.json", candidates)
        dump(OUT / "candidate_duplicates.json", dupes)
        if dupes:
            raise RuntimeError(f"Duplicate candidates ({len(dupes)}), inspect candidate_duplicates.json")
        print(f"EXTRACTED total_candidates={len(candidates)}, extract_jobs={len(shards)}", flush=True)

        batches = [candidates[i:i+VERIFY_BATCH] for i in range(0, len(candidates), VERIFY_BATCH)]
        print(f"STAGE 2: source verification jobs={len(batches)}", flush=True)
        verified_parts = await asyncio.gather(*(verify_job(i, batch) for i, batch in enumerate(batches, 1)), return_exceptions=True)
        errors = [str(v) for v in verified_parts if isinstance(v, BaseException)]
        if errors:
            raise RuntimeError("Verification failures: " + "; ".join(errors))
        claims = [x for part in verified_parts for x in part]
        assert len(claims) == len(candidates)
        dump(OUT / "claims_verified.json", claims)
        print(f"VERIFIED {len(claims)} claims", flush=True)

        print("STAGE 3: semantic check groups", flush=True)
        judge_module.AsyncOpenAI = ScopedJudgeClient
        try:
            check_results = await asyncio.gather(*(check_job(name, group) for name, group in groups.items()), return_exceptions=True)
        finally:
            judge_module.AsyncOpenAI = ORIGINAL_CLIENT
        errors = [str(v) for v in check_results if isinstance(v, BaseException)]
        if errors:
            raise RuntimeError("Semantic checks failed: " + "; ".join(errors))
        checks = [x for part in check_results for x in part]
        assert len(checks) == 10
        assembled = {"checks": checks, "extractions": [{"id": "material_claims", "kind": "material_claims", "items": claims}]}
        dump(OUT / "assembled.json", assembled)
        validate_response(json.dumps(assembled, ensure_ascii=False), plan=plan, payload=check_payload())
        numeric_spans = [s for s in spans if re.search(r"\d|[一二三四五六七八九十][倍态层等奖条篇个]", s["text"])]
        claim_span_ids = {c["answer_span_id"] for c in claims}
        uncovered = [s for s in numeric_spans if s["id"] not in claim_span_ids]
        dump(OUT / "quality_review.json", {
            "numeric_spans_without_claim": uncovered, "all_claims": claims,
            "candidate_duplicates": dupes,
            "note": "Review coverage and whether canonical or noncanonical project/metric/mechanism/value tuples match authoritative source text; schema validation is not semantic correctness.",
        })
        summary = {"success": True, "full_contract_validated": True, "case": "A1",
                   "answer_spans_covered_by_extraction": len(spans), "extract_jobs": len(shards),
                   "verify_jobs": len(batches), "check_jobs": len(groups),
                   "semantic_checks": len(checks), "claims": len(claims),
                   "claims_unsupported": sum(c["support_result"] != "PASS" for c in claims),
                   "numeric_spans_to_review": len(uncovered),
                   "all_finished_stop": all(t["finish_reason"] == "stop" for t in telemetry),
                   "max_output_tokens_per_job": MAX_OUTPUT, "nominal_margin_vs_8192": 8192-MAX_OUTPUT,
                   "elapsed_seconds": round(perf_counter() - total_start, 2),
                   "input_tokens_estimated_total": sum((t.get("input_tokens_estimate") or 0) for t in telemetry),
                   "telemetry": telemetry, "artifact_dir": str(OUT.resolve()),
                   "caveats": ["Provider output usage may be absent: 8192-6144 is nominal, not measured unused output tokens",
                               "Manual semantic audit and repeated trials remain required before production changes"]}
        dump(OUT / "summary.json", summary)
        print("\n===== SUMMARY =====\n" + json.dumps({k: v for k, v in summary.items() if k != "telemetry"}, ensure_ascii=False, indent=2), flush=True)
        return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except Exception as exc:
        print(f"\nPOC FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("Partial artifacts (for resume):", OUT.resolve(), file=sys.stderr, flush=True)
        sys.exit(2)
