#!/usr/bin/env python3
"""A1 read-only two-stage Claims smoke test (extract -> evidence verification).

Run in JobPilot/backend with PYTHONPATH=.; uses the existing successful Agent artifact.
Does not touch DB, Gold, product eval code or existing artifacts.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.evals.compiler import compile_case
from app.evals.datasets import DEFAULT_PACKAGE_ROOT, load_dataset
from app.evals.evidence import EvidencePack
from app.evals.judge import answer_spans, validate_response
from app.infra.token_count import TokenCounter

RUN = Path("../jobpilot_eval_v0.1.1/runs/20261009T070204Z_e428692_6ff74de7")
OUT = Path("../artifacts/judge_replays/A1_two_stage_poc") / datetime.now().strftime("%Y%m%dT%H%M%S_%f")
SPAN_ID = os.environ.get("A1_CLAIM_SPAN", "answer_14")
MAX_OUTPUT = int(os.environ.get("A1_CLAIM_MAX_OUTPUT", "6144"))
TIMEOUT = int(os.environ.get("A1_CLAIM_TIMEOUT", "600"))
VERIFY_BATCH = int(os.environ.get("A1_VERIFY_BATCH", "2"))
assert 1 <= MAX_OUTPUT <= 8192
assert 1 <= VERIFY_BATCH <= 3


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_strict_json(raw: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in items:
            if k in d:
                raise ValueError(f"Repeated JSON key: {k}")
            d[k] = v
        return d
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid JSON {x}")))


async def ask(client: AsyncOpenAI, counter: TokenCounter, model: str,
              name: str, system: str, user: dict[str, Any]) -> dict[str, Any]:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
    estimated = counter.count(json.dumps(messages, ensure_ascii=False))
    reasoning_path, content_path = OUT / f"{name}_reasoning.txt", OUT / f"{name}_content.txt"
    content: list[str] = []
    reasoning_chars = 0
    first_content_at = None
    finish = None
    usage = None
    start = perf_counter()
    print(f"START {name}: input_estimate={estimated} max_output={MAX_OUTPUT}", flush=True)
    with reasoning_path.open("w", encoding="utf-8") as rf, content_path.open("w", encoding="utf-8") as cf:
        try:
            async with asyncio.timeout(TIMEOUT):
                stream = await client.chat.completions.create(
                    model=model, temperature=0, messages=messages,
                    stream=True, max_tokens=MAX_OUTPUT)
                try:
                    async for chunk in stream:
                        if chunk.usage:
                            usage = {"prompt_tokens": chunk.usage.prompt_tokens,
                                     "completion_tokens": chunk.usage.completion_tokens}
                        for choice in chunk.choices:
                            finish = choice.finish_reason or finish
                            delta = choice.delta
                            reason = getattr(delta, "reasoning_content", None) or ""
                            if reason:
                                rf.write(reason)
                                reasoning_chars += len(reason)
                            s = delta.content or ""
                            if s:
                                if first_content_at is None:
                                    first_content_at = perf_counter() - start
                                cf.write(s)
                                content.append(s)
                finally:
                    await stream.close()
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:800]}
            dump(OUT / f"{name}_error.json", error)
            print(f"ERROR {name}: {error}", flush=True)
            raise
    raw = "".join(content)
    telemetry = {"name": name, "input_tokens_estimate": estimated,
                 "reasoning_chars": reasoning_chars, "content_chars": len(raw),
                 "content_tokens_estimate": counter.count(raw),
                 "first_content_seconds": first_content_at,
                 "elapsed_seconds": round(perf_counter() - start, 2),
                 "finish_reason": finish, "provider_usage": usage,
                 "nominal_budget_margin_vs_8192": 8192 - MAX_OUTPUT}
    dump(OUT / f"{name}_telemetry.json", telemetry)
    print(f"DONE  {name}: finish={finish} elapsed={telemetry['elapsed_seconds']}s "
          f"reasoning_chars={reasoning_chars} content_chars={len(raw)}", flush=True)
    if finish != "stop":
        raise RuntimeError(f"{name}: finish={finish}; see {reasoning_path}")
    return {"json": parse_strict_json(raw), "telemetry": telemetry}


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    case = load_dataset(DEFAULT_PACKAGE_ROOT).cases["A1"]
    plan = compile_case(case)
    assert len(plan.extractions) == 1 and plan.extractions[0].kind == "material_claims"
    assert plan.extractions[0].allowed_ids == []  # A1 has open-world, no canonical claim IDs
    rows = [json.loads(line) for line in (RUN / "case_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    row = next(r for r in rows if r["case_id"] == "A1")
    assert row["runtime"]["run_failed"] is False
    obs = row["observable"]
    all_spans = answer_spans(obs["final_answer"])
    span = next(s for s in all_spans if s["id"] == SPAN_ID)
    evidence = EvidencePack.model_validate(obs["evidence"]).judge_items()
    assert evidence and all(e["visible_to_agent"] and e["authoritative_for_grounding"] for e in evidence)
    sources = []
    for e in evidence:
        # Preserve every authoritative excerpt verbatim. Omit repetitive metadata.
        source_text = e.get("content")
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValueError(f"Evidence item {e['id']} missing original content; abort")
        sources.append({"id": e["id"], "content": source_text})
    settings = get_settings()
    base_url = settings.eval_judge_base_url or settings.llm_base_url
    api_key = settings.eval_judge_api_key or settings.llm_api_key
    model = settings.eval_judge_model or settings.llm_model
    assert base_url and api_key and model
    counter = TokenCounter(model)
    print(f"A1 span={SPAN_ID}: {span['text']}", flush=True)
    print(f"evidence_count={len(evidence)}, output_limit={MAX_OUTPUT}, verify_batch={VERIFY_BATCH}", flush=True)
    dump(OUT / "manifest.json", {"case": "A1", "run": str(RUN), "span": span,
                                      "evidence_count": len(evidence), "output_limit": MAX_OUTPUT})

    extract_system = (
        "你是职业事实量化主张识别器。本请求是第一阶段：只识别给定的单条答案 Span 中的全部实质性量化主张，"
        "不要核实真伪、不要寻找证据，也不要分析其他答案段落。"
        "每项记录独立的项目/所属事项、指标、原文数值、机制。教育成绩、奖项年份、实习/项目时间等"
        "若实际出现且与个人职业档案有关，亦应提取；排除纯编号、无具体数值的指标名、一般技术常量。"
        "若一个 Span 中存在多个不同指标，必须分别列出，不得合并或遗漏。"
        "输出仅 JSON：{\"items\":[{\"claim_id\":null,\"project_id\":\"实际项目或领域\","
        "\"metric\":\"指标\",\"values\":[\"原文值\"],\"mechanism\":\"已陈述的机制或空串\","
        "\"answer_span_id\":\"传入Span ID\"}]}。不得输出 evidence_ids 或 support_result。"
    )
    verify_system = (
        "你是职业事实来源支持核验器。本请求是第二阶段：只处理传入的少量 Claim，"
        "逐项判定其project_id、metric、values、mechanism组成的完整关系是否由本次 Agent 实际可见的"
        "权威原文直接支持。严格禁止混用其他项目或其他指标的数值。"
        "只能引用所提供来源ID；Gold、摘要、常识不构成证据。"
        "PASS仅表示该完整事实关系有权威原文支持；明显相反为FAIL；资料不足为UNKNOWN。"
        "不要重新抽取、删减或改写 Claim。输出仅 JSON："
        "{\"results\":[{\"index\":0,\"support_result\":\"PASS|FAIL|UNKNOWN\","
        "\"evidence_ids\":[\"实际支持来源ID\"],\"support_reason\":\"简短理由\"}]}。"
        "index为输入候选的从0开始的序号，每个index恰好出现一次。"
    )
    telemetry: list[dict[str, Any]] = []
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=TIMEOUT)
    try:
        e = await ask(client, counter, model, "extract", extract_system, {"span": span})
        telemetry.append(e["telemetry"])
        parsed = e["json"]
        if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
            raise ValueError("Extract response has no items array")
        candidates = parsed["items"]
        for c in candidates:
            if not isinstance(c, dict) or set(c) != {"claim_id", "project_id", "metric", "values", "mechanism", "answer_span_id"}:
                raise ValueError("Unexpected extraction candidate shape")
            if c["claim_id"] is not None or c["answer_span_id"] != SPAN_ID:
                raise ValueError("Invalid claim_id or answer_span_id")
            if not isinstance(c["values"], list) or not c["values"] or not all(isinstance(v, str) and v.strip() for v in c["values"]):
                raise ValueError("Invalid material values")
            for key in ("project_id", "metric", "mechanism"):
                if not isinstance(c[key], str):
                    raise ValueError(f"{key} must be string")
        if not candidates and re.search(r"\d|[一二三四五六七八九十][倍层态]", span["text"]):
            raise ValueError("Numeric span returned zero candidates: review required")
        dump(OUT / "candidates.json", candidates)
        print(f"EXTRACTED candidates={len(candidates)}", flush=True)
        for i, c in enumerate(candidates):
            print(f"  {i}: {c['project_id']} / {c['metric']} / {c['values']}", flush=True)
        all_supported = []
        for offset in range(0, len(candidates), VERIFY_BATCH):
            batch = candidates[offset: offset + VERIFY_BATCH]
            indexed = [{"index": idx, **c} for idx, c in enumerate(batch)]
            v = await ask(client, counter, model, f"verify_{offset // VERIFY_BATCH + 1:02d}",
                          verify_system, {"claims": indexed, "evidence": sources})
            telemetry.append(v["telemetry"])
            obj = v["json"]
            results = obj.get("results") if isinstance(obj, dict) else None
            if not isinstance(results, list) or len(results) != len(batch):
                raise ValueError("Verifier omitted or duplicated claims")
            by_index = {}
            allowed = {s["id"] for s in sources}
            for vitem in results:
                if set(vitem) != {"index", "support_result", "evidence_ids", "support_reason"}:
                    raise ValueError("Unexpected verification result shape")
                idx = vitem["index"]
                if not isinstance(idx, int) or idx in by_index or idx not in range(len(batch)):
                    raise ValueError("Unknown/repeated claim index")
                if vitem["support_result"] not in ("PASS", "FAIL", "UNKNOWN"):
                    raise ValueError("Invalid support status")
                ids = vitem["evidence_ids"]
                if not isinstance(ids, list) or any(x not in allowed for x in ids):
                    raise ValueError("Unknown evidence ID")
                if vitem["support_result"] == "PASS" and not ids:
                    raise ValueError("PASS without a source")
                if not isinstance(vitem["support_reason"], str):
                    raise ValueError("Invalid support reason")
                by_index[idx] = vitem
            for idx, original_candidate in enumerate(batch):
                support = by_index[idx]
                all_supported.append({**original_candidate, "evidence_ids": support["evidence_ids"],
                                      "support_result": support["support_result"],
                                      "support_reason": support["support_reason"]})
        assembled = {"checks": [], "extractions": [{"id": "material_claims", "kind": "material_claims", "items": all_supported}]}
        dump(OUT / "assembled.json", assembled)
        partial_plan = plan.model_copy(update={"semantic_checks": []})
        full_payload = {"final_answer": obs["final_answer"], "answer_spans": all_spans, "evidence": evidence,
                        "memories": [], "memory_before": [], "memory_events": [], "tool_events": []}
        validated = validate_response(json.dumps(assembled, ensure_ascii=False), plan=partial_plan, payload=full_payload)
        print("ORIGINAL CONTRACT VALIDATED for this partial claims probe", flush=True)
        print(f"  item_count={len(validated.extractions[0].items)}", flush=True)
        summary = {"success": True, "case": "A1", "span_id": SPAN_ID,
                   "contract_validated_for_partial_claims": True,
                   "full_A1_claim_coverage_validated": False,
                   "claim_count": len(all_supported), "checks_count": 0,
                   "telemetry": telemetry,
                   "unknown_or_unsupported": sum(c["support_result"] != "PASS" for c in all_supported),
                   "all_finished_stop": all(t["finish_reason"] == "stop" for t in telemetry),
                   "nominal_output_margin_to_8192": 8192 - MAX_OUTPUT,
                   "note": "Single-span proof; does not certify all A1 claims or all semantic checks."}
        dump(OUT / "summary.json", summary)
        print("===== SUMMARY =====")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        await client.close()
        print("Artifacts:", OUT.resolve(), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
