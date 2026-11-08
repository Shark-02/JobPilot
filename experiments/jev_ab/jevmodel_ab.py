#!/usr/bin/env python3
"""Optional synthetic-only experiment via independent JevModel; no request without --live.

JevModel says it uses typesafe/jev-1.13 via OpenRouter. This claim and account
access have not been independently verified. Never send private source data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import jev_ab as base

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://jevmodel.app/api/v1/systemone"
MODEL_CLAIM = "typesafe/jev-1.13 via OpenRouter (JevModel claim; not independently verified)"
OUTPUT_PATTERN = re.compile(r"jevmodel-result-[0-9]{8}T[0-9]{12}Z\.json")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def open_request(request, timeout):
    # A redirect must never forward the key or synthetic inputs to another URL.
    return build_opener(NoRedirect()).open(request, timeout=timeout)


def call_jevmodel(case: dict, doc: dict, key: str) -> dict:
    state = {"query": case["query"], "candidate": doc["text"]}
    payload = {"state": state, "questions": {"relevance": {
        "type": "score", "instructions": base.INSTRUCTIONS, "criteria": base.CRITERIA,
    }}}
    payload_json = json.dumps(payload, ensure_ascii=False)
    if len(payload_json) > 8000 or len(json.dumps(state, ensure_ascii=False)) > 8000:
        raise ValueError("固定合成请求超出服务文档中的 8000 字符限制")
    request = Request(ENDPOINT, payload_json.encode("utf-8"),
                      {"Authorization": "Bearer " + key, "Content-Type": "application/json",
                       "User-Agent": "JobPilot-JevExperiment/0.1"},
                      method="POST")
    result = {"status": "failed", "model_claim": MODEL_CLAIM}
    started = time.perf_counter()
    try:
        with open_request(request, timeout=45) as response:
            result["http_status"] = response.status
            body = json.loads(response.read())
        if body.get("code") != 0 or not isinstance(body.get("data"), dict):
            raise ValueError("unsuccessful or malformed JevModel envelope")
        data = body["data"]
        answer = data["answers"]["relevance"]
        score = answer["score"]
        if (answer.get("type") != "score" or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score)
                or not 0 <= score <= 2):
            raise ValueError("invalid score answer")
        usage = data.get("usage")
        tokens = usage.get("inputTokens") if isinstance(usage, dict) else None
        result.update({"status": "ok", "score": score,
                       "response": {k: data[k] for k in ("request_id", "answers", "usage", "quota") if k in data},
                       "input_tokens": tokens if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0 else None})
    except HTTPError as exc:
        result.update({"error": "HTTPError", "http_status": exc.code})
        # Do not persist arbitrary response bodies or headers: an intermediary
        # could echo credentials or other sensitive text. Preserve only whether
        # the error uses the documented JevModel JSON envelope.
        try:
            error_body = json.loads(exc.read(16384))
            result["error_format"] = ("jevmodel_json" if isinstance(error_body, dict)
                                      and error_body.get("code") == -1 else "other_json")
        except (ValueError, UnicodeError, OSError):
            result["error_format"] = "non_json_or_unreadable"
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        result["error"] = type(exc).__name__
    finally:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def initialize(selected, docs):
    return [{"id": c["id"], "origin": c.get("origin"), "tags": c.get("tags", []),
             "query": c["query"], "gold": c["gold"], "rrf_ranking": ids,
             "candidates": [{"id": i, "text": docs[i]["text"]} for i in ids],
             "rrf_metrics": base.metrics(ids, c["gold"]), "calls": [],
             "jev_ranking": None, "jev_metrics": None} for c, ids in selected]


def update_rows(rows):
    for row in rows:
        good = [c for c in row["calls"] if c["status"] == "ok"]
        if len(good) == 5 and {c["doc_id"] for c in good} == set(row["rrf_ranking"]):
            # Scores tie in original RRF order, not in resume-time order.
            score = {c["doc_id"]: c["score"] for c in good}
            row["jev_ranking"] = sorted(row["rrf_ranking"], key=lambda i: -score[i])
            row["jev_metrics"] = base.metrics(row["jev_ranking"], row["gold"])
        else:
            row["jev_ranking"] = row["jev_metrics"] = None


def summary(rows):
    calls = [call for row in rows for call in row["calls"]]
    successes = [call for call in calls if call["status"] == "ok"]
    complete = all(row["jev_ranking"] is not None for row in rows)
    answered = [r for r in rows if r["gold"]]
    times = [c["latency_ms"] for c in calls]
    token_counts = [c.get("input_tokens") for c in successes]
    return {"answerable_denominator": len(answered), "unanswerable_count": len(rows)-len(answered),
            "rrf": base.aggregate([r["rrf_metrics"] for r in answered]),
            "jev": base.aggregate([r["jev_metrics"] for r in answered]) if complete else None,
            "comparison_complete": complete, "requests_attempted": len(calls),
            "requests_succeeded": len(successes), "requests_failed": len(calls)-len(successes),
            "latency_ms": {"median_all_attempts": statistics.median(times), "max_all_attempts": max(times)} if times else None,
            "successful_reported_input_tokens": sum(x for x in token_counts if x is not None),
            "all_successes_have_usage": bool(successes) and all(x is not None for x in token_counts),
            "actual_cost_usd": None,
            "cost_note": "本站费用和免费配额需在账户核对；token 使用不等于美元账单，未知金额不按官方直连价格计算。"}


def validate_resume(path: Path, fresh):
    if not OUTPUT_PATTERN.fullmatch(path.name) or path.resolve().parent != HERE.resolve():
        raise ValueError("只允许恢复此目录中的 jevmodel-result-*.json 文件")
    prior = json.loads(path.read_text(encoding="utf-8"))
    if (prior.get("dataset_sha256") != base.DATASET_SHA256
            or prior.get("endpoint") != ENDPOINT or prior.get("model_claim") != MODEL_CLAIM):
        raise ValueError("恢复文件的实验身份不匹配")
    by_id = {row["id"]: row for row in prior["cases"]}
    if len(by_id) != len(fresh) or set(by_id) != {r["id"] for r in fresh}:
        raise ValueError("恢复文件的案例集合不匹配")
    for row in fresh:
        old = by_id[row["id"]]
        if any(old.get(k) != row[k] for k in ("query", "gold", "rrf_ranking", "candidates")):
            raise ValueError("恢复文件与固定合成数据或 RRF 候选不一致")
        successes = set()
        for call in old.get("calls", []):
            if call.get("doc_id") not in row["rrf_ranking"] or call.get("status") not in ("ok", "failed"):
                raise ValueError("恢复文件包含未知请求")
            if call["status"] == "ok":
                score = call.get("score")
                if (call["doc_id"] in successes or isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(score) or not 0 <= score <= 2):
                    raise ValueError("恢复文件包含重复或非法分数")
                successes.add(call["doc_id"])
        row["calls"] = old.get("calls", [])
    return prior.get("created_utc")


def write_checkpoint(out, rows, created):
    update_rows(rows)
    record = {"created_utc": created, "updated_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256": base.DATASET_SHA256, "endpoint": ENDPOINT, "model_claim": MODEL_CLAIM,
              "metric_definition": "文档级：首个 gold 的 MRR、任意 gold 的 Hit@3、全部 gold 的 Evidence Success@k；不可回答不计入均值。",
              "summary": summary(rows), "typical_cases": base.examples(rows), "cases": rows}
    temp = out.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(out)
    return record["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="发送固定合成样本至 JevModel")
    parser.add_argument("--resume", help="恢复本目录 jevmodel-result-*.json；仅重试未成功的候选")
    args = parser.parse_args()
    _, docs, selected = base.load_inputs()
    if args.live and not os.environ.get("JEVMODEL_API_KEY", "").strip():
        print("未设置 JEVMODEL_API_KEY；未发送请求，未生成 JevModel 结果。", file=sys.stderr)
        return 2
    if args.resume and not args.live:
        raise ValueError("--resume 仅可与 --live 一起使用")
    rows = initialize(selected, docs)
    created = datetime.now(timezone.utc).isoformat()
    if args.resume:
        out = Path(args.resume)
        if not out.is_absolute() and out.parent == Path("."):
            out = HERE / out
        created = validate_resume(out, rows) or created
    else:
        out = HERE / ("jevmodel-result-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    if args.live:
        key = os.environ["JEVMODEL_API_KEY"].strip()
        write_checkpoint(out, rows, created)
        for row, (case, _) in zip(rows, selected):
            successful = {c["doc_id"] for c in row["calls"] if c["status"] == "ok"}
            for doc_id in row["rrf_ranking"]:
                if doc_id in successful:
                    continue
                call = call_jevmodel(case, docs[doc_id], key)
                call["doc_id"] = doc_id
                row["calls"].append(call)
                status = write_checkpoint(out, rows, created)
                if call["status"] != "ok":
                    print(f"调用失败，已停止；HTTP 状态: {call.get('http_status', '无')}；"
                          f"错误格式: {call.get('error_format', call.get('error', '未知'))}。"
                          "检查服务状态后可恢复。", file=sys.stderr)
                    print(f"实验记录: {out}")
                    return 1
    status = write_checkpoint(out, rows, created)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"实验记录: {out}")
    return 0 if not args.live or status["comparison_complete"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"实验终止：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
