from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.agent.prompts import SYSTEM_PROMPT
from app.config import Settings
from app.evals.datasets import EvalDataset
from app.evals.retrieval_contract import METRIC_NAMES, RETRIEVAL_CONTRACT_VERSION
from app.knowledge.retrieval import RRF_K


CONTRACT_SETTINGS = (
    "llm_model", "llm_temperature", "llm_context_window", "llm_max_output_tokens",
    "agent_max_model_turns", "agent_max_tool_executions", "agent_model_timeout_seconds",
    "agent_tool_timeout_seconds", "agent_total_deadline_seconds", "core_profile_max_tokens",
    "recent_conversation_target_tokens", "recent_conversation_max_tokens",
    "context_compaction_max_tokens", "context_safety_margin_tokens",
    "knowledge_default_max_modules", "knowledge_max_modules", "knowledge_default_max_tokens",
    "knowledge_max_tokens", "eval_judge_temperature", "eval_judge_timeout_seconds",
    "eval_judge_max_output_tokens", "eval_judge_concurrency", "eval_judge_transport_retries",
    "eval_judge_retry_backoff_seconds",
)
METRICS = METRIC_NAMES
COMPARISON_VERSION = 4


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _files_digest(root: Path, paths: list[Path]) -> str:
    return _digest({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)})


def comparison_contract(
    settings: Settings, dataset: EvalDataset, case_ids: list[str], corpus_identity: dict[str, Any]
) -> dict[str, Any]:
    """Record experiment controls without provider credentials or corpus text."""
    app_root = Path(__file__).resolve().parents[1]
    package_paths = [path for folder in ("cases", "corpora", "fixtures")
                     for path in (dataset.root / folder).rglob("*")
                     if path.is_file() and path.suffix in {".yaml", ".md", ".tex"}]
    package_paths.extend(dataset.root / name for name in (
        "retrieval_evidence.yaml", "retrieval_cases.yaml", "retrieval_protocol.yaml",
    ))
    protocol = dataset.retrieval_protocol
    return {
        "version": COMPARISON_VERSION,
        "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
        "case_ids": case_ids,
        "case_package_sha256": _files_digest(dataset.root, package_paths),
        "implementation_sha256": _files_digest(app_root, list(app_root.rglob("*.py"))),
        "agent_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "corpus_identity": corpus_identity,
        "retrieval": {"rrf_k": RRF_K, "max_modules": protocol.retrieval_k,
                      "max_tokens": protocol.max_context_tokens, "k": protocol.retrieval_k},
        "settings": {**{key: getattr(settings, key) for key in CONTRACT_SETTINGS},
                     "judge_cache_enabled": settings.eval_judge_cache_dir is not None},
        "judge_model": settings.eval_judge_model or settings.llm_model,
        "llm_endpoint_sha256": _digest(settings.llm_base_url),
        "judge_endpoint_sha256": _digest(settings.eval_judge_base_url or settings.llm_base_url),
    }


def _module_key(module: dict[str, Any]) -> str:
    return module.get("module_id") or json.dumps([
        module.get("document_key", module.get("logical_document_key", module.get("document_id"))),
        module.get("section_id"), module.get("section_path"),
    ], ensure_ascii=False, sort_keys=True)


def _validate_retrieval_result(result: dict[str, Any]) -> None:
    if result.get("outcome") in {"BLOCKED", "ERROR"}:
        return
    if (result.get("outcome") != "SCORED"
            or result.get("retrieval_contract_version") != RETRIEVAL_CONTRACT_VERSION
            or set(result.get("metrics", {})) != set(METRICS)):
        raise ValueError("Legacy or incompatible retrieval artifact; stable evidence metrics are required")


def retrieval_case_diff(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    before, after = baseline["result"], candidate["result"]
    _validate_retrieval_result(before)
    _validate_retrieval_result(after)
    before_modules, after_modules = before.get("modules", []), after.get("modules", [])
    before_by_id = {_module_key(module): module for module in before_modules}
    after_by_id = {_module_key(module): module for module in after_modules}
    rank_changes = [
        {"module_id": key, "baseline_rank": before_by_id.get(key, {}).get("rank"), "candidate_rank": after_by_id.get(key, {}).get("rank")}
        for key in sorted(set(before_by_id) | set(after_by_id))
        if before_by_id.get(key, {}).get("rank") != after_by_id.get(key, {}).get("rank")
    ]
    scorable = before["outcome"] == after["outcome"] == "SCORED"
    before_metrics, after_metrics = before.get("metrics", {}), after.get("metrics", {})
    deltas = {
        key: float(after_metrics[key]) - float(before_metrics[key])
        for key in METRICS
        if before_metrics.get(key) is not None and after_metrics.get(key) is not None
    }
    before_evidence, after_evidence = set(before.get("covered_evidence_ids", [])), set(after.get("covered_evidence_ids", []))
    before_facts, after_facts = set(before.get("satisfied_fact_ids", [])), set(after.get("satisfied_fact_ids", []))
    return {
        "case_id": baseline["case_id"],
        "baseline_outcome": before["outcome"], "candidate_outcome": after["outcome"],
        "baseline_top_k": before_modules, "candidate_top_k": after_modules,
        "metric_deltas": deltas if scorable else {},
        "covered_evidence_added": sorted(after_evidence - before_evidence) if scorable else [],
        "covered_evidence_lost": sorted(before_evidence - after_evidence) if scorable else [],
        "satisfied_facts_added": sorted(after_facts - before_facts) if scorable else [],
        "satisfied_facts_lost": sorted(before_facts - after_facts) if scorable else [],
        "ranking_changes": rank_changes if scorable else [],
        "comparison_scorable": scorable,
        "metrics_applicable": scorable and all(before_metrics.get(key) is not None and after_metrics.get(key) is not None for key in METRICS),
    }


def compare_runs(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline_run = json.loads((baseline_dir / "run.json").read_text())
    candidate_run = json.loads((candidate_dir / "run.json").read_text())
    contracts = [item.get("comparison_contract") for item in (baseline_run, candidate_run)]
    if any(not contract for contract in contracts):
        raise ValueError("Both runs must contain comparison_contract; older artifacts cannot establish controlled A/B")
    if any(contract.get("version") != COMPARISON_VERSION
           or contract.get("retrieval_contract_version") != RETRIEVAL_CONTRACT_VERSION
           for contract in contracts):
        raise ValueError("Legacy comparison contract; current stable evidence retrieval contract is required")
    if contracts[0] != contracts[1]:
        differences = sorted(key for key in set(contracts[0]) | set(contracts[1]) if contracts[0].get(key) != contracts[1].get(key))
        raise ValueError("A/B controls differ: " + ", ".join(differences))
    if not baseline_run.get("embedding_profile_id") or not candidate_run.get("embedding_profile_id"):
        raise ValueError("Both runs must have a resolved embedding profile")
    if baseline_run["embedding_profile_id"] == candidate_run["embedding_profile_id"]:
        raise ValueError("A/B requires different embedding profiles")
    def read_results(root: Path) -> dict[str, dict[str, Any]]:
        items = [json.loads(line) for line in (root / "case_results.jsonl").read_text().splitlines() if line.strip()]
        return {item["case_id"]: item for item in items}
    baseline_results, candidate_results = read_results(baseline_dir), read_results(candidate_dir)
    if set(baseline_results) != set(candidate_results):
        raise ValueError("A/B case sets differ")
    case_diffs, agent_diffs = [], []
    for case_id in sorted(baseline_results):
        before, after = baseline_results[case_id], candidate_results[case_id]
        if before["suite"] != after["suite"]:
            raise ValueError(f"A/B suites differ for {case_id}")
        if before["suite"] == "retrieval":
            _validate_retrieval_result(before["result"])
            _validate_retrieval_result(after["result"])
            if before["result"]["outcome"] == "SCORED" and not before.get("retrieval_trace"):
                raise ValueError(f"Baseline retrieval trace is missing for {case_id}")
            if after["result"]["outcome"] == "SCORED" and not after.get("retrieval_trace"):
                raise ValueError(f"Candidate retrieval trace is missing for {case_id}")
            if before.get("retrieval_trace") and after.get("retrieval_trace"):
                if before["retrieval_trace"].get("input") != after["retrieval_trace"].get("input"):
                    raise ValueError(f"A/B retrieval inputs differ for {case_id}")
                lexical_before = before["retrieval_trace"].get("output", {}).get("lexical")
                lexical_after = after["retrieval_trace"].get("output", {}).get("lexical")
                if lexical_before != lexical_after:
                    raise ValueError(f"A/B lexical results differ for {case_id}")
            case_diffs.append(retrieval_case_diff(before, after))
        else:
            agent_diffs.append({"case_id": case_id, "baseline_outcome": before["result"]["outcome"], "candidate_outcome": after["result"]["outcome"], "baseline_reason": before["result"].get("reason"), "candidate_reason": after["result"].get("reason"), "baseline_runtime": before.get("runtime"), "candidate_runtime": after.get("runtime")})
    baseline_summary = json.loads((baseline_dir / "summary.json").read_text())
    candidate_summary = json.loads((candidate_dir / "summary.json").read_text())
    return {
        "controlled_ab": True,
        "baseline_run_id": baseline_run["eval_run_id"], "candidate_run_id": candidate_run["eval_run_id"],
        "baseline_profile": {key: baseline_run[key] for key in ("embedding_profile_id", "embedding_profile_name", "embedding_model", "embedding_dimension")},
        "candidate_profile": {key: candidate_run[key] for key in ("embedding_profile_id", "embedding_profile_name", "embedding_model", "embedding_dimension")},
        "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
        "metrics": {key: {"baseline": baseline_summary["suites"]["retrieval"]["metrics"].get(key), "candidate": candidate_summary["suites"]["retrieval"]["metrics"].get(key)} for key in METRICS},
        "relevance_basis": "Deterministic frozen-source span coverage of stable Evidence Units and required fact proof alternatives. Unanswerable query metrics are not applicable; abstention is not evaluated.",
        "retrieval_cases": case_diffs, "agent_cases": agent_diffs,
        "inconclusive_retrieval_cases": [item["case_id"] for item in case_diffs if not item["comparison_scorable"]],
    }


def comparison_report(comparison: dict[str, Any]) -> str:
    lines = ["# Controlled Embedding Profile A/B", "", "| Metric | Baseline | Candidate |", "| --- | ---: | ---: |"]
    for metric, values in comparison["metrics"].items():
        lines.append(f"| {metric} | {values['baseline']} | {values['candidate']} |")
    lines.extend(["", comparison["relevance_basis"], "", f"Inconclusive retrieval cases: {', '.join(comparison['inconclusive_retrieval_cases']) or 'none'}", ""])
    for case in comparison["retrieval_cases"]:
        lines.extend([f"## {case['case_id']}", "", f"Baseline: {case['baseline_outcome']}; candidate: {case['candidate_outcome']}; metrics applicable={case['metrics_applicable']}", "", "| Profile | Rank | Document | Section |", "| --- | ---: | --- | --- |"])
        for profile, key in (("Baseline", "baseline_top_k"), ("Candidate", "candidate_top_k")):
            for module in case[key]:
                section = " / ".join(module.get("section_path", [])).replace("|", "\\|")
                document = module.get("document_key", module.get("logical_document_key", module.get("document_id")))
                lines.append(f"| {profile} | {module['rank']} | {document} | {section} |")
        details = {key: case[key] for key in ("metric_deltas", "covered_evidence_added", "covered_evidence_lost", "satisfied_facts_added", "satisfied_facts_lost", "ranking_changes")}
        lines.extend(["", "```json", json.dumps(details, ensure_ascii=False, indent=2), "```", ""])
    if comparison["agent_cases"]:
        lines.extend(["## Agent Smoke", "", "```json", json.dumps(comparison["agent_cases"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen eval artifacts without provider calls or database writes.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new output directory; never overwrite an existing comparison")
    args = parser.parse_args()
    comparison = compare_runs(args.baseline, args.candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(comparison_report(comparison), encoding="utf-8")
    print(json.dumps({"comparison_dir": str(args.output), "inconclusive_retrieval_cases": comparison["inconclusive_retrieval_cases"]}))


if __name__ == "__main__":
    main()
