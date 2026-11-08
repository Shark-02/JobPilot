"""Build a local human review worksheet from a current-contract run; no providers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evals.schema import CONTRACT_VERSION


def _cell(value) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text.replace("|", "\\|").replace("\n", "<br>")


def review_run(root: Path, case_id: str | None = None) -> Path:
    metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if metadata.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Human review requires an artifact from the current Eval contract")
    results = [json.loads(line) for line in (root / "case_results.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    selected = [item for item in results if case_id is None or item["case_id"] == case_id]
    if not selected:
        raise ValueError("Requested case is absent from this run")
    if any(item.get("suite") == "retrieval" for item in selected):
        from app.evals.retrieval_contract import RETRIEVAL_CONTRACT_VERSION
        if metadata.get("retrieval_contract_version") != RETRIEVAL_CONTRACT_VERSION:
            raise ValueError("Legacy retrieval artifacts are not the current stable-evidence contract")
        if any(item.get("suite") == "retrieval" and item["result"].get("outcome") == "SCORED"
               and item["result"].get("retrieval_contract_version") != RETRIEVAL_CONTRACT_VERSION
               for item in selected):
            raise ValueError("Retrieval result contract identity mismatch")
    lines = ["# Eval human review worksheet", "", f"Contract: `{CONTRACT_VERSION}`", "",
             "This worksheet does not approve a run or freeze a baseline.", "",
             "Inspect case_results.jsonl and judge_results.jsonl for final answers, raw evidence, exact SourceRefs, "
             "mutation events and provider telemetry. Record the human decision after checking those artifacts.", ""]
    for item in selected:
        result = item["result"]
        lines.extend([f"## {item['case_id']}", "", f"Recorded outcome: **{result['outcome']}**", "",
                      "| Gold path | Compiled check | Evaluator | Expected | Actual | Evidence | Result / reason |",
                      "|---|---|---|---|---|---|---|"])
        for check in result.get("check_results", []):
            values = [check.get("gold_path"), check.get("id"), check.get("evaluator_kind"),
                      check.get("expected"), check.get("actual"), check.get("evidence_ids", []),
                      f"{check.get('result')}: {check.get('reason', '')}"]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
        if not result.get("check_results"):
            lines.extend(["", "No executed check results: inspect the recorded preflight/harness/provider interruption."])
        lines.extend(["", "Human decision: PENDING", "", "Review notes:", ""])
    path = root / (f"human_review_{case_id}.md" if case_id else "human_review.md")
    # An identity must never become an output path.
    if path.parent.resolve() != root.resolve():
        raise ValueError("Invalid case identity for review output")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--case")
    args = parser.parse_args()
    print(review_run(args.run, args.case))


if __name__ == "__main__":
    main()
