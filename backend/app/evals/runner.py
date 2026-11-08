from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from collections import Counter
import subprocess
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select

from app.agent.loop import ThinAgentLoop
from app.config import Settings, get_settings
from app.conversation.repository import ConversationRepository
from app.db.models import AgentTurn, EmbeddingProfile, KnowledgeChunk, KnowledgeSection, MemoryItem, Message, MessageRole, TurnStatus
from app.evals.comparison import comparison_contract
from app.evals.datasets import (
    DEFAULT_PACKAGE_ROOT, DatasetError, EvalDataset, load_dataset,
    resolve_source_embedding_profile, verify_and_copy_corpus,
)
from app.embedding.service import EmbeddingProfileService
from app.evals.evaluators import (
    EvalFailureClassifier,
    case_result,
    collect_turn_data,
    runtime_metrics,
    source_refs_valid,
    summarize,
)
from app.evals.faults import ToolFaultPlan, reset_fault_plan, set_fault_plan
from app.evals.judge import StructuredJudge, JUDGE_CONTRACT_VERSION, answer_spans
from app.evals.compiler import compile_case, CONTRACT_VERSION
from app.evals.database import assert_canonical_eval_database
from app.evals.executor import EvaluationContext, execute_plan
from app.evals.evidence import (EvalObservationCollector, set_observation_collector, reset_observation_collector, set_observation_turn, reset_observation_turn)
from app.evals.schema import EvalCase, AgentCaseBase, RetrievalCase, MemoryCase, RuntimeFailureCase, LocalRevisionCase, InputStep
from app.evals.retrieval_contract import RETRIEVAL_CONTRACT_VERSION, SOURCE_VIEW_VERSION, evaluate_retrieval
from app.evals.retrieval_provenance import map_retrieved_modules
from app.memory.repository import MemoryRepository
from app.infra.provider_errors import normalize_provider_error, classify_provider_error_code
from app.evals.scope import EvalScope, reset_eval_scope, retract_current_case_state, set_eval_scope, scoped_values, apply_eval_scope
from app.infra.db import configure_database, session_factory
from app.infra.token_count import TokenCounter
from app.knowledge.retrieval import KnowledgeRetriever
from app.knowledge.sources import SourceRef, SourceResolver


@dataclass(frozen=True)
class RunOptions:
    suites: set[str] | None = None
    case_ids: set[str] | None = None
    include_variance: bool = False
    package_root: Path = DEFAULT_PACKAGE_ROOT
    embedding_profile: str | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _git_revision(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return "unknown"


class EvalRunner:
    def __init__(self, settings: Settings, dataset: EvalDataset, options: RunOptions, source_database_url: str) -> None:
        self.settings, self.dataset, self.options = settings, dataset, options
        self.source_database_url = source_database_url
        self.run_id = uuid4()
        self.counter = TokenCounter(settings.llm_model)
        self.judge = StructuredJudge(settings)
        self.document_by_key: dict[str, Any] = {}
        self.embedding_profile: EmbeddingProfile | None = None
        self.comparison_contract: dict[str, Any] | None = None

    def profile_metadata(self) -> dict[str, Any]:
        profile = self.embedding_profile
        return {
            "embedding_profile_id": str(profile.id) if profile else None,
            "embedding_profile_name": profile.name if profile else self.options.embedding_profile,
            "embedding_model": profile.model if profile else None,
            "embedding_dimension": profile.dimension if profile else None,
        }

    def _bind_metadata(self, summary, results, judges):
        metadata = self.profile_metadata()
        summary.update(metadata)
        for artifact in [*results, *judges]:
            artifact.update(metadata)
        return summary, results, judges

    async def prepare_corpus(self) -> None:
        self.embedding_profile = await resolve_source_embedding_profile(
            source_database_url=self.source_database_url, settings=self.settings,
            selector=self.options.embedding_profile,
        )
        async with session_factory() as session:
            self.document_by_key = await verify_and_copy_corpus(
                dataset=self.dataset, source_database_url=self.source_database_url,
                target_session=session, settings=self.settings, embedding_profile=self.embedding_profile,
            )
            chunks = list((await session.scalars(select(KnowledgeChunk).where(
                KnowledgeChunk.document_id.in_([row.id for row in self.document_by_key.values()])
            ).order_by(KnowledgeChunk.id))).all())
            contract = [{"id": str(row.id), "document_id": str(row.document_id),
                         "section_id": str(row.section_id), "ordinal": row.ordinal,
                         "content_sha256": hashlib.sha256(row.content.encode()).hexdigest(),
                         "retrieval_text_sha256": hashlib.sha256(row.retrieval_text.encode()).hexdigest(),
                         "start_offset": row.start_offset, "end_offset": row.end_offset,
                         "token_count": row.token_count} for row in chunks]
            self.comparison_contract = comparison_contract(
                self.settings, self.dataset, [case.id for case in self.selected_cases()], {
                    "documents": {key: {"id": str(row.id), "sha256": row.sha256}
                                  for key, row in sorted(self.document_by_key.items())},
                    "chunk_count": len(chunks),
                    "chunks_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                })

    def selected_cases(self) -> list[EvalCase]:
        if self.options.case_ids and self.options.case_ids - set(self.dataset.cases):
            raise DatasetError("Requested case IDs do not exist in the typed dataset")
        selected = [case for case in self.dataset.cases.values()
                    if (not self.options.suites or case.suite in self.options.suites)
                    and (not self.options.case_ids or case.id in self.options.case_ids)]
        if not selected:
            raise DatasetError("Selection has no executable cases")
        order = {"retrieval": 0, "agent_tasks": 1, "memory": 2, "runtime_failure": 3}
        return sorted(selected, key=lambda case: (order[case.suite], case.id))

    async def run(self):
        selected = self.selected_cases()
        results, judges = [], []
        try:
            await self.prepare_corpus()
        except Exception as exc:
            import traceback

            print("\n=== PREPARE CORPUS ERROR ===", flush=True)
            print("Exception:", repr(exc), flush=True)
            traceback.print_exc()
            results = [self._harness_case_result(case, exc) for case in selected]
            summary = summarize(results)
            summary["variance_runs"] = []
            return self._bind_metadata(summary, results, judges)
        for case in selected:
            result, judge = await self.run_case(case)
            results.append(result)
            if judge is not None:
                judges.append(judge)
        if self.options.include_variance:
            for case in selected:
                for repeat in range(1, getattr(case, "variance_repeats", 0) + 1):
                    repeated = case.model_copy(update={"id": f"{case.id}#repeat{repeat}"})
                    result, judge = await self.run_case(repeated)
                    result["variance_repeat"] = repeat
                    results.append(result)
                    if judge is not None:
                        judges.append(judge)
        summary = summarize([item for item in results if "variance_repeat" not in item])
        summary["variance_runs"] = [item for item in results if "variance_repeat" in item]
        summary["variance"] = variance_summary(summary["variance_runs"])
        return self._bind_metadata(summary, results, judges)

    @staticmethod
    def _harness_case_result(case: EvalCase, exc: Exception):
        code = normalize_provider_error(exc, timeout_code="MODEL_TIMEOUT")
        provider_kind = classify_provider_error_code(code)
        classification = (EvalFailureClassifier._provider_interruption(provider_kind, source="MAIN_AGENT")
                          if provider_kind else EvalFailureClassifier.harness_error(
                              "EVAL_DATASET_ERROR" if isinstance(exc, DatasetError) else "EVAL_RUNNER_ERROR"))
        # Do not include DB URLs, provider bodies or private fixture content in errors.
        return {"case_id": case.id, "suite": case.suite,
                "result": case_result(classification, harness_error=type(exc).__name__)}

    async def run_case(self, case: EvalCase):
        plan = compile_case(case)
        scope_token = set_eval_scope(EvalScope(
            run_id=self.run_id, case_id=case.id,
            corpus_document_ids=frozenset(row.id for row in self.document_by_key.values()),
            embedding_profile_id=self.embedding_profile.id if self.embedding_profile else None,
        ))
        collector = EvalObservationCollector()
        collector_token = set_observation_collector(collector)
        fault = ToolFaultPlan.from_config(case.fault_injection) if isinstance(case, RuntimeFailureCase) else None
        fault_token = set_fault_plan(fault)
        judge_result = None
        try:
            async with session_factory() as session:
                assert_canonical_eval_database(session.bind.url, self.settings.database_url)
                if isinstance(case, RetrievalCase):
                    protocol = self.dataset.retrieval_protocol
                    embedding = await EmbeddingProfileService(session, self.settings).client_for_active()
                    modules, trace = await KnowledgeRetriever(session, embedding, self.counter).search_with_debug(
                        case.query, document_ids=None, max_modules=protocol.retrieval_k,
                        max_tokens=protocol.max_context_tokens,
                    )
                    chunk_ids = [ref.parts()[3] for module in modules for ref in module.source_refs
                                 if ref.parts()[2] == "chunk"]
                    chunk_rows = list((await session.scalars(select(KnowledgeChunk).where(
                        KnowledgeChunk.id.in_(chunk_ids)))).all())
                    mapped, records = map_retrieved_modules(
                        modules, logical_key_by_document={row.id: key for key, row in self.document_by_key.items()},
                        documents={row.id: row for row in self.document_by_key.values()},
                        chunks={row.id: row for row in chunk_rows}, sources=self.dataset.retrieval_sources)
                    score = evaluate_retrieval(case, self.dataset.retrieval_evidence, mapped, protocol)
                    retrieval = {"outcome": "SCORED", "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
                                 **score.model_dump(mode="json"), "modules": records, "tags": case.tags}
                    context = EvaluationContext(retrieval_result=retrieval)
                    result = execute_plan(plan, context)
                    return {"case_id": case.id, "suite": case.suite,
                            "compiled_plan": plan.model_dump(mode="json"), "result": result,
                            "retrieval_trace": trace}, None
                turn_ids, metadata = await self._run_inputs(session, case)
                data = await collect_turn_data(session, turn_ids)
                data["tool_events"] = collector.tool_events
                collector.validate_capture(data)
                evidence = collector.evidence_pack()
                allowed = set(self.dataset.manifest.evidence_policy.authoritative_origins)
                origin_groups = {"knowledge_module": "knowledge", "resolved_source": "knowledge",
                                 "user_message": "user_message", "memory_derived": "memory",
                                 "core_profile_derived": "core_profile"}
                for item in evidence.items:
                    group = origin_groups.get(item.origin_type)
                    if item.origin_type == "resolved_source" and (item.source_ref or "").startswith("message://"):
                        group = "user_message"
                    if group not in allowed:
                        item.authoritative_for_grounding = False
                valid, invalid = await source_refs_valid(
                    session, data["tool_calls"], self.counter,
                    visible_refs=[ref for item in evidence.items for ref in item.source_refs],
                )
                final_answer = data["final_answers"][-1] if data["final_answers"] else ""
                final_refs = re.findall(r"(?:message://[0-9a-fA-F-]{36}|document://[0-9a-fA-F-]{36}/(?:section|chunk)/[0-9a-fA-F-]{36})", final_answer)
                for ref in final_refs:
                    try:
                        await SourceResolver(session, self.counter).validate(SourceRef(ref))
                    except (ValueError, LookupError):
                        invalid.append(ref)
                context = EvaluationContext(
                    data=data, evidence_pack=evidence, fixture_source_refs=metadata["fixture_source_refs"],
                    step_turn_ids=metadata["step_turn_ids"], step_source_refs=metadata["step_source_refs"],
                    final_answers_by_step=metadata["final_answers_by_step"],
                    memory_before=metadata["memory_before"], memory_after=await self._memory_snapshot(session),
                    memory_events=collector.memory_events, tool_events=collector.tool_events,
                    fault_events=fault.events if fault else [], source_refs_valid=valid and not invalid,
                    invalid_source_refs=sorted(set(invalid)),
                )
                interruption = EvalFailureClassifier.agent_interruption(data)
                if interruption is None and not evidence.capture_complete:
                    interruption = EvalFailureClassifier.harness_error("EVAL_EVIDENCE_CAPTURE_ERROR")
                if interruption is None and (plan.semantic_checks or plan.extractions):
                    judge_result = await self.judge.judge(plan=plan, payload=self._judge_payload(case, context))
                    judge_result["case_id"] = case.id
                    interruption = EvalFailureClassifier.judge_interruption(judge_result)
                result = execute_plan(plan, context, judge_result)
                return {"case_id": case.id, "suite": case.suite, "compiled_plan": plan.model_dump(mode="json"),
                        "result": result, "runtime": runtime_metrics(data),
                        "observable": {"final_answer": final_answer,
                                       "final_answers_by_step": context.final_answers_by_step,
                                       "fixture_source_refs": context.fixture_source_refs,
                                       "step_turn_ids": context.step_turn_ids,
                                       "step_source_refs": context.step_source_refs,
                                       "evidence": evidence.model_dump(mode="json"),
                                       "memory_before": context.memory_before, "memory_after": context.memory_after,
                                       "memory_events": context.memory_events, "tool_events": context.tool_events,
                                       "fault_events": context.fault_events}}, judge_result
        except Exception as exc:
            import traceback

            print("Exception repr:", repr(exc), flush=True)
            print("Exception str:", str(exc), flush=True)

            print("\n========== EVAL EXCEPTION ==========", flush=True)
            print("Exception:", type(exc).__name__, flush=True)

            orig = getattr(exc, "orig", None)
            if orig is not None:
                print("Original type:", type(orig).__name__, flush=True)
                print("Original message:", str(orig)[:1000], flush=True)

            print("\nTraceback frames:", flush=True)
            for frame in traceback.extract_tb(exc.__traceback__):
                print(
                    f"{frame.filename}:{frame.lineno} "
                    f"in {frame.name}",
                    flush=True,
                )

            print("====================================\n", flush=True)

            result = self._harness_case_result(case, exc)
            result["compiled_plan"] = plan.model_dump(mode="json")
            return result, judge_result
        finally:
            reset_fault_plan(fault_token)
            reset_observation_collector(collector_token)
            try:
                async with session_factory() as session:
                    await retract_current_case_state(session)
            finally:
                reset_eval_scope(scope_token)

    async def _add_fixtures(self, session, conversation_id, fixtures, refs):
        for fixture in fixtures:
            if fixture.id in refs:
                raise DatasetError(f"Fixture must be injected exactly once: {fixture.id}")
            message = await ConversationRepository(session).add_message(
                conversation_id, MessageRole.USER, self.dataset.fixture(fixture.path),
                {"eval_fixture": fixture.id},
            )
            refs[fixture.id] = f"message://{message.id}"
            await session.commit()

    async def _run_turn(
            self, session, conversation_id, content,
            *, conversation="A", phase="target"
    ):
        await session.commit()

        message = await ConversationRepository(session).add_message(
            conversation_id, MessageRole.USER, content
        )
        message_id = message.id

        turn = AgentTurn(
            conversation_id=conversation_id,
            user_message_id=message_id,
            status=TurnStatus.RUNNING,
            **scoped_values(),
        )
        session.add(turn)
        await session.flush()
        turn_id = turn.id
        await session.commit()

        origin_token = set_observation_turn(
            turn_id, conversation, phase
        )
        try:
            await ThinAgentLoop(
                session, self.settings
            ).run(turn_id)
        finally:
            reset_observation_turn(origin_token)

        # 显式异步查询，不依赖可能过期的 ORM 实例
        assistant_message_id = await session.scalar(
            apply_eval_scope(
                select(AgentTurn.assistant_message_id).where(
                    AgentTurn.id == turn_id
                ),
                AgentTurn,
            )
        )

        answer = ""
        if assistant_message_id is not None:
            answer = (
                         await session.scalar(
                             select(Message.content).where(
                                 Message.id == assistant_message_id
                             )
                         )
                     ) or ""

        return turn_id, f"message://{message_id}", answer

    async def _run_inputs(self, session, case):
        metadata = {"fixture_source_refs": {}, "step_turn_ids": {}, "step_source_refs": {},
                    "final_answers_by_step": {}, "memory_before": []}
        turns, conversations = [], {}

        async def conversation_id(phase, label):
            key = (phase, label)
            if key not in conversations:
                created = await ConversationRepository(session).create(title=f"eval:{self.run_id}:{case.id}:{phase}:{label}")
                # A turn may rollback this shared session and expire every ORM
                # instance, regardless of expire_on_commit. Retain only identity.
                conversations[key] = created.id
            return conversations[key]

        for step in case.setup.seed_steps:
            identifier = await conversation_id("seed", step.conversation)
            await self._add_fixtures(session, identifier, step.user_fixtures, metadata["fixture_source_refs"])
            turn_id, _, _ = await self._run_turn(session, identifier, step.user, conversation=step.conversation, phase="seed")
            turns.append(turn_id)
        metadata["memory_before"] = await self._memory_snapshot(session)
        identifier = await conversation_id("target", "A")
        await self._add_fixtures(session, identifier, case.setup.user_fixtures, metadata["fixture_source_refs"])
        if isinstance(case, LocalRevisionCase):
            draft = "\n".join(f"- {item.id}: {item.text}" for item in case.gold.draft)
            await ConversationRepository(session).add_message(identifier, MessageRole.USER,
                                                              f"当前简历草稿：\n{draft}", {"eval_fixture": "embedded_draft"})
            await session.commit()
        steps = case.steps if isinstance(case, MemoryCase) else [InputStep(id="task", user=case.prompt)]
        for step in steps:
            identifier = await conversation_id("target", step.conversation)
            await self._add_fixtures(session, identifier, step.user_fixtures, metadata["fixture_source_refs"])
            turn_id, ref, answer = await self._run_turn(session, identifier, step.user, conversation=step.conversation)
            turns.append(turn_id)
            metadata["step_turn_ids"][step.id] = str(turn_id)
            metadata["step_source_refs"][step.id] = ref
            metadata["final_answers_by_step"][step.id] = answer
        return turns, metadata

    async def _memory_snapshot(self, session):
        rows = list((await session.scalars(apply_eval_scope(select(MemoryItem).order_by(MemoryItem.created_at), MemoryItem,
                                                          include_retracted=True))).all())
        sources = await MemoryRepository(session).source_rows([item.id for item in rows])
        result = []
        for item in rows:
            refs, bound = [], []
            for source in sources:
                if source.memory_id != item.id:
                    continue
                if source.message_id:
                    ref = SourceRef.message(source.message_id)
                elif source.document_section_id:
                    section = await session.get(KnowledgeSection, source.document_section_id)
                    ref = SourceRef.section(section.document_id, section.id) if section else None
                else:
                    chunk = await session.get(KnowledgeChunk, source.document_chunk_id)
                    ref = SourceRef.chunk(chunk.document_id, chunk.id) if chunk else None
                if ref is None:
                    raise DatasetError("Memory source binding is missing")
                refs.append(ref.uri)
                resolved = await SourceResolver(session, self.counter).read([ref], max_tokens=None)
                for raw in resolved:
                    identity = hashlib.sha256((raw.source_ref.uri + raw.content).encode()).hexdigest()[:20]
                    bound.append({"id": f"bound_source_{identity}", "source_ref": raw.source_ref.uri,
                                  "content": raw.content, "visible_to_agent": False,
                                  "authoritative_for_grounding": False, "metadata": raw.metadata})
            result.append({"id": str(item.id), "kind": item.kind.value, "status": item.status.value,
                           "content": item.content, "attributes": item.attributes, "source_refs": sorted(set(refs)),
                           "bound_sources": bound})
        return result

    @staticmethod
    def _judge_payload(case, context):
        final_answer = context.data["final_answers"][-1] if context.data.get("final_answers") else ""
        spans = answer_spans(final_answer)
        for step, answer in context.final_answers_by_step.items():
            spans.extend(answer_spans(answer, prefix=f"step_{step}"))
        return {"task": case.prompt if isinstance(case, AgentCaseBase) or isinstance(case, RuntimeFailureCase)
                else [step.model_dump(mode="json") for step in case.steps],
                "source_policy": case.source_policy.model_dump(mode="json"),
                "final_answer": final_answer, "answer_spans": spans,
                "final_answers_by_step": context.final_answers_by_step,
                "evidence": context.evidence_pack.judge_items(), "memories": context.memory_after,
                "memory_before": context.memory_before, "memory_events": context.memory_events,
                "tool_events": context.tool_events,
                "fixture_source_refs": context.fixture_source_refs,
                "step_turn_ids": context.step_turn_ids, "step_source_refs": context.step_source_refs}


def variance_summary(results):
    groups = {}
    for row in results:
        identity = row["case_id"].rsplit("#repeat", 1)[0]
        groups.setdefault(identity, []).append(row["result"]["outcome"])
    return {key: {"repeat_count": len(values), "outcome_counts": dict(Counter(values)),
                  "scorable_count": sum(item in {"PASS", "FAIL"} for item in values),
                  "pass_rate": values.count("PASS") / sum(item in {"PASS", "FAIL"} for item in values)
                  if any(item in {"PASS", "FAIL"} for item in values) else None}
            for key, values in groups.items()}

def write_artifacts(
    *, package_root: Path, run_id: UUID, summary: dict[str, Any], results: list[dict[str, Any]], judges: list[dict[str, Any]], settings: Settings,
    embedding_profile_metadata: dict[str, Any],
    comparison_metadata: dict[str, Any] | None = None,
    retrieval_inputs: dict[str, Path] | None = None,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    revision = _git_revision(package_root.parent)
    root = package_root / "runs" / f"{timestamp}_{revision}_{str(run_id)[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    run = {
        "eval_run_id": str(run_id), "started_at": timestamp,
        "model": settings.llm_model, **embedding_profile_metadata,
        "judge_model": settings.eval_judge_model or settings.llm_model,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "corpus_version": "corpus_v1", "git_revision": revision, "contract_version": CONTRACT_VERSION,
        "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
        "retrieval_source_view_version": SOURCE_VIEW_VERSION,
        "comparison_contract": comparison_metadata,
    }
    _write_json(root / "run.json", run)
    for name, path in (retrieval_inputs or {}).items():
        if name not in {"retrieval_evidence.yaml", "retrieval_cases.yaml", "retrieval_protocol.yaml"}:
            raise ValueError("Unknown retrieval input artifact")
        shutil.copyfile(path, root / name)
    _write_json(root / "compiled_plans.json", [item["compiled_plan"] for item in results if "compiled_plan" in item])
    with (root / "case_results.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")
    with (root / "judge_results.jsonl").open("w", encoding="utf-8") as handle:
        for item in judges:
            handle.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")
    _write_json(root / "summary.json", summary)
    report_summary = {
        "run_status": summary["run_status"],
        "outcomes": summary["outcomes"],
        "suites": summary["suites"],
        "runtime": summary["runtime"],
    }
    lines = [
        "# JobPilot Evaluation Run",
        "",
        f"- eval_run_id: `{run_id}`",
        f"- embedding_profile: `{embedding_profile_metadata['embedding_profile_name']}` (`{embedding_profile_metadata['embedding_profile_id']}`)",
        f"- embedding_model / dimension: `{embedding_profile_metadata['embedding_model']}` / `{embedding_profile_metadata['embedding_dimension']}`",
        f"- cases: {summary['case_count']}",
        f"- run_status: `{summary['run_status']}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Case outcomes",
        "",
    ]
    for item in results:
        result = item["result"]
        reason = f" ({result['reason']})" if result.get("reason") else ""
        lines.append(f"- {item['case_id']} ({item['suite']}): {result['outcome']}{reason}")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


async def _async_main(args: argparse.Namespace) -> int:
    try:
        dataset = load_dataset(Path(args.package_root) if args.package_root else DEFAULT_PACKAGE_ROOT)
    except DatasetError:
        print(json.dumps({"outcome": "ERROR", "reason": "EVAL_DATASET_PREFLIGHT_ERROR"}))
        return 2
    if args.preflight:
        if args.plan_output:
            _write_json(Path(args.plan_output), [plan.model_dump(mode="json") for plan in dataset.plans.values()])
        print(json.dumps({"outcome": "PASS", "phase": "dataset_preflight",
                          "case_count": len(dataset.cases), "contract_version": CONTRACT_VERSION,
                          "check_count": sum(len(plan.checks) for plan in dataset.plans.values()),
                          "provider_calls": 0}, ensure_ascii=False))
        return 0
    settings = get_settings()
    eval_url = args.database_url or settings.eval_database_url
    if not eval_url:
        raise SystemExit("Configure the canonical jobpilot_eval database")
    source_url = args.source_database_url or os.environ.get("EVAL_SOURCE_DATABASE_URL") or settings.database_url
    assert_canonical_eval_database(eval_url, settings.database_url)
    assert_canonical_eval_database(eval_url, source_url)
    configure_database(eval_url)
    options = RunOptions(suites=set(args.suite) if args.suite else None,
                         case_ids=set(args.case) if args.case else None,
                         include_variance=args.include_variance, package_root=dataset.root,
                         embedding_profile=args.embedding_profile)
    runner = EvalRunner(settings, dataset, options, source_url)
    summary, results, judges = await runner.run()
    artifact = write_artifacts(package_root=dataset.root, run_id=runner.run_id, summary=summary,
                               results=results, judges=judges, settings=settings,
                               embedding_profile_metadata=runner.profile_metadata(),
                               comparison_metadata=runner.comparison_contract,
                               retrieval_inputs={name: dataset.root / name for name in (
                                   "retrieval_evidence.yaml", "retrieval_cases.yaml", "retrieval_protocol.yaml")}
                               if any(item["suite"] == "retrieval" for item in results) else None)
    print(json.dumps({"eval_run_id": str(runner.run_id), "artifact_dir": str(artifact), "summary": summary},
                     ensure_ascii=False, default=_json_default))
    return 0 if summary["run_status"] == "COMPLETED" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JobPilot's offline evaluation suite against an isolated database.")
    parser.add_argument("--preflight", action="store_true", help="Validate and compile all cases without DB/provider calls")
    parser.add_argument("--plan-output", help="Optional compiled-plan JSON path for preflight")
    parser.add_argument("--database-url", help="isolated jobpilot_eval SQLAlchemy URL")
    parser.add_argument("--source-database-url", help="read-only source database URL used only to verify/copy corpus_v1")
    parser.add_argument("--package-root", help="path to jobpilot_eval_v0.1.1")
    parser.add_argument("--embedding-profile", help="source profile name or UUID; defaults to source database active profile")
    parser.add_argument("--suite", action="append", choices=["retrieval", "agent_tasks", "memory", "runtime_failure"])
    parser.add_argument("--case", action="append")
    parser.add_argument("--include-variance", action="store_true", help="run case-declared variance repeats")
    raise SystemExit(asyncio.run(_async_main(parser.parse_args())))


if __name__ == "__main__":
    main()
