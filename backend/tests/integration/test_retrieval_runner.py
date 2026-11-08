"""Real SQL/provenance/scoring wiring with a synthetic retriever and no providers."""
from dataclasses import replace
from hashlib import sha256

import pytest
from sqlalchemy import func, select

from app.config import Settings
from app.db.models import DocumentFormat, KnowledgeChunk, KnowledgeChunkEmbedding, KnowledgeDocument, KnowledgeSection
from app.evals.compiler import compile_case
from app.evals.corpus import setup_local_document
from app.evals.datasets import load_dataset
from app.evals.retrieval_contract import (
    METRIC_NAMES, RETRIEVAL_CONTRACT_VERSION, EvidenceFile, EvidenceUnit, RequiredFact,
    RetrievalCase, RetrievalProtocol, Span,
)
from app.evals.retrieval_provenance import FrozenSource
from app.evals.runner import EvalRunner, RunOptions
from app.knowledge.parser import DocumentParser
from app.knowledge.retrieval import KnowledgeModule
from app.knowledge.sources import SourceRef


@pytest.mark.parametrize("answerability", ["answerable", "unanswerable"])
async def test_runner_reads_real_chunks_and_scores_verified_provenance_without_providers(
        profile_database_factory, monkeypatch, answerability):
    import app.agent.model_client as model_module
    import app.evals.corpus as corpus_module
    import app.evals.judge as judge_module
    import app.evals.judge_scheduler as scheduler_module
    import app.evals.runner as runner_module
    import app.infra.embedding as embedding_module

    def forbidden_provider(*args, **kwargs):
        raise AssertionError("Retrieval contract integration must never construct or call a provider")

    for module in (model_module, judge_module, scheduler_module, embedding_module):
        monkeypatch.setattr(module, "AsyncOpenAI", forbidden_provider)
    monkeypatch.setattr(runner_module, "ThinAgentLoop", forbidden_provider)

    class Counter:
        def __init__(self, *args):
            pass

        def count(self, text):
            return len(text)

        def truncate(self, text, limit):
            return text[:limit]

        def split(self, text, limit):
            return [text[start:start + limit] for start in range(0, len(text), limit)]

    class NoJudge:
        def __init__(self, *args):
            pass

        async def judge(self, **kwargs):
            forbidden_provider()

    class NoEmbedding:
        async def embed_one(self, *args, **kwargs):
            forbidden_provider()

        async def embed_many(self, *args, **kwargs):
            forbidden_provider()

    monkeypatch.setattr(corpus_module, "TokenCounter", Counter)
    monkeypatch.setattr(runner_module, "TokenCounter", Counter)
    monkeypatch.setattr(runner_module, "StructuredJudge", NoJudge)
    database = profile_database_factory()
    monkeypatch.setattr(runner_module, "session_factory", database.sessions)
    raw = "# Education\nThe school is Example University.\n"
    snapshot_hash = sha256(raw.encode()).hexdigest()
    spec = {"key": "synthetic-source", "sha256": snapshot_hash,
            "filename": "synthetic.md", "title": "Synthetic source",
            "source_role": "USER_CAREER_SOURCE", "snapshot_path": "corpora/synthetic.md"}
    async with database.sessions() as session:
        await setup_local_document(session, spec, raw, "runner-contract-test")
        document = await session.scalar(select(KnowledgeDocument).where(KnowledgeDocument.sha256 == snapshot_hash))
        chunk = await session.scalar(select(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id))
        section = await session.get(KnowledgeSection, chunk.section_id)
        assert await session.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding)) == 0
    parsed = DocumentParser().parse(raw, DocumentFormat.MARKDOWN)
    canonical_hash = sha256(parsed.normalized_text.encode()).hexdigest()
    text = "The school is Example University."
    start = parsed.normalized_text.index(text)
    unit = EvidenceUnit(
        id="eu.synthetic.school", document_key=spec["key"], snapshot_hash=snapshot_hash,
        canonical_hash=canonical_hash, section_path=section.section_path,
        span=Span(start=start, end=start + len(text)), span_hash=sha256(text.encode()).hexdigest(),
    )
    case = RetrievalCase(
        id="opaque-task-identity", query="What is the school?" if answerability == "answerable" else "What is the tuition?",
        answerability=answerability, tags=["synthetic"],
        facts=[RequiredFact(id="requested-fact", statement="Requested corpus fact",
                            proofs=[[unit.id]] if answerability == "answerable" else [])],
    )
    protocol = RetrievalProtocol(retrieval_k=5, max_context_tokens=8000)
    dataset = replace(
        load_dataset(), cases={case.id: case}, plans={case.id: compile_case(case)},
        retrieval_evidence=EvidenceFile(evidence=[unit]), retrieval_protocol=protocol,
        retrieval_sources={spec["key"]: FrozenSource(spec["key"], snapshot_hash, canonical_hash,
                                                    parsed.normalized_text, parsed.sections)},
    )
    module = KnowledgeModule(
        document.id, document.title, section.id, section.section_path, chunk.content,
        [SourceRef.section(document.id, section.id), SourceRef.chunk(document.id, chunk.id)], chunk.token_count,
    )
    embedding = NoEmbedding()

    async def client_for_active(self):
        return embedding

    calls = []

    async def search_with_debug(self, query, *, document_ids, max_modules, max_tokens):
        assert self.embedding is embedding
        calls.append((query, document_ids, max_modules, max_tokens))
        return [module], {"input": {"query": query}, "output": {"lexical": []}}

    monkeypatch.setattr(runner_module.EmbeddingProfileService, "client_for_active", client_for_active)
    monkeypatch.setattr(runner_module.KnowledgeRetriever, "search_with_debug", search_with_debug)
    runner = EvalRunner(Settings(_env_file=None, llm_model="synthetic"), dataset,
                        RunOptions(case_ids={case.id}), "unused-read-only-source")
    runner.document_by_key = {spec["key"]: document}
    row, judged = await runner.run_case(case)
    result = row["result"]
    assert calls == [(case.query, None, 5, 8000)]
    assert judged is None and result["outcome"] == "SCORED", result
    assert result["retrieval_contract_version"] == RETRIEVAL_CONTRACT_VERSION
    assert result["metrics"] == dict.fromkeys(METRIC_NAMES, 1.0 if answerability == "answerable" else None)
    assert result["abstention"] == "NOT_EVALUATED"
    assert result["modules"][0]["included_chunk_ids"] == [str(chunk.id)]
    assert result["modules"][0]["spans"][0]["document_key"] == spec["key"]
    assert result["check_results"][0]["result"] == "SCORED"
    if answerability == "answerable":
        assert result["covered_evidence_ids"] == [unit.id]
        assert result["fact_witnesses"] == {"requested-fact": [unit.id]}
