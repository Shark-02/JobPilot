"""Synthetic source/DB-row fakes; no database, embedding, or provider calls."""
from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.evals import retrieval_provenance as provenance
from app.evals.retrieval_contract import (
    EvidenceFile, EvidenceUnit, RequiredFact, RetrievalCase, Span, evaluate_retrieval,
)

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
SECTION_ID = UUID("22222222-2222-2222-2222-222222222222")
CHUNK_A = UUID("33333333-3333-3333-3333-333333333333")
CHUNK_B = UUID("44444444-4444-4444-4444-444444444444")
OTHER_ID = UUID("55555555-5555-5555-5555-555555555555")


def ref(kind, identity, document_id=DOCUMENT_ID):
    return SimpleNamespace(parts=lambda: ("document", document_id, kind, identity))


@pytest.fixture
def graph(tmp_path):
    raw = b"  # Synthetic\r\nAlpha beta gamma delta.\r\n## Other\r\nUnrelated synthetic detail.  "
    path = tmp_path / "corpora/synthetic.md"
    path.parent.mkdir()
    path.write_bytes(raw)
    spec = SimpleNamespace(key="doc.synthetic", snapshot_path="corpora/synthetic.md",
                           filename="synthetic.md", sha256=sha256(raw).hexdigest())
    manifest = SimpleNamespace(frozen_documents=[spec])
    sources = provenance.frozen_sources(tmp_path, manifest)
    source = sources[spec.key]
    start = source.text.index("Alpha")
    split = source.text.index("gamma")
    end = source.text.index("delta.") + len("delta.")
    chunks = {
        CHUNK_A: SimpleNamespace(document_id=DOCUMENT_ID, section_id=SECTION_ID,
                                 start_offset=start, end_offset=split,
                                 content=source.text[start:split]),
        CHUNK_B: SimpleNamespace(document_id=DOCUMENT_ID, section_id=SECTION_ID,
                                 start_offset=split, end_offset=end,
                                 content=source.text[split:end]),
    }
    document = SimpleNamespace(sha256=spec.sha256, normalized_text=source.text)
    eu = EvidenceUnit(id="eu.synthetic", document_key=spec.key, snapshot_hash=spec.sha256,
                      canonical_hash=source.canonical_hash, section_path=["Synthetic"],
                      span=Span(start=start, end=end), span_hash=provenance.digest(source.text[start:end]))
    evidence = EvidenceFile(evidence=[eu])
    task = RetrievalCase(id="synthetic-task", query="Require the complete synthetic statement",
                         answerability="answerable", tags=["single_fact"],
                         facts=[RequiredFact(id="statement", statement="Full synthetic statement",
                                             proofs=[[eu.id]])])
    return SimpleNamespace(root=tmp_path, manifest=manifest, spec=spec, sources=sources,
                           source=source, chunks=chunks, document=document,
                           evidence=evidence, task=task)


def module(graph, chunk_ids=(CHUNK_A, CHUNK_B), *, include_section=True, content=None):
    refs = ([ref("section", SECTION_ID)] if include_section else []) + [
        ref("chunk", identity) for identity in chunk_ids
    ]
    return SimpleNamespace(document_id=DOCUMENT_ID, section_id=SECTION_ID,
                           section_path=["Synthetic"], source_refs=refs,
                           content="\n\n".join(graph.chunks[key].content for key in chunk_ids)
                           if content is None else content)


def map_modules(graph, modules):
    return provenance.map_retrieved_modules(
        modules, logical_key_by_document={DOCUMENT_ID: graph.spec.key},
        documents={DOCUMENT_ID: graph.document}, chunks=graph.chunks, sources=graph.sources,
    )


def test_frozen_source_hashes_are_raw_bytes_and_parser_normalized_codepoints(graph):
    source = graph.source
    assert source.text.startswith("# Synthetic\nAlpha")
    assert "\r" not in source.text
    assert source.snapshot_hash != source.canonical_hash
    assert source.canonical_hash == provenance.digest(source.text)
    provenance.validate_evidence_sources(graph.evidence, graph.sources)


def test_raw_snapshot_hash_mismatch_is_preflight_error(graph):
    graph.spec.sha256 = "0" * 64
    with pytest.raises(provenance.RetrievalProvenanceError, match="snapshot hash"):
        provenance.frozen_sources(graph.root, graph.manifest)


@pytest.mark.parametrize("identity_field", ["document_key", "snapshot_hash", "canonical_hash"])
def test_bad_evidence_source_identity_or_hash_is_preflight_error(graph, identity_field):
    value = "outside.corpus" if identity_field == "document_key" else "0" * 64
    changed = graph.evidence.evidence[0].model_copy(update={identity_field: value})
    with pytest.raises(provenance.RetrievalProvenanceError, match="identity/hash"):
        provenance.validate_evidence_sources(EvidenceFile(evidence=[changed]), graph.sources)


def test_bad_evidence_span_hash_is_preflight_error(graph):
    changed = graph.evidence.evidence[0].model_copy(update={"span_hash": "0" * 64})
    with pytest.raises(provenance.RetrievalProvenanceError, match="span/hash"):
        provenance.validate_evidence_sources(EvidenceFile(evidence=[changed]), graph.sources)


def test_out_of_bounds_evidence_span_is_preflight_error(graph):
    changed = graph.evidence.evidence[0].model_copy(update={
        "span": Span(start=0, end=len(graph.source.text) + 1),
    })
    with pytest.raises(provenance.RetrievalProvenanceError, match="span/hash"):
        provenance.validate_evidence_sources(EvidenceFile(evidence=[changed]), graph.sources)


def test_wrong_evidence_section_path_is_preflight_error(graph):
    changed = graph.evidence.evidence[0].model_copy(update={"section_path": ["Other"]})
    with pytest.raises(provenance.RetrievalProvenanceError, match="section/span"):
        provenance.validate_evidence_sources(EvidenceFile(evidence=[changed]), graph.sources)


def test_evidence_cannot_claim_one_section_while_spanning_two_sections(graph):
    start = graph.evidence.evidence[0].span.start
    end = len(graph.source.text)
    changed = graph.evidence.evidence[0].model_copy(update={
        "span": Span(start=start, end=end), "span_hash": provenance.digest(graph.source.text[start:end]),
    })
    with pytest.raises(provenance.RetrievalProvenanceError, match="section/span"):
        provenance.validate_evidence_sources(EvidenceFile(evidence=[changed]), graph.sources)


@pytest.mark.parametrize("location", ["outside", "missing"])
def test_snapshot_must_exist_inside_frozen_corpus_directory(graph, location):
    if location == "outside":
        path = graph.root / "outside.md"
        path.write_text("Synthetic outside source")
        graph.spec.snapshot_path = "outside.md"
    else:
        graph.spec.snapshot_path = "corpora/missing.md"
    with pytest.raises(provenance.RetrievalProvenanceError, match="outside-corpus"):
        provenance.frozen_sources(graph.root, graph.manifest)


def test_changed_parser_version_requires_explicit_adapter_review(graph, monkeypatch):
    monkeypatch.setattr(provenance, "PARSER_VERSION", "future-v2")
    with pytest.raises(provenance.RetrievalProvenanceError, match="parser-v1"):
        provenance.frozen_sources(graph.root, graph.manifest)


def test_verified_module_chunks_can_jointly_cover_one_evidence_unit(graph):
    mapped, records = map_modules(graph, [module(graph)])
    result = evaluate_retrieval(graph.task, graph.evidence, mapped)
    assert result.metrics.model_dump() == {
        "evidence_recall": 1.0, "mrr": 1.0, "fact_recall": 1.0, "complete_support": 1.0,
    }
    assert len(mapped[0].spans) == 2
    assert records[0]["included_chunk_ids"] == [str(CHUNK_A), str(CHUNK_B)]


def test_chunks_in_separate_modules_can_jointly_complete_the_fact(graph):
    mapped, _ = map_modules(graph, [module(graph, (CHUNK_A,)), module(graph, (CHUNK_B,))])
    result = evaluate_retrieval(graph.task, graph.evidence, mapped)
    assert result.metrics.evidence_recall == result.metrics.fact_recall == 1.0
    assert result.metrics.mrr == 0.5


def test_section_reference_never_grants_undelivered_rest_of_section(graph):
    mapped, _ = map_modules(graph, [module(graph, (CHUNK_A,))])
    result = evaluate_retrieval(graph.task, graph.evidence, mapped)
    assert len(mapped[0].spans) == 1
    assert mapped[0].spans[0].span.end < graph.evidence.evidence[0].span.end
    assert result.metrics.fact_recall == result.metrics.evidence_recall == 0.0


def test_section_reference_without_chunk_delivery_is_error(graph):
    with pytest.raises(provenance.RetrievalProvenanceError, match="included chunk spans"):
        map_modules(graph, [module(graph, (), content=graph.source.text)])


@pytest.mark.parametrize("change", ["chunk_body", "missing_chunk", "chunk_document", "chunk_section"])
def test_invalid_indexed_chunk_or_reference_fails_closed(graph, change):
    delivered = module(graph)
    if change == "chunk_body":
        graph.chunks[CHUNK_A].content = "Tampered synthetic content"
    elif change == "missing_chunk":
        del graph.chunks[CHUNK_A]
    elif change == "chunk_document":
        graph.chunks[CHUNK_A].document_id = OTHER_ID
    else:
        graph.chunks[CHUNK_A].section_id = OTHER_ID
    with pytest.raises(provenance.RetrievalProvenanceError):
        map_modules(graph, [delivered])


@pytest.mark.parametrize("offsets", [(-1, 10), (10, 10), (20, 10), (0, 100000)])
def test_bad_chunk_offsets_fail_closed(graph, offsets):
    delivered = module(graph)
    graph.chunks[CHUNK_A].start_offset, graph.chunks[CHUNK_A].end_offset = offsets
    with pytest.raises(provenance.RetrievalProvenanceError, match="frozen source slice"):
        map_modules(graph, [delivered])


@pytest.mark.parametrize("document_field", ["sha256", "normalized_text"])
def test_indexed_document_version_must_match_frozen_source(graph, document_field):
    setattr(graph.document, document_field, "0" * 64 if document_field == "sha256" else "Changed text")
    with pytest.raises(provenance.RetrievalProvenanceError, match="Indexed document"):
        map_modules(graph, [module(graph)])


@pytest.mark.parametrize("change", ["outside_document", "wrong_ref_document", "wrong_ref_section"])
def test_module_and_source_ref_identities_are_verified(graph, change):
    delivered = module(graph)
    if change == "outside_document":
        delivered.document_id = OTHER_ID
    elif change == "wrong_ref_document":
        delivered.source_refs[1] = ref("chunk", CHUNK_A, OTHER_ID)
    else:
        delivered.source_refs[0] = ref("section", OTHER_ID)
    with pytest.raises(provenance.RetrievalProvenanceError):
        map_modules(graph, [delivered])


@pytest.mark.parametrize("change", ["truncate", "reorder", "extra_content"])
def test_delivered_module_must_equal_the_exact_included_chunk_bodies(graph, change):
    delivered = module(graph)
    if change == "truncate":
        delivered.content = delivered.content[:-1]
    elif change == "reorder":
        delivered.content = "\n\n".join([graph.chunks[CHUNK_B].content, graph.chunks[CHUNK_A].content])
    else:
        delivered.content += " Added unrelated generated prose."
    with pytest.raises(provenance.RetrievalProvenanceError, match="included chunk spans"):
        map_modules(graph, [delivered])


def test_changed_chunk_ids_with_identical_delivered_source_spans_leave_metrics_identical(graph):
    original, _ = map_modules(graph, [module(graph)])
    graph.chunks[OTHER_ID] = graph.chunks.pop(CHUNK_A)
    remapped, _ = map_modules(graph, [module(graph, (OTHER_ID, CHUNK_B))])
    assert original == remapped
    assert evaluate_retrieval(graph.task, graph.evidence, original) == evaluate_retrieval(
        graph.task, graph.evidence, remapped,
    )


def test_different_chunking_over_same_visible_source_preserves_coverage(graph):
    original, _ = map_modules(graph, [module(graph)])
    eu = graph.evidence.evidence[0]
    graph.chunks[OTHER_ID] = SimpleNamespace(
        document_id=DOCUMENT_ID, section_id=SECTION_ID,
        start_offset=eu.span.start, end_offset=eu.span.end,
        content=graph.source.text[eu.span.start:eu.span.end],
    )
    whole, _ = map_modules(graph, [module(graph, (OTHER_ID,))])
    assert evaluate_retrieval(graph.task, graph.evidence, original).metrics == evaluate_retrieval(
        graph.task, graph.evidence, whole,
    ).metrics
