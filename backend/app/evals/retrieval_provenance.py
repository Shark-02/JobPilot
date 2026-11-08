"""Eval-only verified mapping from delivered chunk references to frozen source spans.

Text equality here validates delivery/provenance, never semantic relevance.
No provider, embedding, or product retrieval behavior is changed by this adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID

from app.db.models import DocumentFormat
from app.evals.retrieval_contract import EvidenceFile, RetrievedModule, SourceSpan, Span
from app.knowledge.parser import PARSER_VERSION, DocumentParser, ParsedSection


class RetrievalProvenanceError(ValueError):
    pass


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenSource:
    document_key: str
    snapshot_hash: str
    canonical_hash: str
    text: str
    sections: list[ParsedSection]


def frozen_sources(root: Path, manifest: Any) -> dict[str, FrozenSource]:
    """Pin Unicode [start,end) coordinates to the verified parser-v1 source view."""
    if PARSER_VERSION != "v1":
        raise RetrievalProvenanceError("Frozen evidence requires the reviewed parser-v1 source view")
    sources = {}
    for spec in manifest.frozen_documents:
        path = (root / spec.snapshot_path).resolve()
        if not path.is_relative_to((root / "corpora").resolve()) or not path.is_file():
            raise RetrievalProvenanceError("Missing or outside-corpus source snapshot")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != spec.sha256:
            raise RetrievalProvenanceError("Frozen source snapshot hash mismatch")
        format_ = DocumentFormat.LATEX if Path(spec.filename).suffix == ".tex" else DocumentFormat.MARKDOWN
        parsed = DocumentParser().parse(raw.decode("utf-8", errors="strict"), format_)
        sources[spec.key] = FrozenSource(spec.key, spec.sha256, digest(parsed.normalized_text),
                                         parsed.normalized_text, parsed.sections)
    return sources


def validate_evidence_sources(evidence: EvidenceFile, sources: dict[str, FrozenSource]) -> None:
    for unit in evidence.evidence:
        source = sources.get(unit.document_key)
        if source is None or (unit.snapshot_hash, unit.canonical_hash) != (
                source.snapshot_hash, source.canonical_hash):
            raise RetrievalProvenanceError("Evidence unit source identity/hash mismatch")
        if unit.span.end > len(source.text) or digest(source.text[unit.span.start:unit.span.end]) != unit.span_hash:
            raise RetrievalProvenanceError("Evidence unit span/hash mismatch")
        if not any(section.start_offset <= unit.span.start < unit.span.end <= section.end_offset
                   and section.section_path == unit.section_path for section in source.sections):
            raise RetrievalProvenanceError("Evidence unit section/span provenance mismatch")


def map_retrieved_modules(
    modules: list[Any], *, logical_key_by_document: dict[UUID, str],
    documents: dict[UUID, Any], chunks: dict[UUID, Any], sources: dict[str, FrozenSource],
) -> tuple[list[RetrievedModule], list[dict[str, Any]]]:
    """Credit only verified included chunks, never an entire section reference.

    The existing retriever delivers whole chunk bodies joined by two newlines.
    If assembly/truncation changes, fail closed until its adapter is reviewed.
    """
    mapped, records = [], []
    for rank, module in enumerate(modules, 1):
        key = logical_key_by_document.get(module.document_id)
        source = sources.get(key)
        document = documents.get(module.document_id)
        if source is None or document is None:
            raise RetrievalProvenanceError("Retrieved module source is outside frozen corpus")
        if document.sha256 != source.snapshot_hash or digest(document.normalized_text) != source.canonical_hash:
            raise RetrievalProvenanceError("Indexed document differs from frozen source coordinates")
        spans, bodies, chunk_ids = [], [], []
        for ref in module.source_refs:
            scheme, document_id, kind, source_id = ref.parts()
            if scheme != "document" or document_id != module.document_id:
                raise RetrievalProvenanceError("Retrieved module SourceRef document mismatch")
            if kind == "section":
                if source_id != module.section_id:
                    raise RetrievalProvenanceError("Retrieved module section reference mismatch")
                continue
            if kind != "chunk":
                raise RetrievalProvenanceError("Retrieved module has an unsupported source kind")
            chunk = chunks.get(source_id)
            if (chunk is None or chunk.document_id != module.document_id
                    or chunk.section_id != module.section_id):
                raise RetrievalProvenanceError("Retrieved module chunk reference mismatch")
            start, end = chunk.start_offset, chunk.end_offset
            if not 0 <= start < end <= len(source.text) or chunk.content != source.text[start:end]:
                raise RetrievalProvenanceError("Indexed chunk is not its declared frozen source slice")
            spans.append(SourceSpan(document_key=key, snapshot_hash=source.snapshot_hash,
                                    canonical_hash=source.canonical_hash, span=Span(start=start, end=end)))
            bodies.append(chunk.content)
            chunk_ids.append(str(source_id))
        if not spans or module.content != "\n\n".join(bodies):
            raise RetrievalProvenanceError("Delivered module content is not its included chunk spans")
        mapped.append(RetrievedModule(module_id=str(module.section_id), spans=spans))
        records.append({"rank": rank, "document_id": str(module.document_id),
                        "logical_document_key": key, "section_id": str(module.section_id),
                        "section_path": module.section_path, "included_chunk_ids": chunk_ids,
                        "spans": [span.model_dump(mode="json") for span in spans]})
    return mapped, records
