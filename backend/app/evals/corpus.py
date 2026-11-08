"""Frozen corpus setup/check and explicitly requested canonical Eval-only backfill."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.models import (
    DocumentFormat, DocumentRole, DocumentStatus, KnowledgeChunk, KnowledgeChunkEmbedding,
    EmbeddingProfile, KnowledgeDocument, KnowledgeSection,
)
from app.evals.database import assert_canonical_eval_database
from app.infra.token_count import TokenCounter
from app.knowledge.chunker import CHUNKER_VERSION, Chunker
from app.knowledge.parser import PARSER_VERSION, DocumentParser


def frozen_snapshot_texts(dataset) -> dict[str, str]:
    from app.evals.datasets import DatasetError
    manifest = dataset.manifest.model_dump(mode="json")
    result = {}
    root = dataset.root.resolve()
    for item in manifest["frozen_documents"]:
        relative = item.get("snapshot_path")
        if not relative:
            raise DatasetError(f"Frozen snapshot_path is required: {item['key']}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise DatasetError(f"Missing or outside-package frozen snapshot: {item['key']}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise DatasetError(f"Frozen snapshot SHA256 mismatch: {item['key']}")
        try:
            result[item["key"]] = raw.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise DatasetError(f"Frozen snapshot is not UTF-8: {item['key']}") from exc
    return result


async def setup_local_document(session, spec: dict, raw_text: str, corpus_version: str) -> None:
    from app.evals.datasets import DatasetError
    settings = get_settings()
    assert_canonical_eval_database(str(session.bind.url), settings.database_url)
    if await session.scalar(text("SELECT current_database()")) != "jobpilot_eval":
        raise DatasetError("Local corpus writes require canonical jobpilot_eval")
    document_id = uuid5(NAMESPACE_URL, f"jobpilot:eval:{corpus_version}:{spec['key']}:{spec['sha256']}")
    format = DocumentFormat.LATEX if Path(spec["filename"]).suffix == ".tex" else DocumentFormat.MARKDOWN
    parsed = DocumentParser().parse(raw_text, format)
    drafts = Chunker(TokenCounter()).chunk(parsed, spec["title"])
    existing = await session.scalar(select(KnowledgeDocument).where(KnowledgeDocument.sha256 == spec["sha256"]))
    if existing:
        if existing.id != document_id or existing.raw_text != raw_text or existing.normalized_text != parsed.normalized_text:
            raise DatasetError("Local snapshot identity/content changed; reset canonical jobpilot_eval")
        chunks = list((await session.scalars(select(KnowledgeChunk).where(
            KnowledgeChunk.document_id == document_id).order_by(KnowledgeChunk.ordinal)
        )).all())
        if len(chunks) != len(drafts) or any(
            chunk.content != draft.content or chunk.retrieval_text != draft.retrieval_text
            for chunk, draft in zip(chunks, drafts, strict=True)
        ):
            raise DatasetError("Local snapshot chunks differ; reset canonical jobpilot_eval")
        return
    document = KnowledgeDocument(
        id=document_id, title=spec["title"], filename=spec["filename"], format=format,
        source_role=DocumentRole(spec["source_role"]), raw_text=raw_text,
        normalized_text=parsed.normalized_text, sha256=spec["sha256"], status=DocumentStatus.READY,
        parser_version=PARSER_VERSION, chunker_version=CHUNKER_VERSION,
        metadata_={"eval_corpus": corpus_version, "logical_document_key": spec["key"],
                   "snapshot_path": spec["snapshot_path"], "eval_index_origin": "local_frozen_snapshot",
                   "vector_readiness": "verified_per_profile_before_run"},
    )
    session.add(document)
    await session.flush()
    sections = {}
    for section in parsed.sections:
        parent = next((record for ordinal, record in reversed(list(sections.items()))
                       if parsed.sections[ordinal].level < section.level), None)
        record = KnowledgeSection(
            id=uuid5(document_id, f"section:{section.ordinal}"), document_id=document_id,
            parent_section_id=parent.id if parent else None, heading=section.heading, level=section.level,
            section_path=section.section_path, ordinal=section.ordinal,
            start_offset=section.start_offset, end_offset=section.end_offset,
        )
        session.add(record)
        await session.flush()
        sections[section.ordinal] = record
    for draft in drafts:
        session.add(KnowledgeChunk(
            id=uuid5(document_id, f"chunk:{draft.ordinal}:{hashlib.sha256(draft.content.encode()).hexdigest()}"),
            document_id=document_id, section_id=sections[draft.section_ordinal].id,
            ordinal=draft.ordinal, content=draft.content, retrieval_text=draft.retrieval_text,
            start_offset=draft.start_offset, end_offset=draft.end_offset, token_count=draft.token_count,
        ))
    await session.commit()


async def run(action: str, selector: str | None) -> dict:
    from app.embedding.service import EmbeddingProfileService
    from app.evals.datasets import load_dataset, resolve_source_embedding_profile, verify_and_copy_corpus
    settings = get_settings()
    if action == "backfill" and not selector:
        raise ValueError("Canonical Eval backfill requires explicit --embedding-profile")
    target = assert_canonical_eval_database(settings.eval_database_url, settings.database_url)
    dataset = load_dataset()
    manifest = dataset.manifest.model_dump(mode="json")
    frozen_snapshot_texts(dataset)
    engine = create_async_engine(target, pool_pre_ping=True)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            if action in {"setup", "check"}:
                profile = await resolve_source_embedding_profile(
                    source_database_url=settings.database_url, settings=settings, selector=selector)
                documents = await verify_and_copy_corpus(
                    dataset=dataset, source_database_url=settings.database_url, target_session=session,
                    settings=settings, embedding_profile=profile, require_vectors=action == "check")
            else:
                service = EmbeddingProfileService(session, settings)
                profile = await service.get(selector)
                await service.backfill(profile)
                documents = {item.metadata_["logical_document_key"]: item for item in (await session.scalars(
                    select(KnowledgeDocument).where(KnowledgeDocument.sha256.in_(
                        [item["sha256"] for item in manifest["frozen_documents"]]
                    ))
                )).all()}
            rows = []
            for key, document in documents.items():
                chunks = await session.scalar(select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id))
                vectors = await session.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding).join(
                    KnowledgeChunk, KnowledgeChunk.id == KnowledgeChunkEmbedding.chunk_id).where(
                        KnowledgeChunk.document_id == document.id, KnowledgeChunkEmbedding.profile_id == profile.id))
                rows.append({"key": key, "document_id": str(document.id), "sha256": document.sha256,
                             "chunks": chunks, "vectors": vectors, "missing_vectors": chunks - vectors})
            return {"database": "jobpilot_eval", "profile": profile.name, "documents": rows,
                    "profile_status": (await session.scalar(select(EmbeddingProfile.status).where(
                        EmbeddingProfile.id == profile.id))).value,
                    "ready_for_model_eval": all(item["missing_vectors"] == 0 for item in rows),
                    "provider_calls": action == "backfill"}
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "check", "backfill"])
    parser.add_argument("--embedding-profile")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.action, args.embedding_profile)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
