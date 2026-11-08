from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentFormat, DocumentRole, DocumentStatus, KnowledgeChunk, KnowledgeChunkEmbedding, KnowledgeDocument, KnowledgeSection
from app.infra.embedding import EmbeddingClient
from app.infra.token_count import TokenCounter
from app.knowledge.chunker import CHUNKER_VERSION, Chunker
from app.knowledge.parser import PARSER_VERSION, DocumentParser
from app.knowledge.repository import KnowledgeRepository


@dataclass(frozen=True)
class IngestRequest:
    title: str
    content: str
    format: DocumentFormat
    source_role: DocumentRole
    filename: str | None
    metadata: dict


@dataclass(frozen=True)
class IngestResult:
    document: KnowledgeDocument
    deduplicated: bool


class KnowledgeIngestService:
    def __init__(self, session: AsyncSession, embedding: EmbeddingClient, counter: TokenCounter) -> None:
        self.session = session
        self.embedding = embedding
        self.counter = counter
        self.repo = KnowledgeRepository(session)
        self.parser = DocumentParser()
        self.chunker = Chunker(counter)

    async def ingest(self, request: IngestRequest) -> IngestResult:
        await self.embedding.assert_resolved_profile(self.session, for_write=True)
        if request.format not in {DocumentFormat.MARKDOWN, DocumentFormat.TEXT, DocumentFormat.LATEX}:
            raise ValueError("Only MARKDOWN, TEXT, and LATEX documents are supported")
        raw_text = request.content.encode("utf-8", errors="strict").decode("utf-8")
        sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        # Serialize same-content uploads before the lookup/insert window. The unique schema
        # constraint remains the last line of defence if a caller bypasses this service.
        lock_id = int(sha256[:16], 16)
        if lock_id >= 2**63:
            lock_id -= 2**64
        await self.session.execute(select(func.pg_advisory_xact_lock(lock_id)))
        existing = await self.repo.by_sha256(sha256)
        if existing is not None:
            if existing.status is not DocumentStatus.FAILED:
                return IngestResult(existing, deduplicated=True)
            # A transient parser/embedding fault must not permanently poison this source hash.
            await self.session.delete(existing)
            await self.session.flush()

        document = KnowledgeDocument(
            title=request.title.strip(), filename=request.filename, format=request.format,
            source_role=request.source_role, raw_text=raw_text, normalized_text="", sha256=sha256,
            status=DocumentStatus.PROCESSING, parser_version=PARSER_VERSION, chunker_version=CHUNKER_VERSION,
            metadata_=request.metadata,
        )
        self.session.add(document)
        await self.session.flush()
        try:
            # The outer transaction keeps the hash lock and PROCESSING document. A nested
            # transaction guarantees a failure cannot leave partial sections/chunks behind.
            async with self.session.begin_nested():
                parsed = self.parser.parse(raw_text, request.format)
                document.normalized_text = parsed.normalized_text
                sections_by_ordinal: dict[int, KnowledgeSection] = {}
                for section in parsed.sections:
                    parent = next(
                        (
                            candidate for ordinal, candidate in reversed(list(sections_by_ordinal.items()))
                            if parsed.sections[ordinal].level < section.level
                        ),
                        None,
                    )
                    record = KnowledgeSection(
                        document_id=document.id, parent_section_id=parent.id if parent else None, heading=section.heading,
                        level=section.level, section_path=section.section_path, ordinal=section.ordinal,
                        start_offset=section.start_offset, end_offset=section.end_offset,
                    )
                    self.session.add(record)
                    await self.session.flush()
                    sections_by_ordinal[section.ordinal] = record
                chunk_drafts = self.chunker.chunk(parsed, document.title)
                embeddings = await self.embedding.embed_many([draft.retrieval_text for draft in chunk_drafts])
                for draft, vector in zip(chunk_drafts, embeddings, strict=True):
                    chunk = KnowledgeChunk(
                        document_id=document.id, section_id=sections_by_ordinal[draft.section_ordinal].id,
                        ordinal=draft.ordinal, content=draft.content, retrieval_text=draft.retrieval_text,
                        start_offset=draft.start_offset, end_offset=draft.end_offset, token_count=draft.token_count,
                    )
                    self.session.add(chunk)
                    await self.session.flush()
                    self.session.add(KnowledgeChunkEmbedding(
                        chunk_id=chunk.id, profile_id=self.embedding.profile.id, embedding=vector,
                    ))
                document.status = DocumentStatus.READY
            await self.session.commit()
            await self.session.refresh(document)
            return IngestResult(document, deduplicated=False)
        except Exception:
            document.status = DocumentStatus.FAILED
            document.normalized_text = ""
            await self.session.commit()
            raise
