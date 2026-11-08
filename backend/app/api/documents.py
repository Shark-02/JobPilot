from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import DocumentCreate, DocumentDeleteOut, DocumentDetail, DocumentOut
from app.config import get_settings
from app.infra.db import get_session
from app.embedding.service import EmbeddingProfileService
from app.infra.token_count import TokenCounter
from app.knowledge.ingest import IngestRequest, KnowledgeIngestService
from app.knowledge.repository import KnowledgeRepository
from app.memory.repository import MemoryRepository

router = APIRouter(prefix="/api/documents", tags=["documents"])


def detail(document: object, section_count: int, chunk_count: int) -> DocumentDetail:
    return DocumentDetail.model_validate(document).model_copy(update={"section_count": section_count, "chunk_count": chunk_count})


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(payload: DocumentCreate, session: AsyncSession = Depends(get_session)) -> DocumentOut:
    embedding = await EmbeddingProfileService(session, get_settings()).client_for_active(for_write=True)
    service = KnowledgeIngestService(session, embedding, TokenCounter(get_settings().llm_model))
    result = await service.ingest(IngestRequest(**payload.model_dump()))
    return result.document


@router.get("", response_model=list[DocumentOut])
async def list_documents(query: str | None = None, limit: int = 20, session: AsyncSession = Depends(get_session)) -> list[DocumentOut]:
    return await KnowledgeRepository(session).list(query, max(1, min(limit, 100)))


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: UUID, session: AsyncSession = Depends(get_session)) -> DocumentDetail:
    repo = KnowledgeRepository(session)
    document = await repo.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    sections, chunks = await repo.counts(document_id)
    return detail(document, sections, chunks)


@router.post("/{document_id}/archive", response_model=DocumentOut)
async def archive_document(document_id: UUID, session: AsyncSession = Depends(get_session)) -> DocumentOut:
    document = await KnowledgeRepository(session).archive(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await session.commit()
    await session.refresh(document)
    return document


@router.delete("/{document_id}", response_model=DocumentDeleteOut)
async def delete_document(document_id: UUID, session: AsyncSession = Depends(get_session)) -> DocumentDeleteOut:
    repo = KnowledgeRepository(session)
    async with session.begin():
        if await repo.get(document_id) is None:
            raise HTTPException(status_code=404, detail="Document not found")
        sections, chunks = await repo.counts(document_id)
        await repo.delete(document_id)
        await MemoryRepository(session).cleanup_orphans()
    return DocumentDeleteOut(deleted=True, document_id=document_id, deleted_sections=sections, deleted_chunks=chunks)
