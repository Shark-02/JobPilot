from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from pydantic import RootModel, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import KnowledgeChunk, KnowledgeDocument, KnowledgeSection, Message
from app.evals.scope import apply_eval_scope, get_eval_scope
from app.infra.token_count import TokenCounter


class SourceRef(RootModel[str]):
    """Stable, agent-visible pointer to evidence; never a database foreign key."""

    @property
    def uri(self) -> str:
        return self.root

    @model_validator(mode="after")
    def validate_uri(self) -> "SourceRef":
        try:
            scheme, value = self.root.split("://", maxsplit=1)
            if scheme == "message":
                UUID(value)
            elif scheme == "document":
                document_id, kind, source_id = value.split("/")
                UUID(document_id)
                UUID(source_id)
                if kind not in {"section", "chunk"}:
                    raise ValueError("document source kind must be section or chunk")
            else:
                raise ValueError("source scheme must be message or document")
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid SourceRef: {self.root}") from exc
        return self

    @classmethod
    def message(cls, message_id: UUID) -> "SourceRef":
        return cls(f"message://{message_id}")

    @classmethod
    def section(cls, document_id: UUID, section_id: UUID) -> "SourceRef":
        return cls(f"document://{document_id}/section/{section_id}")

    @classmethod
    def chunk(cls, document_id: UUID, chunk_id: UUID) -> "SourceRef":
        return cls(f"document://{document_id}/chunk/{chunk_id}")

    def parts(self) -> tuple[str, UUID, str | None, UUID | None]:
        scheme, value = self.root.split("://", maxsplit=1)
        if scheme == "message":
            return scheme, UUID(value), None, None
        document_id, kind, source_id = value.split("/")
        return scheme, UUID(document_id), kind, UUID(source_id)


@dataclass(frozen=True)
class ResolvedSource:
    source_ref: SourceRef
    title: str
    content: str
    metadata: dict


class SourceResolver:
    def __init__(self, session: AsyncSession, token_counter: TokenCounter) -> None:
        self.session = session
        self.token_counter = token_counter

    async def validate(self, ref: SourceRef) -> None:
        await self.read([ref], max_tokens=1)

    async def read(self, refs: list[SourceRef], max_tokens: int | None) -> list[ResolvedSource]:
        """Read budgeted Tool evidence, or complete internal evaluator sources.

        None is used only by the offline evaluator's bound-source observation;
        public Tool arguments remain bounded integers.
        """
        if len(refs) > 12:
            raise ValueError("source_refs must contain at most 12 entries")
        remaining = max_tokens
        resolved: list[ResolvedSource] = []
        for ref in refs:
            scheme, first_id, kind, source_id = ref.parts()
            scope = get_eval_scope()
            if scheme == "document" and scope is not None and first_id not in scope.corpus_document_ids:
                raise LookupError(f"Source outside evaluation corpus: {ref.uri}")
            if scheme == "message":
                item = await self.session.scalar(apply_eval_scope(
                    select(Message).where(Message.id == first_id), Message
                ))
                if item is None:
                    raise LookupError(f"Source not found: {ref.uri}")
                title, content, metadata = "Conversation message", item.content, {"role": item.role.value}
            elif kind == "section":
                item = await self.session.scalar(
                    select(KnowledgeSection).where(
                        KnowledgeSection.id == source_id, KnowledgeSection.document_id == first_id
                    )
                )
                if item is None:
                    raise LookupError(f"Source not found: {ref.uri}")
                document = await self.session.scalar(select(KnowledgeDocument).where(KnowledgeDocument.id == first_id))
                content = (document.normalized_text if document else "")[item.start_offset : item.end_offset]
                title, metadata = (document.title if document else "Document section"), {
                    "section_path": item.section_path,
                    "source_role": document.source_role.value if document else None,
                }
            else:
                item = await self.session.scalar(
                    select(KnowledgeChunk).where(
                        KnowledgeChunk.id == source_id, KnowledgeChunk.document_id == first_id
                    )
                )
                if item is None:
                    raise LookupError(f"Source not found: {ref.uri}")
                document = await self.session.scalar(select(KnowledgeDocument).where(KnowledgeDocument.id == first_id))
                title, content, metadata = (document.title if document else "Document chunk"), item.content, {
                    "source_role": document.source_role.value if document else None,
                }
            if remaining is not None and remaining <= 0:
                break
            included = self.token_counter.truncate(content, remaining) if remaining is not None else content
            resolved.append(ResolvedSource(ref, title, included, metadata))
            if remaining is not None:
                remaining -= self.token_counter.count(included)
        return resolved
