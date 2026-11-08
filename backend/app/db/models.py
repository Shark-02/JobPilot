from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, CheckConstraint, DateTime, Enum, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass


def _uuid() -> UUID:
    return uuid4()


class MessageRole(str, enum.Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class DocumentFormat(str, enum.Enum):
    MARKDOWN = "MARKDOWN"
    TEXT = "TEXT"
    LATEX = "LATEX"


class DocumentRole(str, enum.Enum):
    USER_CAREER_SOURCE = "USER_CAREER_SOURCE"
    REFERENCE_SOURCE = "REFERENCE_SOURCE"


class DocumentStatus(str, enum.Enum):
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class MemoryKind(str, enum.Enum):
    BACKGROUND = "BACKGROUND"
    EXPERIENCE = "EXPERIENCE"
    ACHIEVEMENT = "ACHIEVEMENT"
    SKILL = "SKILL"
    CAREER_GOAL = "CAREER_GOAL"
    PREFERENCE = "PREFERENCE"
    LEARNING = "LEARNING"
    OTHER = "OTHER"


class MemoryImportance(str, enum.Enum):
    CORE = "CORE"
    NORMAL = "NORMAL"


class MemoryStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    ARCHIVED = "ARCHIVED"


class SourceType(str, enum.Enum):
    MESSAGE = "MESSAGE"
    DOCUMENT_SECTION = "DOCUMENT_SECTION"
    DOCUMENT_CHUNK = "DOCUMENT_CHUNK"


class SourceRelation(str, enum.Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"


class TurnStatus(str, enum.Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ABORTED = "ABORTED"


class EmbeddingProfileStatus(str, enum.Enum):
    REGISTERED = "REGISTERED"
    BACKFILLING = "BACKFILLING"
    READY = "READY"
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class EmbeddingProfile(Base):
    __tablename__ = "embedding_profile"
    __table_args__ = (
        CheckConstraint("dimension BETWEEN 1 AND 2000", name="ck_embedding_profile_dimension"),
        CheckConstraint("max_input_tokens > 0 AND batch_limit > 0", name="ck_embedding_profile_limits"),
        CheckConstraint(
            "(identity_version = 1 AND legacy_identity_json IS NOT NULL AND space_revision = '') "
            "OR (identity_version = 2 AND legacy_identity_json IS NULL)",
            name="ck_embedding_profile_identity_version",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    provider: Mapped[str] = mapped_column(String(64), default="openai-compatible")
    base_url: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(200))
    dimension: Mapped[int] = mapped_column(Integer)
    max_input_tokens: Mapped[int] = mapped_column(Integer)
    batch_limit: Mapped[int] = mapped_column(Integer)
    send_dimensions: Mapped[bool] = mapped_column(Boolean)
    normalize: Mapped[bool] = mapped_column(Boolean)
    query_prefix: Mapped[str] = mapped_column(Text, default="", server_default="")
    document_prefix: Mapped[str] = mapped_column(Text, default="", server_default="")
    tokenizer_encoding: Mapped[str] = mapped_column(String(100))
    api_key_env: Mapped[str] = mapped_column(String(100))
    space_revision: Mapped[str] = mapped_column(String(200), default="", server_default="")
    identity_version: Mapped[int] = mapped_column(Integer, default=2, server_default="2")
    legacy_identity_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[EmbeddingProfileStatus] = mapped_column(Enum(EmbeddingProfileStatus, name="embedding_profile_status"))
    validation_passed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    validation_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ActiveEmbeddingProfile(Base):
    __tablename__ = "active_embedding_profile"
    __table_args__ = (CheckConstraint("singleton_id = 1", name="ck_active_embedding_profile_singleton"),)

    singleton_id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    profile_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_profile.id", ondelete="RESTRICT"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Conversation(Base):
    __tablename__ = "conversation"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    title: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    eval_run_id: Mapped[UUID | None] = mapped_column(index=True)
    eval_case_id: Mapped[str | None] = mapped_column(String(64), index=True)
    retracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "message"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversation.id", ondelete="CASCADE"), index=True)
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole, name="message_role"))
    content: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    eval_run_id: Mapped[UUID | None] = mapped_column(index=True)
    eval_case_id: Mapped[str | None] = mapped_column(String(64), index=True)
    retracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_document"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(Text)
    filename: Mapped[str | None] = mapped_column(Text)
    format: Mapped[DocumentFormat] = mapped_column(Enum(DocumentFormat, name="document_format"))
    source_role: Mapped[DocumentRole] = mapped_column(Enum(DocumentRole, name="document_role"))
    raw_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus, name="document_status"), index=True)
    parser_version: Mapped[str] = mapped_column(String(64))
    chunker_version: Mapped[str] = mapped_column(String(64))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    sections: Mapped[list["KnowledgeSection"]] = relationship(cascade="all, delete-orphan")
    chunks: Mapped[list["KnowledgeChunk"]] = relationship(cascade="all, delete-orphan")


class KnowledgeSection(Base):
    __tablename__ = "knowledge_section"
    __table_args__ = (UniqueConstraint("document_id", "ordinal", name="uq_document_section_ordinal"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("knowledge_document.id", ondelete="CASCADE"), index=True)
    parent_section_id: Mapped[UUID | None] = mapped_column(ForeignKey("knowledge_section.id", ondelete="CASCADE"))
    heading: Mapped[str | None] = mapped_column(Text)
    level: Mapped[int] = mapped_column(Integer)
    section_path: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    ordinal: Mapped[int] = mapped_column(Integer)
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunk"
    __table_args__ = (UniqueConstraint("document_id", "ordinal", name="uq_document_chunk_ordinal"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("knowledge_document.id", ondelete="CASCADE"), index=True)
    section_id: Mapped[UUID] = mapped_column(ForeignKey("knowledge_section.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    retrieval_text: Mapped[str] = mapped_column(Text)
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeChunkEmbedding(Base):
    __tablename__ = "knowledge_chunk_embedding"

    chunk_id: Mapped[UUID] = mapped_column(ForeignKey("knowledge_chunk.id", ondelete="CASCADE"), primary_key=True)
    profile_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_profile.id", ondelete="RESTRICT"), primary_key=True, index=True)
    embedding: Mapped[list[float]] = mapped_column(Vector())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MemoryItem(Base):
    __tablename__ = "memory_item"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    kind: Mapped[MemoryKind] = mapped_column(Enum(MemoryKind, name="memory_kind"), index=True)
    content: Mapped[str] = mapped_column(Text)
    attributes: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    importance: Mapped[MemoryImportance] = mapped_column(Enum(MemoryImportance, name="memory_importance"), index=True)
    status: Mapped[MemoryStatus] = mapped_column(Enum(MemoryStatus, name="memory_status"), index=True)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    eval_run_id: Mapped[UUID | None] = mapped_column(index=True)
    eval_case_id: Mapped[str | None] = mapped_column(String(64), index=True)
    retracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sources: Mapped[list["MemorySource"]] = relationship(cascade="all, delete-orphan")


class MemoryItemEmbedding(Base):
    __tablename__ = "memory_item_embedding"

    memory_item_id: Mapped[UUID] = mapped_column(ForeignKey("memory_item.id", ondelete="CASCADE"), primary_key=True)
    profile_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_profile.id", ondelete="RESTRICT"), primary_key=True, index=True)
    embedding: Mapped[list[float]] = mapped_column(Vector())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MemorySource(Base):
    __tablename__ = "memory_source"
    __table_args__ = (
        CheckConstraint(
            "(CASE WHEN message_id IS NOT NULL THEN 1 ELSE 0 END) + "
            "(CASE WHEN document_section_id IS NOT NULL THEN 1 ELSE 0 END) + "
            "(CASE WHEN document_chunk_id IS NOT NULL THEN 1 ELSE 0 END) = 1",
            name="ck_memory_source_exactly_one_source",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    memory_id: Mapped[UUID] = mapped_column(ForeignKey("memory_item.id", ondelete="CASCADE"), index=True)
    source_type: Mapped[SourceType] = mapped_column(Enum(SourceType, name="source_type"))
    message_id: Mapped[UUID | None] = mapped_column(ForeignKey("message.id", ondelete="CASCADE"), index=True)
    document_section_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("knowledge_section.id", ondelete="CASCADE"), index=True
    )
    document_chunk_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("knowledge_chunk.id", ondelete="CASCADE"), index=True
    )
    relation: Mapped[SourceRelation] = mapped_column(Enum(SourceRelation, name="source_relation"))
    excerpt: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentTurn(Base):
    __tablename__ = "agent_turn"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversation.id", ondelete="CASCADE"), index=True)
    user_message_id: Mapped[UUID] = mapped_column(ForeignKey("message.id", ondelete="CASCADE"))
    assistant_message_id: Mapped[UUID | None] = mapped_column(ForeignKey("message.id", ondelete="SET NULL"))
    status: Mapped[TurnStatus] = mapped_column(Enum(TurnStatus, name="turn_status"), index=True)
    model_turn_count: Mapped[int] = mapped_column(Integer, default=0)
    tool_execution_count: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    eval_run_id: Mapped[UUID | None] = mapped_column(index=True)
    eval_case_id: Mapped[str | None] = mapped_column(String(64), index=True)


class ModelCallTrace(Base):
    __tablename__ = "model_call_trace"
    __table_args__ = (UniqueConstraint("turn_id", "call_type", "sequence", name="uq_model_trace_turn_type_sequence"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("agent_turn.id", ondelete="CASCADE"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    call_type: Mapped[str] = mapped_column(String(32), default="MAIN_AGENT", server_default="MAIN_AGENT")
    provider: Mapped[str | None] = mapped_column(String(200))
    model: Mapped[str] = mapped_column(String(200))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    finish_reason: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(50))
    error_code: Mapped[str | None] = mapped_column(String(128))
    egress_preview: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    eval_run_id: Mapped[UUID | None] = mapped_column(index=True)
    eval_case_id: Mapped[str | None] = mapped_column(String(64), index=True)


class ToolCallTrace(Base):
    __tablename__ = "tool_call_trace"
    __table_args__ = (UniqueConstraint("turn_id", "sequence", name="uq_tool_trace_turn_sequence"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("agent_turn.id", ondelete="CASCADE"), index=True)
    provider_call_id: Mapped[str | None] = mapped_column(String(255))
    sequence: Mapped[int] = mapped_column(Integer)
    tool_name: Mapped[str] = mapped_column(String(100))
    arguments: Mapped[dict] = mapped_column(JSONB)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(50))
    ui_summary: Mapped[str | None] = mapped_column(Text)
    source_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    eval_run_id: Mapped[UUID | None] = mapped_column(index=True)
    eval_case_id: Mapped[str | None] = mapped_column(String(64), index=True)


class AgentObservation(Base):
    __tablename__ = "agent_observation"
    __table_args__ = (UniqueConstraint("turn_id", "sequence", name="uq_agent_observation_turn_sequence"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("agent_turn.id", ondelete="CASCADE"), index=True)
    parent_observation_id: Mapped[UUID | None] = mapped_column(ForeignKey("agent_observation.id", ondelete="CASCADE"))
    sequence: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(32))
    name: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32))
    input_payload: Mapped[dict | None] = mapped_column(JSONB)
    output_payload: Mapped[dict | None] = mapped_column(JSONB)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, server_default="{}")
    source_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    eval_run_id: Mapped[UUID | None] = mapped_column(index=True)
    eval_case_id: Mapped[str | None] = mapped_column(String(64), index=True)


Index("idx_knowledge_chunk_trgm", KnowledgeChunk.retrieval_text, postgresql_using="gin", postgresql_ops={"retrieval_text": "gin_trgm_ops"})
Index("idx_memory_content_trgm", MemoryItem.content, postgresql_using="gin", postgresql_ops={"content": "gin_trgm_ops"})
Index("uq_embedding_profile_one_active", EmbeddingProfile.status, unique=True,
      postgresql_where=EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE)
