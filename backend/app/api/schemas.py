from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import DocumentFormat, DocumentRole, DocumentStatus, MemoryImportance, MemoryKind, MemoryStatus, MessageRole, TurnStatus


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ConversationCreate(APIModel):
    title: str | None = Field(default=None, max_length=300)


class ConversationOut(APIModel):
    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class MessageOut(APIModel):
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(alias="metadata_")
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class RunCreate(APIModel):
    content: str = Field(min_length=1, max_length=100_000)
    attachment_document_ids: list[UUID] = Field(default_factory=list, max_length=12)


class RunOut(APIModel):
    turn_id: UUID
    status: TurnStatus
    events_url: str


class AgentTurnOut(APIModel):
    id: UUID
    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID | None
    status: TurnStatus
    model_turn_count: int
    tool_execution_count: int
    error_code: str | None
    started_at: datetime
    finished_at: datetime | None


class DocumentCreate(APIModel):
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=2_000_000)
    format: DocumentFormat
    source_role: DocumentRole = DocumentRole.USER_CAREER_SOURCE
    filename: str | None = Field(default=None, max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentOut(APIModel):
    id: UUID
    title: str
    filename: str | None
    format: DocumentFormat
    source_role: DocumentRole
    status: DocumentStatus
    sha256: str
    metadata: dict[str, Any] = Field(alias="metadata_")
    created_at: datetime
    updated_at: datetime


class DocumentDetail(DocumentOut):
    section_count: int = 0
    chunk_count: int = 0


class DocumentDeleteOut(APIModel):
    deleted: bool
    document_id: UUID
    deleted_sections: int
    deleted_chunks: int


class MemoryOut(APIModel):
    id: UUID
    kind: MemoryKind
    content: str
    attributes: dict[str, Any]
    importance: MemoryImportance
    status: MemoryStatus
    valid_from: datetime | None
    valid_to: datetime | None
    created_at: datetime
    updated_at: datetime


class ModelCallOut(APIModel):
    id: UUID
    turn_id: UUID
    sequence: int
    call_type: str
    provider: str | None
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int | None
    finish_reason: str | None
    status: str
    error_code: str | None
    created_at: datetime


class ToolCallOut(APIModel):
    id: UUID
    turn_id: UUID
    provider_call_id: str | None
    sequence: int
    tool_name: str
    arguments: dict[str, Any]
    fingerprint: str
    status: str
    ui_summary: str | None
    source_refs: list[str]
    latency_ms: int | None
    error_code: str | None
    created_at: datetime
