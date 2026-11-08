from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.db.models import MemoryImportance, MemoryKind
from app.knowledge.sources import SourceRef


class ModelRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_tokens: int
    temperature: float


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ReadWorkingContextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_refs(self) -> "ReadWorkingContextArgs":
        if len(set(self.refs)) != len(self.refs):
            raise ValueError("Working context refs must be unique")
        return self


class TaskStateEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    refs: list[str] = Field(default_factory=list)


class CheckpointTaskStateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    established_facts_add: list[TaskStateEntry] = Field(default_factory=list)
    decisions_add: list[TaskStateEntry] = Field(default_factory=list)
    open_questions_add: list[TaskStateEntry] = Field(default_factory=list)
    completed_actions_add: list[TaskStateEntry] = Field(default_factory=list)


class ModelResponse(BaseModel):
    assistant_message: dict[str, Any]
    final_text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class TextDelta(BaseModel):
    text: str


class ModelCompleted(BaseModel):
    response: ModelResponse


ModelEvent = TextDelta | ModelCompleted


class ModelClient(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...


class SearchMemoryArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    kinds: list[MemoryKind] | None = None
    limit: int = Field(default=8, ge=1, le=20)


class MemoryMutation(BaseModel):
    operation: Literal["ADD", "MERGE", "SUPERSEDE", "ARCHIVE"]
    target_memory_id: UUID | None = None
    kind: MemoryKind | None = None
    content: str | None = Field(default=None, max_length=20_000)
    attributes: dict[str, Any] = Field(default_factory=dict)
    importance: MemoryImportance = MemoryImportance.NORMAL
    source_refs: list[SourceRef] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_operation(self) -> "MemoryMutation":
        if self.operation == "ADD":
            if self.target_memory_id is not None or self.kind is None or not self.content:
                raise ValueError("ADD requires kind and content and does not accept target_memory_id")
        elif self.operation == "MERGE":
            if self.target_memory_id is None or not self.content:
                raise ValueError("MERGE requires target_memory_id and content")
        elif self.operation == "SUPERSEDE":
            if self.target_memory_id is None or self.kind is None or not self.content:
                raise ValueError("SUPERSEDE requires target_memory_id, kind, and content")
        elif self.operation == "ARCHIVE":
            if self.target_memory_id is None:
                raise ValueError("ARCHIVE requires target_memory_id")
            if self.source_refs:
                raise ValueError("ARCHIVE does not accept source_refs")
        if self.operation != "ARCHIVE" and not self.source_refs:
            raise ValueError("A non-ARCHIVE mutation requires at least one source_ref")
        return self


class UpdateUserModelArgs(BaseModel):
    mutations: list[MemoryMutation] = Field(min_length=1, max_length=20)


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[UUID] | None = Field(default=None, max_length=50)
    max_modules: int = Field(default=6, ge=1, le=16)
    max_tokens: int = Field(default=8000, ge=1, le=24_000)


class ReadSourceArgs(BaseModel):
    source_refs: list[SourceRef] = Field(min_length=1, max_length=12)
    max_tokens: int = Field(default=8000, ge=1, le=24_000)


class ListDocumentsArgs(BaseModel):
    query: str | None = Field(default=None, max_length=500)
    limit: int = Field(default=20, ge=1, le=100)


class SaveSourceArgs(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=2_000_000)
    format: Literal["MARKDOWN", "TEXT", "LATEX"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolError(BaseModel):
    code: Literal["VALIDATION_ERROR", "NOT_FOUND", "DOMAIN_ERROR", "TIMEOUT", "INTERNAL_ERROR", "CANCELLED", "POSSIBLE_DUPLICATE"]
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ok: bool
    agent_payload: Any | None = None
    ui_summary: str = ""
    source_refs: list[SourceRef] = Field(default_factory=list)
    result_ref: str | None = None
    error: ToolError | None = None
    observation_payload: dict[str, Any] | None = Field(default=None, exclude=True)

    def provider_payload(self) -> dict[str, Any]:
        result = self.model_dump(mode="json")
        result.pop("ui_summary", None)
        result.pop("result_ref", None)
        return result
