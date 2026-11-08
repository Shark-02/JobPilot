from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.model import (
    CheckpointTaskStateArgs, ListDocumentsArgs, ReadSourceArgs, ReadWorkingContextArgs,
    SaveSourceArgs, SearchKnowledgeArgs, SearchMemoryArgs,
    ToolResult, UpdateUserModelArgs,
)
from app.config import Settings
from app.db.models import DocumentFormat, DocumentRole
from app.embedding.service import EmbeddingProfileService
from app.infra.token_count import TokenCounter
from app.knowledge.ingest import IngestRequest, KnowledgeIngestService
from app.knowledge.repository import KnowledgeRepository
from app.knowledge.retrieval import KnowledgeRetriever
from app.knowledge.sources import SourceResolver
from app.memory.retrieval import MemoryRetriever
from app.memory.service import MemoryService, MutationCommand


@dataclass(frozen=True)
class ToolUIMetadata:
    label: str
    icon: str


ToolCallable = Callable[[BaseModel], Awaitable[ToolResult]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]
    executor: ToolCallable
    ui: ToolUIMetadata
    internal: bool = False


class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._definitions = {item.name: item for item in definitions}

    def get(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {
                "name": item.name, "description": item.description,
                "parameters": item.args_model.model_json_schema(),
            }}
            for item in self._definitions.values()
        ]


def build_default_registry(session: AsyncSession, settings: Settings) -> ToolRegistry:
    counter = TokenCounter(settings.llm_model)
    profiles = EmbeddingProfileService(session, settings)
    resolver = SourceResolver(session, counter)

    async def runtime_only(args: BaseModel) -> ToolResult:
        # These tools require the current pending interaction and are dispatched
        # by the loop's atomic tool batch, never by the ordinary cached executor.
        raise RuntimeError("Working context tools require the current AgentContext")

    async def search_memory(args: BaseModel) -> ToolResult:
        typed = SearchMemoryArgs.model_validate(args)
        embedding = await profiles.client_for_active()
        matches = await MemoryRetriever(session, embedding).search(typed.query, typed.kinds, typed.limit)
        refs = [ref for match in matches for ref in match.source_refs]
        return ToolResult(
            ok=True, agent_payload={"memories": [item.agent_payload() for item in matches]},
            ui_summary=f"Found {len(matches)} relevant memories", source_refs=refs,
            observation_payload={
                "input": {"query": typed.query, "kinds": [item.value for item in typed.kinds or []], "limit": typed.limit},
                "output": {"memory_ids": [str(item.id) for item in matches]},
                "metadata": {"algorithm": "lexical + cosine vector + reciprocal-rank fusion", **embedding.profile_metadata()},
                "source_refs": [item.uri for item in refs],
            },
        )

    async def update_user_model(args: BaseModel) -> ToolResult:
        typed = UpdateUserModelArgs.model_validate(args)
        embedding = await profiles.client_for_active(for_write=True)
        commands = [
            MutationCommand(
                operation=item.operation, target_memory_id=item.target_memory_id, kind=item.kind,
                content=item.content, attributes=item.attributes, importance=item.importance,
                source_refs=item.source_refs,
            )
            for item in typed.mutations
        ]
        changed = await MemoryService(session, embedding, resolver).mutate(commands)
        return ToolResult(
            ok=True,
            agent_payload={"changed_memory_ids": [str(item.id) for item in changed]},
            ui_summary=f"Updated {len(changed)} long-term memory item(s)",
        )

    async def search_knowledge(args: BaseModel) -> ToolResult:
        typed = SearchKnowledgeArgs.model_validate(args)
        embedding = await profiles.client_for_active()
        modules, debug = await KnowledgeRetriever(session, embedding, counter).search_with_debug(
            typed.query, document_ids=typed.document_ids, max_modules=typed.max_modules, max_tokens=typed.max_tokens
        )
        refs = [ref for module in modules for ref in module.source_refs]
        return ToolResult(
            ok=True, agent_payload={"modules": [item.agent_payload() for item in modules]},
            ui_summary=f"Found {len(modules)} source module(s)", source_refs=refs, observation_payload=debug,
        )

    async def read_source(args: BaseModel) -> ToolResult:
        typed = ReadSourceArgs.model_validate(args)
        sources = await resolver.read(typed.source_refs, typed.max_tokens)
        return ToolResult(
            ok=True,
            agent_payload={"sources": [
                {"source_ref": item.source_ref.uri, "title": item.title, "content": item.content, "metadata": item.metadata}
                for item in sources
            ]},
            ui_summary=f"Read {len(sources)} source item(s)", source_refs=[item.source_ref for item in sources],
        )

    async def list_documents(args: BaseModel) -> ToolResult:
        typed = ListDocumentsArgs.model_validate(args)
        documents = await KnowledgeRepository(session).list(typed.query, typed.limit)
        return ToolResult(
            ok=True,
            agent_payload={"documents": [
                {"id": str(item.id), "title": item.title, "filename": item.filename, "format": item.format.value,
                 "source_role": item.source_role.value, "status": item.status.value, "metadata": item.metadata_}
                for item in documents
            ]},
            ui_summary=f"Listed {len(documents)} document(s)",
        )

    async def save_source(args: BaseModel) -> ToolResult:
        typed = SaveSourceArgs.model_validate(args)
        embedding = await profiles.client_for_active(for_write=True)
        result = await KnowledgeIngestService(session, embedding, counter).ingest(IngestRequest(
            title=typed.title, content=typed.content, format=DocumentFormat(typed.format),
            source_role=DocumentRole.USER_CAREER_SOURCE, filename=None,
            metadata={"saved_by": "agent", "user_explicit_request": True, **typed.metadata},
        ))
        return ToolResult(
            ok=True, agent_payload={"document_id": str(result.document.id), "deduplicated": result.deduplicated},
            ui_summary="Used existing long-term source" if result.deduplicated else "Saved long-term source",
        )

    return ToolRegistry([
        ToolDefinition(
            "read_working_context", "Read exact complete interactions by ctx:// refs from this turn only. No search; capacity errors are explicit.",
            ReadWorkingContextArgs, runtime_only, ToolUIMetadata("Read working context", "history"), internal=True,
        ),
        ToolDefinition(
            "checkpoint_task_state", "Add small facts, decisions, open questions or completed actions at a meaningful task boundary. Use existing ctx:// or original source refs; do not summarize every round.",
            CheckpointTaskStateArgs, runtime_only, ToolUIMetadata("Checkpoint task state", "list-check"), internal=True,
        ),
        ToolDefinition("search_memory", "Search relevant active long-term user memory.", SearchMemoryArgs, search_memory, ToolUIMetadata("Search memory", "memory")),
        ToolDefinition("update_user_model", "Create or change evidence-backed long-term user memory.", UpdateUserModelArgs, update_user_model, ToolUIMetadata("Update profile", "user-round")),
        ToolDefinition("search_knowledge", "Search uploaded original career material by relevance.", SearchKnowledgeArgs, search_knowledge, ToolUIMetadata("Search sources", "search")),
        ToolDefinition("read_source", "Read an original message, document section, or document chunk by SourceRef.", ReadSourceArgs, read_source, ToolUIMetadata("Read source", "file-text")),
        ToolDefinition("list_documents", "List document metadata when a source needs to be located.", ListDocumentsArgs, list_documents, ToolUIMetadata("List sources", "files")),
        ToolDefinition("save_source", "Save content as a long-term source only after the user explicitly requests it.", SaveSourceArgs, save_source, ToolUIMetadata("Save source", "save")),
    ])
