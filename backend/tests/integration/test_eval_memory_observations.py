"""Real canonical Eval DB transactions with deterministic local embeddings."""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.config import get_settings
from app.db.models import Conversation, Message, MessageRole, MemoryImportance, MemoryItem, MemoryKind, MemoryStatus
from app.embedding.service import EmbeddingProfileService
from app.evals.evidence import (EvalObservationCollector, set_observation_collector, reset_observation_collector,
                                set_observation_turn, reset_observation_turn)
from app.infra.embedding import EmbeddingClient
from app.knowledge.sources import SourceRef, SourceResolver
from app.memory.service import MemoryService, MutationCommand


class LocalCounter:
    def count(self, value):
        return len(value)

    def truncate(self, value, limit):
        return value[:limit]


class LocalEmbedding(EmbeddingClient):
    def __init__(self, profile):
        self.profile = profile

    async def embed_one(self, value, *, purpose="document"):
        return [1.0] + [0.0] * (self.profile.dimension - 1)


async def user_source(session, content):
    conversation = Conversation(title="Synthetic eval event source")
    session.add(conversation)
    await session.flush()
    message = Message(conversation_id=conversation.id, role=MessageRole.USER, content=content, metadata_={})
    session.add(message)
    await session.commit()
    return SourceRef.message(message.id)


def command(operation, target, source, content):
    return MutationCommand(operation=operation, target_memory_id=target, kind=MemoryKind.LEARNING,
                           content=content, attributes={}, importance=MemoryImportance.NORMAL, source_refs=[source])


async def test_mutation_events_capture_actual_committed_before_after_target_source_and_history(profile_database_factory):
    database = profile_database_factory()
    collector = EvalObservationCollector()
    token = set_observation_collector(collector)
    try:
        async with database.sessions() as session:
            profile = await EmbeddingProfileService(session, get_settings()).resolve_active()
            service = MemoryService(session, LocalEmbedding(profile), SourceResolver(session, LocalCounter()))
            initial = await user_source(session, "用户自报 HashMap 学习状态不稳定")
            repeated = await user_source(session, "第二轮仍然不稳定，新增一次面试 evidence")
            stable = await user_source(session, "用户自报最近几轮已稳定，可以降频")
            origin = set_observation_turn("seed_turn", "A", "seed")
            try:
                [old] = await service.mutate([command("ADD", None, initial, "HashMap unstable")])
            finally:
                reset_observation_turn(origin)
            old_id = old.id
            origin = set_observation_turn("target_turn", "A", "target")
            try:
                await service.mutate([command("MERGE", old_id, repeated, "HashMap unstable with new interview evidence")])
                [new] = await service.mutate([command("SUPERSEDE", old_id, stable, "HashMap stable by user report")])
            finally:
                reset_observation_turn(origin)
            old_row = await session.get(MemoryItem, old_id)
            assert old_row.status is MemoryStatus.SUPERSEDED
            assert new.status is MemoryStatus.ACTIVE
            add_event, merge_event, replace_event = collector.memory_events
            assert add_event["phase"] == "seed" and add_event["turn_id"] == "seed_turn"
            assert merge_event["phase"] == "target" and merge_event["turn_id"] == "target_turn"
            assert all(event["success"] for event in collector.memory_events)
            assert merge_event["before"][0]["content"] == "HashMap unstable"
            assert merge_event["after"][0]["content"] == "HashMap unstable with new interview evidence"
            assert initial.uri in merge_event["after"][0]["source_refs"]
            assert repeated.uri in merge_event["after"][0]["source_refs"]
            assert replace_event["target_memory_id"] == str(old_id)
            assert replace_event["result_memory_id"] == str(new.id)
            states = {item["id"]: item["status"] for item in replace_event["after"]}
            assert states == {str(old_id): "SUPERSEDED", str(new.id): "ACTIVE"}
            assert replace_event["sources"][0]["source_ref"] == stable.uri
            assert replace_event["sources"][0]["content"] == "用户自报最近几轮已稳定，可以降频"
    finally:
        reset_observation_collector(token)


async def test_rolled_back_batch_records_failed_attempts_without_false_successful_mutation(profile_database_factory):
    database = profile_database_factory()
    collector = EvalObservationCollector()
    token = set_observation_collector(collector)
    origin = set_observation_turn("target_turn")
    try:
        async with database.sessions() as session:
            profile = await EmbeddingProfileService(session, get_settings()).resolve_active()
            service = MemoryService(session, LocalEmbedding(profile), SourceResolver(session, LocalCounter()))
            source = await user_source(session, "Synthetic user state")
            missing = SourceRef.message(uuid4())
            with pytest.raises(LookupError):
                await service.mutate([command("ADD", None, source, "temporary pending write"), command("ADD", None, missing, "never committed")])
            assert await session.scalar(select(func.count()).select_from(MemoryItem)) == 0
            assert len(collector.memory_events) == 2
            assert all(not event["success"] and not event["after"] for event in collector.memory_events)
            assert {event["new_source_refs"][0] for event in collector.memory_events} == {source.uri, missing.uri}
    finally:
        reset_observation_turn(origin)
        reset_observation_collector(token)
