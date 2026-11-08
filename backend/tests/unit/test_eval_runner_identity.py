"""Runner identities survive transaction expiration without external I/O."""
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import make_transient_to_detached

from app.db.models import AgentTurn, Conversation
from app.evals.datasets import load_dataset
from app.evals.runner import EvalRunner


@pytest.mark.parametrize("case_id", ["M4", "M5", "M6"])
async def test_seed_and_cross_conversation_inputs_reuse_plain_ids_after_runtime_rollback(monkeypatch, case_id):
    import app.evals.runner as module

    runner = object.__new__(EvalRunner)
    runner.run_id = uuid4()
    created = []
    calls = []

    class Repository:
        def __init__(self, session):
            self.session = session

        async def create(self, title):
            entity = Conversation(id=uuid4(), title=title)
            make_transient_to_detached(entity)
            self.session.add(entity)
            created.append(entity)
            return entity

    async def fixtures(session, conversation_id, values, refs):
        assert isinstance(conversation_id, UUID)

    async def turn(session, conversation_id, content, *, conversation="A", phase="target"):
        assert isinstance(conversation_id, UUID)
        calls.append((phase, conversation, conversation_id))
        # This is the Agent error-finalization behavior. Even with
        # expire_on_commit=False, SQLAlchemy expires persistent attributes.
        await session.rollback()
        assert all(inspect(entity).expired for entity in created)
        return uuid4(), "message://synthetic", "Synthetic completed answer"

    async def snapshot(session):
        return []

    monkeypatch.setattr(module, "ConversationRepository", Repository)
    monkeypatch.setattr(runner, "_add_fixtures", fixtures)
    monkeypatch.setattr(runner, "_run_turn", turn)
    monkeypatch.setattr(runner, "_memory_snapshot", snapshot)
    case = load_dataset().cases[case_id]
    async with AsyncSession(expire_on_commit=False) as session:
        turns, metadata = await runner._run_inputs(session, case)
    assert len(turns) == len(case.setup.seed_steps) + len(case.steps)
    assert set(metadata["step_turn_ids"]) == {step.id for step in case.steps}
    grouped = {}
    for phase, label, identifier in calls:
        grouped.setdefault((phase, label), set()).add(identifier)
    assert all(len(ids) == 1 for ids in grouped.values())
    assert len({identifier for ids in grouped.values() for identifier in ids}) == len(grouped)


async def test_run_turn_captures_identity_before_commit_and_reads_answer_with_explicit_query(monkeypatch):
    import app.evals.runner as module

    runner = object.__new__(EvalRunner)
    runner.settings = SimpleNamespace()
    expected_turn = uuid4()
    assistant_id = uuid4()
    executed = []
    queries = []

    class Session(AsyncSession):
        def add(self, entity, **kwargs):
            if isinstance(entity, AgentTurn):
                entity.id = expected_turn
                # Attach a synthetic persistent entity without any DB engine.
                make_transient_to_detached(entity)
            return super().add(entity, **kwargs)

        async def scalar(self, statement, **kwargs):
            selected = list(statement.selected_columns)[0].key
            queries.append(selected)
            return assistant_id if selected == "assistant_message_id" else "Synthetic final answer"

    class Repository:
        def __init__(self, session):
            pass

        async def add_message(self, conversation_id, role, content):
            return SimpleNamespace(id=uuid4())

    class Runtime:
        def __init__(self, session, settings):
            self.session = session

        async def run(self, turn_id):
            executed.append(turn_id)
            await self.session.rollback()

    monkeypatch.setattr(module, "ConversationRepository", Repository)
    monkeypatch.setattr(module, "ThinAgentLoop", Runtime)
    # Commit expiration is also an explicit transaction boundary: the runner
    # must not depend on a still-loaded ORM ID even under this configuration.
    async with Session(expire_on_commit=True) as session:
        turn_id, source_ref, answer = await runner._run_turn(session, uuid4(), "Synthetic input")
    assert turn_id == expected_turn and executed == [expected_turn]
    assert source_ref.startswith("message://")
    assert answer == "Synthetic final answer"
    assert queries == ["assistant_message_id", "content"]
