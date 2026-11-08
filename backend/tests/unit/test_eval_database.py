from types import SimpleNamespace

import pytest
from sqlalchemy.engine import make_url

from app.evals.database import (
    CanonicalEvalDatabase, EvalDatabaseSafetyError, assert_canonical_eval_database,
    assert_disposable_eval_name,
)


def settings(product="jobpilot", target="jobpilot_eval"):
    return SimpleNamespace(database_url=f"postgresql+psycopg://localhost/{product}",
                           eval_database_url=f"postgresql+psycopg://localhost/{target}")


@pytest.mark.parametrize("action", ["inventory", "setup", "reset", "cleanup", "migrate", "smoke"])
def test_every_lifecycle_path_rejects_product_before_constructing_engine(action):
    calls = []
    with pytest.raises(EvalDatabaseSafetyError, match="Product database"):
        lifecycle = CanonicalEvalDatabase(settings(target="jobpilot"),
                                          engine_factory=lambda *a, **k: calls.append(a))
        getattr(lifecycle, "setup" if action in {"reset", "cleanup"} else action)()
    assert calls == []


def test_same_name_on_different_host_is_still_product():
    with pytest.raises(EvalDatabaseSafetyError):
        assert_canonical_eval_database("postgresql://different-host/jobpilot", "postgresql://localhost/jobpilot")


@pytest.mark.parametrize("name", ["postgres", "jobpilot", "jobpilot_eval;DROP DATABASE jobpilot", "other_eval", "jobpilot_eval-2026"])
def test_drop_name_guard_rejects_product_and_non_eval_identifiers(name):
    with pytest.raises(EvalDatabaseSafetyError):
        assert_disposable_eval_name(name, "jobpilot")


def test_noncanonical_new_database_is_rejected():
    with pytest.raises(EvalDatabaseSafetyError, match="canonical"):
        CanonicalEvalDatabase(settings(target="jobpilot_eval_20261008"))


def test_drop_guard_also_protects_custom_configured_product():
    with pytest.raises(EvalDatabaseSafetyError, match="Product database"):
        assert_disposable_eval_name("jobpilot_eval_test_product", "jobpilot_eval_test_product")


class FakeConnection:
    def __init__(self, state):
        self.state = state
        self.dialect = SimpleNamespace(identifier_preparer=SimpleNamespace(quote_identifier=lambda name: f'"{name}"'))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        statement = str(sql)
        self.state["statements"].append(statement)
        if "pg_database_size" in statement:
            return [SimpleNamespace(_mapping={"name": name, "owner": "test", "size_bytes": 0})
                    for name in sorted(self.state["databases"])]
        if statement.startswith("DROP DATABASE"):
            self.state["databases"].discard(statement.split('"')[1])
        if statement.startswith("CREATE DATABASE"):
            self.state["databases"].add(statement.split('"')[1])

    def scalar(self, sql, params=None):
        if "pg_database WHERE" in str(sql):
            return params["name"] in self.state["databases"]
        raise AssertionError(str(sql))


def test_canonical_setup_and_reset_are_idempotent_and_cleanup_never_targets_product(monkeypatch):
    state = {"databases": {"jobpilot", "postgres", "jobpilot_eval", "jobpilot_eval_phaseb_20261006"}, "statements": []}

    def factory(url, **kwargs):
        assert make_url(url).database == "postgres"
        return SimpleNamespace(connect=lambda: FakeConnection(state), dispose=lambda: None)

    lifecycle = CanonicalEvalDatabase(settings(), engine_factory=factory)
    migrations = []
    monkeypatch.setattr(lifecycle, "migrate", lambda: migrations.append("head"))
    monkeypatch.setattr(lifecycle, "smoke", lambda: {"database": "jobpilot_eval"})
    first = lifecycle.setup(cleanup=True)
    assert set(first["dropped"]) == {"jobpilot_eval", "jobpilot_eval_phaseb_20261006"}
    assert state["databases"] == {"jobpilot", "postgres", "jobpilot_eval"}
    lifecycle.setup()
    lifecycle.setup()
    lifecycle.setup(reset=True)
    lifecycle.setup(reset=True)
    assert state["databases"] == {"jobpilot", "postgres", "jobpilot_eval"}
    assert len(migrations) == 5
    assert not any('DATABASE "jobpilot"' in value for value in state["statements"])
def test_scoped_runtime_short_session_refuses_product_before_opening(monkeypatch):
    from types import SimpleNamespace
    from uuid import uuid4
    from app.infra import db
    from app.evals.scope import EvalScope, set_eval_scope, reset_eval_scope
    from app.evals.database import EvalDatabaseSafetyError
    monkeypatch.setattr(db, "engine", SimpleNamespace(url="postgresql://synthetic@localhost/jobpilot"))
    token = set_eval_scope(EvalScope(run_id=uuid4(), case_id="identity"))
    try:
        with pytest.raises(EvalDatabaseSafetyError):
            db.session_factory()
    finally:
        reset_eval_scope(token)
