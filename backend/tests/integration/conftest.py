from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.config import get_settings


@dataclass
class ProfileDatabase:
    schema: str
    engine: object
    async_engine: AsyncEngine

    @property
    def sessions(self):
        return async_sessionmaker(self.async_engine, expire_on_commit=False)

    def upgrade(self, revision: str) -> None:
        configuration = Config()
        configuration.set_main_option("script_location", str(Path(__file__).resolve().parents[2] / "migrations"))
        with self.engine.begin() as connection:
            assert connection.scalar(text("SELECT current_schema()")) == self.schema
            configuration.attributes["connection"] = connection
            command.upgrade(configuration, revision)


@pytest.fixture
async def profile_database_factory():
    """Opt-in real pgvector tests; only randomized test schemas are ever modified.

    JOBPILOT_TEST_POSTGRES=1 enables local integration tests. TEST_DATABASE_URL
    must point to the canonical eval database. Disposable isolated schemas
    never use the product database, including when configuration is missing.
    """
    if os.environ.get("JOBPILOT_TEST_POSTGRES") != "1":
        pytest.skip("Set JOBPILOT_TEST_POSTGRES=1 to run isolated PostgreSQL/pgvector tests")
    from sqlalchemy.engine import make_url
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().eval_database_url
    if not url or make_url(url).database != "jobpilot_eval":
        raise ValueError("Integration tests require canonical jobpilot_eval; product database is forbidden")
    administration = create_engine(url)
    databases = []

    def create(revision: str = "head") -> ProfileDatabase:
        schema = f"jobpilot_test_{uuid4().hex}"
        assert re.fullmatch(r"jobpilot_test_[0-9a-f]{32}", schema)
        with administration.begin() as connection:
            # Extensions must already be installed by normal database setup;
            # the fixture never installs or alters shared extensions.
            installed = set(connection.scalars(text(
                "SELECT extname FROM pg_extension WHERE extname IN ('vector','pg_trgm')"
            )))
            assert installed == {"vector", "pg_trgm"}
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            # Own version table prevents public.alembic_version being visible
            # via search_path before Alembic creates its first version record.
            connection.execute(text(
                f'CREATE TABLE "{schema}".alembic_version (version_num varchar(32) PRIMARY KEY)'
            ))
        options = {"options": f"-csearch_path={schema},public"}
        database = ProfileDatabase(
            schema=schema,
            engine=create_engine(url, connect_args=options),
            async_engine=create_async_engine(url, connect_args=options),
        )
        databases.append(database)
        database.upgrade(revision)
        return database

    yield create

    for database in databases:
        await database.async_engine.dispose()
        database.engine.dispose()
        with administration.begin() as connection:
            assert re.fullmatch(r"jobpilot_test_[0-9a-f]{32}", database.schema)
            connection.execute(text(f'DROP SCHEMA "{database.schema}" CASCADE'))
    administration.dispose()
