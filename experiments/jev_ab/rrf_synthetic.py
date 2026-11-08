#!/usr/bin/env python3
"""Run the real memory hybrid retriever on a synthetic corpus in an isolated eval schema.

Requires an explicit JOBPILOT_JEV_EVAL_DATABASE_URL for a local jobpilot_eval
PostgreSQL database with vector and pg_trgm installed. Never opens the product
database or an embedding API. All table/schema writes are isolated and removed.
The character n-gram vectors are a deterministic test embedding, not a model
or a representation of production embedding retrieval quality.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.models import (
    EmbeddingProfile, EmbeddingProfileStatus, MemoryImportance, MemoryItem,
    MemoryItemEmbedding, MemoryKind, MemoryStatus,
)
from app.evals.scope import EvalScope, reset_eval_scope, set_eval_scope
from app.embedding.profiles import ProfileDefinition
from app.infra.embedding import EmbeddingClient
from app.memory.retrieval import MemoryRetriever

HERE = Path(__file__).resolve().parent
DIMENSION = 256


def synthetic_vector(value: str) -> list[float]:
    """Deterministic local lexical character n-gram hashing; no provider calls."""
    normalized = re.sub(r"\s+", " ", value.casefold()).strip()
    grams = [normalized[i:i + size] for size in (1, 2, 3)
             for i in range(len(normalized) - size + 1)]
    result = [0.0] * DIMENSION
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        result[int.from_bytes(digest[:4], "big") % DIMENSION] += 1.0
    norm = math.sqrt(sum(x * x for x in result))
    if not norm:
        raise ValueError("Empty synthetic text cannot be embedded")
    return [x / norm for x in result]


class SyntheticEmbedding(EmbeddingClient):
    def __init__(self, profile: EmbeddingProfile):
        self.profile = profile

    async def embed_one(self, text: str, *, purpose: str = "document") -> list[float]:
        return synthetic_vector(text)


def validate_dataset() -> dict:
    dataset = json.loads((HERE / "dataset.json").read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object")
    cases, documents = dataset.get("cases"), dataset.get("documents")
    if not isinstance(cases, list) or not 10 <= len(cases) <= 20:
        raise ValueError("dataset must contain 10 to 20 cases")
    if not isinstance(documents, list) or len(documents) < 5:
        raise ValueError("dataset must contain at least five documents")
    if any(not isinstance(d, dict) or not isinstance(d.get("id"), str)
           or not d["id"] or not isinstance(d.get("text"), str) or not d["text"].strip()
           for d in documents):
        raise ValueError("document ID and text must be nonempty strings")
    ids = [d["id"] for d in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate document IDs")
    if any(not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"]
           or not isinstance(c.get("query"), str) or not c["query"].strip()
           or not isinstance(c.get("gold"), list)
           or any(not isinstance(g, str) or g not in ids for g in c["gold"])
           or not isinstance(c.get("origin"), str) or not c["origin"] for c in cases):
        raise ValueError("each case needs an ID, query, valid gold IDs (possibly empty), and origin")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    return dataset


def database_url() -> str:
    raw = os.environ.get("JOBPILOT_JEV_EVAL_DATABASE_URL")
    if not raw:
        raise ValueError("Set JOBPILOT_JEV_EVAL_DATABASE_URL to a local, isolated jobpilot_eval database")
    url = make_url(raw)
    if (url.drivername != "postgresql+psycopg" or url.host not in {"localhost", "127.0.0.1", "::1"}
            or url.database != "jobpilot_eval"):
        raise ValueError("Only postgresql+psycopg on localhost with database jobpilot_eval is permitted")
    # Prevent using the configured product database even if it was renamed.
    product = make_url(get_settings().database_url)
    def identity(u):
        return ("loopback" if u.host in {"localhost", "127.0.0.1", "::1"} else u.host,
                u.port or 5432, u.database)
    if identity(url) == identity(product):
        raise ValueError("The eval database must differ from the configured product database")
    return raw


async def retrieve(dataset: dict, url: str) -> list[dict]:
    schema = "jobpilot_jev_" + uuid4().hex
    assert re.fullmatch(r"jobpilot_jev_[0-9a-f]{32}", schema)
    admin = create_engine(url)
    engine = None
    created = False
    try:
        with admin.begin() as connection:
            installed = set(connection.scalars(text(
                "SELECT extname FROM pg_extension WHERE extname IN ('vector','pg_trgm')")))
            if installed != {"vector", "pg_trgm"}:
                raise RuntimeError("The isolated eval database requires vector and pg_trgm extensions")
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            connection.execute(text(f'CREATE TABLE "{schema}".alembic_version (version_num varchar(32) PRIMARY KEY)'))
        created = True
        options = {"options": f"-csearch_path={schema},public"}
        sync_engine = create_engine(url, connect_args=options)
        try:
            config = Config()
            config.set_main_option("script_location", str(ROOT / "backend" / "migrations"))
            with sync_engine.begin() as connection:
                if connection.scalar(text("SELECT current_schema()")) != schema:
                    raise RuntimeError("Schema isolation check failed")
                config.attributes["connection"] = connection
                command.upgrade(config, "head")
        finally:
            sync_engine.dispose()
        engine = create_async_engine(url, connect_args=options)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            definition = ProfileDefinition(
                name="jev-ab-synthetic-ngram", provider="openai-compatible",
                base_url="https://example.invalid", model="hashed-character-ngram",
                dimension=DIMENSION, max_input_tokens=100000, batch_limit=100,
                send_dimensions=False, normalize=True, query_prefix="", document_prefix="",
                tokenizer_encoding="cl100k_base", api_key_env="NOT_USED", space_revision="v1",
                identity_version=2, legacy_identity_json=None,
            )
            profile = EmbeddingProfile(id=definition.id, **asdict(definition), status=EmbeddingProfileStatus.READY)
            session.add(profile)
            await session.flush()
            run_id = uuid4()
            # Each case gets its own synthetic memory rows because the real retriever
            # intentionally isolates memory by eval case.
            rows = []
            for case in dataset["cases"]:
                id_map = {}
                for doc in dataset["documents"]:
                    memory = MemoryItem(id=uuid4(), kind=MemoryKind.SKILL, content=doc["text"],
                                        attributes={}, importance=MemoryImportance.NORMAL,
                                        status=MemoryStatus.ACTIVE, eval_run_id=run_id,
                                        eval_case_id=case["id"])
                    session.add(memory)
                    session.add(MemoryItemEmbedding(memory_item_id=memory.id, profile_id=profile.id,
                                                    embedding=synthetic_vector(doc["text"])))
                    id_map[memory.id] = doc["id"]
                await session.flush()
                scope = set_eval_scope(EvalScope(run_id=run_id, case_id=case["id"],
                                                 embedding_profile_id=profile.id))
                try:
                    matches = await MemoryRetriever(session, SyntheticEmbedding(profile)).search(
                        case["query"], kinds=None, limit=5)
                finally:
                    reset_eval_scope(scope)
                if len(matches) != 5 or any(item.id not in id_map for item in matches):
                    raise RuntimeError("Retrieval did not return five in-scope synthetic candidates")
                rows.append({"id": case["id"], "query": case["query"],
                             "candidates": [{"id": id_map[item.id], "text": item.content} for item in matches]})
            # No commit: synthetic rows are rolled back at session exit.
            return rows
    finally:
        if engine is not None:
            await engine.dispose()
        if created:
            with admin.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def main() -> None:
    dataset = validate_dataset()
    rows = asyncio.run(retrieve(dataset, database_url()))
    output = {"metadata": {"retriever": "MemoryRetriever.search", "limit": 5,
                           "embedding": "local deterministic hashed character n-gram; not production embedding",
                           "dataset_sha256": hashlib.sha256((HERE / "dataset.json").read_bytes()).hexdigest()},
              "cases": rows}
    tmp = HERE / "candidates.json.tmp"
    tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(HERE / "candidates.json")
    print(f"Saved real hybrid RRF Top 5 for {len(rows)} synthetic cases to candidates.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Avoid revealing database URLs or credentials in exception messages.
        print(f"Retrieval failed ({type(exc).__name__}); no candidate output written. Check dataset and isolated database setup.", file=sys.stderr)
        raise SystemExit(1) from None
