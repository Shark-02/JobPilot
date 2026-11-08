"""Canonical disposable Eval database lifecycle. Never mutates the product database."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url

from app.config import Settings, get_settings

CANONICAL_EVAL_DATABASE = "jobpilot_eval"
PROTECTED_PRODUCT_DATABASE = "jobpilot"


class EvalDatabaseSafetyError(ValueError):
    pass


def assert_canonical_eval_database(target: str | URL, product: str | URL) -> URL:
    target_url, product_url = make_url(target), make_url(product)
    if target_url.database in {PROTECTED_PRODUCT_DATABASE, product_url.database}:
        raise EvalDatabaseSafetyError("Product database is forbidden for any Eval write or reset")
    if target_url.database != CANONICAL_EVAL_DATABASE:
        raise EvalDatabaseSafetyError("Eval writes require canonical jobpilot_eval; reset it for freshness")
    return target_url


def assert_disposable_eval_name(name: str, product_name: str | None) -> None:
    if name in {PROTECTED_PRODUCT_DATABASE, product_name}:
        raise EvalDatabaseSafetyError("Product database is protected from DROP/reset")
    if not re.fullmatch(r"jobpilot_eval(?:_[a-zA-Z0-9_]+)?", name):
        raise EvalDatabaseSafetyError("DROP is limited to inventoried jobpilot_eval databases")


class CanonicalEvalDatabase:
    def __init__(self, settings: Settings, *, engine_factory: Callable = create_engine) -> None:
        if not settings.eval_database_url:
            raise EvalDatabaseSafetyError("EVAL_DATABASE_URL must be configured as jobpilot_eval")
        self.product_name = make_url(settings.database_url).database
        self.target = assert_canonical_eval_database(settings.eval_database_url, settings.database_url)
        self.engine_factory = engine_factory

    def _admin_engine(self):
        # Connect directly to postgres. No management statement connects to product.
        return self.engine_factory(self.target.set(database="postgres"), isolation_level="AUTOCOMMIT")

    def inventory(self) -> dict:
        engine = self._admin_engine()
        try:
            with engine.connect() as connection:
                rows = [dict(row._mapping) for row in connection.execute(text(
                    "SELECT datname AS name, pg_get_userbyid(datdba) AS owner, "
                    "pg_database_size(oid) AS size_bytes FROM pg_database "
                    "WHERE NOT datistemplate ORDER BY datname"
                ))]
        finally:
            engine.dispose()
        return {"product_database": self.product_name,
                "configured_eval_database": self.target.database,
                "databases": rows,
                "eval_databases": [row["name"] for row in rows if row["name"].startswith("jobpilot_eval")]}

    def _drop(self, connection, name: str) -> None:
        assert_disposable_eval_name(name, self.product_name)
        quoted = connection.dialect.identifier_preparer.quote_identifier(name)
        connection.execute(text(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname=:name AND pid <> pg_backend_pid()"
        ), {"name": name})
        connection.execute(text(f"DROP DATABASE IF EXISTS {quoted}"))

    def setup(self, *, reset: bool = False, cleanup: bool = False) -> dict:
        before = self.inventory()
        dropped = []
        engine = self._admin_engine()
        try:
            with engine.connect() as connection:
                targets = before["eval_databases"] if cleanup else ([CANONICAL_EVAL_DATABASE] if reset else [])
                # Validate the whole manifest before even terminating one connection.
                for name in targets:
                    assert_disposable_eval_name(name, self.product_name)
                for name in targets:
                    self._drop(connection, name)
                    if name in before["eval_databases"]:
                        dropped.append(name)
                exists = connection.scalar(text("SELECT 1 FROM pg_database WHERE datname=:name"),
                                           {"name": CANONICAL_EVAL_DATABASE})
                if not exists:
                    quoted = connection.dialect.identifier_preparer.quote_identifier(CANONICAL_EVAL_DATABASE)
                    connection.execute(text(f"CREATE DATABASE {quoted}"))
        finally:
            engine.dispose()
        self.migrate()
        return {"before": before, "dropped": dropped, "after": self.inventory(), "smoke": self.smoke()}

    def migrate(self) -> None:
        assert_canonical_eval_database(self.target, self.product_name_url)
        engine = self.engine_factory(self.target)
        try:
            with engine.begin() as connection:
                if connection.scalar(text("SELECT current_database()")) != CANONICAL_EVAL_DATABASE:
                    raise EvalDatabaseSafetyError("Connected database differs from canonical Eval target")
                configuration = Config()
                configuration.set_main_option("script_location", str(Path(__file__).resolve().parents[2] / "migrations"))
                configuration.attributes["connection"] = connection
                command.upgrade(configuration, "head")
                # The shared historical migration bootstraps a product ACTIVE
                # profile for an empty corpus. Eval uses only explicit binding;
                # importing new snapshots must not inherit that empty-corpus
                # readiness or an operational default pointer.
                connection.execute(text("DELETE FROM active_embedding_profile"))
                connection.execute(text(
                    "UPDATE embedding_profile SET status='REGISTERED', "
                    "validation_passed_at=NULL, validation_note=NULL WHERE status='ACTIVE'"
                ))
        finally:
            engine.dispose()

    @property
    def product_name_url(self) -> URL:
        # Only its database field is consumed by the safety predicate.
        return self.target.set(database=self.product_name)

    def smoke(self) -> dict:
        engine = self.engine_factory(self.target)
        try:
            with engine.connect() as connection:
                return {"database": connection.scalar(text("SELECT current_database()")),
                        "migration_revision": connection.scalar(text("SELECT version_num FROM alembic_version")),
                        "extensions": sorted(connection.scalars(text(
                            "SELECT extname FROM pg_extension WHERE extname IN ('vector','pg_trgm')"
                        ))),
                        "documents": connection.scalar(text("SELECT count(*) FROM knowledge_document")),
                        "memories": connection.scalar(text("SELECT count(*) FROM memory_item")),
                        "active_pointer_count": connection.scalar(text("SELECT count(*) FROM active_embedding_profile")),
                        "profile_states": [dict(row._mapping) for row in connection.execute(text(
                            "SELECT name, status::text AS status FROM embedding_profile ORDER BY name"
                        ))]}
        finally:
            engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "setup", "reset", "cleanup", "smoke"])
    parser.add_argument("--inventory-output", type=Path,
                        help="Save a credential-free pre-cleanup inventory before destructive actions")
    args = parser.parse_args()
    lifecycle = CanonicalEvalDatabase(get_settings())
    inventory = lifecycle.inventory()
    if args.inventory_output:
        args.inventory_output.parent.mkdir(parents=True, exist_ok=True)
        args.inventory_output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.action == "inventory":
        result = inventory
    elif args.action == "smoke":
        result = lifecycle.smoke()
    else:
        if args.action in {"reset", "cleanup"} and not args.inventory_output:
            parser.error("reset/cleanup requires --inventory-output to preserve the prior inventory")
        result = lifecycle.setup(reset=args.action == "reset", cleanup=args.action == "cleanup")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
