"""Separate embedding-space identity from mutable runtime metadata.

Revision ID: 0006_embedding_identity_v2
Revises: 0005_embedding_profiles
"""

import hashlib
import json
from uuid import UUID, uuid5

from alembic import op
import sqlalchemy as sa

revision = "0006_embedding_identity_v2"
down_revision = "0005_embedding_profiles"
branch_labels = None
depends_on = None

LEGACY_FIELDS = (
    "name", "base_url", "model", "dimension", "provider", "max_input_tokens",
    "batch_limit", "send_dimensions", "normalize", "query_prefix",
    "document_prefix", "tokenizer_encoding", "api_key_env",
)
SPACE_FIELDS = (
    "provider", "model", "dimension", "send_dimensions", "normalize",
    "query_prefix", "document_prefix", "space_revision",
)


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _guard(fields: tuple[str, ...]) -> None:
    old = ", ".join(f"OLD.{name}" for name in fields)
    new = ", ".join(f"NEW.{name}" for name in fields)
    op.execute(f"""
        CREATE OR REPLACE FUNCTION guard_embedding_profile_immutable()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF ROW({old}) IS DISTINCT FROM ROW({new}) THEN
                RAISE EXCEPTION 'Embedding profile identity and encoding rules are immutable';
            END IF;
            RETURN NEW;
        END $$
    """)


def upgrade() -> None:
    connection = op.get_bind()
    connection.execute(sa.text("LOCK TABLE embedding_profile IN ACCESS EXCLUSIVE MODE"))
    # No existing UUID, vector, ANN index, lifecycle, or active pointer changes.
    # Freeze the v1 payload before allowing metadata changes under its old ID.
    rows = connection.execute(sa.text("SELECT * FROM embedding_profile ORDER BY id")).mappings().all()
    snapshots = []
    for row in rows:
        canonical = _canonical({field: row[field] for field in LEGACY_FIELDS})
        expected = uuid5(UUID("b251dd17-0f84-4cab-b7aa-a34589ac7b37"), hashlib.sha256(canonical.encode()).hexdigest())
        if expected != row["id"]:
            raise RuntimeError("Existing embedding profile does not match its v1 identity; migration aborted")
        snapshots.append({"id": row["id"], "snapshot": canonical})
    op.add_column("embedding_profile", sa.Column("space_revision", sa.String(200), nullable=False, server_default=""))
    op.add_column("embedding_profile", sa.Column("identity_version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("embedding_profile", sa.Column("legacy_identity_json", sa.Text()))
    if snapshots:
        connection.execute(sa.text("UPDATE embedding_profile SET legacy_identity_json=:snapshot WHERE id=:id"), snapshots)
    op.alter_column("embedding_profile", "identity_version", server_default="2")
    op.create_check_constraint(
        "ck_embedding_profile_identity_version", "embedding_profile",
        "(identity_version = 1 AND legacy_identity_json IS NOT NULL AND space_revision = '') "
        "OR (identity_version = 2 AND legacy_identity_json IS NULL)",
    )
    _guard(("id", "name", *SPACE_FIELDS, "identity_version", "legacy_identity_json"))


def downgrade() -> None:
    connection = op.get_bind()
    connection.execute(sa.text("LOCK TABLE embedding_profile IN ACCESS EXCLUSIVE MODE"))
    rows = connection.execute(sa.text("SELECT * FROM embedding_profile")).mappings().all()
    for row in rows:
        current = _canonical({field: row[field] for field in LEGACY_FIELDS})
        if row["identity_version"] != 1 or current != row["legacy_identity_json"]:
            raise RuntimeError("Identity downgrade requires only unchanged grandfathered v1 profiles")
    _guard(("id", *LEGACY_FIELDS))
    op.drop_constraint("ck_embedding_profile_identity_version", "embedding_profile", type_="check")
    op.drop_column("embedding_profile", "legacy_identity_json")
    op.drop_column("embedding_profile", "identity_version")
    op.drop_column("embedding_profile", "space_revision")
