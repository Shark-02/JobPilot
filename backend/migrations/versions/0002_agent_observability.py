"""persist complete agent observations

Revision ID: 0002_agent_observability
Revises: 0001_initial
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_agent_observability"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    uuid = postgresql.UUID(as_uuid=True)

    op.create_table(
        "agent_observation",
        sa.Column("id", uuid, primary_key=True),
        sa.Column(
            "turn_id",
            uuid,
            sa.ForeignKey("agent_turn.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_observation_id",
            uuid,
            sa.ForeignKey("agent_observation.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("input_payload", postgresql.JSONB, nullable=True),
        sa.Column("output_payload", postgresql.JSONB, nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "source_refs",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "turn_id",
            "sequence",
            name="uq_agent_observation_turn_sequence",
        ),
    )
    # uq_agent_observation_turn_sequence already creates an index beginning
    # with turn_id, so a second (turn_id, sequence) index would be redundant.
    op.create_index(
        "idx_agent_observation_kind",
        "agent_observation",
        ["kind"],
    )

    for table in ("model_call_trace", "tool_call_trace"):
        # Add nullable first so historical rows can be backfilled with their
        # original trace creation time instead of the migration execution time.
        op.add_column(
            table,
            sa.Column(
                "started_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
        )
        op.add_column(
            table,
            sa.Column(
                "finished_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
        )

        op.execute(
            sa.text(
                f"""
                UPDATE {table}
                SET started_at = created_at
                WHERE started_at IS NULL
                """
            )
        )

        # Historical terminal traces can only be reconstructed approximately.
        # latency_ms gives a better end time than blindly setting finished_at
        # equal to created_at. RUNNING rows intentionally keep finished_at NULL.
        op.execute(
            sa.text(
                f"""
                UPDATE {table}
                SET finished_at = started_at
                    + (latency_ms * INTERVAL '1 millisecond')
                WHERE status <> 'RUNNING'
                  AND finished_at IS NULL
                  AND latency_ms IS NOT NULL
                """
            )
        )

        op.alter_column(
            table,
            "started_at",
            existing_type=sa.DateTime(timezone=True),
            existing_nullable=True,
            nullable=False,
            server_default=sa.text("now()"),
        )


def downgrade() -> None:
    for table in ("tool_call_trace", "model_call_trace"):
        op.drop_column(table, "finished_at")
        op.drop_column(table, "started_at")

    op.drop_index("idx_agent_observation_kind", table_name="agent_observation")
    op.drop_table("agent_observation")
