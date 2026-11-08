"""add context-compaction model-call classification

Revision ID: 0004_context_compaction
Revises: 0003_eval_scope
"""

from alembic import op
import sqlalchemy as sa


revision = "0004_context_compaction"
down_revision = "0003_eval_scope"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_call_trace",
        sa.Column("call_type", sa.String(32), nullable=False, server_default="MAIN_AGENT"),
    )
    op.drop_constraint("uq_model_trace_turn_sequence", "model_call_trace", type_="unique")
    op.create_unique_constraint(
        "uq_model_trace_turn_type_sequence",
        "model_call_trace",
        ["turn_id", "call_type", "sequence"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_model_trace_turn_type_sequence", "model_call_trace", type_="unique")
    op.create_unique_constraint("uq_model_trace_turn_sequence", "model_call_trace", ["turn_id", "sequence"])
    op.drop_column("model_call_trace", "call_type")
