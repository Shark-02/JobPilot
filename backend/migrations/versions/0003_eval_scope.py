"""add offline evaluation scope and logical retraction

Revision ID: 0003_eval_scope
Revises: 0002_agent_observability
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0003_eval_scope"
down_revision = "0002_agent_observability"
branch_labels = None
depends_on = None


def _scoped(table: str, *, retracted: bool = False) -> None:
    uuid = postgresql.UUID(as_uuid=True)
    op.add_column(table, sa.Column("eval_run_id", uuid, nullable=True))
    op.add_column(table, sa.Column("eval_case_id", sa.String(64), nullable=True))
    if retracted:
        op.add_column(table, sa.Column("retracted_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(f"idx_{table}_eval_scope", table, ["eval_run_id", "eval_case_id"])


def upgrade() -> None:
    for table in ("conversation", "message", "memory_item"):
        _scoped(table, retracted=True)
    for table in ("agent_turn", "model_call_trace", "tool_call_trace", "agent_observation"):
        _scoped(table)


def downgrade() -> None:
    for table in ("agent_observation", "tool_call_trace", "model_call_trace", "agent_turn", "memory_item", "message", "conversation"):
        op.drop_index(f"idx_{table}_eval_scope", table_name=table)
        if table in {"conversation", "message", "memory_item"}:
            op.drop_column(table, "retracted_at")
        op.drop_column(table, "eval_case_id")
        op.drop_column(table, "eval_run_id")
