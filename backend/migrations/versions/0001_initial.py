"""Initial JobPilot V1 schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-28
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    uuid = postgresql.UUID(as_uuid=True)
    jsonb = postgresql.JSONB(astext_type=sa.Text())
    message_role = postgresql.ENUM("USER", "ASSISTANT", name="message_role", create_type=False)
    document_format = postgresql.ENUM("MARKDOWN", "TEXT", "LATEX", name="document_format", create_type=False)
    document_role = postgresql.ENUM("USER_CAREER_SOURCE", "REFERENCE_SOURCE", name="document_role", create_type=False)
    document_status = postgresql.ENUM("PROCESSING", "READY", "FAILED", "ARCHIVED", name="document_status", create_type=False)
    memory_kind = postgresql.ENUM("BACKGROUND", "EXPERIENCE", "ACHIEVEMENT", "SKILL", "CAREER_GOAL", "PREFERENCE", "LEARNING", "OTHER", name="memory_kind", create_type=False)
    memory_importance = postgresql.ENUM("CORE", "NORMAL", name="memory_importance", create_type=False)
    memory_status = postgresql.ENUM("ACTIVE", "SUPERSEDED", "ARCHIVED", name="memory_status", create_type=False)
    source_type = postgresql.ENUM("MESSAGE", "DOCUMENT_SECTION", "DOCUMENT_CHUNK", name="source_type", create_type=False)
    source_relation = postgresql.ENUM("SUPPORTS", "CONTRADICTS", name="source_relation", create_type=False)
    turn_status = postgresql.ENUM("RUNNING", "COMPLETED", "FAILED", "CANCELLED", "ABORTED", name="turn_status", create_type=False)
    for enum in [message_role, document_format, document_role, document_status, memory_kind, memory_importance, memory_status, source_type, source_relation, turn_status]:
        enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "conversation",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("title", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_table(
        "message",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("conversation_id", uuid, sa.ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", message_role, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", jsonb, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_message_conversation_id", "message", ["conversation_id"])
    op.create_table(
        "knowledge_document",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("title", sa.Text(), nullable=False), sa.Column("filename", sa.Text()),
        sa.Column("format", document_format, nullable=False), sa.Column("source_role", document_role, nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False), sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False, unique=True), sa.Column("status", document_status, nullable=False),
        sa.Column("parser_version", sa.String(64), nullable=False), sa.Column("chunker_version", sa.String(64), nullable=False),
        sa.Column("metadata", jsonb, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_knowledge_document_sha256", "knowledge_document", ["sha256"])
    op.create_index("ix_knowledge_document_status", "knowledge_document", ["status"])
    op.create_table(
        "knowledge_section",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("document_id", uuid, sa.ForeignKey("knowledge_document.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_section_id", uuid, sa.ForeignKey("knowledge_section.id", ondelete="CASCADE")),
        sa.Column("heading", sa.Text()), sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("section_path", jsonb, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("ordinal", sa.Integer(), nullable=False), sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_document_section_ordinal"),
    )
    op.create_index("ix_knowledge_section_document_id", "knowledge_section", ["document_id"])
    op.create_table(
        "knowledge_chunk",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("document_id", uuid, sa.ForeignKey("knowledge_document.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_id", uuid, sa.ForeignKey("knowledge_section.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False), sa.Column("content", sa.Text(), nullable=False),
        sa.Column("retrieval_text", sa.Text(), nullable=False), sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False), sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(1536)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_document_chunk_ordinal"),
    )
    op.create_index("ix_knowledge_chunk_document_id", "knowledge_chunk", ["document_id"])
    op.create_index("ix_knowledge_chunk_section_id", "knowledge_chunk", ["section_id"])
    op.execute("CREATE INDEX idx_knowledge_chunk_trgm ON knowledge_chunk USING GIN (retrieval_text gin_trgm_ops)")
    op.execute("CREATE INDEX idx_knowledge_chunk_embedding_hnsw ON knowledge_chunk USING hnsw (embedding vector_cosine_ops)")
    op.create_table(
        "memory_item",
        sa.Column("id", uuid, primary_key=True), sa.Column("kind", memory_kind, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("attributes", jsonb, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("importance", memory_importance, nullable=False), sa.Column("status", memory_status, nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True)), sa.Column("valid_to", sa.DateTime(timezone=True)),
        sa.Column("embedding", Vector(1536)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_memory_item_kind", "memory_item", ["kind"])
    op.create_index("ix_memory_item_importance", "memory_item", ["importance"])
    op.create_index("ix_memory_item_status", "memory_item", ["status"])
    op.execute("CREATE INDEX idx_memory_active_kind ON memory_item(status, kind)")
    op.execute("CREATE INDEX idx_memory_core ON memory_item(status, importance)")
    op.execute("CREATE INDEX idx_memory_content_trgm ON memory_item USING GIN (content gin_trgm_ops)")
    op.execute("CREATE INDEX idx_memory_embedding_hnsw ON memory_item USING hnsw (embedding vector_cosine_ops)")
    op.create_table(
        "memory_source",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("memory_id", uuid, sa.ForeignKey("memory_item.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_type", source_type, nullable=False),
        sa.Column("message_id", uuid, sa.ForeignKey("message.id", ondelete="CASCADE")),
        sa.Column("document_section_id", uuid, sa.ForeignKey("knowledge_section.id", ondelete="CASCADE")),
        sa.Column("document_chunk_id", uuid, sa.ForeignKey("knowledge_chunk.id", ondelete="CASCADE")),
        sa.Column("relation", source_relation, nullable=False), sa.Column("excerpt", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("(CASE WHEN message_id IS NOT NULL THEN 1 ELSE 0 END) + (CASE WHEN document_section_id IS NOT NULL THEN 1 ELSE 0 END) + (CASE WHEN document_chunk_id IS NOT NULL THEN 1 ELSE 0 END) = 1", name="ck_memory_source_exactly_one_source"),
    )
    op.create_index("ix_memory_source_memory_id", "memory_source", ["memory_id"])
    op.create_index("ix_memory_source_message_id", "memory_source", ["message_id"])
    op.create_index("ix_memory_source_document_section_id", "memory_source", ["document_section_id"])
    op.create_index("ix_memory_source_document_chunk_id", "memory_source", ["document_chunk_id"])
    op.create_table(
        "agent_turn",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("conversation_id", uuid, sa.ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_message_id", uuid, sa.ForeignKey("message.id", ondelete="CASCADE"), nullable=False),
        sa.Column("assistant_message_id", uuid, sa.ForeignKey("message.id", ondelete="SET NULL")),
        sa.Column("status", turn_status, nullable=False), sa.Column("model_turn_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_execution_count", sa.Integer(), nullable=False, server_default="0"), sa.Column("error_code", sa.String(128)),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")), sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_agent_turn_conversation_id", "agent_turn", ["conversation_id"])
    op.create_index("ix_agent_turn_status", "agent_turn", ["status"])
    op.execute("CREATE UNIQUE INDEX uq_agent_turn_one_running_per_conversation ON agent_turn (conversation_id) WHERE status = 'RUNNING'")
    op.create_table(
        "model_call_trace",
        sa.Column("id", uuid, primary_key=True), sa.Column("turn_id", uuid, sa.ForeignKey("agent_turn.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False), sa.Column("provider", sa.String(200)), sa.Column("model", sa.String(200), nullable=False),
        sa.Column("input_tokens", sa.Integer()), sa.Column("output_tokens", sa.Integer()), sa.Column("latency_ms", sa.Integer()),
        sa.Column("finish_reason", sa.String(100)), sa.Column("status", sa.String(50), nullable=False), sa.Column("error_code", sa.String(128)),
        sa.Column("egress_preview", jsonb, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("turn_id", "sequence", name="uq_model_trace_turn_sequence"),
    )
    op.create_index("ix_model_call_trace_turn_id", "model_call_trace", ["turn_id"])
    op.create_table(
        "tool_call_trace",
        sa.Column("id", uuid, primary_key=True), sa.Column("turn_id", uuid, sa.ForeignKey("agent_turn.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provider_call_id", sa.String(255)), sa.Column("sequence", sa.Integer(), nullable=False), sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("arguments", jsonb, nullable=False), sa.Column("fingerprint", sa.String(64), nullable=False), sa.Column("status", sa.String(50), nullable=False),
        sa.Column("ui_summary", sa.Text()), sa.Column("source_refs", jsonb, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("latency_ms", sa.Integer()), sa.Column("error_code", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("turn_id", "sequence", name="uq_tool_trace_turn_sequence"),
    )
    op.create_index("ix_tool_call_trace_turn_id", "tool_call_trace", ["turn_id"])
    op.create_index("ix_tool_call_trace_fingerprint", "tool_call_trace", ["fingerprint"])


def downgrade() -> None:
    for table in ["tool_call_trace", "model_call_trace", "agent_turn", "memory_source", "memory_item", "knowledge_chunk", "knowledge_section", "knowledge_document", "message", "conversation"]:
        op.drop_table(table)
    for enum in ["turn_status", "source_relation", "source_type", "memory_status", "memory_importance", "memory_kind", "document_status", "document_role", "document_format", "message_role"]:
        sa.Enum(name=enum).drop(op.get_bind(), checkfirst=True)
