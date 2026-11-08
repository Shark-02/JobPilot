from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    database_url: str = "postgresql+psycopg://jobpilot:jobpilot@localhost:5432/jobpilot"

    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_context_window: int = 131_072
    llm_max_output_tokens: int = 8192
    llm_temperature: float = 0.2

    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    # Bootstrap settings are consumed by the one-time profile migration only.
    # Online encoding resolves the immutable profile from the database.
    embedding_dim: int = Field(default=1536, ge=1, le=2000)
    embedding_max_input_tokens: int = Field(default=8192, ge=1)
    embedding_tokenizer_encoding: str = "o200k_base"

    agent_max_model_turns: int = Field(default=12, ge=1)
    agent_max_tool_executions: int = Field(default=20, ge=1)
    agent_model_timeout_seconds: int = Field(default=120, ge=1)
    agent_tool_timeout_seconds: int = Field(default=60, ge=1)
    agent_total_deadline_seconds: int = Field(default=300, ge=1)

    core_profile_max_tokens: int = Field(default=1200, ge=1)
    recent_conversation_target_tokens: int = Field(default=12_000, ge=1)
    recent_conversation_max_tokens: int = Field(default=20_000, ge=1)
    context_eviction_trigger_tokens: int | None = Field(default=None, ge=1)
    recent_raw_runtime_keep_count: int = Field(default=1, ge=1)
    working_task_state_max_tokens: int = Field(default=6000, ge=1)
    context_compaction_max_tokens: int = Field(default=6000, ge=256)
    context_safety_margin_tokens: int = Field(default=4096, ge=1)

    knowledge_default_max_modules: int = Field(default=6, ge=1, le=16)
    knowledge_max_modules: int = Field(default=16, ge=1)
    knowledge_default_max_tokens: int = Field(default=8000, ge=1, le=24_000)
    knowledge_max_tokens: int = Field(default=24_000, ge=1)

    trace_enabled: bool = True
    trace_retention_days: int = Field(default=30, ge=1)
    model_egress_preview_enabled: bool = True
    model_egress_preview_store_content: bool = False
    model_egress_preview_max_chars: int = Field(default=30_000, ge=1)
    observability_enabled: bool = True
    observability_capture_payload: bool = True

    eval_database_url: str | None = None
    eval_judge_base_url: str | None = None
    eval_judge_api_key: str | None = None
    eval_judge_model: str | None = None
    eval_judge_reasoning_effort: Literal["none", "low", "high", "max"] | None = None
    eval_judge_temperature: float = 0.0
    eval_judge_timeout_seconds: int = Field(default=600, ge=1)
    eval_judge_max_output_tokens: int = Field(default=6144, ge=1)
    eval_judge_concurrency: int = Field(default=2, ge=1, le=32)
    eval_judge_transport_retries: int = Field(default=1, ge=0, le=5)
    eval_judge_retry_backoff_seconds: float = Field(default=0.5, ge=0, le=30)
    eval_judge_cache_dir: Path | None = None

    @model_validator(mode="after")
    def validate_context_budget(self) -> "Settings":
        if self.llm_context_window <= self.llm_max_output_tokens + self.context_safety_margin_tokens:
            raise ValueError("LLM context window must leave room for output and safety margin")
        if self.context_compaction_max_tokens >= self.hard_input_limit:
            raise ValueError("CONTEXT_COMPACTION_MAX_TOKENS must fit within the hard input limit")
        if self.context_eviction_trigger > self.hard_input_limit:
            raise ValueError("CONTEXT_EVICTION_TRIGGER_TOKENS must not exceed the hard input limit")
        return self

    @property
    def hard_input_limit(self) -> int:
        return self.llm_context_window - self.llm_max_output_tokens - self.context_safety_margin_tokens

    @property
    def soft_input_limit(self) -> int:
        """Compatibility name for the eviction trigger; no LLM is required."""
        return self.context_eviction_trigger

    @property
    def context_eviction_trigger(self) -> int:
        if self.context_eviction_trigger_tokens is not None:
            return self.context_eviction_trigger_tokens
        return min(48_000, int(self.hard_input_limit * 0.70))


@lru_cache
def get_settings() -> Settings:
    return Settings()
