from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import UUID, uuid5

from app.embedding.tokenizers import load_token_counter

PROFILE_NAMESPACE = UUID("b251dd17-0f84-4cab-b7aa-a34589ac7b37")
MAX_HNSW_VECTOR_DIMENSION = 2000

# v1 hashed every configuration field. Keep this tuple frozen so migration
# history and already stored UUIDs never depend on future dataclass additions.
LEGACY_FIELDS = (
    "name", "base_url", "model", "dimension", "provider", "max_input_tokens",
    "batch_limit", "send_dimensions", "normalize", "query_prefix",
    "document_prefix", "tokenizer_encoding", "api_key_env",
)
SPACE_FIELDS = (
    "provider", "model", "dimension", "send_dimensions", "normalize",
    "query_prefix", "document_prefix", "space_revision",
)
RUNTIME_METADATA_FIELDS = (
    "base_url", "api_key_env", "batch_limit", "max_input_tokens", "tokenizer_encoding",
)


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def identity_uuid(canonical: str) -> UUID:
    return uuid5(PROFILE_NAMESPACE, hashlib.sha256(canonical.encode()).hexdigest())


@dataclass(frozen=True)
class ProfileDefinition:
    """Encoding contract plus mutable admission/transport/credential metadata."""

    name: str
    base_url: str
    model: str
    dimension: int
    provider: str = "openai-compatible"
    max_input_tokens: int = 8192
    batch_limit: int = 20
    send_dimensions: bool = True
    normalize: bool = False
    query_prefix: str = ""
    document_prefix: str = ""
    tokenizer_encoding: str = "o200k_base"
    api_key_env: str = "EMBEDDING_API_KEY"
    space_revision: str = ""
    identity_version: int = 2
    legacy_identity_json: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}", self.name):
            raise ValueError("Profile name must contain 1..120 letters, digits, '.', '_' or '-'")
        url = urlsplit(self.base_url)
        if url.scheme not in {"https", "http"} or not url.hostname:
            raise ValueError("Profile base_url must be an absolute HTTP(S) provider URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Profile base_url must not contain credentials, a query, or a fragment")
        if self.provider != "openai-compatible":
            raise ValueError("Only the openai-compatible provider is currently supported")
        if not self.model.strip() or len(self.model) > 200:
            raise ValueError("Profile model must contain 1..200 characters")
        if not 1 <= self.dimension <= MAX_HNSW_VECTOR_DIMENSION:
            raise ValueError("Profile dimension must be 1..2000 for the current full-precision pgvector HNSW strategy")
        if self.max_input_tokens < 1 or self.batch_limit < 1:
            raise ValueError("Profile input limit and batch limit must be positive")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", self.api_key_env):
            raise ValueError("Profile api_key_env must name an environment credential")
        if len(self.space_revision) > 200:
            raise ValueError("Profile space_revision must contain at most 200 characters")
        if self.identity_version not in {1, 2}:
            raise ValueError("Unsupported embedding profile identity version")
        if self.identity_version == 2 and self.legacy_identity_json is not None:
            raise ValueError("Only grandfathered v1 profiles can contain a legacy identity snapshot")
        if self.legacy_identity_json is not None:
            legacy = json.loads(self.legacy_identity_json)
            if not isinstance(legacy, dict) or set(legacy) != set(LEGACY_FIELDS):
                raise ValueError("Invalid legacy embedding profile identity snapshot")
            if canonical_json(legacy) != self.legacy_identity_json:
                raise ValueError("Legacy embedding profile identity snapshot must be canonical JSON")
            # Metadata may now change, but an old UUID must never silently
            # acquire different accepted-input encoding semantics.
            for field in set(SPACE_FIELDS) - {"space_revision"}:
                if legacy[field] != getattr(self, field):
                    raise ValueError("Legacy profile encoding contract differs from its frozen identity")
            if self.space_revision:
                raise ValueError("Legacy profile cannot acquire a new semantic space revision")
        # This is an explicit local counting policy, not a claim about a
        # provider's undocumented tokenizer. Compatibility still needs a probe.
        load_token_counter(self.tokenizer_encoding)

    @property
    def id(self) -> UUID:
        if self.identity_version == 1:
            return identity_uuid(self.legacy_identity_json or canonical_json(self.legacy_payload()))
        return self.space_id

    def legacy_payload(self) -> dict:
        return {field: getattr(self, field) for field in LEGACY_FIELDS}

    def space_payload(self) -> dict:
        return {"identity_version": 2, **{field: getattr(self, field) for field in SPACE_FIELDS}}

    @property
    def space_id(self) -> UUID:
        """Current semantic fingerprint, also comparable for legacy rows."""
        return identity_uuid(canonical_json(self.space_payload()))

    def metadata(self) -> dict:
        return {"id": str(self.id), "name": self.name, "model": self.model, "dimension": self.dimension}


DEFINITION_FIELDS = tuple(ProfileDefinition.__dataclass_fields__)
IMMUTABLE_FIELDS = ("name", *SPACE_FIELDS, "identity_version", "legacy_identity_json")


def profile_definition(profile: object) -> ProfileDefinition:
    return ProfileDefinition(**{name: getattr(profile, name) for name in DEFINITION_FIELDS})


def validate_profile(profile: object) -> ProfileDefinition:
    definition = profile_definition(profile)
    if definition.identity_version == 1 and definition.legacy_identity_json is None:
        raise ValueError("Legacy embedding profile is missing its frozen identity snapshot")
    if definition.id != getattr(profile, "id"):
        raise ValueError("Embedding profile identity does not match its immutable configuration")
    return definition


def index_name(entity: str, profile_id: UUID) -> str:
    if entity not in {"knowledge", "memory"}:
        raise ValueError("Unknown embedding entity")
    return f"idx_{entity}_embedding_{profile_id.hex}_hnsw"


def index_ddl(entity: str, profile_id: UUID, dimension: int) -> str:
    """Only validated identifiers/integers reach the PostgreSQL DDL string."""
    if not 1 <= dimension <= MAX_HNSW_VECTOR_DIMENSION:
        raise ValueError("Unsupported full-precision HNSW dimension")
    profile_id = UUID(str(profile_id))
    table = {"knowledge": "knowledge_chunk_embedding", "memory": "memory_item_embedding"}[entity]
    return (
        f"CREATE INDEX IF NOT EXISTS {index_name(entity, profile_id)} ON {table} "
        f"USING hnsw ((embedding::vector({dimension})) vector_cosine_ops) "
        f"WHERE profile_id = '{profile_id}'::uuid"
    )
