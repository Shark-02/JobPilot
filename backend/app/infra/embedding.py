from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from app.db.models import EmbeddingProfile
from app.embedding.profiles import ProfileDefinition
from app.embedding.tokenizers import load_token_counter


class EmbeddingClient:
    """Encode one immutable profile's space; credentials never belong to its identity."""

    def __init__(self, profile: EmbeddingProfile | ProfileDefinition, api_key: str, counter: object | None = None) -> None:
        self.profile = profile
        if not api_key:
            raise RuntimeError(f"Embedding credentials are missing for profile {profile.name}")
        if not profile.base_url or not profile.model or profile.dimension <= 0:
            raise ValueError("Embedding profile configuration is incomplete")
        if profile.batch_limit <= 0 or profile.max_input_tokens <= 0:
            raise ValueError("Embedding profile input and batch limits must be positive")
        self._counter = counter or load_token_counter(profile.tokenizer_encoding)
        self.http_attempts = 0
        self.embedding_batches = 0
        # Preserve the existing OpenAI SDK's two retries for connection/429/5xx failures.
        self._client = AsyncOpenAI(
            base_url=profile.base_url, api_key=api_key, max_retries=2,
            http_client=DefaultAsyncHttpxClient(event_hooks={"request": [self._record_http_attempt]}),
        )

    async def _record_http_attempt(self, request) -> None:
        # HTTPX invokes this hook for retries as well. Count only; never retain
        # a request URL, headers, credential, or private embedding input.
        if request.method == "POST" and request.url.path.rstrip("/").endswith("/embeddings"):
            self.http_attempts += 1

    @property
    def enabled(self) -> bool:
        return True

    def profile_metadata(self) -> dict:
        return {
            "embedding_profile_id": str(self.profile.id),
            "embedding_profile_name": self.profile.name,
            "embedding_model": self.profile.model,
            "embedding_dimension": self.profile.dimension,
        }

    async def assert_resolved_profile(self, session: object, *, for_write: bool = False) -> None:
        from app.config import get_settings
        from app.embedding.service import EmbeddingProfileService

        resolved = await EmbeddingProfileService(session, get_settings()).resolve_active(for_write=for_write)
        if resolved.id != self.profile.id:
            raise ValueError("Embedding client profile differs from the resolved active/evaluation profile")

    async def embed_one(self, text: str, *, purpose: Literal["query", "document"] = "document") -> list[float]:
        return (await self.embed_many([text], purpose=purpose))[0]

    async def embed_many(
        self, texts: Sequence[str], *, purpose: Literal["query", "document"] = "document"
    ) -> list[list[float]]:
        if purpose not in {"query", "document"}:
            raise ValueError("Embedding purpose must be query or document")
        prefix = self.profile.query_prefix if purpose == "query" else self.profile.document_prefix
        inputs = [prefix + text for text in texts]
        # Validate the entire request before any batch is sent. Never truncate or rechunk.
        for position, text in enumerate(inputs):
            tokens = self._counter.count(text)
            if tokens > self.profile.max_input_tokens:
                raise ValueError(
                    f"Embedding input incompatible with profile {self.profile.name}: "
                    f"item={position}, tokens={tokens}, limit={self.profile.max_input_tokens}, "
                    f"tokenizer={self.profile.tokenizer_encoding}"
                )

        vectors: list[list[float]] = []
        for start in range(0, len(inputs), self.profile.batch_limit):
            batch = inputs[start:start + self.profile.batch_limit]
            request = {"model": self.profile.model, "input": batch, "encoding_format": "float"}
            if self.profile.send_dimensions:
                request["dimensions"] = self.profile.dimension
            self.embedding_batches += 1
            response = await self._client.embeddings.create(**request)
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(batch))):
                raise RuntimeError("Embedding provider returned missing, duplicate or invalid response indices")
            for item in items:
                vector = [float(value) for value in item.embedding]
                if len(vector) != self.profile.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch for profile {self.profile.name}: "
                        f"expected={self.profile.dimension}, actual={len(vector)}"
                    )
                if not all(math.isfinite(value) for value in vector):
                    raise ValueError("Embedding provider returned a non-finite vector")
                norm = math.hypot(*vector)
                if norm == 0:
                    raise ValueError("Embedding provider returned a zero vector incompatible with cosine retrieval")
                if self.profile.normalize:
                    vector = [value / norm for value in vector]
                vectors.append(vector)
        return vectors
