from __future__ import annotations

import tiktoken


class TokenCounter:
    """A deterministic approximation usable without an LLM provider."""

    def __init__(self, model: str | None = None) -> None:
        try:
            self._encoder = tiktoken.encoding_for_model(model or "gpt-4o")
        except KeyError:
            self._encoder = tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if self.count(text) <= max_tokens:
            return text
        return self._encoder.decode(self._encoder.encode(text)[:max_tokens])

    def split(self, text: str, max_tokens: int) -> list[str]:
        """Split a pathological unbroken block without exceeding its token budget."""
        encoded = self._encoder.encode(text)
        return [self._encoder.decode(encoded[index : index + max_tokens]) for index in range(0, len(encoded), max_tokens)]
