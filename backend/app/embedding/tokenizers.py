from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path

import tiktoken

from app.config import PROJECT_ROOT


class TiktokenCounter:
    def __init__(self, encoding: str) -> None:
        self.encoder = tiktoken.get_encoding(encoding)

    def count(self, value: str) -> int:
        return len(self.encoder.encode(value, disallowed_special=()))


class JsonTokenizerCounter:
    def __init__(self, digest: str) -> None:
        directory = Path(os.environ.get("EMBEDDING_TOKENIZER_DIR") or PROJECT_ROOT / "artifacts" / "tokenizers")
        path = directory / f"{digest}.json"
        if not path.is_file():
            raise ValueError(f"Embedding tokenizer artifact is missing: hf-json:{digest}")
        stat = path.stat()
        self.encoder = _load_json_encoder(str(path), digest, stat.st_mtime_ns, stat.st_size)

    def count(self, value: str) -> int:
        return len(self.encoder.encode(value, add_special_tokens=True).ids)


@lru_cache(maxsize=8)
def _load_json_encoder(path: str, digest: str, modified_ns: int, size: int):
    from tokenizers import Tokenizer

    # File metadata participates in the cache key so changing a local artifact
    # forces a new hash check, while normal profile reads avoid reparsing it.
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError(f"Embedding tokenizer artifact hash mismatch: hf-json:{digest}")
    encoder = Tokenizer.from_file(path)
    # Admission counts must include every input token, independent of any
    # truncation/padding options that came with the downloaded artifact.
    encoder.no_truncation()
    encoder.no_padding()
    return encoder


def load_token_counter(encoding: str):
    """Local admission policy; provider compatibility requires observed probe evidence."""
    if encoding.startswith("hf-json:"):
        digest = encoding.removeprefix("hf-json:")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("hf-json tokenizer encoding must contain a lowercase SHA-256 digest")
        return JsonTokenizerCounter(digest)
    return TiktokenCounter(encoding)
