from __future__ import annotations

import hashlib

import pytest

from app.embedding.tokenizers import load_token_counter


def test_hf_json_counter_counts_special_tokens_and_disables_truncation_padding(tmp_path, monkeypatch):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[UNK]": 0, "hello": 1, "world": 2, "[CLS]": 3, "[SEP]": 4}, unk_token="[UNK]",
    ))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 3), ("[SEP]", 4)],
    )
    tokenizer.enable_truncation(max_length=3)
    tokenizer.enable_padding(length=10)
    temporary = tmp_path / "tokenizer.json"
    tokenizer.save(str(temporary))
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.rename(tmp_path / f"{digest}.json")
    monkeypatch.setenv("EMBEDDING_TOKENIZER_DIR", str(tmp_path))

    counter = load_token_counter(f"hf-json:{digest}")
    assert counter.count("hello world") == 4
    assert counter.count("hello world hello world") == 6
    assert counter.count("") == 2


def test_hf_json_counter_rejects_missing_and_changed_artifacts(tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    monkeypatch.setenv("EMBEDDING_TOKENIZER_DIR", str(tmp_path))
    encoding = "hf-json:" + "0" * 64
    with pytest.raises(ValueError, match="missing"):
        load_token_counter(encoding)
    (tmp_path / ("0" * 64 + ".json")).write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_token_counter(encoding)


@pytest.mark.parametrize("suffix", ["", "0" * 63, "G" * 64, "../tokenizer.json"])
def test_hf_json_counter_requires_a_sha256_identifier(suffix):
    with pytest.raises(ValueError, match="SHA-256"):
        load_token_counter(f"hf-json:{suffix}")


def test_tiktoken_counting_policy_remains_supported():
    counter = load_token_counter("o200k_base")
    assert counter.count("hello world") == 2
