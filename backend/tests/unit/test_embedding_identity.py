from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.embedding.profiles import ProfileDefinition, canonical_json, validate_profile


@pytest.fixture(autouse=True)
def offline_counting_policy_validation(monkeypatch):
    # These tests exercise hash membership, not tokenizer assets or networking.
    monkeypatch.setattr("tiktoken.get_encoding", lambda name: SimpleNamespace(encode=lambda text: []))


def definition(**changes) -> ProfileDefinition:
    return ProfileDefinition(
        **({"name": "qwen-current-v1", "base_url": "https://embedding.invalid/v1",
            "model": "test-model", "dimension": 1536} | changes)
    )


@pytest.mark.parametrize("changes", [
    {"api_key_env": "SCHOOL_EMBEDDING_API_KEY"},
    {"base_url": "https://moved-gateway.invalid/v1"},
    {"batch_limit": 2},
    {"max_input_tokens": 1024},
    {"tokenizer_encoding": "cl100k_base"},
    {"name": "same-space-new-label"},
])
def test_runtime_admission_credential_and_label_changes_preserve_space_identity(changes):
    original = definition()
    updated = replace(original, **changes)

    assert updated.id == original.id
    assert updated.space_payload() == original.space_payload()


@pytest.mark.parametrize("changes", [
    {"model": "different-weights"}, {"dimension": 1024},
    {"send_dimensions": False}, {"normalize": True},
    {"query_prefix": "query: "}, {"document_prefix": "document: "},
    {"space_revision": "deployment-with-different-server-preprocessing"},
])
def test_accepted_input_encoding_contract_changes_create_another_space(changes):
    assert replace(definition(), **changes).id != definition().id


def test_grandfathered_v1_id_survives_operational_changes_and_uses_same_space_fingerprint():
    original = definition(identity_version=1)
    snapshot = canonical_json(original.legacy_payload())
    grandfathered = replace(original, legacy_identity_json=snapshot)
    # Fixed pre-v2 result protects migration compatibility without recomputing
    # the legacy algorithm as the assertion.
    assert grandfathered.id == UUID("ed038183-6e80-5760-ab93-1eadd04d2250")
    updated = replace(grandfathered, api_key_env="SCHOOL_EMBEDDING_API_KEY",
                      base_url="https://moved-gateway.invalid/v1", batch_limit=2,
                      max_input_tokens=1024, tokenizer_encoding="cl100k_base")
    assert updated.id == grandfathered.id
    assert updated.space_id == definition().id
    assert validate_profile(SimpleNamespace(id=updated.id, **asdict(updated))) == updated


def test_legacy_snapshot_cannot_hide_a_changed_semantic_contract():
    old = definition(identity_version=1)
    with pytest.raises(ValueError, match="encoding contract differs"):
        replace(old, legacy_identity_json=canonical_json(old.legacy_payload()), model="different")


def test_profile_validation_requires_legacy_snapshot_and_correct_versioned_uuid():
    old = definition(identity_version=1)
    with pytest.raises(ValueError, match="missing its frozen identity snapshot"):
        validate_profile(SimpleNamespace(id=old.id, **asdict(old)))
    current = definition()
    with pytest.raises(ValueError, match="identity does not match"):
        validate_profile(SimpleNamespace(id=old.id, **asdict(current)))
