from __future__ import annotations

import math
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.infra.embedding import EmbeddingClient


class CharacterCounter:
    def count(self, text: str) -> int:
        return len(text)


def _profile(**changes: object) -> SimpleNamespace:
    values = {
        "id": uuid4(),
        "name": "test-space-v1",
        "base_url": "https://embedding.invalid/v1",
        "model": "test-model",
        "dimension": 3,
        "max_input_tokens": 100,
        "batch_limit": 2,
        "send_dimensions": True,
        "normalize": False,
        "query_prefix": "",
        "document_prefix": "",
        "tokenizer_encoding": "o200k_base",
    }
    return SimpleNamespace(**(values | changes))


def _client(monkeypatch: pytest.MonkeyPatch, *, profile: object | None = None, responses: list | None = None):
    requests: list[dict] = []
    constructor_options: list[dict] = []

    async def create(**request: object) -> SimpleNamespace:
        requests.append(request)
        data = responses.pop(0) if responses is not None else [
            SimpleNamespace(index=index, embedding=[3.0, 4.0, 0.0])
            for index, _ in enumerate(request["input"])
        ]
        return SimpleNamespace(data=data)

    def openai(**options: object) -> SimpleNamespace:
        constructor_options.append(options)
        return SimpleNamespace(embeddings=SimpleNamespace(create=create))

    monkeypatch.setattr("app.infra.embedding.AsyncOpenAI", openai)
    return EmbeddingClient(profile or _profile(), "test-secret", counter=CharacterCounter()), requests, constructor_options


@pytest.mark.parametrize("send_dimensions", [False, True])
async def test_dimensions_is_a_profile_capability(monkeypatch, send_dimensions) -> None:
    client, requests, _ = _client(monkeypatch, profile=_profile(send_dimensions=send_dimensions))

    assert await client.embed_one("example") == [3.0, 4.0, 0.0]

    assert ("dimensions" in requests[0]) is send_dimensions
    if send_dimensions:
        assert requests[0]["dimensions"] == 3
    assert requests[0]["model"] == "test-model"
    assert requests[0]["encoding_format"] == "float"


async def test_batching_and_query_document_preprocessing_follow_profile(monkeypatch) -> None:
    client, requests, options = _client(
        monkeypatch, profile=_profile(query_prefix="query: ", document_prefix="document: ", batch_limit=2)
    )

    assert len(await client.embed_many(["a", "b", "c"])) == 3
    await client.embed_one("lookup", purpose="query")

    assert [request["input"] for request in requests] == [
        ["document: a", "document: b"], ["document: c"], ["query: lookup"]
    ]
    assert options[0]["max_retries"] == 2


async def test_over_limit_input_fails_before_any_batch_and_is_never_truncated(monkeypatch) -> None:
    client, requests, _ = _client(
        monkeypatch, profile=_profile(max_input_tokens=5, document_prefix="d:", batch_limit=1)
    )

    with pytest.raises(ValueError, match=r"item=1, tokens=6, limit=5"):
        await client.embed_many(["a", "1234"])

    assert requests == []


async def test_normalization_is_explicit_and_disabled_profile_preserves_provider_values(monkeypatch) -> None:
    normalized, _, _ = _client(monkeypatch, profile=_profile(normalize=True))
    assert await normalized.embed_one("example") == pytest.approx([0.6, 0.8, 0.0])

    unchanged, _, _ = _client(monkeypatch, profile=_profile(normalize=False))
    assert await unchanged.embed_one("example") == [3.0, 4.0, 0.0]


async def test_response_indices_restore_request_order(monkeypatch) -> None:
    responses = [[
        SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
        SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
    ]]
    client, _, _ = _client(monkeypatch, responses=responses)

    assert await client.embed_many(["first", "second"]) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


@pytest.mark.parametrize("indices", [[], [0], [0, 0], [0, 2], [-1, 0]])
async def test_missing_duplicate_and_invalid_response_indices_are_rejected(monkeypatch, indices) -> None:
    responses = [[SimpleNamespace(index=index, embedding=[1.0, 0.0, 0.0]) for index in indices]]
    client, _, _ = _client(monkeypatch, responses=responses)

    with pytest.raises(RuntimeError, match="response indices"):
        await client.embed_many(["first", "second"])


@pytest.mark.parametrize(
    ("vector", "error"),
    [
        ([1.0, 0.0], "dimension mismatch"),
        ([1.0, math.inf, 0.0], "non-finite"),
        ([1.0, math.nan, 0.0], "non-finite"),
        ([0.0, 0.0, 0.0], "zero vector"),
    ],
)
async def test_invalid_vectors_never_reach_storage(monkeypatch, vector, error) -> None:
    client, _, _ = _client(monkeypatch, responses=[[SimpleNamespace(index=0, embedding=vector)]])

    with pytest.raises(ValueError, match=error):
        await client.embed_one("example")


async def test_empty_batch_has_no_provider_call(monkeypatch) -> None:
    client, requests, _ = _client(monkeypatch)

    assert await client.embed_many([]) == []
    assert requests == []
    assert client.embedding_batches == 0
    assert client.http_attempts == 0


async def test_http_attempt_count_includes_sdk_retries_without_changing_policy(monkeypatch) -> None:
    import httpx

    requests = []

    async def handler(request):
        requests.append(request)
        if len(requests) < 3:
            return httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "retry"}})
        return httpx.Response(200, json={
            "object": "list", "model": "test-model",
            "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    monkeypatch.setattr("app.infra.embedding.DefaultAsyncHttpxClient", lambda **options: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), **options,
    ))
    client = EmbeddingClient(_profile(), "test-secret", counter=CharacterCounter())
    try:
        assert await client.embed_one("synthetic input") == [1.0, 0.0, 0.0]
        assert client.embedding_batches == 1
        assert client.http_attempts == 3
        assert client._client.max_retries == 2
    finally:
        await client._client.close()


async def test_invalid_encoding_purpose_is_rejected_before_request(monkeypatch) -> None:
    client, requests, _ = _client(monkeypatch)

    with pytest.raises(ValueError, match="purpose"):
        await client.embed_one("example", purpose="invalid")

    assert requests == []


def test_profile_identity_is_exposed_to_retrieval_and_eval(monkeypatch) -> None:
    profile = _profile()
    client, _, _ = _client(monkeypatch, profile=profile)

    assert client.profile_metadata() == {
        "embedding_profile_id": str(profile.id),
        "embedding_profile_name": profile.name,
        "embedding_model": profile.model,
        "embedding_dimension": profile.dimension,
    }
