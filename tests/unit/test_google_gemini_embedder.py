"""Tests for the Gemini embedding adapter (ADR-0005), using a fake genai.Client (no network)."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from google import genai
from google.genai import errors

from ragforge.adapters.llm_cache import FileLLMCache
from ragforge.embeddings.errors import EmbeddingError
from ragforge.embeddings.google_gemini_embedder import GoogleGeminiEmbedder


def test_raises_when_no_api_key_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key (env or explicit) fails fast with no network call, unlike a missing model."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(EmbeddingError, match="no Gemini API key"):
        GoogleGeminiEmbedder("gemini-embedding-001")


class _FakeEmbedding:
    def __init__(self, values: list[float] | None) -> None:
        self.values = values


class _FakeResponse:
    def __init__(self, embeddings: list[_FakeEmbedding]) -> None:
        self.embeddings = embeddings


class _FakeModels:
    """Stands in for ``genai.Client().models``, behaving per the scripted handler."""

    def __init__(self, handler: Callable[[list[str]], _FakeResponse]) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def embed_content(self, *, model: str, contents: list[str], config: object) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._handler(contents)


class _FakeClient:
    def __init__(self, handler: Callable[[list[str]], _FakeResponse]) -> None:
        self.models = _FakeModels(handler)


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[list[str]], _FakeResponse]
) -> _FakeClient:
    fake_client = _FakeClient(handler)
    monkeypatch.setattr(genai, "Client", lambda api_key: fake_client)
    return fake_client


def _vector_response(n: int, dims: int = 3) -> _FakeResponse:
    return _FakeResponse([_FakeEmbedding([1.0] * dims) for _ in range(n)])


def test_embed_returns_one_vector_per_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful call maps each input text to one embedding vector, in order."""
    _install_fake_client(monkeypatch, lambda contents: _vector_response(len(contents)))
    embedder = GoogleGeminiEmbedder("gemini-embedding-001", api_key="key")

    vectors = embedder.embed(["a", "b"])

    assert len(vectors) == 2
    assert all(len(v) == embedder.dimensions for v in vectors)


def test_embed_batches_requests_at_the_configured_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Texts are split into batches of 100 (the default), not sent in one call."""
    fake_client = _install_fake_client(
        monkeypatch, lambda contents: _vector_response(len(contents))
    )
    embedder = GoogleGeminiEmbedder("gemini-embedding-001", api_key="key")
    fake_client.models.calls.clear()  # drop the constructor's probe call

    embedder.embed(["x"] * 150)

    assert [len(call["contents"]) for call in fake_client.models.calls] == [100, 50]


def test_single_text_models_send_one_text_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """gemini-embedding-2's verified single-text quirk forces batch_size=1."""
    fake_client = _install_fake_client(
        monkeypatch, lambda contents: _vector_response(len(contents))
    )
    embedder = GoogleGeminiEmbedder("gemini-embedding-2", api_key="key")
    fake_client.models.calls.clear()

    embedder.embed(["x", "y", "z"])

    assert [len(call["contents"]) for call in fake_client.models.calls] == [1, 1, 1]


def test_output_dimensionality_is_forwarded_to_the_request_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured output_dimensionality reaches the real embed_content config."""
    fake_client = _install_fake_client(
        monkeypatch, lambda contents: _vector_response(len(contents), dims=5)
    )

    GoogleGeminiEmbedder("gemini-embedding-001", api_key="key", output_dimensionality=1536)

    probe_call = fake_client.models.calls[0]
    assert probe_call["config"].output_dimensionality == 1536


def test_retries_once_on_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single 429 is retried and the call succeeds without raising."""
    attempts = {"n": 0}

    def handler(contents: list[str]) -> _FakeResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise errors.ClientError(429, {"error": {"message": "rate limited"}})
        return _vector_response(len(contents))

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _install_fake_client(monkeypatch, handler)

    embedder = GoogleGeminiEmbedder("gemini-embedding-001", api_key="key")

    assert embedder.dimensions == 3
    assert attempts["n"] == 2


def test_raises_after_exhausting_retries_on_persistent_rate_limiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent 429s exhaust every retry and surface as EmbeddingError."""

    def handler(contents: list[str]) -> _FakeResponse:
        raise errors.ClientError(429, {"error": {"message": "rate limited"}})

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _install_fake_client(monkeypatch, handler)

    with pytest.raises(EmbeddingError, match="failed"):
        GoogleGeminiEmbedder("gemini-embedding-001", api_key="key")


def test_raises_immediately_on_a_non_rate_limit_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-429 ClientError is not retried."""
    attempts = {"n": 0}

    def handler(contents: list[str]) -> _FakeResponse:
        attempts["n"] += 1
        raise errors.ClientError(400, {"error": {"message": "bad request"}})

    _install_fake_client(monkeypatch, handler)

    with pytest.raises(EmbeddingError, match="failed"):
        GoogleGeminiEmbedder("gemini-embedding-001", api_key="key")

    assert attempts["n"] == 1


def test_raises_on_unexpected_exception_during_embed_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-ClientError exception is wrapped and not retried."""

    def handler(contents: list[str]) -> _FakeResponse:
        raise RuntimeError("boom")

    _install_fake_client(monkeypatch, handler)

    with pytest.raises(EmbeddingError, match="failed"):
        GoogleGeminiEmbedder("gemini-embedding-001", api_key="key")


def test_raises_when_response_embedding_count_does_not_match_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response with a different embedding count than requested is rejected."""

    def handler(contents: list[str]) -> _FakeResponse:
        if len(contents) == 1:
            return _vector_response(1)
        return _vector_response(len(contents) - 1)

    _install_fake_client(monkeypatch, handler)
    embedder = GoogleGeminiEmbedder("gemini-embedding-001", api_key="key")

    with pytest.raises(EmbeddingError, match="expected 2 embeddings"):
        embedder.embed(["a", "b"])


def test_raises_when_an_embedding_has_no_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response embedding with values=None is rejected rather than silently kept."""

    def handler(contents: list[str]) -> _FakeResponse:
        return _FakeResponse([_FakeEmbedding(None) for _ in contents])

    _install_fake_client(monkeypatch, handler)

    with pytest.raises(EmbeddingError, match="no values"):
        GoogleGeminiEmbedder("gemini-embedding-001", api_key="key")


def test_raises_when_client_creation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genai.Client construction failure is translated to EmbeddingError."""

    def raise_error(api_key: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(genai, "Client", raise_error)

    with pytest.raises(EmbeddingError, match="failed to create Gemini client"):
        GoogleGeminiEmbedder("gemini-embedding-001", api_key="key")


def test_a_cache_hit_skips_the_api_call_entirely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The second embed() for the same text reuses the cached vector, no new API call."""
    fake_client = _install_fake_client(
        monkeypatch, lambda contents: _vector_response(len(contents))
    )
    cache = FileLLMCache(tmp_path)
    embedder = GoogleGeminiEmbedder("gemini-embedding-001", api_key="key", cache=cache)
    fake_client.models.calls.clear()  # drop the constructor's probe call

    first = embedder.embed(["texto repetido"])
    calls_after_first = len(fake_client.models.calls)
    second = embedder.embed(["texto repetido"])

    assert calls_after_first == 1
    assert len(fake_client.models.calls) == 1, "the second embed() made no additional API call"
    assert second == first


def test_only_uncached_texts_reach_the_api_and_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A mix of cached and new texts sends only the new ones, in original relative order."""
    fake_client = _install_fake_client(
        monkeypatch, lambda contents: _vector_response(len(contents))
    )
    cache = FileLLMCache(tmp_path)
    embedder = GoogleGeminiEmbedder("gemini-embedding-001", api_key="key", cache=cache)
    fake_client.models.calls.clear()

    embedder.embed(["a", "b"])
    fake_client.models.calls.clear()

    vectors = embedder.embed(["a", "c", "b"])

    assert len(vectors) == 3
    sent_texts = [text for call in fake_client.models.calls for text in call["contents"]]
    assert sent_texts == ["c"], "only the uncached text 'c' should reach the API"
