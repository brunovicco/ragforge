"""Tests for the Gemini<->LightRAG glue functions, using fakes (no network)."""

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from google import genai
from google.genai import errors

from ragforge.generation.errors import GenerationError
from ragforge.retrieval.graph.lightrag_gemini import (
    build_gemini_embedding_func,
    build_gemini_llm_model_func,
)


class _FakeEmbedder:
    dimensions = 3

    def __init__(self) -> None:
        self.embedded_texts: list[str] | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts = texts
        return [[1.0, 2.0, 3.0] for _ in texts]


def test_gemini_embedding_func_wraps_the_sync_embedder() -> None:
    """The async embedding_func calls the sync embedder and returns a float32 array."""
    embedder = _FakeEmbedder()
    embedding_func = build_gemini_embedding_func(embedder)  # type: ignore[arg-type]

    result = asyncio.run(embedding_func.func(["texto um", "texto dois"]))

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    assert embedder.embedded_texts == ["texto um", "texto dois"]
    assert embedding_func.embedding_dim == 3


def test_raises_when_no_api_key_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key (env or explicit) fails fast with no network call."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="no Gemini API key"):
        build_gemini_llm_model_func("gemini-3.1-flash-lite")


class _FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, handler: Callable[[list[object]], _FakeResponse]) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def generate_content(
        self, *, model: str, contents: list[object], config: object
    ) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._handler(contents)


class _FakeClient:
    def __init__(self, handler: Callable[[list[object]], _FakeResponse]) -> None:
        self.models = _FakeModels(handler)


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[list[object]], _FakeResponse]
) -> _FakeClient:
    fake_client = _FakeClient(handler)
    monkeypatch.setattr(genai, "Client", lambda api_key: fake_client)
    return fake_client


def test_llm_model_func_returns_response_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful call returns the model's text."""
    _install_fake_client(monkeypatch, lambda contents: _FakeResponse("resposta"))
    llm_model_func = build_gemini_llm_model_func("gemini-3.1-flash-lite", api_key="key")

    result = asyncio.run(llm_model_func("pergunta"))

    assert result == "resposta"


def test_llm_model_func_includes_history_messages_as_prior_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History messages become prior Content turns, assistant mapped to the model role."""
    fake_client = _install_fake_client(monkeypatch, lambda contents: _FakeResponse("ok"))
    llm_model_func = build_gemini_llm_model_func("gemini-3.1-flash-lite", api_key="key")

    asyncio.run(
        llm_model_func(
            "pergunta atual",
            history_messages=[
                {"role": "user", "content": "pergunta anterior"},
                {"role": "assistant", "content": "resposta anterior"},
            ],
        )
    )

    contents = fake_client.models.calls[0]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]


def test_llm_model_func_returns_empty_string_for_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """LightRAG's contract expects a string; None text becomes an empty string, not an error."""
    _install_fake_client(monkeypatch, lambda contents: _FakeResponse(None))
    llm_model_func = build_gemini_llm_model_func("gemini-3.1-flash-lite", api_key="key")

    result = asyncio.run(llm_model_func("pergunta"))

    assert result == ""


def test_llm_model_func_retries_once_on_rate_limit_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single 429 is retried and the call succeeds without raising."""
    attempts = {"n": 0}

    def handler(contents: list[object]) -> _FakeResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise errors.ClientError(429, {"error": {"message": "rate limited"}})
        return _FakeResponse("ok")

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    _install_fake_client(monkeypatch, handler)
    llm_model_func = build_gemini_llm_model_func("gemini-3.1-flash-lite", api_key="key")

    result = asyncio.run(llm_model_func("pergunta"))

    assert result == "ok"
    assert attempts["n"] == 2


def test_llm_model_func_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent 429s exhaust every retry and surface as GenerationError."""

    def handler(contents: list[object]) -> _FakeResponse:
        raise errors.ClientError(429, {"error": {"message": "rate limited"}})

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    _install_fake_client(monkeypatch, handler)
    llm_model_func = build_gemini_llm_model_func("gemini-3.1-flash-lite", api_key="key")

    with pytest.raises(GenerationError, match="failed"):
        asyncio.run(llm_model_func("pergunta"))


def test_llm_model_func_raises_immediately_on_non_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-429 ClientError is not retried."""
    attempts = {"n": 0}

    def handler(contents: list[object]) -> _FakeResponse:
        attempts["n"] += 1
        raise errors.ClientError(400, {"error": {"message": "bad request"}})

    _install_fake_client(monkeypatch, handler)
    llm_model_func = build_gemini_llm_model_func("gemini-3.1-flash-lite", api_key="key")

    with pytest.raises(GenerationError, match="failed"):
        asyncio.run(llm_model_func("pergunta"))

    assert attempts["n"] == 1


def test_raises_when_client_creation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genai.Client construction failure is translated to GenerationError."""

    def raise_error(api_key: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(genai, "Client", raise_error)

    with pytest.raises(GenerationError, match="failed to create Gemini client"):
        build_gemini_llm_model_func("gemini-3.1-flash-lite", api_key="key")
