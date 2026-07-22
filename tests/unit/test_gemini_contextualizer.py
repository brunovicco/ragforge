"""Tests for the Gemini contextualization adapter, using a fake genai.Client (no network)."""

import time
from collections.abc import Callable
from typing import Any

import pytest
from google import genai
from google.genai import errors

from ragforge.generation.errors import GenerationError
from ragforge.generation.gemini_contextualizer import GeminiContextualizer


def test_raises_when_no_api_key_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key (env or explicit) fails fast with no network call, unlike a missing model."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="no Gemini API key"):
        GeminiContextualizer("gemini-3.1-flash-lite")


class _FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeModels:
    """Stands in for ``genai.Client().models``, behaving per the scripted handler."""

    def __init__(self, handler: Callable[[str], _FakeResponse]) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: str, config: object) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._handler(contents)


class _FakeClient:
    def __init__(self, handler: Callable[[str], _FakeResponse]) -> None:
        self.models = _FakeModels(handler)


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[str], _FakeResponse]
) -> _FakeClient:
    fake_client = _FakeClient(handler)
    monkeypatch.setattr(genai, "Client", lambda api_key: fake_client)
    return fake_client


def test_contextualize_returns_stripped_response_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful call returns the model's text, stripped of surrounding whitespace."""
    _install_fake_client(monkeypatch, lambda contents: _FakeResponse("  a short context.  "))
    contextualizer = GeminiContextualizer("gemini-3.1-flash-lite", api_key="key")

    context = contextualizer.contextualize("documento completo", "Art. 1º")

    assert context == "a short context."


def test_contextualize_sends_the_document_and_chunk_in_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The document and chunk text both reach the real prompt sent to the API."""
    fake_client = _install_fake_client(monkeypatch, lambda contents: _FakeResponse("context"))
    contextualizer = GeminiContextualizer("gemini-3.1-flash-lite", api_key="key")

    contextualizer.contextualize("DOCUMENTO_MARCADOR", "CHUNK_MARCADOR")

    prompt = fake_client.models.calls[0]["contents"]
    assert "DOCUMENTO_MARCADOR" in prompt
    assert "CHUNK_MARCADOR" in prompt


def test_retries_once_on_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single 429 is retried and the call succeeds without raising."""
    attempts = {"n": 0}

    def handler(contents: str) -> _FakeResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise errors.ClientError(429, {"error": {"message": "rate limited"}})
        return _FakeResponse("context")

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _install_fake_client(monkeypatch, handler)
    contextualizer = GeminiContextualizer("gemini-3.1-flash-lite", api_key="key")

    context = contextualizer.contextualize("documento", "chunk")

    assert context == "context"
    assert attempts["n"] == 2


def test_raises_after_exhausting_retries_on_persistent_rate_limiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent 429s exhaust every retry and surface as GenerationError."""

    def handler(contents: str) -> _FakeResponse:
        raise errors.ClientError(429, {"error": {"message": "rate limited"}})

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _install_fake_client(monkeypatch, handler)
    contextualizer = GeminiContextualizer("gemini-3.1-flash-lite", api_key="key")

    with pytest.raises(GenerationError, match="failed"):
        contextualizer.contextualize("documento", "chunk")


def test_raises_immediately_on_a_non_rate_limit_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-429 ClientError is not retried."""
    attempts = {"n": 0}

    def handler(contents: str) -> _FakeResponse:
        attempts["n"] += 1
        raise errors.ClientError(400, {"error": {"message": "bad request"}})

    _install_fake_client(monkeypatch, handler)
    contextualizer = GeminiContextualizer("gemini-3.1-flash-lite", api_key="key")

    with pytest.raises(GenerationError, match="failed"):
        contextualizer.contextualize("documento", "chunk")

    assert attempts["n"] == 1


def test_raises_on_unexpected_exception_during_generate_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-ClientError exception is wrapped and not retried."""

    def handler(contents: str) -> _FakeResponse:
        raise RuntimeError("boom")

    _install_fake_client(monkeypatch, handler)
    contextualizer = GeminiContextualizer("gemini-3.1-flash-lite", api_key="key")

    with pytest.raises(GenerationError, match="failed"):
        contextualizer.contextualize("documento", "chunk")


def test_raises_when_the_model_returns_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty/None response text is rejected rather than silently propagated."""
    _install_fake_client(monkeypatch, lambda contents: _FakeResponse(None))
    contextualizer = GeminiContextualizer("gemini-3.1-flash-lite", api_key="key")

    with pytest.raises(GenerationError, match="no text"):
        contextualizer.contextualize("documento", "chunk")


def test_raises_when_client_creation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genai.Client construction failure is translated to GenerationError."""

    def raise_error(api_key: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(genai, "Client", raise_error)

    with pytest.raises(GenerationError, match="failed to create Gemini client"):
        GeminiContextualizer("gemini-3.1-flash-lite", api_key="key")
