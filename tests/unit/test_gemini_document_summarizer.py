"""Tests for the Gemini document-summarization adapter, using a fake genai.Client (no network)."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from google import genai
from google.genai import errors

from ragforge.adapters.llm_cache import FileLLMCache
from ragforge.generation.errors import GenerationError
from ragforge.generation.gemini_document_summarizer import GeminiDocumentSummarizer


def test_raises_when_no_api_key_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key (env or explicit) fails fast with no network call, unlike a missing model."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="no Gemini API key"):
        GeminiDocumentSummarizer("gemini-3.1-flash-lite")


class _FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeModels:
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


def test_summarize_document_returns_a_document_summary_with_generation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful call returns the model's text plus the document/generation identity."""
    _install_fake_client(monkeypatch, lambda contents: _FakeResponse("Resumo do documento."))
    summarizer = GeminiDocumentSummarizer("gemini-3.1-flash-lite", api_key="key")

    result = summarizer.summarize_document("NORM/2020", "sha256-abc", "texto completo da norma")

    assert result.summary == "Resumo do documento."
    assert result.document_id == "NORM/2020"
    assert result.document_version == "sha256-abc"
    assert result.generation_model == "gemini-3.1-flash-lite"
    assert result.prompt_version == "sac-summary-ptbr-v1"


def test_summarize_document_sends_the_document_text_in_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The document text reaches the real prompt sent to the API."""
    fake_client = _install_fake_client(monkeypatch, lambda contents: _FakeResponse("ok"))
    summarizer = GeminiDocumentSummarizer("gemini-3.1-flash-lite", api_key="key")

    summarizer.summarize_document("NORM/2020", "sha256-abc", "TEXTO_MARCADOR_DA_NORMA")

    prompt = fake_client.models.calls[0]["contents"]
    assert "TEXTO_MARCADOR_DA_NORMA" in prompt


def test_retries_once_on_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single 429 is retried and the call succeeds without raising."""
    attempts = {"n": 0}

    def handler(contents: str) -> _FakeResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise errors.ClientError(429, {"error": {"message": "rate limited"}})
        return _FakeResponse("ok")

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _install_fake_client(monkeypatch, handler)
    summarizer = GeminiDocumentSummarizer("gemini-3.1-flash-lite", api_key="key")

    result = summarizer.summarize_document("NORM/2020", "sha256-abc", "texto")

    assert result.summary == "ok"
    assert attempts["n"] == 2


def test_raises_when_the_model_returns_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty/None response text is rejected rather than silently propagated."""
    _install_fake_client(monkeypatch, lambda contents: _FakeResponse(None))
    summarizer = GeminiDocumentSummarizer("gemini-3.1-flash-lite", api_key="key")

    with pytest.raises(GenerationError, match="no text"):
        summarizer.summarize_document("NORM/2020", "sha256-abc", "texto")


def test_raises_when_client_creation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genai.Client construction failure is translated to GenerationError."""

    def raise_error(api_key: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(genai, "Client", raise_error)

    with pytest.raises(GenerationError, match="failed to create Gemini client"):
        GeminiDocumentSummarizer("gemini-3.1-flash-lite", api_key="key")


def test_a_cache_hit_skips_the_api_call_entirely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The second call for the same document identity reuses the cached summary, no new call."""
    fake_client = _install_fake_client(
        monkeypatch, lambda contents: _FakeResponse("Resumo do documento.")
    )
    cache = FileLLMCache(tmp_path)
    summarizer = GeminiDocumentSummarizer("gemini-3.1-flash-lite", api_key="key", cache=cache)

    first = summarizer.summarize_document("NORM/2020", "sha256-abc", "texto completo")
    calls_after_first = len(fake_client.models.calls)
    second = summarizer.summarize_document("NORM/2020", "sha256-abc", "texto completo")

    assert calls_after_first == 1
    assert len(fake_client.models.calls) == 1, "the second call made no additional API call"
    assert second == first


def test_a_changed_document_version_is_a_cache_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A different source hash for the same document_id never reuses a stale cached summary."""
    fake_client = _install_fake_client(monkeypatch, lambda contents: _FakeResponse("resumo"))
    cache = FileLLMCache(tmp_path)
    summarizer = GeminiDocumentSummarizer("gemini-3.1-flash-lite", api_key="key", cache=cache)

    summarizer.summarize_document("NORM/2020", "sha256-old", "texto v1")
    summarizer.summarize_document("NORM/2020", "sha256-new", "texto v2")

    assert len(fake_client.models.calls) == 2
