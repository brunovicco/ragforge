"""Tests for the Gemini answer-generation adapter, using a fake genai.Client (no network)."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from google import genai
from google.genai import errors

from ragforge.adapters.llm_cache import FileLLMCache
from ragforge.domain.models import Chunk, Query, RetrievalResult
from ragforge.generation.errors import GenerationError
from ragforge.generation.gemini_answer_generator import (
    GeminiAnswerGenerator,
    _extract_citations,
    _format_context,
)


def _result(chunk_id: str, text: str, structural_ids: tuple[str, ...]) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id, source_text=text, retrieval_text=text, structural_ids=structural_ids
    )
    return RetrievalResult(chunk=chunk, score=1.0, strategy="dense")


def test_extract_citations_returns_well_formed_structural_ids_in_first_cited_order() -> None:
    """Only valid structural IDs are extracted, deduplicated, in first-seen order."""
    text = "Regra A [LC-105/2001::art-10]. Regra B [RES-CMN-4893/2021::art-2::par-1]."

    citations = _extract_citations(text)

    assert citations == ("LC-105/2001::art-10", "RES-CMN-4893/2021::art-2::par-1")


def test_extract_citations_skips_malformed_brackets() -> None:
    """A bracketed span that isn't a valid structural ID is skipped, not raised."""
    text = "See [this note] and [LC-105/2001::art-10]."

    citations = _extract_citations(text)

    assert citations == ("LC-105/2001::art-10",)


def test_extract_citations_deduplicates_repeated_citations() -> None:
    """The same structural ID cited twice appears once, at its first position."""
    text = "First [LC-105/2001::art-10]. Again [LC-105/2001::art-10]."

    citations = _extract_citations(text)

    assert citations == ("LC-105/2001::art-10",)


def test_extract_citations_returns_empty_tuple_for_no_citations() -> None:
    """Text with no brackets at all yields no citations, not an error."""
    assert _extract_citations("Uma resposta sem citações.") == ()


def test_format_context_includes_each_result_text_and_structural_ids() -> None:
    """Every result's structural_ids and text appear in the formatted context block."""
    results = [
        _result("c1", "Texto do artigo 1.", ("LC-105/2001::art-1",)),
        _result("c2", "Texto do artigo 2.", ("LC-105/2001::art-2",)),
    ]

    context = _format_context(results)

    assert "LC-105/2001::art-1" in context
    assert "Texto do artigo 1." in context
    assert "LC-105/2001::art-2" in context
    assert "Texto do artigo 2." in context


def test_raises_when_no_api_key_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key (env or explicit) fails fast with no network call, unlike a missing model."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="no Gemini API key"):
        GeminiAnswerGenerator("gemini-3.1-flash-lite")


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


def test_generate_returns_an_answer_with_extracted_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful call returns Answer.text and its parsed citations."""
    _install_fake_client(
        monkeypatch, lambda contents: _FakeResponse("Resposta [LC-105/2001::art-10].")
    )
    generator = GeminiAnswerGenerator("gemini-3.1-flash-lite", api_key="key")
    results = [_result("c1", "Texto.", ("LC-105/2001::art-10",))]

    answer = generator.generate(Query(text="pergunta"), results)

    assert answer.text == "Resposta [LC-105/2001::art-10]."
    assert answer.citations == ("LC-105/2001::art-10",)


def test_generate_sends_the_query_and_context_in_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The query text and every result's content reach the real prompt sent to the API."""
    fake_client = _install_fake_client(monkeypatch, lambda contents: _FakeResponse("ok"))
    generator = GeminiAnswerGenerator("gemini-3.1-flash-lite", api_key="key")
    results = [_result("c1", "CONTEUDO_MARCADOR", ("LC-105/2001::art-1",))]

    generator.generate(Query(text="PERGUNTA_MARCADOR"), results)

    prompt = fake_client.models.calls[0]["contents"]
    assert "PERGUNTA_MARCADOR" in prompt
    assert "CONTEUDO_MARCADOR" in prompt


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
    generator = GeminiAnswerGenerator("gemini-3.1-flash-lite", api_key="key")

    answer = generator.generate(Query(text="pergunta"), [])

    assert answer.text == "ok"
    assert attempts["n"] == 2


def test_raises_when_the_model_returns_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty/None response text is rejected rather than silently propagated."""
    _install_fake_client(monkeypatch, lambda contents: _FakeResponse(None))
    generator = GeminiAnswerGenerator("gemini-3.1-flash-lite", api_key="key")

    with pytest.raises(GenerationError, match="no text"):
        generator.generate(Query(text="pergunta"), [])


def test_raises_when_client_creation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genai.Client construction failure is translated to GenerationError."""

    def raise_error(api_key: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(genai, "Client", raise_error)

    with pytest.raises(GenerationError, match="failed to create Gemini client"):
        GeminiAnswerGenerator("gemini-3.1-flash-lite", api_key="key")


def test_a_cache_hit_skips_the_api_call_entirely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The second generate() for the same query+context reuses the cached answer, no new call."""
    fake_client = _install_fake_client(
        monkeypatch, lambda contents: _FakeResponse("Resposta [LC-105/2001::art-10].")
    )
    cache = FileLLMCache(tmp_path)
    generator = GeminiAnswerGenerator("gemini-3.1-flash-lite", api_key="key", cache=cache)
    results = [_result("c1", "Texto.", ("LC-105/2001::art-10",))]

    first = generator.generate(Query(text="pergunta"), results)
    calls_after_first = len(fake_client.models.calls)
    second = generator.generate(Query(text="pergunta"), results)

    assert calls_after_first == 1
    assert len(fake_client.models.calls) == 1, "the second generate() made no additional API call"
    assert second == first
