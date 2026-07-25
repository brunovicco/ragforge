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
from ragforge.generation.gemini_answer_generator import GeminiAnswerGenerator, _format_context


def _result(chunk_id: str, text: str, structural_ids: tuple[str, ...]) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id, source_text=text, retrieval_text=text, structural_ids=structural_ids
    )
    return RetrievalResult(chunk=chunk, score=1.0, strategy="dense")


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


class _FakeUsageMetadata:
    def __init__(
        self,
        prompt_token_count: int = 10,
        candidates_token_count: int = 5,
        total_token_count: int = 15,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count


class _FakeResponse:
    def __init__(self, text: str | None, usage_metadata: _FakeUsageMetadata | None = None) -> None:
        self.text = text
        self.usage_metadata = usage_metadata if usage_metadata is not None else _FakeUsageMetadata()


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


def test_drain_generation_lineage_captures_token_usage_and_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real (uncached) call records real token counts and cache_hit=False (ADR-0017)."""
    _install_fake_client(
        monkeypatch,
        lambda contents: _FakeResponse(
            "Resposta [LC-105/2001::art-10].",
            usage_metadata=_FakeUsageMetadata(
                prompt_token_count=100, candidates_token_count=20, total_token_count=120
            ),
        ),
    )
    generator = GeminiAnswerGenerator("gemini-3.1-flash-lite", api_key="key")
    results = [_result("c1", "Texto.", ("LC-105/2001::art-10",))]

    answer = generator.generate(Query(text="pergunta"), results)
    [lineage] = generator.drain_generation_lineage()

    assert lineage.provider == "gemini"
    assert lineage.model == "gemini-3.1-flash-lite"
    assert lineage.input_chunk_ids == ("c1",)
    assert lineage.parsed_citations == answer.citations
    assert lineage.prompt_tokens == 100
    assert lineage.completion_tokens == 20
    assert lineage.total_tokens == 120
    assert lineage.cache_hit is False
    assert lineage.latency_seconds >= 0.0


def test_drain_generation_lineage_reports_cache_hit_with_no_token_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The second (cached) generate() reports cache_hit=True and no token counts."""
    _install_fake_client(
        monkeypatch, lambda contents: _FakeResponse("Resposta [LC-105/2001::art-10].")
    )
    cache = FileLLMCache(tmp_path)
    generator = GeminiAnswerGenerator("gemini-3.1-flash-lite", api_key="key", cache=cache)
    results = [_result("c1", "Texto.", ("LC-105/2001::art-10",))]

    generator.generate(Query(text="pergunta"), results)
    generator.generate(Query(text="pergunta"), results)
    first_lineage, second_lineage = generator.drain_generation_lineage()

    assert first_lineage.cache_hit is False
    assert second_lineage.cache_hit is True
    assert second_lineage.prompt_tokens is None
    assert second_lineage.completion_tokens is None
    assert second_lineage.total_tokens is None


def test_drain_generation_lineage_clears_the_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Draining resets the buffer - a second drain with no new call in between is empty."""
    _install_fake_client(monkeypatch, lambda contents: _FakeResponse("ok"))
    generator = GeminiAnswerGenerator("gemini-3.1-flash-lite", api_key="key")

    generator.generate(Query(text="pergunta"), [])
    first_drain = generator.drain_generation_lineage()
    second_drain = generator.drain_generation_lineage()

    assert len(first_drain) == 1
    assert second_drain == []
