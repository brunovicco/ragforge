"""Tests for the OpenAI answer rewriter adapter, using a fake instructor client."""

from pathlib import Path
from typing import Any

import instructor
import pytest

from ragforge.adapters.llm_cache import FileLLMCache
from ragforge.evaluation.audit_ports import (
    AnswerClaim,
    ClaimAudit,
    DeterministicCitationCheck,
)
from ragforge.generation.errors import GenerationError
from ragforge.generation.openai_answer_rewriter import OpenAIAnswerRewriter


class _FakeChatCompletions:
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def create(
        self, *, model: str, messages: list[dict[str, str]], response_model: Any, **kwargs: Any
    ) -> Any:
        self.calls.append(
            {"model": model, "messages": messages, "response_model": response_model, **kwargs}
        )
        return self._handler(messages, response_model)


class _FakeChat:
    def __init__(self, handler: Any) -> None:
        self.completions = _FakeChatCompletions(handler)


class _FakeInstructorClient:
    def __init__(self, handler: Any) -> None:
        self.chat = _FakeChat(handler)


def _install_fake_instructor_client(
    monkeypatch: pytest.MonkeyPatch, handler: Any
) -> _FakeInstructorClient:
    fake_client = _FakeInstructorClient(handler)
    monkeypatch.setattr(instructor, "from_provider", lambda *args, **kwargs: fake_client)
    return fake_client


def _handler_returning(text: str) -> Any:
    def handler(messages: list[dict[str, str]], response_model: Any) -> Any:
        return response_model(rewritten_answer=text)

    return handler


def _findings() -> tuple[ClaimAudit, ...]:
    claim = AnswerClaim(
        claim_id="claim-0",
        text="Devem adotar controles [LC-105/2001::art-99].",
        cited_structural_ids=("LC-105/2001::art-99",),
        sentence_index=0,
        material=True,
    )
    check = DeterministicCitationCheck(
        structural_id="LC-105/2001::art-99",
        well_formed=True,
        exists_in_corpus=False,
        belongs_to_selected_document_version=True,
        present_in_retrieved_context=False,
        source_text_hash_matches=False,
    )
    return (ClaimAudit(claim=claim, citation_checks=(check,), semantic_support=None),)


def test_raises_when_no_api_key_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key (env or explicit) fails fast with no network call."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="no OpenAI API key"):
        OpenAIAnswerRewriter("gpt-5.4-mini-2026-03-17")


def test_raises_when_client_creation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """An instructor.from_provider construction failure is translated to GenerationError."""

    def raise_error(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(instructor, "from_provider", raise_error)

    with pytest.raises(GenerationError, match="failed to create OpenAI rewriter client"):
        OpenAIAnswerRewriter("gpt-5.4-mini-2026-03-17", api_key="key")


def test_rewrite_returns_the_model_produced_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful call returns the rewritten answer text, stripped."""
    _install_fake_instructor_client(
        monkeypatch, _handler_returning("  Resposta reescrita [LC-105/2001::art-1].  ")
    )
    rewriter = OpenAIAnswerRewriter("gpt-5.4-mini-2026-03-17", api_key="key")

    result = rewriter.rewrite("pergunta", "resposta original", ("texto válido",), _findings())

    assert result == "Resposta reescrita [LC-105/2001::art-1]."


def test_rewrite_sends_question_original_answer_and_valid_sources_in_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The question, original answer, and valid source texts all reach the real prompt."""
    fake_client = _install_fake_instructor_client(monkeypatch, _handler_returning("ok"))
    rewriter = OpenAIAnswerRewriter("gpt-5.4-mini-2026-03-17", api_key="key")

    rewriter.rewrite(
        "PERGUNTA_MARCADOR",
        "RESPOSTA_ORIGINAL_MARCADOR",
        ("FONTE_VALIDA_MARCADOR",),
        _findings(),
    )

    prompt = fake_client.chat.completions.calls[0]["messages"][-1]["content"]
    assert "PERGUNTA_MARCADOR" in prompt
    assert "RESPOSTA_ORIGINAL_MARCADOR" in prompt
    assert "FONTE_VALIDA_MARCADOR" in prompt


def test_a_cache_hit_skips_the_api_call_entirely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The second rewrite() for the same inputs reuses the cache, no new call."""
    fake_client = _install_fake_instructor_client(monkeypatch, _handler_returning("resposta"))
    cache = FileLLMCache(tmp_path)
    rewriter = OpenAIAnswerRewriter("gpt-5.4-mini-2026-03-17", api_key="key", cache=cache)
    findings = _findings()

    first = rewriter.rewrite("pergunta", "original", ("fonte",), findings)
    calls_after_first = len(fake_client.chat.completions.calls)
    second = rewriter.rewrite("pergunta", "original", ("fonte",), findings)

    assert calls_after_first == 1
    assert len(fake_client.chat.completions.calls) == 1, "the second rewrite() made no extra call"
    assert second == first


def test_rewrite_raises_generation_error_when_the_underlying_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure in the underlying instructor call is translated to GenerationError."""

    def failing_handler(messages: list[dict[str, str]], response_model: Any) -> Any:
        raise RuntimeError("model refused")

    _install_fake_instructor_client(monkeypatch, failing_handler)
    rewriter = OpenAIAnswerRewriter("gpt-5.4-mini-2026-03-17", api_key="key")

    with pytest.raises(GenerationError, match="OpenAI answer rewrite failed"):
        rewriter.rewrite("pergunta", "original", ("fonte",), _findings())
