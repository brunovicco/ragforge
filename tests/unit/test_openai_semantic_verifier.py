"""Tests for the OpenAI semantic support verifier adapter, using a fake instructor client."""

from pathlib import Path
from typing import Any

import instructor
import pytest

from ragforge.adapters.llm_cache import FileLLMCache
from ragforge.evaluation.audit_ports import SupportVerdict
from ragforge.generation.errors import GenerationError
from ragforge.generation.openai_semantic_verifier import OpenAISemanticSupportVerifier


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


def _supported_handler(messages: list[dict[str, str]], response_model: Any) -> Any:
    return response_model(
        verdict="supported",
        rationale="o texto sustenta a afirmação",
        supported_citation_ids=["LC-105/2001::art-1"],
        unsupported_citation_ids=[],
        missing_evidence=[],
    )


def test_raises_when_no_api_key_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key (env or explicit) fails fast with no network call."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="no OpenAI API key"):
        OpenAISemanticSupportVerifier("gpt-5.4-mini-2026-03-17")


def test_raises_when_client_creation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """An instructor.from_provider construction failure is translated to GenerationError."""

    def raise_error(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(instructor, "from_provider", raise_error)

    with pytest.raises(GenerationError, match="failed to create OpenAI verifier client"):
        OpenAISemanticSupportVerifier("gpt-5.4-mini-2026-03-17", api_key="key")


def test_identity_reflects_the_configured_model_and_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """.identity exposes provider/model/reasoning_effort - recorded in the run manifest."""
    _install_fake_instructor_client(monkeypatch, _supported_handler)
    verifier = OpenAISemanticSupportVerifier(
        "gpt-5.4-mini-2026-03-17", reasoning_effort="medium", api_key="key"
    )

    assert verifier.identity.provider == "openai"
    assert verifier.identity.model == "gpt-5.4-mini-2026-03-17"
    assert verifier.identity.reasoning_effort == "medium"


def test_verify_returns_the_structured_support_judgment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful call returns the parsed verdict/rationale/citation attribution."""
    _install_fake_instructor_client(monkeypatch, _supported_handler)
    verifier = OpenAISemanticSupportVerifier("gpt-5.4-mini-2026-03-17", api_key="key")

    result = verifier.verify(
        "pergunta", "afirmação de teste", (("LC-105/2001::art-1", "texto-fonte"),)
    )

    assert result.verdict == SupportVerdict.SUPPORTED
    assert result.rationale == "o texto sustenta a afirmação"
    assert result.supported_citation_ids == ("LC-105/2001::art-1",)


def test_verify_sends_the_question_claim_and_citations_in_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The question, claim text, and cited source text all reach the real prompt."""
    fake_client = _install_fake_instructor_client(monkeypatch, _supported_handler)
    verifier = OpenAISemanticSupportVerifier("gpt-5.4-mini-2026-03-17", api_key="key")

    verifier.verify(
        "PERGUNTA_MARCADOR",
        "AFIRMACAO_MARCADOR",
        (("LC-105/2001::art-1", "FONTE_MARCADOR"),),
    )

    prompt = fake_client.chat.completions.calls[0]["messages"][-1]["content"]
    assert "PERGUNTA_MARCADOR" in prompt
    assert "AFIRMACAO_MARCADOR" in prompt
    assert "FONTE_MARCADOR" in prompt


def test_a_cache_hit_skips_the_api_call_entirely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The second verify() for the same triple reuses the cached result, no new call."""
    fake_client = _install_fake_instructor_client(monkeypatch, _supported_handler)
    cache = FileLLMCache(tmp_path)
    verifier = OpenAISemanticSupportVerifier("gpt-5.4-mini-2026-03-17", api_key="key", cache=cache)
    citations = (("LC-105/2001::art-1", "texto-fonte"),)

    first = verifier.verify("pergunta", "afirmação", citations)
    calls_after_first = len(fake_client.chat.completions.calls)
    second = verifier.verify("pergunta", "afirmação", citations)

    assert calls_after_first == 1
    assert len(fake_client.chat.completions.calls) == 1, "the second verify() made no extra call"
    assert second == first


def test_verify_raises_generation_error_when_the_underlying_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure in the underlying instructor call is translated to GenerationError."""

    def failing_handler(messages: list[dict[str, str]], response_model: Any) -> Any:
        raise RuntimeError("model refused")

    _install_fake_instructor_client(monkeypatch, failing_handler)
    verifier = OpenAISemanticSupportVerifier("gpt-5.4-mini-2026-03-17", api_key="key")

    with pytest.raises(GenerationError, match="OpenAI semantic verification failed"):
        verifier.verify("pergunta", "afirmação", (("LC-105/2001::art-1", "texto"),))
