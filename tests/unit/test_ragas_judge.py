"""Tests for RagasJudge, using fakes for RAGAS metrics and the abstention LLM (no network)."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ragforge.adapters.llm_cache import FileLLMCache
from ragforge.evaluation.judge_ports import JudgeSample, ModelIdentity
from ragforge.evaluation.ragas_judge import (
    RagasJudge,
    build_gemini_ragas_judge,
    build_openai_ragas_judge,
)
from ragforge.generation.errors import GenerationError

_IDENTITY = ModelIdentity(
    provider="test", model="test-model", reasoning_effort=None, output_schema_version=1
)


class _FakeMetricResult:
    def __init__(self, value: float) -> None:
        self.value = value


class _FakeMetric:
    def __init__(self, handler: Callable[..., _FakeMetricResult]) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def score(self, **kwargs: Any) -> _FakeMetricResult:
        self.calls.append(kwargs)
        return self._handler(**kwargs)


class _FakeAbstentionResult:
    def __init__(self, appropriate: bool, rationale: str) -> None:
        self.appropriate = appropriate
        self.rationale = rationale


class _FakeAbstentionLLM:
    def __init__(self, appropriate: bool = True, rationale: str = "ok") -> None:
        self._appropriate = appropriate
        self._rationale = rationale
        self.calls: list[tuple[str, Any]] = []

    def generate(self, prompt: str, response_model: Any) -> Any:
        self.calls.append((prompt, response_model))
        return _FakeAbstentionResult(self._appropriate, self._rationale)


def _sample(
    question: str = "pergunta", answer: str = "resposta", *, unanswerable: bool = False
) -> JudgeSample:
    return JudgeSample(
        question=question,
        contexts=("contexto",),
        answer=answer,
        query_class=None,
        unanswerable=unanswerable,
    )


def test_evaluate_returns_faithfulness_answer_relevancy_and_abstention() -> None:
    """A successful evaluate() call returns all three dimensions in one JudgeResult."""
    faithfulness = _FakeMetric(lambda **kwargs: _FakeMetricResult(0.9))
    answer_relevancy = _FakeMetric(lambda **kwargs: _FakeMetricResult(0.8))
    abstention_llm = _FakeAbstentionLLM(
        appropriate=True, rationale="respondeu com base no contexto"
    )
    judge = RagasJudge(faithfulness, answer_relevancy, abstention_llm, _IDENTITY)

    result = judge.evaluate(_sample())

    assert result.faithfulness.score == 0.9
    assert result.answer_relevancy.score == 0.8
    assert result.abstention.appropriate is True
    assert result.abstention.rationale == "respondeu com base no contexto"
    assert result.schema_version == 1


def test_identity_property_returns_the_configured_identity() -> None:
    """.identity exposes the exact judge configuration, for the run manifest."""
    judge = RagasJudge(
        _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0)),
        _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0)),
        _FakeAbstentionLLM(),
        _IDENTITY,
    )

    assert judge.identity == _IDENTITY


def test_evaluate_passes_question_contexts_and_answer_to_faithfulness() -> None:
    """Faithfulness receives the question, contexts, and generated answer."""
    faithfulness = _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0))
    answer_relevancy = _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0))
    judge = RagasJudge(faithfulness, answer_relevancy, _FakeAbstentionLLM(), _IDENTITY)
    sample = JudgeSample(
        question="PERGUNTA",
        contexts=("CONTEXTO_1", "CONTEXTO_2"),
        answer="RESPOSTA",
        query_class=None,
        unanswerable=False,
    )

    judge.evaluate(sample)

    assert faithfulness.calls == [
        {
            "user_input": "PERGUNTA",
            "response": "RESPOSTA",
            "retrieved_contexts": ["CONTEXTO_1", "CONTEXTO_2"],
        }
    ]


def test_evaluate_passes_question_and_answer_to_answer_relevancy() -> None:
    """Answer Relevancy receives the question and generated answer, no context."""
    faithfulness = _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0))
    answer_relevancy = _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0))
    judge = RagasJudge(faithfulness, answer_relevancy, _FakeAbstentionLLM(), _IDENTITY)

    judge.evaluate(_sample(question="PERGUNTA", answer="RESPOSTA"))

    assert answer_relevancy.calls == [{"user_input": "PERGUNTA", "response": "RESPOSTA"}]


def test_evaluate_raises_generation_error_when_faithfulness_fails() -> None:
    """A failure in the underlying metric call is translated to GenerationError."""

    def raise_error(**kwargs: Any) -> _FakeMetricResult:
        raise RuntimeError("boom")

    faithfulness = _FakeMetric(raise_error)
    answer_relevancy = _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0))
    judge = RagasJudge(faithfulness, answer_relevancy, _FakeAbstentionLLM(), _IDENTITY)

    with pytest.raises(GenerationError, match="RAGAS judge scoring failed"):
        judge.evaluate(_sample())


def test_evaluate_raises_generation_error_when_the_abstention_call_fails() -> None:
    """A failure in this project's own abstention call is also translated to GenerationError."""

    class _FailingAbstentionLLM:
        def generate(self, prompt: str, response_model: Any) -> Any:
            raise RuntimeError("boom")

    judge = RagasJudge(
        _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0)),
        _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0)),
        _FailingAbstentionLLM(),
        _IDENTITY,
    )

    with pytest.raises(GenerationError, match="RAGAS judge scoring failed"):
        judge.evaluate(_sample())


def test_build_gemini_ragas_judge_raises_when_no_api_key_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key (env or explicit) fails fast with no network call."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="no Gemini API key"):
        build_gemini_ragas_judge("gemini-3.1-flash-lite", "gemini-embedding-001")


def test_build_openai_ragas_judge_raises_when_no_api_key_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key (env or explicit) fails fast with no network call."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="no OpenAI API key"):
        build_openai_ragas_judge("gpt-5.4-mini-2026-03-17", "text-embedding-3-small")


def test_a_cache_hit_skips_the_metric_and_abstention_calls_entirely(tmp_path: Path) -> None:
    """The second evaluate() for the same sample reuses the cache, no new calls at all."""
    faithfulness = _FakeMetric(lambda **kwargs: _FakeMetricResult(0.9))
    answer_relevancy = _FakeMetric(lambda **kwargs: _FakeMetricResult(0.8))
    abstention_llm = _FakeAbstentionLLM(appropriate=True, rationale="ok")
    cache = FileLLMCache(tmp_path)
    judge = RagasJudge(faithfulness, answer_relevancy, abstention_llm, _IDENTITY, cache=cache)

    first = judge.evaluate(_sample())
    second = judge.evaluate(_sample())

    assert first == second
    assert len(faithfulness.calls) == 1, "the second evaluate() made no additional metric call"
    assert len(answer_relevancy.calls) == 1
    assert len(abstention_llm.calls) == 1
