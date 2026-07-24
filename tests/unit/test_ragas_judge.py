"""Tests for RagasJudge, using fake RAGAS metric objects (no network, no real RAGAS calls)."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ragforge.adapters.llm_cache import FileLLMCache
from ragforge.evaluation.ragas_judge import RagasJudge
from ragforge.generation.errors import GenerationError


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


def test_score_returns_faithfulness_and_answer_relevancy() -> None:
    """A successful scoring call returns both metric values, keyed by name."""
    faithfulness = _FakeMetric(lambda **kwargs: _FakeMetricResult(0.9))
    answer_relevancy = _FakeMetric(lambda **kwargs: _FakeMetricResult(0.8))
    judge = RagasJudge(faithfulness, answer_relevancy)

    scores = judge.score("pergunta", ["contexto"], "resposta")

    assert scores == {"faithfulness": 0.9, "answer_relevancy": 0.8}


def test_score_passes_query_context_and_answer_to_faithfulness() -> None:
    """Faithfulness receives the question, contexts, and generated answer."""
    faithfulness = _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0))
    answer_relevancy = _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0))
    judge = RagasJudge(faithfulness, answer_relevancy)

    judge.score("PERGUNTA", ["CONTEXTO_1", "CONTEXTO_2"], "RESPOSTA")

    assert faithfulness.calls == [
        {
            "user_input": "PERGUNTA",
            "response": "RESPOSTA",
            "retrieved_contexts": ["CONTEXTO_1", "CONTEXTO_2"],
        }
    ]


def test_score_passes_query_and_answer_to_answer_relevancy() -> None:
    """Answer Relevancy receives the question and generated answer, no context."""
    faithfulness = _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0))
    answer_relevancy = _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0))
    judge = RagasJudge(faithfulness, answer_relevancy)

    judge.score("PERGUNTA", ["CONTEXTO"], "RESPOSTA")

    assert answer_relevancy.calls == [{"user_input": "PERGUNTA", "response": "RESPOSTA"}]


def test_score_raises_generation_error_when_faithfulness_fails() -> None:
    """A failure in the underlying metric call is translated to GenerationError."""

    def raise_error(**kwargs: Any) -> _FakeMetricResult:
        raise RuntimeError("boom")

    faithfulness = _FakeMetric(raise_error)
    answer_relevancy = _FakeMetric(lambda **kwargs: _FakeMetricResult(1.0))
    judge = RagasJudge(faithfulness, answer_relevancy)

    with pytest.raises(GenerationError, match="RAGAS judge scoring failed"):
        judge.score("pergunta", ["contexto"], "resposta")


def test_build_gemini_ragas_judge_raises_when_no_api_key_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key (env or explicit) fails fast with no network call."""
    from ragforge.evaluation.ragas_judge import build_gemini_ragas_judge

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="no Gemini API key"):
        build_gemini_ragas_judge("gemini-3.1-flash-lite", "gemini-embedding-001")


def test_a_cache_hit_skips_the_metric_calls_entirely(tmp_path: Path) -> None:
    """The second score() for the same triple reuses the cached metrics, no new metric calls."""
    faithfulness = _FakeMetric(lambda **kwargs: _FakeMetricResult(0.9))
    answer_relevancy = _FakeMetric(lambda **kwargs: _FakeMetricResult(0.8))
    cache = FileLLMCache(tmp_path)
    judge = RagasJudge(faithfulness, answer_relevancy, model_name="judge-model", cache=cache)

    first = judge.score("pergunta", ["contexto"], "resposta")
    second = judge.score("pergunta", ["contexto"], "resposta")

    assert first == {"faithfulness": 0.9, "answer_relevancy": 0.8}
    assert second == first
    assert len(faithfulness.calls) == 1, "the second score() made no additional metric call"
    assert len(answer_relevancy.calls) == 1
