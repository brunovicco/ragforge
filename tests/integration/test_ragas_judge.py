"""Integration test for the RAGAS judge (ADR-0007/ADR-0018).

Opt-in only: excluded from the default `pytest` run by the `-m "not integration"`
addopts, because it makes several real, metered API calls (RAGAS's internal
LLM calls for statement generation/verification, plus embeddings for Answer
Relevancy, plus this project's own abstention call). Run explicitly with
`uv run pytest -m integration`, with GEMINI_API_KEY/GOOGLE_API_KEY (Gemini)
or OPENAI_API_KEY (OpenAI, the ADR-0018 canonical judge) set.

Scores are unvalidated (ADR-0007/ADR-0018's human calibration hasn't
happened - see judge_calibration.py) - these tests only prove the real
provider/ragas/instructor wiring works, not that the scores are trustworthy
for the published benchmark.
"""

import os

import pytest

from ragforge.evaluation.judge_ports import JudgeSample

_TEST_GEMINI_LLM_MODEL = "gemini-3.1-flash-lite"
_TEST_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
_TEST_OPENAI_LLM_MODEL = "gpt-5.4-mini-2026-03-17"
_TEST_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

_SAMPLE = JudgeSample(
    question="O que as instituições financeiras devem adotar?",
    contexts=(
        "Art. 2º As instituições financeiras devem adotar controles de "
        "segurança da informação proporcionais ao seu perfil de risco.",
    ),
    answer="As instituições financeiras devem adotar controles de segurança da informação.",
    query_class=None,
    unanswerable=False,
)


def _has_gemini_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _has_openai_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not _has_gemini_api_key(), reason="GEMINI_API_KEY/GOOGLE_API_KEY not set")
def test_gemini_judge_scores_a_faithful_grounded_answer_highly() -> None:
    """A real Gemini-backed judge scores an answer that's fully grounded in its context highly."""
    from ragforge.evaluation.ragas_judge import build_gemini_ragas_judge

    judge = build_gemini_ragas_judge(_TEST_GEMINI_LLM_MODEL, _TEST_GEMINI_EMBEDDING_MODEL)

    result = judge.evaluate(_SAMPLE)

    assert result.faithfulness.score >= 0.5
    assert 0.0 <= result.answer_relevancy.score <= 1.0
    assert isinstance(result.abstention.appropriate, bool)


@pytest.mark.integration
@pytest.mark.skipif(not _has_openai_api_key(), reason="OPENAI_API_KEY not set")
def test_openai_judge_scores_a_faithful_grounded_answer_highly() -> None:
    """A real OpenAI-backed judge (ADR-0018 canonical) scores a grounded answer highly."""
    from ragforge.evaluation.ragas_judge import build_openai_ragas_judge

    judge = build_openai_ragas_judge(_TEST_OPENAI_LLM_MODEL, _TEST_OPENAI_EMBEDDING_MODEL)

    result = judge.evaluate(_SAMPLE)

    assert result.faithfulness.score >= 0.5
    assert 0.0 <= result.answer_relevancy.score <= 1.0
    assert isinstance(result.abstention.appropriate, bool)
