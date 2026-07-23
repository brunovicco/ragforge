"""Integration test for the RAGAS judge (ADR-0007).

Opt-in only: excluded from the default `pytest` run by the `-m "not integration"`
addopts, because it makes several real, metered API calls (RAGAS's internal
LLM calls for statement generation/verification, plus embeddings for Answer
Relevancy). Run explicitly with `uv run pytest -m integration`, with
GEMINI_API_KEY or GOOGLE_API_KEY set.

Scores are unvalidated (ADR-0007's human calibration hasn't happened) - this
test only proves the real Gemini/ragas/instructor wiring works, not that the
scores are trustworthy for the published benchmark.
"""

import os

import pytest

_TEST_LLM_MODEL = "gemini-3.1-flash-lite"
_TEST_EMBEDDING_MODEL = "gemini-embedding-001"


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not _has_api_key(), reason="GEMINI_API_KEY/GOOGLE_API_KEY not set")
def test_scores_a_faithful_grounded_answer_highly() -> None:
    """A real RAGAS judge scores an answer that's fully grounded in its context highly."""
    from ragforge.evaluation.ragas_judge import build_gemini_ragas_judge

    judge = build_gemini_ragas_judge(_TEST_LLM_MODEL, _TEST_EMBEDDING_MODEL)

    scores = judge.score(
        query_text="O que as instituições financeiras devem adotar?",
        contexts=[
            "Art. 2º As instituições financeiras devem adotar controles de "
            "segurança da informação proporcionais ao seu perfil de risco."
        ],
        answer_text="As instituições financeiras devem adotar controles de "
        "segurança da informação.",
    )

    assert set(scores) == {"faithfulness", "answer_relevancy"}
    assert scores["faithfulness"] >= 0.5
    assert 0.0 <= scores["answer_relevancy"] <= 1.0
