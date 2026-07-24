"""Integration test for the Gemini answer-generation adapter.

Opt-in only: excluded from the default `pytest` run by the `-m "not integration"`
addopts, because it makes a real, metered API call. Run explicitly with
`uv run pytest -m integration`, with GEMINI_API_KEY or GOOGLE_API_KEY set.
"""

import os

import pytest

_TEST_MODEL = "gemini-3.1-flash-lite"


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not _has_api_key(), reason="GEMINI_API_KEY/GOOGLE_API_KEY not set")
def test_generates_a_grounded_cited_answer_for_a_real_question() -> None:
    """A real Gemini model answers using the given context and cites its structural ID."""
    from ragforge.domain.models import Chunk, Query, RetrievalResult
    from ragforge.generation.gemini_answer_generator import GeminiAnswerGenerator

    generator = GeminiAnswerGenerator(_TEST_MODEL)
    chunk = Chunk(
        chunk_id="c1",
        source_text="Art. 2º As instituições financeiras devem adotar controles de "
        "segurança da informação proporcionais ao seu perfil de risco.",
        retrieval_text="Art. 2º As instituições financeiras devem adotar controles de "
        "segurança da informação proporcionais ao seu perfil de risco.",
        structural_ids=("RES-CMN-4893/2021::art-2",),
    )
    results = [RetrievalResult(chunk=chunk, score=1.0, strategy="dense")]
    query = Query(text="O que as instituições financeiras devem adotar?")

    answer = generator.generate(query, results)

    assert "controles de segurança" in answer.text.lower() or "segurança" in answer.text.lower()
    assert answer.citations == ("RES-CMN-4893/2021::art-2",)
