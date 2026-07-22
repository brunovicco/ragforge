"""Integration test for the Gemini summarization adapter.

Opt-in only: excluded from the default `pytest` run by the `-m "not integration"`
addopts, because it makes a real, metered API call. Run explicitly with
`uv run pytest -m integration`, with GEMINI_API_KEY or GOOGLE_API_KEY set.

Uses gemini-3.1-flash-lite, the same model verified available against the
live API for test_gemini_contextualizer.py (2026-07-22).
"""

import os

import pytest

_TEST_MODEL = "gemini-3.1-flash-lite"


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not _has_api_key(), reason="GEMINI_API_KEY/GOOGLE_API_KEY not set")
def test_summarizes_real_excerpts_into_non_empty_text() -> None:
    """A real Gemini model returns a non-empty summary covering both excerpts."""
    from ragforge.generation.gemini_summarizer import GeminiSummarizer

    summarizer = GeminiSummarizer(_TEST_MODEL)
    texts = [
        "Art. 1º Esta Resolução estabelece requisitos para a política de "
        "segurança cibernética das instituições financeiras.",
        "Art. 2º As instituições devem adotar controles de segurança da "
        "informação proporcionais ao seu perfil de risco.",
    ]

    summary = summarizer.summarize(texts)

    assert isinstance(summary, str)
    assert len(summary.strip()) > 0
