"""Integration test for the Gemini contextualization adapter.

Opt-in only: excluded from the default `pytest` run by the `-m "not integration"`
addopts, because it makes a real, metered API call. Run explicitly with
`uv run pytest -m integration`, with GEMINI_API_KEY or GOOGLE_API_KEY set.

Model verified against the live API (2026-07-22): gemini-2.5-flash-lite has
been retired ("no longer available to new users"), so this uses
gemini-3.1-flash-lite, confirmed available via client.models.list() and a
real generate_content call at the time this test was written.
"""

import os

import pytest

_TEST_MODEL = "gemini-3.1-flash-lite"


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not _has_api_key(), reason="GEMINI_API_KEY/GOOGLE_API_KEY not set")
def test_generates_a_context_string_for_a_real_chunk() -> None:
    """A real Gemini model returns non-empty text situating the chunk in the document."""
    from ragforge.generation.gemini_contextualizer import GeminiContextualizer

    contextualizer = GeminiContextualizer(_TEST_MODEL)
    document_text = (
        "RESOLUÇÃO CMN Nº 4.893, DE 2021. Dispõe sobre a política de segurança "
        "cibernética. Art. 1º Esta Resolução estabelece requisitos para a "
        "política de segurança cibernética. Art. 2º As instituições devem "
        "adotar controles de segurança da informação."
    )
    chunk_text = "Art. 2º As instituições devem adotar controles de segurança da informação."

    context = contextualizer.contextualize(document_text, chunk_text)

    assert isinstance(context, str)
    assert len(context.strip()) > 0
