"""Integration test for the Gemini embedding adapter (ADR-0005): real, metered API calls.

Opt-in only (`pytest -m integration`). Needs GEMINI_API_KEY or GOOGLE_API_KEY in
the environment; skipped (not failed) when absent, since this credential is the
project's one paid-API dependency and most environments won't have it configured.
Run with: GEMINI_API_KEY=... uv run pytest -m integration -k google_gemini
"""

import os

import pytest

from ragforge.embeddings.google_gemini_embedder import GoogleGeminiEmbedder

pytestmark = pytest.mark.integration


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


@pytest.mark.skipif(not _has_api_key(), reason="GEMINI_API_KEY/GOOGLE_API_KEY not set")
def test_embeds_real_text_with_the_real_gemini_api() -> None:
    """A real Gemini embedding call returns a vector matching the model's own declared dimension."""
    embedder = GoogleGeminiEmbedder("gemini-embedding-001")

    vectors = embedder.embed(["Art. 1º Esta Resolução dispõe sobre o objeto."])

    assert len(vectors) == 1
    assert len(vectors[0]) == embedder.dimensions
    assert embedder.dimensions > 0
