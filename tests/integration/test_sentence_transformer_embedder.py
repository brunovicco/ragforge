"""Integration test for the sentence-transformers embedding adapter (ADR-0005).

Opt-in only: excluded from the default `pytest` run by the `-m "not integration"`
addopts, because loading a real model is slow (~17s import, network on first
download). Run explicitly with `uv run pytest -m integration`.

Uses a small, widely-cached test model (all-MiniLM-L6-v2), not one of the
BGE-M3/Qwen3-Embedding candidates ADR-0005 actually compares - this test proves
the adapter's wiring to sentence-transformers works, not which production model
wins the comparison.
"""

import pytest

_TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.integration
def test_embeds_real_text_with_a_real_model() -> None:
    """A real sentence-transformers model loads and encodes to the declared dimension."""
    from ragforge.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(_TEST_MODEL)

    vectors = embedder.embed(["Art. 1º Esta Resolução dispõe sobre o objeto."])

    assert len(vectors) == 1
    assert len(vectors[0]) == embedder.dimensions
    assert embedder.dimensions > 0
