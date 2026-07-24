"""Integration test for the cross-encoder reranking adapter.

Opt-in only: excluded from the default `pytest` run by the `-m "not integration"`
addopts, because loading a real model is slow (~model download on first use).
Run explicitly with `uv run pytest -m integration`.

Uses a small, widely-cached test model (ms-marco-MiniLM-L-6-v2), not a
production PT-BR candidate - this test proves the adapter's wiring to
sentence-transformers' CrossEncoder works, not which model wins any future
reranking comparison.
"""

import pytest

_TEST_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.mark.integration
def test_scores_relevant_text_higher_than_irrelevant_text() -> None:
    """A real cross-encoder ranks an on-topic chunk above an off-topic one."""
    from ragforge.domain.models import Chunk, Query
    from ragforge.reranking.cross_encoder_reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(_TEST_MODEL)
    query = Query(text="What is the capital of France?")
    relevant = Chunk(
        chunk_id="relevant",
        source_text="Paris is the capital of France.",
        retrieval_text="Paris is the capital of France.",
        structural_ids=("relevant",),
    )
    irrelevant = Chunk(
        chunk_id="irrelevant",
        source_text="Bananas are a good source of potassium.",
        retrieval_text="Bananas are a good source of potassium.",
        structural_ids=("irrelevant",),
    )

    scores = reranker.score(query, [irrelevant, relevant])

    assert len(scores) == 2
    assert scores[1] > scores[0]
