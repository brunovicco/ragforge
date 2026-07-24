"""Tests for the SparseRetrieval strategy (ADR-0005), using a fake store."""

from ragforge.domain.models import Chunk, Query, RetrievalResult
from ragforge.retrieval.sparse.strategy import SparseRetrieval


class _FakeStore:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.searched_with: tuple[str, int] | None = None

    def search(self, query_text: str, top_k: int) -> list[RetrievalResult]:
        self.searched_with = (query_text, top_k)
        return self._results[:top_k]


def test_retrieve_searches_with_the_query_text_directly() -> None:
    """SparseRetrieval passes the query's raw text to the store - no embedding step."""
    chunk = Chunk(
        chunk_id="c1", source_text="Art. 1º", retrieval_text="Art. 1º", structural_ids=("c1",)
    )
    store = _FakeStore([RetrievalResult(chunk=chunk, score=3.2, strategy="sparse")])
    strategy = SparseRetrieval(store)

    results = strategy.retrieve(Query(text="sigilo bancário"), top_k=5)

    assert store.searched_with == ("sigilo bancário", 5)
    assert results[0].chunk.chunk_id == "c1"
    assert strategy.name == "sparse"


def test_retrieve_respects_top_k_from_the_store() -> None:
    """The strategy returns whatever the store returns, capped at top_k."""
    chunks = [
        RetrievalResult(
            chunk=Chunk(
                chunk_id=f"c{i}", source_text="x", retrieval_text="x", structural_ids=(f"c{i}",)
            ),
            score=10.0 - i,
            strategy="sparse",
        )
        for i in range(5)
    ]
    store = _FakeStore(chunks)
    strategy = SparseRetrieval(store)

    results = strategy.retrieve(Query(text="q"), top_k=2)

    assert len(results) == 2
