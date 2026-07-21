"""Tests for the DenseRetrieval strategy (ADR-0005), using fakes for both collaborators."""

from ragforge.domain.models import Chunk, Query, RetrievalResult
from ragforge.retrieval.dense.strategy import DenseRetrieval


class _FakeEmbedder:
    name = "fake-embedder"
    dimensions = 3

    def __init__(self) -> None:
        self.embedded_texts: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.append(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


class _FakeStore:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.searched_with: tuple[list[float], int] | None = None

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        self.searched_with = (query_embedding, top_k)
        return self._results[:top_k]


def test_retrieve_embeds_the_query_text_and_searches_the_store() -> None:
    """DenseRetrieval embeds exactly the query's text and forwards the vector to the store."""
    chunk = Chunk(chunk_id="c1", text="Art. 1º", structural_ids=("c1",))
    store = _FakeStore([RetrievalResult(chunk=chunk, score=0.9, strategy="dense")])
    embedder = _FakeEmbedder()
    strategy = DenseRetrieval(store, embedder)

    results = strategy.retrieve(Query(text="o que diz o artigo 1?"), top_k=5)

    assert embedder.embedded_texts == [["o que diz o artigo 1?"]]
    assert store.searched_with == ([1.0, 0.0, 0.0], 5)
    assert results[0].chunk.chunk_id == "c1"


def test_retrieve_respects_top_k_from_the_store() -> None:
    """The strategy returns whatever the store returns, capped at top_k."""
    chunks = [
        RetrievalResult(
            chunk=Chunk(chunk_id=f"c{i}", text="x", structural_ids=(f"c{i}",)),
            score=1.0 - i * 0.1,
            strategy="dense",
        )
        for i in range(5)
    ]
    store = _FakeStore(chunks)
    strategy = DenseRetrieval(store, _FakeEmbedder())

    results = strategy.retrieve(Query(text="q"), top_k=2)

    assert len(results) == 2
    assert strategy.name == "dense"
