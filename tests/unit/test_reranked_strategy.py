"""Tests for the RerankedRetrieval strategy, using fakes for both collaborators."""

from ragforge.domain.models import Chunk, Query, RetrievalResult
from ragforge.retrieval.reranked.strategy import RerankedRetrieval


class _FakeBase:
    name = "fake-base"

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.retrieved_with: tuple[str, int] | None = None

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        self.retrieved_with = (query.text, top_k)
        return self._results[:top_k]


class _FakeReranker:
    name = "fake-reranker"

    def __init__(self, scores_by_chunk_id: dict[str, float]) -> None:
        self._scores_by_chunk_id = scores_by_chunk_id
        self.scored_chunks: list[Chunk] | None = None

    def score(self, query: Query, chunks: list[Chunk]) -> list[float]:
        self.scored_chunks = chunks
        return [self._scores_by_chunk_id[chunk.chunk_id] for chunk in chunks]


def _result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=chunk_id, text=chunk_id, structural_ids=(chunk_id,)),
        score=1.0,
        strategy="fake-base",
    )


def test_retrieve_pulls_the_configured_pool_size_from_the_base_strategy() -> None:
    """The strategy asks the base retriever for pool_size candidates, not top_k."""
    base = _FakeBase([_result("c1"), _result("c2")])
    reranker = _FakeReranker({"c1": 0.5, "c2": 0.9})
    strategy = RerankedRetrieval(base, reranker, pool_size=50)

    strategy.retrieve(Query(text="pergunta"), top_k=5)

    assert base.retrieved_with == ("pergunta", 50)


def test_retrieve_widens_the_pool_when_top_k_exceeds_pool_size() -> None:
    """A larger top_k than the configured pool never shrinks the candidate pool."""
    base = _FakeBase([_result("c1")])
    reranker = _FakeReranker({"c1": 1.0})
    strategy = RerankedRetrieval(base, reranker, pool_size=10)

    strategy.retrieve(Query(text="pergunta"), top_k=20)

    assert base.retrieved_with == ("pergunta", 20)


def test_retrieve_reorders_candidates_by_reranker_score() -> None:
    """Results come back ordered by the reranker's score, not the base strategy's order."""
    base = _FakeBase([_result("low"), _result("high"), _result("mid")])
    reranker = _FakeReranker({"low": 0.1, "high": 0.9, "mid": 0.5})
    strategy = RerankedRetrieval(base, reranker)

    results = strategy.retrieve(Query(text="pergunta"), top_k=3)

    assert [r.chunk.chunk_id for r in results] == ["high", "mid", "low"]
    assert [r.score for r in results] == [0.9, 0.5, 0.1]


def test_retrieve_truncates_to_top_k_after_reranking() -> None:
    """Only the top_k highest-scored candidates survive, even with a wider pool."""
    base = _FakeBase([_result("a"), _result("b"), _result("c")])
    reranker = _FakeReranker({"a": 0.2, "b": 0.8, "c": 0.5})
    strategy = RerankedRetrieval(base, reranker, pool_size=3)

    results = strategy.retrieve(Query(text="pergunta"), top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["b", "c"]


def test_retrieve_labels_results_with_the_reranked_strategy_name() -> None:
    """Reranked results carry this strategy's name, not the base strategy's."""
    base = _FakeBase([_result("c1")])
    reranker = _FakeReranker({"c1": 1.0})
    strategy = RerankedRetrieval(base, reranker)

    results = strategy.retrieve(Query(text="pergunta"), top_k=1)

    assert strategy.name == "reranked"
    assert results[0].strategy == "reranked"
