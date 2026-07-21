"""Tests for ParentChildRetrieval's small-to-big expansion (ADR-0006)."""

from ragforge.domain.models import Chunk, Query, RetrievalResult
from ragforge.retrieval.hierarchical.strategy import ParentChildRetrieval

QUERY = Query(text="q")


def _result(chunk: Chunk, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(chunk=chunk, score=score, strategy="dense")


class _FakeInner:
    name = "fake-inner"

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        return self._results[:top_k]


class _FakeStore:
    def __init__(self, chunks_by_id: dict[str, Chunk]) -> None:
        self._chunks_by_id = chunks_by_id

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks_by_id.get(chunk_id)


def test_a_fragment_hit_is_expanded_to_its_parent_article() -> None:
    """A result with parent_id gets replaced by the full parent chunk."""
    article = Chunk(chunk_id="art-2", text="full article text", structural_ids=("art-2",))
    fragment = Chunk(
        chunk_id="art-2::par-1",
        text="§ 1º fragment",
        structural_ids=("art-2", "art-2::par-1"),
        parent_id="art-2",
    )
    inner = _FakeInner([_result(fragment)])
    store = _FakeStore({"art-2": article})
    strategy = ParentChildRetrieval(inner, store)

    [result] = strategy.retrieve(QUERY, top_k=5)

    assert result.chunk.chunk_id == "art-2"
    assert result.chunk.text == "full article text"
    assert strategy.name == "parent-child"


def test_a_result_without_a_parent_id_passes_through_unchanged() -> None:
    """An already article-level hit (no parent_id) is returned as-is."""
    article = Chunk(chunk_id="art-1", text="article text", structural_ids=("art-1",))
    inner = _FakeInner([_result(article)])
    strategy = ParentChildRetrieval(inner, _FakeStore({}))

    [result] = strategy.retrieve(QUERY, top_k=5)

    assert result.chunk is article


def test_two_fragments_of_the_same_parent_collapse_into_one_result() -> None:
    """Expanding two fragments of the same article yields one result, not two."""
    article = Chunk(chunk_id="art-2", text="full article", structural_ids=("art-2",))
    fragment_1 = Chunk(
        chunk_id="art-2::par-1",
        text="p1",
        structural_ids=("art-2", "art-2::par-1"),
        parent_id="art-2",
    )
    fragment_2 = Chunk(
        chunk_id="art-2::par-2",
        text="p2",
        structural_ids=("art-2", "art-2::par-2"),
        parent_id="art-2",
    )
    inner = _FakeInner([_result(fragment_1, score=0.9), _result(fragment_2, score=0.8)])
    store = _FakeStore({"art-2": article})
    strategy = ParentChildRetrieval(inner, store)

    results = strategy.retrieve(QUERY, top_k=5)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "art-2"
    assert results[0].score == 0.9


def test_falls_back_to_the_fragment_when_the_parent_is_not_indexed() -> None:
    """A dangling parent_id (parent not in the store) keeps the original fragment."""
    fragment = Chunk(
        chunk_id="art-2::par-1",
        text="§ 1º fragment",
        structural_ids=("art-2", "art-2::par-1"),
        parent_id="art-2",
    )
    inner = _FakeInner([_result(fragment)])
    strategy = ParentChildRetrieval(inner, _FakeStore({}))

    [result] = strategy.retrieve(QUERY, top_k=5)

    assert result.chunk.chunk_id == "art-2::par-1"
