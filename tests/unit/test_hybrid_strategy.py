"""Tests for HybridRetrieval's Reciprocal Rank Fusion (ADR-0005)."""

from ragforge.domain.models import Chunk, Query, RetrievalResult
from ragforge.retrieval.hybrid.strategy import HybridRetrieval

QUERY = Query(text="q")


def _result(chunk_id: str, strategy: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=chunk_id, text=chunk_id, structural_ids=(chunk_id,)),
        score=score,
        strategy=strategy,
    )


class _FakeStrategy:
    def __init__(self, name: str, results: list[RetrievalResult]) -> None:
        self.name = name
        self._results = results

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        return self._results[:top_k]


def test_a_chunk_found_by_both_strategies_outranks_one_found_by_only_one() -> None:
    """The classic RRF property: agreement across strategies beats a single rank-1 hit."""
    dense = _FakeStrategy("dense", [_result("agreed", "dense"), _result("dense-only", "dense")])
    sparse = _FakeStrategy(
        "sparse", [_result("sparse-only", "sparse"), _result("agreed", "sparse")]
    )
    hybrid = HybridRetrieval(dense, sparse)

    results = hybrid.retrieve(QUERY, top_k=3)

    assert results[0].chunk.chunk_id == "agreed"


def test_rrf_score_matches_the_formula_for_a_known_rank_and_k() -> None:
    """1 / (k + rank), summed across every ranking a chunk appears in."""
    dense = _FakeStrategy("dense", [_result("only", "dense")])
    sparse = _FakeStrategy("sparse", [])
    hybrid = HybridRetrieval(dense, sparse, rrf_k=60)

    [result] = hybrid.retrieve(QUERY, top_k=1)

    assert result.score == 1.0 / (60 + 1)
    assert result.strategy == "hybrid"


def test_result_truncates_to_top_k() -> None:
    """Fused results are capped at top_k even when more chunks were seen."""
    dense = _FakeStrategy("dense", [_result(f"d{i}", "dense") for i in range(5)])
    sparse = _FakeStrategy("sparse", [])
    hybrid = HybridRetrieval(dense, sparse)

    results = hybrid.retrieve(QUERY, top_k=2)

    assert len(results) == 2


def test_handles_an_empty_ranking_from_one_strategy() -> None:
    """Fusion works even when one underlying strategy returns nothing."""
    dense = _FakeStrategy("dense", [_result("only", "dense")])
    sparse = _FakeStrategy("sparse", [])
    hybrid = HybridRetrieval(dense, sparse)

    results = hybrid.retrieve(QUERY, top_k=5)

    assert [r.chunk.chunk_id for r in results] == ["only"]
