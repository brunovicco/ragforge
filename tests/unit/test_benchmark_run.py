"""Tests for the pure wiring/formatting logic in the benchmark runner (ADR-0004).

The runner's main() orchestrates real infra (Postgres, OpenSearch, Gemini,
LightRAG) end-to-end and is exercised manually via `make bench-live`, not
here. These tests cover the parts of run.py that don't need real infra:
mode validation, strategy wiring from already-indexed (fake) stores, and the
pure formatting/record-building functions.
"""

from ragforge.domain.models import Chunk, Query, RetrievalResult
from ragforge.evaluation.run import (
    build_base_strategies,
    build_contextual_strategy,
    build_run_record,
    format_results_table,
)
from ragforge.retrieval.reranked.strategy import RerankedRetrieval


def test_reject_cache_mode_raises_for_cache() -> None:
    """--mode cache fails fast with an explanation, not a silent no-op."""
    import pytest

    from ragforge.evaluation.run import _reject_cache_mode

    with pytest.raises(SystemExit, match="cache"):
        _reject_cache_mode("cache")


def test_reject_cache_mode_allows_live() -> None:
    """--mode live passes through without raising."""
    from ragforge.evaluation.run import _reject_cache_mode

    _reject_cache_mode("live")  # must not raise


class _FakeEmbedder:
    name = "fake-embedder"
    dimensions = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class _FakeDenseStore:
    def __init__(self, chunks: dict[str, Chunk]) -> None:
        self._chunks = chunks

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(chunk=chunk, score=1.0, strategy="dense")
            for chunk in list(self._chunks.values())[:top_k]
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)


class _FakeSparseStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def search(self, query_text: str, top_k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(chunk=chunk, score=1.0, strategy="sparse")
            for chunk in self._chunks[:top_k]
        ]


class _FakeReranker:
    name = "fake-reranker"

    def score(self, query: Query, chunks: list[Chunk]) -> list[float]:
        return [1.0 for _ in chunks]


def test_build_base_strategies_returns_all_five_labels() -> None:
    """Every base-index strategy is present, keyed by its benchmark-v01.yaml label."""
    chunk = Chunk(chunk_id="c1", text="Art. 1º", structural_ids=("s1",))
    dense_store = _FakeDenseStore({"c1": chunk})
    sparse_store = _FakeSparseStore([chunk])

    strategies = build_base_strategies(
        dense_store, sparse_store, _FakeEmbedder(), _FakeReranker(), rerank_pool=50
    )

    assert set(strategies) == {"dense", "sparse_bm25", "hybrid_rrf", "reranked", "parent_child"}


def test_build_base_strategies_wires_each_strategy_to_retrieve_correctly() -> None:
    """Each returned strategy actually retrieves through its wired collaborators."""
    chunk = Chunk(chunk_id="c1", text="Art. 1º", structural_ids=("s1",))
    dense_store = _FakeDenseStore({"c1": chunk})
    sparse_store = _FakeSparseStore([chunk])

    strategies = build_base_strategies(
        dense_store, sparse_store, _FakeEmbedder(), _FakeReranker(), rerank_pool=50
    )

    query = Query(text="pergunta")
    for strategy in strategies.values():
        results = strategy.retrieve(query, top_k=5)
        assert results
        assert results[0].chunk.chunk_id == "c1"


def test_build_base_strategies_passes_rerank_pool_through() -> None:
    """The reranked strategy pools from the configured rerank_pool size."""
    chunk = Chunk(chunk_id="c1", text="Art. 1º", structural_ids=("s1",))
    dense_store = _FakeDenseStore({"c1": chunk})
    sparse_store = _FakeSparseStore([chunk])

    strategies = build_base_strategies(
        dense_store, sparse_store, _FakeEmbedder(), _FakeReranker(), rerank_pool=7
    )

    reranked = strategies["reranked"]
    assert isinstance(reranked, RerankedRetrieval)
    assert reranked._pool_size == 7


def test_build_contextual_strategy_retrieves_through_hybrid() -> None:
    """The contextual strategy is a working Hybrid retrieval over the given stores."""
    chunk = Chunk(chunk_id="c1", text="Contexto: Art. 1º", structural_ids=("s1",))
    dense_store = _FakeDenseStore({"c1": chunk})
    sparse_store = _FakeSparseStore([chunk])

    strategy = build_contextual_strategy(dense_store, sparse_store, _FakeEmbedder())

    results = strategy.retrieve(Query(text="pergunta"), top_k=5)
    assert results[0].chunk.chunk_id == "c1"


def test_format_results_table_includes_every_configured_strategy() -> None:
    """Every strategy label from the config appears in the table, in order."""
    metrics = {
        "dense": {"recall_at_k": 0.8, "precision_at_k": 0.2, "ndcg_at_k": 0.7, "mrr": 0.6, "n": 5},
    }

    table = format_results_table(["dense", "sparse_bm25"], metrics)

    lines = table.splitlines()
    assert "dense" in lines[1]
    assert "0.800" in lines[1]
    assert "(not run)" in lines[2]


def test_build_run_record_includes_all_run_metadata() -> None:
    """The run record carries the run id, mode, embedding config, and per-strategy metrics."""
    metrics = {"dense": {"recall_at_k": 0.8, "n": 5.0}}

    record = build_run_record(
        run_id="20260101T000000Z",
        mode="live",
        config_path="configs/experiments/benchmark-v01.yaml",
        embedding_model="gemini-embedding-001",
        embedding_dimensions=1536,
        n_chunks=465,
        top_k=5,
        run_metrics=metrics,
    )

    assert record["run_id"] == "20260101T000000Z"
    assert record["mode"] == "live"
    assert record["embedding"] == {"model": "gemini-embedding-001", "dimensions": 1536}
    assert record["n_chunks"] == 465
    assert record["metrics"] == metrics
