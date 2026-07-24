"""Tests for GraphRagRetrieval, using a fake LightRAG query_data (ADR-0010)."""

from typing import Any

from ragforge.domain.models import Chunk, Query
from ragforge.retrieval.graph.strategy import GraphRagRetrieval


class _FakeRag:
    def __init__(self, result: dict[str, object]) -> None:
        self._result = result
        self.query_calls: list[tuple[str, Any]] = []

    def query_data(self, query_text: str, param: Any) -> dict[str, object]:
        self.query_calls.append((query_text, param))
        return self._result


def _success(chunks_content: list[str]) -> dict[str, object]:
    return {"status": "success", "data": {"chunks": [{"content": c} for c in chunks_content]}}


def test_retrieve_maps_returned_content_back_to_indexed_chunks() -> None:
    """Each returned chunk's content is matched back to the real Chunk via the content index."""
    chunk1 = Chunk(
        chunk_id="c1", source_text="Art. 1º", retrieval_text="Art. 1º", structural_ids=("s1",)
    )
    chunk2 = Chunk(
        chunk_id="c2", source_text="Art. 2º", retrieval_text="Art. 2º", structural_ids=("s2",)
    )
    rag = _FakeRag(_success(["Art. 1º", "Art. 2º"]))
    content_index = {"Art. 1º": chunk1, "Art. 2º": chunk2}
    strategy = GraphRagRetrieval(rag, content_index)

    results = strategy.retrieve(Query(text="pergunta"), top_k=5)

    assert [r.chunk.chunk_id for r in results] == ["c1", "c2"]
    assert strategy.name == "graphrag"


def test_retrieve_assigns_descending_rank_based_scores() -> None:
    """Results are scored 1/rank, preserving LightRAG's own return order."""
    chunk1 = Chunk(chunk_id="c1", source_text="a", retrieval_text="a", structural_ids=("s1",))
    chunk2 = Chunk(chunk_id="c2", source_text="b", retrieval_text="b", structural_ids=("s2",))
    rag = _FakeRag(_success(["a", "b"]))
    strategy = GraphRagRetrieval(rag, {"a": chunk1, "b": chunk2})

    results = strategy.retrieve(Query(text="pergunta"), top_k=5)

    assert results[0].score == 1.0
    assert results[1].score == 0.5


def test_retrieve_skips_unmatched_content_without_raising() -> None:
    """A chunk LightRAG returns that isn't in the content index is dropped, not an error."""
    chunk1 = Chunk(chunk_id="c1", source_text="a", retrieval_text="a", structural_ids=("s1",))
    rag = _FakeRag(_success(["a", "content not in our index"]))
    strategy = GraphRagRetrieval(rag, {"a": chunk1})

    results = strategy.retrieve(Query(text="pergunta"), top_k=5)

    assert [r.chunk.chunk_id for r in results] == ["c1"]


def test_retrieve_truncates_to_top_k() -> None:
    """Only the first top_k returned chunks are considered."""
    chunk1 = Chunk(chunk_id="c1", source_text="a", retrieval_text="a", structural_ids=("s1",))
    chunk2 = Chunk(chunk_id="c2", source_text="b", retrieval_text="b", structural_ids=("s2",))
    rag = _FakeRag(_success(["a", "b"]))
    strategy = GraphRagRetrieval(rag, {"a": chunk1, "b": chunk2})

    results = strategy.retrieve(Query(text="pergunta"), top_k=1)

    assert [r.chunk.chunk_id for r in results] == ["c1"]


def test_retrieve_returns_empty_list_on_failure_status() -> None:
    """A failed query_data (e.g. no results found) yields an empty result list, not an error."""
    rag = _FakeRag({"status": "failure", "message": "no results", "data": {}})
    strategy = GraphRagRetrieval(rag, {})

    results = strategy.retrieve(Query(text="pergunta"), top_k=5)

    assert results == []


def test_retrieve_passes_the_configured_mode_to_query_params() -> None:
    """The strategy's mode (local/global/...) is forwarded to LightRAG's QueryParam."""
    rag = _FakeRag(_success([]))
    strategy = GraphRagRetrieval(rag, {}, mode="global")

    strategy.retrieve(Query(text="pergunta"), top_k=5)

    _, param = rag.query_calls[0]
    assert param.mode == "global"
