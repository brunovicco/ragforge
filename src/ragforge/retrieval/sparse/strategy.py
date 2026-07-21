"""Sparse BM25 retrieval strategy over OpenSearch (ADR-0005)."""

from ragforge.domain.models import Query, RetrievalResult
from ragforge.retrieval.sparse.ports import TextSearchStore


class SparseRetrieval:
    """Searches the BM25 index directly with the query's text; no embedding step."""

    name = "sparse"

    def __init__(self, store: TextSearchStore) -> None:
        """Wire the strategy to its text search store."""
        self._store = store

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        """Return up to ``top_k`` ranked results for the query."""
        return self._store.search(query.text, top_k)
