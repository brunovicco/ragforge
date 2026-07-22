"""Ports the reranked retrieval strategy depends on, defined near its use case (ADR-0001)."""

from typing import Protocol, runtime_checkable

from ragforge.domain.models import Chunk, Query


@runtime_checkable
class Reranker(Protocol):
    """Scores how relevant each chunk is to a query, for reordering a candidate pool."""

    name: str

    def score(self, query: Query, chunks: list[Chunk]) -> list[float]:
        """Return one relevance score per chunk, same order, higher meaning more relevant."""
        ...
