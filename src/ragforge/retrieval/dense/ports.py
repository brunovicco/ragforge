"""Ports the dense retrieval strategy depends on, defined near their use case (ADR-0001)."""

from typing import Protocol, runtime_checkable

from ragforge.domain.models import RetrievalResult


@runtime_checkable
class ChunkSearchStore(Protocol):
    """Searches an indexed store of chunk embeddings by vector similarity."""

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """Return the top_k chunks most similar to query_embedding."""
        ...
