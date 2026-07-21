"""Ports the sparse retrieval strategy depends on, defined near their use case (ADR-0001)."""

from typing import Protocol, runtime_checkable

from ragforge.domain.models import RetrievalResult


@runtime_checkable
class TextSearchStore(Protocol):
    """Searches an indexed store of chunks by BM25 text relevance."""

    def search(self, query_text: str, top_k: int) -> list[RetrievalResult]:
        """Return the top_k chunks most relevant to query_text."""
        ...
