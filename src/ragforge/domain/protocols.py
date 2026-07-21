"""Strategy protocols. LLMs and SDKs live at the edges; the core depends only on these."""

from typing import Protocol, runtime_checkable

from ragforge.domain.models import Query, RetrievalResult


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Contract implemented by every benchmarked retrieval strategy."""

    name: str

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        """Return up to ``top_k`` ranked results for the query."""
        ...
