"""Ports the hierarchical retrieval strategy depends on, defined near their use case (ADR-0001)."""

from typing import Protocol, runtime_checkable

from ragforge.domain.models import Chunk


@runtime_checkable
class ChunkFetchStore(Protocol):
    """Fetches a single indexed chunk by its id."""

    def get(self, chunk_id: str) -> Chunk | None:
        """Return the chunk with chunk_id, or None if it isn't indexed."""
        ...
