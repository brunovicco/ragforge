"""Ports the contextual retrieval pipeline depends on, defined near its use case (ADR-0001)."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Contextualizer(Protocol):
    """Generates a short context situating a chunk within its source document."""

    name: str

    def contextualize(self, document_text: str, chunk_text: str) -> str:
        """Return a short context string for ``chunk_text``, given the full ``document_text``."""
        ...
