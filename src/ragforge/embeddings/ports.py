"""Ports the embedding pipeline depends on, defined near their use case (ADR-0001)."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """Encodes text into dense vector embeddings for indexing and querying.

    ``dimensions`` matters beyond introspection: it is the vector column width a
    pgvector index must declare, so it has to be known before indexing starts.
    """

    name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
        ...
