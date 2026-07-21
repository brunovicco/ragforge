"""Errors raised by the embeddings package (ADR-0005)."""


class EmbeddingError(Exception):
    """Raised when a model fails to load or fails to encode text."""
