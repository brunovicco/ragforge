"""Errors raised by the corpus ingestion pipeline (ADR-0006)."""


class ExtractionError(Exception):
    """Raised when a document could not be converted to plain text."""


class SnapshotError(Exception):
    """Raised when a document's content could not be hashed for reproducibility."""
