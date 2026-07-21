"""Ports the ingestion pipeline depends on, defined near their use case (ADR-0001)."""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class NormTextExtractor(Protocol):
    """Converts one norm document into plain text carrying its legal structure."""

    def extract(self, path: Path) -> str:
        """Return the extracted plain text of the document at ``path``.

        Raises:
            ExtractionError: If the document cannot be converted.
        """
        ...
