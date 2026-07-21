"""Docling-based extraction: the primary path for NormTextExtractor (ADR-0006).

Docling loads layout-analysis models on first conversion (multi-second import and
first-call cost). This adapter is deliberately not exercised by the unit test suite,
which must stay network-free and fast; verify it manually or via an opt-in
integration check against a real corpus document (see ``datasets/corpus/``).
"""

from pathlib import Path

from docling.document_converter import DocumentConverter

from ragforge.ingestion.errors import ExtractionError


class DoclingExtractor:
    """Extracts plain text from a document using Docling's layout-aware conversion."""

    def __init__(self) -> None:
        """Build the underlying converter once, so model loading is paid only here."""
        self._converter = DocumentConverter()

    def extract(self, path: Path) -> str:
        """Return the document's text content, exported in reading order.

        Raises:
            ExtractionError: If Docling cannot convert the document.
        """
        try:
            result = self._converter.convert(path)
        except Exception as exc:  # Docling's backends raise their own varied exceptions
            raise ExtractionError(f"docling failed to extract {path}: {exc}") from exc
        return result.document.export_to_text()
