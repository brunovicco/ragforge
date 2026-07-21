"""PyMuPDF-based extraction: the fallback path for NormTextExtractor (ADR-0006)."""

from pathlib import Path

import fitz

from ragforge.ingestion.errors import ExtractionError


class PyMuPdfExtractor:
    """Extracts plain text from a PDF using PyMuPDF, page by page in reading order."""

    def extract(self, path: Path) -> str:
        """Return the concatenated text of every page.

        Raises:
            ExtractionError: If the file is missing or not a readable PDF.
        """
        try:
            with fitz.open(path) as document:
                return "\n".join(page.get_text() for page in document)
        except RuntimeError as exc:
            raise ExtractionError(f"pymupdf failed to extract {path}: {exc}") from exc
