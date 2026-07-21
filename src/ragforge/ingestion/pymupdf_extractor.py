"""PyMuPDF-based extraction: the fallback path for NormTextExtractor (ADR-0006)."""

from pathlib import Path

import fitz

from ragforge.ingestion.errors import ExtractionError

# Some embedded PDF fonts (confirmed via RES-CMN-4893/2021's subset TrueType
# fonts) have a broken ToUnicode CMap that maps the "fi"/"fl" ligature glyphs
# to unrelated Central-European letters instead of decomposing them, e.g.
# "confidencialidade" extracts as "conądencialidade". These two characters
# never legitimately appear in the project's Portuguese-language corpus, so
# the substitution is safe to apply unconditionally.
_LIGATURE_FIXES = {
    "ą": "fi",
    "Ć": "fl",
}


def _fix_broken_ligatures(text: str) -> str:
    """Repair the known "fi"/"fl" ligature mis-mapping described above."""
    for broken, fixed in _LIGATURE_FIXES.items():
        text = text.replace(broken, fixed)
    return text


class PyMuPdfExtractor:
    """Extracts plain text from a PDF using PyMuPDF, page by page in reading order."""

    def extract(self, path: Path) -> str:
        """Return the concatenated text of every page.

        Raises:
            ExtractionError: If the file is missing or not a readable PDF.
        """
        try:
            with fitz.open(path) as document:
                text = "\n".join(page.get_text() for page in document)
        except RuntimeError as exc:
            raise ExtractionError(f"pymupdf failed to extract {path}: {exc}") from exc
        return _fix_broken_ligatures(text)
