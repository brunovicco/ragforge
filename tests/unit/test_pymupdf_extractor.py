"""Tests for the PyMuPDF fallback extractor (ADR-0006)."""

from pathlib import Path

import pytest

from ragforge.ingestion.errors import ExtractionError
from ragforge.ingestion.pymupdf_extractor import PyMuPdfExtractor

CORPUS_PDF = Path(__file__).resolve().parents[2] / "datasets/corpus/cvm/ICVM-607-2019.pdf"
LIGATURE_BUG_PDF = (
    Path(__file__).resolve().parents[2] / "datasets/corpus/bacen/RES-CMN-4893-2021.pdf"
)


def test_extracts_text_from_a_real_pdf_in_reading_order() -> None:
    """Extracted text contains the norm's title and preserves page order."""
    text = PyMuPdfExtractor().extract(CORPUS_PDF)
    assert "INSTRUÇÃO CVM Nº 607" in text
    assert text.index("INSTRUÇÃO CVM Nº 607") < text.index("Art. 1")


def test_repairs_the_broken_fi_fl_ligature_mapping_in_a_real_pdf() -> None:
    """RES-CMN-4893/2021's embedded fonts mis-map fi/fl; extraction repairs both.

    Regression test: this PDF's font ToUnicode CMap turns "confidencialidade"
    into "conądencialidade" and "conflito" into "conĆito" unless repaired.
    """
    text = PyMuPdfExtractor().extract(LIGATURE_BUG_PDF)
    assert "confidencialidade" in text
    assert "conflito" in text
    assert "ą" not in text
    assert "Ć" not in text


def test_raises_extraction_error_for_a_missing_file(tmp_path: Path) -> None:
    """A missing path is translated into the ingestion-level ExtractionError."""
    missing = tmp_path / "does-not-exist.pdf"
    with pytest.raises(ExtractionError, match=str(missing)):
        PyMuPdfExtractor().extract(missing)


def test_raises_extraction_error_for_a_corrupt_file(tmp_path: Path) -> None:
    """A file that isn't a real PDF is translated into ExtractionError, not a raw fitz error."""
    garbage = tmp_path / "garbage.pdf"
    garbage.write_bytes(b"not a real pdf file")
    with pytest.raises(ExtractionError, match=str(garbage)):
        PyMuPdfExtractor().extract(garbage)
