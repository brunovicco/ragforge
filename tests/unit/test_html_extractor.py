"""Tests for the stdlib HTML extractor (ADR-0006)."""

from pathlib import Path

import pytest

from ragforge.ingestion.errors import ExtractionError
from ragforge.ingestion.html_extractor import HtmlTextExtractor

CORPUS_HTML = Path(__file__).resolve().parents[2] / "datasets/corpus/lc-lgpd/LC-105-2001.htm"


def test_extracts_correct_diacritics_from_a_real_undeclared_charset_page() -> None:
    """Windows-1252 fallback recovers Portuguese diacritics that UTF-8 would mangle."""
    text = HtmlTextExtractor().extract(CORPUS_HTML)
    assert "Dispõe" in text
    assert "instituições financeiras" in text
    assert "�" not in text


def test_each_article_lands_on_its_own_line() -> None:
    """Block-level tags (each article's own <p>) become line breaks the parser needs."""
    text = HtmlTextExtractor().extract(CORPUS_HTML)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert any(line.startswith("Art. 1") for line in lines)
    assert any(line.startswith("Art. 13") for line in lines)


def test_script_and_style_content_is_dropped() -> None:
    """Non-visible script/style text never leaks into the extracted plain text."""
    text = HtmlTextExtractor().extract(CORPUS_HTML)
    assert "function(" not in text


def test_raises_extraction_error_for_a_missing_file(tmp_path: Path) -> None:
    """A missing path is translated into the ingestion-level ExtractionError."""
    missing = tmp_path / "does-not-exist.htm"
    with pytest.raises(ExtractionError, match=str(missing)):
        HtmlTextExtractor().extract(missing)
