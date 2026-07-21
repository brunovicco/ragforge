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


def test_collapses_a_source_line_break_between_marker_and_body(tmp_path: Path) -> None:
    """A hand-wrapped newline right after "Art. N" (seen in Lei 6.385/1976) is collapsed.

    Some norm pages break the source line right after the article's ordinal marker,
    before its body text, e.g. ``Art. 1<sup><u>o</u></sup>\\nCorpo do artigo.``. If
    that newline survived, `chunking.legal_parser` would see "Art. 1o" alone on its
    own line (no trailing delimiter to match) and silently drop the article.
    """
    page = tmp_path / "norm.htm"
    page.write_text(
        '<p><a name="art1"></a>Art. 1<sup><u>o</u></sup>\nCorpo do artigo continua aqui.</p>',
        encoding="utf-8",
    )

    text = HtmlTextExtractor().extract(page)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert any(line.startswith("Art. 1") and "Corpo do artigo" in line for line in lines)
