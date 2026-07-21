"""Tests for the extract-with-fallback orchestration (ADR-0006).

Uses fakes for both extractors: Docling is too expensive to invoke in unit tests
(model loading), so the orchestration logic is verified independently of any real
extractor implementation.
"""

from pathlib import Path

import pytest

from ragforge.ingestion.errors import ExtractionError
from ragforge.ingestion.extraction import extract_norm_text

PATH = Path("norm.pdf")


class _FakeExtractor:
    """A NormTextExtractor test double that records calls and returns a fixed result."""

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.calls = 0
        self._text = text
        self._error = error

    def extract(self, path: Path) -> str:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._text


def test_returns_primary_text_without_calling_fallback_when_primary_succeeds() -> None:
    """The fallback extractor is never invoked when the primary result is usable."""
    primary = _FakeExtractor(text="Art. 1º Texto suficientemente longo para passar no limiar.")
    fallback = _FakeExtractor(text="should never be used")

    result = extract_norm_text(PATH, primary, fallback)

    assert result == primary._text
    assert fallback.calls == 0


def test_falls_back_when_primary_raises_extraction_error() -> None:
    """A primary extractor failure triggers the fallback extractor."""
    primary = _FakeExtractor(error=ExtractionError("docling exploded"))
    fallback = _FakeExtractor(text="Art. 1º Texto do fallback.")

    result = extract_norm_text(PATH, primary, fallback)

    assert result == fallback._text
    assert fallback.calls == 1


def test_falls_back_when_primary_text_is_too_short() -> None:
    """A scanned/empty-text result from the primary extractor also triggers fallback."""
    primary = _FakeExtractor(text="   ")
    fallback = _FakeExtractor(text="Art. 1º Texto do fallback com conteúdo real.")

    result = extract_norm_text(PATH, primary, fallback)

    assert result == fallback._text
    assert fallback.calls == 1


def test_raises_when_both_extractors_fail() -> None:
    """If the fallback also fails, its ExtractionError propagates to the caller."""
    primary = _FakeExtractor(error=ExtractionError("docling exploded"))
    fallback = _FakeExtractor(error=ExtractionError("pymupdf exploded too"))

    with pytest.raises(ExtractionError, match="pymupdf exploded too"):
        extract_norm_text(PATH, primary, fallback)
