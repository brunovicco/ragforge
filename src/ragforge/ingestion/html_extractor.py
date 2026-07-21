"""Stdlib HTML-to-text extraction for norm pages (ADR-0006).

Docling and PyMuPDF both mis-detect the character encoding of older Brazilian
government HTML pages (many omit an explicit ``<meta charset>``), corrupting
Portuguese diacritics into mojibake. This extractor decodes the raw bytes with an
explicit, ordered encoding guess instead of delegating to either library's HTML
backend, and reinserts line breaks at block-level tags so that each article,
paragraph, and inciso (each their own ``<p>`` in these documents) lands on its own
line for `chunking.legal_parser`.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

from ragforge.ingestion.errors import ExtractionError

_WHITESPACE_RE = re.compile(r"\s+")
_ENCODINGS = ("utf-8", "cp1252")
_SKIP_TAGS = frozenset({"script", "style"})
_BLOCK_TAGS = frozenset({"p", "div", "br", "li", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6"})


class _TextCollector(HTMLParser):
    """Collects text data, breaking lines at block-level tag boundaries."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        """Track skip-tag nesting and insert a line break before block-level tags."""
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Close skip-tag nesting and insert a line break after block-level tags."""
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        """Collect text data, dropping content inside script/style tags.

        Collapses internal whitespace (including source line-wraps) to a single
        space, the way a browser would render it: line breaks in the output come
        only from the explicit block-tag markers above, not from incidental
        newlines inside the source HTML's own hand-wrapped text nodes (some norm
        pages break a line right after "Art. N", before its body text).
        """
        if not self._skip_depth:
            self._chunks.append(_WHITESPACE_RE.sub(" ", data))

    def text(self) -> str:
        """Return the collected text."""
        return "".join(self._chunks)


def _decode(raw: bytes) -> str:
    """Decode HTML bytes, preferring UTF-8 and falling back to Windows-1252.

    Windows-1252 (a superset of ISO-8859-1) is the practical default for the
    undeclared-charset Brazilian government pages this extractor targets, and it
    never raises: every byte value has a defined mapping.
    """
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp1252")


class HtmlTextExtractor:
    """Extracts plain text from an HTML norm page, decoding bytes explicitly."""

    def extract(self, path: Path) -> str:
        """Return the page's plain text, one block element per line.

        Raises:
            ExtractionError: If the file cannot be read.
        """
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise ExtractionError(f"failed to read {path}: {exc}") from exc
        collector = _TextCollector()
        collector.feed(_decode(raw))
        return collector.text()
