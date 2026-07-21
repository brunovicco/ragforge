"""Extract-with-fallback orchestration for norm documents (ADR-0006).

Docling is the primary extractor (layout-aware); PyMuPDF is the fallback for when
Docling fails outright or returns text too short to be the document's real content
(e.g. a scanned/image-only PDF Docling declines to OCR under the caller's config).
"""

from pathlib import Path

from ragforge.ingestion.errors import ExtractionError
from ragforge.ingestion.ports import NormTextExtractor

MIN_EXTRACTED_CHARS = 20


def extract_norm_text(path: Path, primary: NormTextExtractor, fallback: NormTextExtractor) -> str:
    """Extract plain text from ``path``, falling back if the primary extractor fails.

    Args:
        path: Path to the source document.
        primary: The preferred extractor (e.g. Docling).
        fallback: Used when ``primary`` raises ExtractionError or returns text too
            short to plausibly be the document's content.

    Raises:
        ExtractionError: If both extractors fail.
    """
    try:
        text = primary.extract(path)
    except ExtractionError:
        text = ""
    if len(text.strip()) >= MIN_EXTRACTED_CHARS:
        return text
    return fallback.extract(path)
