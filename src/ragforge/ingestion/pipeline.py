"""Ingest one norm document into validated, provenance-stamped chunks (ADR-0006).

Ties together the pieces built for this ADR: snapshot hashing (reproducibility,
ADR-0004), structural parsing, the per-norm article-count gate that blocks
indexing on a mismatch, and chunk derivation. Text extraction itself is the
caller's responsibility (see ``ragforge.ingestion.*_extractor`` and
``extract_norm_text``), since the right extractor depends on the source format.
"""

from dataclasses import replace
from pathlib import Path

from ragforge.chunking.chunker import chunk_norm
from ragforge.chunking.legal_parser import parse_norm
from ragforge.chunking.validation import validate_article_count
from ragforge.domain.models import Chunk
from ragforge.ingestion.snapshot import snapshot_hash


def ingest_norm(
    norm_id: str, source_path: Path, text: str, expected_article_count: int
) -> list[Chunk]:
    """Parse, validate, and chunk one norm's already-extracted text.

    Args:
        norm_id: Canonical norm identifier, e.g. ``RES-CMN-4893/2021``.
        source_path: Path to the original source document. Hashed for
            reproducibility and stamped onto every returned chunk's metadata
            under ``source_hash``, so a later re-run can detect whether the
            upstream document changed since these chunks were produced.
        text: The document's already-extracted plain text.
        expected_article_count: The curated article count for this norm.

    Raises:
        ArticleCountMismatchError: If the parsed article count does not match
            ``expected_article_count``. No chunks are returned in that case -
            this is the indexing gate from ADR-0006 decision 5.
        SnapshotError: If ``source_path`` cannot be read.
    """
    digest = snapshot_hash(source_path)
    tree = parse_norm(norm_id, text)
    validate_article_count(tree, expected_article_count)
    chunks = chunk_norm(tree)
    return [replace(chunk, metadata={**chunk.metadata, "source_hash": digest}) for chunk in chunks]
