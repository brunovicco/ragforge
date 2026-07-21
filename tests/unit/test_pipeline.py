"""Tests for the ingest_norm orchestration (ADR-0006)."""

from pathlib import Path

import pytest

from ragforge.chunking.validation import ArticleCountMismatchError
from ragforge.ingestion.html_extractor import HtmlTextExtractor
from ragforge.ingestion.pipeline import ingest_norm
from ragforge.ingestion.snapshot import snapshot_hash

NORM_ID = "RES-CMN-4893/2021"
TEXT = """
Art. 1º Primeiro artigo.

Art. 2º Segundo artigo.
"""

CORPUS_HTML = Path(__file__).resolve().parents[2] / "datasets/corpus/lc-lgpd/LC-105-2001.htm"


def test_ingest_norm_returns_chunks_stamped_with_the_source_hash(tmp_path: Path) -> None:
    """Every returned chunk carries the source file's snapshot hash in its metadata."""
    source = tmp_path / "norm.txt"
    source.write_text(TEXT, encoding="utf-8")

    chunks = ingest_norm(NORM_ID, source, TEXT, expected_article_count=2)

    assert len(chunks) == 2
    expected_hash = snapshot_hash(source)
    assert all(chunk.metadata["source_hash"] == expected_hash for chunk in chunks)


def test_ingest_norm_preserves_existing_chunk_metadata(tmp_path: Path) -> None:
    """Stamping the source hash does not drop the role/norm/section metadata chunk_norm sets."""
    source = tmp_path / "norm.txt"
    source.write_text(TEXT, encoding="utf-8")

    chunks = ingest_norm(NORM_ID, source, TEXT, expected_article_count=2)

    assert all(chunk.metadata["norm"] == NORM_ID for chunk in chunks)
    assert all(chunk.metadata["role"] == "article" for chunk in chunks)


def test_ingest_norm_raises_and_returns_nothing_on_article_count_mismatch(
    tmp_path: Path,
) -> None:
    """A wrong curated count blocks indexing: the gate raises, no chunks are produced."""
    source = tmp_path / "norm.txt"
    source.write_text(TEXT, encoding="utf-8")

    with pytest.raises(ArticleCountMismatchError):
        ingest_norm(NORM_ID, source, TEXT, expected_article_count=3)


def test_ingest_norm_against_a_real_corpus_document() -> None:
    """The full parse-validate-chunk-stamp pipeline works end-to-end on a real norm."""
    text = HtmlTextExtractor().extract(CORPUS_HTML)

    chunks = ingest_norm("LC-105-2001", CORPUS_HTML, text, expected_article_count=13)

    article_chunks = [c for c in chunks if c.metadata.get("role") == "article"]
    assert len(article_chunks) == 13
    assert all("source_hash" in c.metadata for c in chunks)
