"""Tests for safe reusable-index completion markers."""

from pathlib import Path

from ragforge.domain.models import Chunk
from ragforge.evaluation.index_cache import FileIndexRegistry, index_fingerprint


def _chunk(text: str = "text") -> Chunk:
    return Chunk(
        chunk_id="c1",
        source_text=text,
        retrieval_text=text,
        structural_ids=("N::art-1",),
    )


def test_index_fingerprint_changes_with_retrieval_text_or_derivation() -> None:
    """Synthetic text and producing model both participate in cache identity."""
    base = index_fingerprint(
        stage="contextual",
        index_namespace="namespace",
        chunks=[_chunk("one")],
        derivation_identity="model-a",
    )
    changed_text = index_fingerprint(
        stage="contextual",
        index_namespace="namespace",
        chunks=[_chunk("two")],
        derivation_identity="model-a",
    )
    changed_model = index_fingerprint(
        stage="contextual",
        index_namespace="namespace",
        chunks=[_chunk("one")],
        derivation_identity="model-b",
    )

    assert len({base, changed_text, changed_model}) == 3


def test_registry_matches_only_an_exact_completed_index(tmp_path: Path) -> None:
    """A marker is reusable only with the same fingerprint, count, and store set."""
    registry = FileIndexRegistry(tmp_path)
    registry.mark_complete(
        stage="base",
        fingerprint="fingerprint",
        chunk_count=10,
        dense=True,
        sparse=True,
    )

    assert registry.matches(
        stage="base",
        fingerprint="fingerprint",
        chunk_count=10,
        dense=True,
        sparse=True,
    )
    assert not registry.matches(
        stage="base",
        fingerprint="different",
        chunk_count=10,
        dense=True,
        sparse=True,
    )


def test_registry_invalidate_removes_completion(tmp_path: Path) -> None:
    """A rebuild invalidates the old marker before touching external state."""
    registry = FileIndexRegistry(tmp_path)
    registry.mark_complete(
        stage="base",
        fingerprint="fingerprint",
        chunk_count=1,
        dense=True,
        sparse=False,
    )

    registry.invalidate("base")

    assert registry.load("base") is None
