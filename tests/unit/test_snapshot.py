"""Tests for corpus snapshot hashing (ADR-0004)."""

import hashlib
from pathlib import Path

import pytest

from ragforge.ingestion.errors import SnapshotError
from ragforge.ingestion.snapshot import snapshot_hash


def test_hash_is_stable_for_the_same_content(tmp_path: Path) -> None:
    """Hashing the same file twice yields the same digest."""
    path = tmp_path / "norm.txt"
    path.write_bytes(b"Art. 1o Texto de exemplo.")

    assert snapshot_hash(path) == snapshot_hash(path)


def test_hash_differs_for_different_content(tmp_path: Path) -> None:
    """Two files with different bytes hash to different digests."""
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_bytes(b"Art. 1o Texto A.")
    second.write_bytes(b"Art. 1o Texto B.")

    assert snapshot_hash(first) != snapshot_hash(second)


def test_hash_matches_independently_computed_sha256(tmp_path: Path) -> None:
    """The digest is exactly SHA-256 of the raw bytes, not some other algorithm."""
    path = tmp_path / "norm.txt"
    content = b"Art. 1o Texto de exemplo com mais bytes para o buffer de leitura."
    path.write_bytes(content)

    assert snapshot_hash(path) == hashlib.sha256(content).hexdigest()


def test_hash_is_correct_across_multiple_read_chunks(tmp_path: Path) -> None:
    """Content larger than one internal read buffer still hashes correctly."""
    path = tmp_path / "large.bin"
    content = b"x" * (3 * (1 << 20) + 7)  # spans multiple 1 MiB read chunks
    path.write_bytes(content)

    assert snapshot_hash(path) == hashlib.sha256(content).hexdigest()


def test_raises_snapshot_error_for_a_missing_file(tmp_path: Path) -> None:
    """A missing path is translated into the ingestion-level SnapshotError."""
    missing = tmp_path / "does-not-exist.pdf"
    with pytest.raises(SnapshotError, match=str(missing)):
        snapshot_hash(missing)
