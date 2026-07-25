"""Tests for atomic artifact writes and checksums (ADR-0017)."""

from pathlib import Path

from ragforge.evaluation.artifact_writer import (
    compute_checksums,
    write_atomic,
    write_checksums_file,
)
from ragforge.ingestion.snapshot import snapshot_hash


def test_write_atomic_creates_parent_directories(tmp_path: Path) -> None:
    """write_atomic creates any missing parent directories before writing."""
    target = tmp_path / "a" / "b" / "c.json"

    write_atomic(target, "hello")

    assert target.read_text(encoding="utf-8") == "hello"


def test_write_atomic_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    """After a successful write, no .tmp sibling file remains."""
    target = tmp_path / "file.json"

    write_atomic(target, "content")

    assert sorted(p.name for p in tmp_path.iterdir()) == ["file.json"]


def test_write_atomic_overwrites_existing_content(tmp_path: Path) -> None:
    """A second write_atomic call replaces the prior content entirely."""
    target = tmp_path / "file.json"
    write_atomic(target, "first")

    write_atomic(target, "second")

    assert target.read_text(encoding="utf-8") == "second"


def test_compute_checksums_covers_every_file_under_root(tmp_path: Path) -> None:
    """compute_checksums returns one entry per regular file, keyed by relative posix path."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("bbb", encoding="utf-8")

    checksums = compute_checksums(tmp_path)

    assert checksums == {
        "a.txt": snapshot_hash(tmp_path / "a.txt"),
        "sub/b.txt": snapshot_hash(tmp_path / "sub" / "b.txt"),
    }


def test_compute_checksums_excludes_the_checksums_file_itself(tmp_path: Path) -> None:
    """A pre-existing checksums.sha256 under root is never included in its own listing."""
    (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
    (tmp_path / "checksums.sha256").write_text("stale", encoding="utf-8")

    checksums = compute_checksums(tmp_path)

    assert "checksums.sha256" not in checksums
    assert "a.txt" in checksums


def test_write_checksums_file_matches_compute_checksums(tmp_path: Path) -> None:
    """The written checksums.sha256 file's contents exactly reflect compute_checksums' output."""
    (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bbb", encoding="utf-8")

    write_checksums_file(tmp_path)

    lines = (tmp_path / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    parsed = dict(line.split("  ", 1)[::-1] for line in lines)
    expected = compute_checksums(tmp_path)
    assert parsed == expected


def test_write_checksums_file_on_empty_directory_writes_empty_file(tmp_path: Path) -> None:
    """An empty root produces an empty (not missing) checksums.sha256."""
    write_checksums_file(tmp_path)

    assert (tmp_path / "checksums.sha256").read_text(encoding="utf-8") == ""
