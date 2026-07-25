"""Tests for the cross-process benchmark lock."""

from pathlib import Path

import pytest

from ragforge.evaluation.run_lock import BenchmarkAlreadyRunningError, BenchmarkRunLock


def test_benchmark_lock_rejects_a_second_owner(tmp_path: Path) -> None:
    """Only one benchmark can mutate shared indexes in a repository."""
    first = BenchmarkRunLock(tmp_path / "benchmark.lock")
    second = BenchmarkRunLock(tmp_path / "benchmark.lock")

    with first, pytest.raises(BenchmarkAlreadyRunningError):
        second.acquire()


def test_benchmark_lock_can_be_reacquired_after_release(tmp_path: Path) -> None:
    """A completed process leaves the lock reusable."""
    path = tmp_path / "benchmark.lock"

    with BenchmarkRunLock(path):
        pass

    with BenchmarkRunLock(path):
        pass
