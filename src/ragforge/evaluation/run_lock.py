"""Cross-process lock preventing concurrent mutation of shared benchmark indexes."""

import fcntl
import os
from pathlib import Path
from types import TracebackType
from typing import TextIO


class BenchmarkAlreadyRunningError(RuntimeError):
    """Raised when another benchmark process owns the repository lock."""


class BenchmarkRunLock:
    """Hold an advisory repository lock for the complete benchmark process."""

    def __init__(self, path: Path) -> None:
        """Bind the lock to ``path`` without acquiring it yet."""
        self._path = path
        self._handle: TextIO | None = None

    def acquire(self) -> None:
        """Acquire the lock without waiting.

        Raises:
            BenchmarkAlreadyRunningError: If another process owns the lock.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise BenchmarkAlreadyRunningError(
                "another benchmark is already running for this repository"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        """Release the lock when currently held."""
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "BenchmarkRunLock":
        """Acquire and return this lock."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Always release the lock."""
        self.release()
