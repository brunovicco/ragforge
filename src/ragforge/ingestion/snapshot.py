"""Content-addressed snapshot hashing for corpus reproducibility (ADR-0004).

A stable hash of a norm's raw source bytes lets the reproducibility pipeline
detect when an upstream document changed since the last ingestion run, so that
derived chunks, embeddings, and indexes can be invalidated deterministically
instead of silently drifting from what was actually indexed.
"""

import hashlib
from pathlib import Path

from ragforge.ingestion.errors import SnapshotError

_READ_CHUNK_SIZE = 1 << 20  # 1 MiB


def snapshot_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of the raw bytes at ``path``.

    Raises:
        SnapshotError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(_READ_CHUNK_SIZE), b""):
                digest.update(block)
    except OSError as exc:
        raise SnapshotError(f"failed to hash {path}: {exc}") from exc
    return digest.hexdigest()
