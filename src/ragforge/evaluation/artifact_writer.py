"""Atomic artifact writes and checksums for a run's evidence directory (ADR-0017).

Every artifact under ``artifacts/runs/<run_id>/`` is written temp-file-then-
rename, never in place - the exact same atomicity pattern
``adapters/llm_cache.FileLLMCache.put`` already established for cache
entries (ADR-0004), reused here instead of duplicated. Checksums are
computed only after every file is closed, so ``checksums.sha256`` never
covers a partially-written file.
"""

from pathlib import Path

from ragforge.ingestion.snapshot import snapshot_hash

_CHECKSUMS_FILENAME = "checksums.sha256"


def write_atomic(path: Path, content: str) -> None:
    """Write ``content`` to ``path``: temp file then atomic rename, never a partial file.

    Creates ``path``'s parent directories if they don't already exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def compute_checksums(root: Path) -> dict[str, str]:
    """Return ``{relative_posix_path: sha256_hex}`` for every regular file under ``root``.

    The checksums file itself (if already present from a prior partial
    write) is excluded - checksums.sha256 never lists its own hash.
    """
    checksums = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == _CHECKSUMS_FILENAME:
            continue
        checksums[relative] = snapshot_hash(path)
    return checksums


def write_checksums_file(root: Path) -> None:
    """Write ``root/checksums.sha256`` in standard ``sha256sum`` format (verifiable externally).

    Every existing file under ``root`` at call time is included - callers
    should call this only after every other artifact for the run has been
    written.
    """
    checksums = compute_checksums(root)
    lines = [f"{digest}  {relative}" for relative, digest in sorted(checksums.items())]
    write_atomic(root / _CHECKSUMS_FILENAME, "\n".join(lines) + "\n" if lines else "")
