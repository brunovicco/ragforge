"""Safe completion markers for reusable benchmark indexes."""

import json
from dataclasses import dataclass
from pathlib import Path

from ragforge.domain.models import Chunk
from ragforge.evaluation.artifact_writer import write_atomic
from ragforge.evaluation.canonical_hash import canonical_json_hash

_SCHEMA_VERSION = 1


def index_fingerprint(
    *,
    stage: str,
    index_namespace: str,
    chunks: list[Chunk],
    derivation_identity: str,
) -> str:
    """Hash every input that determines one dense/sparse index's contents."""
    return canonical_json_hash(
        {
            "schema_version": _SCHEMA_VERSION,
            "stage": stage,
            "index_namespace": index_namespace,
            "derivation_identity": derivation_identity,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_text": chunk.source_text,
                    "retrieval_text": chunk.retrieval_text,
                    "structural_ids": chunk.structural_ids,
                    "parent_id": chunk.parent_id,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class IndexCompletion:
    """One reusable index's validated completion identity."""

    schema_version: int
    stage: str
    fingerprint: str
    chunk_count: int
    dense: bool
    sparse: bool


class FileIndexRegistry:
    """Atomically persist completion markers outside immutable run evidence."""

    def __init__(self, root: Path) -> None:
        """Create a registry rooted at a repository-local cache directory."""
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, stage: str) -> Path:
        return self._root / f"{stage}.json"

    def load(self, stage: str) -> IndexCompletion | None:
        """Load a completion marker, or return None when none exists."""
        path = self._path(stage)
        if not path.exists():
            return None
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"index marker {path} must be a JSON object")
        return IndexCompletion(
            schema_version=int(payload["schema_version"]),
            stage=str(payload["stage"]),
            fingerprint=str(payload["fingerprint"]),
            chunk_count=int(payload["chunk_count"]),
            dense=bool(payload["dense"]),
            sparse=bool(payload["sparse"]),
        )

    def matches(
        self,
        *,
        stage: str,
        fingerprint: str,
        chunk_count: int,
        dense: bool,
        sparse: bool,
    ) -> bool:
        """Return whether the persisted marker exactly matches expected content."""
        marker = self.load(stage)
        return marker == IndexCompletion(
            schema_version=_SCHEMA_VERSION,
            stage=stage,
            fingerprint=fingerprint,
            chunk_count=chunk_count,
            dense=dense,
            sparse=sparse,
        )

    def mark_complete(
        self,
        *,
        stage: str,
        fingerprint: str,
        chunk_count: int,
        dense: bool,
        sparse: bool,
    ) -> None:
        """Atomically publish a marker only after every requested index is complete."""
        completion = IndexCompletion(
            schema_version=_SCHEMA_VERSION,
            stage=stage,
            fingerprint=fingerprint,
            chunk_count=chunk_count,
            dense=dense,
            sparse=sparse,
        )
        write_atomic(
            self._path(stage),
            json.dumps(
                {
                    "schema_version": completion.schema_version,
                    "stage": completion.stage,
                    "fingerprint": completion.fingerprint,
                    "chunk_count": completion.chunk_count,
                    "dense": completion.dense,
                    "sparse": completion.sparse,
                },
                indent=2,
            ),
        )

    def invalidate(self, stage: str) -> None:
        """Remove a stale marker before rebuilding its external indexes."""
        self._path(stage).unlink(missing_ok=True)
