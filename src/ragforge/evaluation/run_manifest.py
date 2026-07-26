"""Run manifest lifecycle for a run's evidence directory (ADR-0017).

The manifest starts ``status="running"`` the moment a run begins and only
becomes ``status="completed"`` after every other artifact is written and
checksummed - a reviewer who finds a "running" manifest for an old run_id
knows the run crashed before finishing, rather than mistaking a partial
evidence directory for a validated one. A completed directory is never
overwritten: reusing a run_id whose manifest already says "completed" fails
closed, the same fail-closed posture ``run.py``'s ``--resume`` identity
check already uses for a different kind of mismatch.
"""

import json
import shutil
import subprocess  # nosec B404
from datetime import UTC, datetime
from pathlib import Path

from ragforge.evaluation.lineage_ports import RunManifest

_SCHEMA_VERSION = 1
_UNKNOWN_GIT_SHA = "unknown"
_GENERATED_OUTPUT_PREFIXES = ("artifacts/", "experiments/", ".ragforge/")


def resolve_git_sha(repository_root: Path | None = None) -> str:
    """Return the current commit's full SHA, or "unknown" if it can't be determined.

    Never raises: git absent, not a repository, or any other failure all
    fall back to the same sentinel - this is manifest metadata, not a
    correctness gate a run should fail over.
    """
    git_path = shutil.which("git")
    if git_path is None:
        return _UNKNOWN_GIT_SHA
    try:
        # git_path resolved via shutil.which; args are fixed literals, no
        # user input reaches this command.
        result = subprocess.run(  # noqa: S603  # nosec B603
            [git_path, "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return _UNKNOWN_GIT_SHA
    sha = result.stdout.strip()
    return sha if sha else _UNKNOWN_GIT_SHA


def require_clean_worktree(repository_root: Path) -> None:
    """Fail closed when reproducibility-relevant Git content is not committed.

    Untracked benchmark output directories are ignored because creating and
    resuming runs necessarily populates them. Tracked modifications under
    those directories still fail the gate.

    Raises:
        SystemExit: If Git is unavailable, status cannot be read, or relevant
            staged, unstaged, or untracked changes exist.
    """
    git_path = shutil.which("git")
    if git_path is None:
        raise SystemExit("auditable benchmark requires Git, but git was not found")
    try:
        result = subprocess.run(  # noqa: S603  # nosec B603
            [git_path, "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"failed to inspect Git worktree: {exc}") from exc

    dirty_entries = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        path = line[3:] if len(line) > 3 else ""
        is_generated_output = line.startswith("?? ") and path.startswith(_GENERATED_OUTPUT_PREFIXES)
        if not is_generated_output:
            dirty_entries.append(line)
    if dirty_entries:
        preview = "\n".join(f"- {entry}" for entry in dirty_entries[:20])
        raise SystemExit(
            "auditable benchmark requires a clean Git worktree; commit or discard "
            f"these changes before running:\n{preview}"
        )


def load_run_manifest(path: Path) -> RunManifest:
    """Load and validate this project's own persisted manifest contract."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"run manifest {path} must contain a JSON object")

    def _required_str(field: str) -> str:
        value = raw.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"run manifest field {field!r} must be a non-empty string")
        return value

    schema_version = raw.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("run manifest field 'schema_version' must be an integer")
    completed_at_raw = raw.get("completed_at")
    if completed_at_raw is not None and not isinstance(completed_at_raw, str):
        raise ValueError("run manifest field 'completed_at' must be a string or null")
    artifact_root_hash_raw = raw.get("artifact_root_hash")
    if artifact_root_hash_raw is not None and not isinstance(artifact_root_hash_raw, str):
        raise ValueError("run manifest field 'artifact_root_hash' must be a string or null")

    models_raw = raw.get("models")
    if not isinstance(models_raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in models_raw.items()
    ):
        raise ValueError("run manifest field 'models' must map strings to strings")
    strategies_raw = raw.get("strategies")
    if not isinstance(strategies_raw, list) or any(
        not isinstance(value, str) for value in strategies_raw
    ):
        raise ValueError("run manifest field 'strategies' must be a list of strings")
    execution_raw = raw.get("execution")
    if not isinstance(execution_raw, dict) or any(
        not isinstance(key, str) for key in execution_raw
    ):
        raise ValueError("run manifest field 'execution' must be an object")

    return RunManifest(
        schema_version=schema_version,
        run_id=_required_str("run_id"),
        status=_required_str("status"),
        git_sha=_required_str("git_sha"),
        started_at=_required_str("started_at"),
        completed_at=completed_at_raw,
        corpus_hash=_required_str("corpus_hash"),
        dataset_hash=_required_str("dataset_hash"),
        split_hash=_required_str("split_hash"),
        configuration_hash=_required_str("configuration_hash"),
        models={str(key): str(value) for key, value in models_raw.items()},
        strategies=tuple(str(value) for value in strategies_raw),
        execution={str(key): value for key, value in execution_raw.items()},
        artifact_root_hash=artifact_root_hash_raw,
    )


def build_initial_manifest(
    *,
    run_id: str,
    git_sha: str,
    corpus_hash: str,
    dataset_hash: str,
    split_hash: str,
    configuration_hash: str,
    models: dict[str, str],
    strategies: tuple[str, ...],
    execution: dict[str, object],
) -> RunManifest:
    """Return the manifest a run starts with: status="running", nothing finalized yet."""
    return RunManifest(
        schema_version=_SCHEMA_VERSION,
        run_id=run_id,
        status="running",
        git_sha=git_sha,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=None,
        corpus_hash=corpus_hash,
        dataset_hash=dataset_hash,
        split_hash=split_hash,
        configuration_hash=configuration_hash,
        models=models,
        strategies=strategies,
        execution=execution,
        artifact_root_hash=None,
    )


def finalize_manifest(manifest: RunManifest, artifact_root_hash: str) -> RunManifest:
    """Return ``manifest`` marked complete, after every artifact is written and checksummed."""
    return RunManifest(
        schema_version=manifest.schema_version,
        run_id=manifest.run_id,
        status="completed",
        git_sha=manifest.git_sha,
        started_at=manifest.started_at,
        completed_at=datetime.now(UTC).isoformat(),
        corpus_hash=manifest.corpus_hash,
        dataset_hash=manifest.dataset_hash,
        split_hash=manifest.split_hash,
        configuration_hash=manifest.configuration_hash,
        models=manifest.models,
        strategies=manifest.strategies,
        execution=manifest.execution,
        artifact_root_hash=artifact_root_hash,
    )


def reject_if_already_completed(manifest: RunManifest) -> None:
    """Fail closed if a resumed run_id's manifest already says "completed".

    Raises:
        SystemExit: If ``manifest.status == "completed"``.
    """
    if manifest.status == "completed":
        raise SystemExit(
            f"run {manifest.run_id!r} already has a completed evidence directory - "
            "a completed artifacts/runs/<run_id>/ is never overwritten; use a new run_id"
        )
