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

import shutil
import subprocess  # nosec B404
from datetime import UTC, datetime

from ragforge.evaluation.lineage_ports import RunManifest

_SCHEMA_VERSION = 1
_UNKNOWN_GIT_SHA = "unknown"


def resolve_git_sha() -> str:
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
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return _UNKNOWN_GIT_SHA
    sha = result.stdout.strip()
    return sha if sha else _UNKNOWN_GIT_SHA


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
