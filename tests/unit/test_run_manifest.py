"""Tests for run manifest lifecycle (ADR-0017)."""

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest

from ragforge.evaluation.lineage_ports import RunManifest
from ragforge.evaluation.run_manifest import (
    build_initial_manifest,
    finalize_manifest,
    load_run_manifest,
    reject_if_already_completed,
    require_clean_worktree,
    resolve_git_sha,
)


def _manifest() -> RunManifest:
    return build_initial_manifest(
        run_id="run-1",
        git_sha="abc123",
        corpus_hash="corpus-hash",
        dataset_hash="dataset-hash",
        split_hash="split-hash",
        configuration_hash="config-hash",
        models={"generator": "gemini"},
        strategies=("base",),
        execution={"max_workers": 4},
    )


def test_resolve_git_sha_returns_a_non_empty_string_in_this_repository() -> None:
    """Running inside a real git checkout, resolve_git_sha returns a real SHA, not the fallback."""
    sha = resolve_git_sha()

    assert sha != "unknown"
    assert len(sha) == 40


def test_resolve_git_sha_falls_back_to_unknown_when_git_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the git executable can't be found, resolve_git_sha falls back to "unknown"."""
    monkeypatch.setattr("ragforge.evaluation.run_manifest.shutil.which", lambda _name: None)

    assert resolve_git_sha() == "unknown"


def test_resolve_git_sha_falls_back_to_unknown_when_git_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If invoking git raises, resolve_git_sha returns "unknown" instead of propagating."""
    monkeypatch.setattr(
        "ragforge.evaluation.run_manifest.shutil.which", lambda _name: "/usr/bin/git"
    )

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["git"])

    monkeypatch.setattr("ragforge.evaluation.run_manifest.subprocess.run", _raise)

    assert resolve_git_sha() == "unknown"


def test_build_initial_manifest_starts_running_with_no_completion_fields() -> None:
    """A freshly built manifest is status="running", with completed_at/artifact_root_hash unset."""
    manifest = _manifest()

    assert manifest.status == "running"
    assert manifest.completed_at is None
    assert manifest.artifact_root_hash is None
    assert manifest.run_id == "run-1"


def test_load_run_manifest_preserves_the_original_start_identity(tmp_path: Path) -> None:
    """A resumed run reloads, rather than reconstructs, its running manifest."""
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dataclasses.asdict(manifest)), encoding="utf-8")

    loaded = load_run_manifest(path)

    assert loaded == manifest
    assert loaded.started_at == manifest.started_at
    assert loaded.git_sha == manifest.git_sha


def test_require_clean_worktree_allows_only_untracked_generated_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run output directories do not prevent a later clean resume."""
    monkeypatch.setattr("ragforge.evaluation.run_manifest.shutil.which", lambda _name: "git")
    result = subprocess.CompletedProcess(
        args=["git"],
        returncode=0,
        stdout=(
            "?? artifacts/runs/run-1/report.json\n"
            "?? experiments/run-1/results.json\n"
            "?? .ragforge/cache/indexes/marker.json\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(
        "ragforge.evaluation.run_manifest.subprocess.run",
        lambda *_args, **_kwargs: result,
    )

    require_clean_worktree(tmp_path)


def test_require_clean_worktree_rejects_staged_unstaged_or_untracked_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest Git SHA may not conceal code or documentation changes."""
    monkeypatch.setattr("ragforge.evaluation.run_manifest.shutil.which", lambda _name: "git")
    result = subprocess.CompletedProcess(
        args=["git"],
        returncode=0,
        stdout=" M src/ragforge/app.py\nA  wiki/Home.md\n?? configs/local.yaml\n",
        stderr="",
    )
    monkeypatch.setattr(
        "ragforge.evaluation.run_manifest.subprocess.run",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(SystemExit, match="clean Git worktree"):
        require_clean_worktree(tmp_path)


def test_require_clean_worktree_rejects_a_tracked_output_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only untracked generated output is exempt; tracked evidence remains protected."""
    monkeypatch.setattr("ragforge.evaluation.run_manifest.shutil.which", lambda _name: "git")
    result = subprocess.CompletedProcess(
        args=["git"],
        returncode=0,
        stdout=" M experiments/published/results.json\n",
        stderr="",
    )
    monkeypatch.setattr(
        "ragforge.evaluation.run_manifest.subprocess.run",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(SystemExit, match="clean Git worktree"):
        require_clean_worktree(tmp_path)


def test_finalize_manifest_marks_completed_and_sets_artifact_root_hash() -> None:
    """finalize_manifest returns a copy with status="completed" and the given root hash set."""
    manifest = _manifest()

    finalized = finalize_manifest(manifest, artifact_root_hash="root-hash")

    assert finalized.status == "completed"
    assert finalized.artifact_root_hash == "root-hash"
    assert finalized.completed_at is not None
    assert finalized.run_id == manifest.run_id
    assert finalized.started_at == manifest.started_at


def test_reject_if_already_completed_raises_for_a_completed_manifest() -> None:
    """A manifest already marked completed fails closed rather than being silently reused."""
    manifest = finalize_manifest(_manifest(), artifact_root_hash="root-hash")

    with pytest.raises(SystemExit):
        reject_if_already_completed(manifest)


def test_reject_if_already_completed_is_a_no_op_for_a_running_manifest() -> None:
    """A manifest still status="running" passes through without raising."""
    manifest = _manifest()

    reject_if_already_completed(manifest)
