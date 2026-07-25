"""Tests for scripts/verify_run.py: checksum, event-chain, and manifest verification (ADR-0017)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ragforge.evaluation.artifact_writer import write_atomic, write_checksums_file
from ragforge.evaluation.event_log import EventLog

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_run.py"


def _load_verify_run() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_run", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify_run = _load_verify_run()


def _build_clean_run(artifacts_dir: Path, *, run_id: str = "run-1") -> Path:
    """Build a minimal, internally-consistent evidence directory, the way run.py would."""
    run_dir = artifacts_dir / run_id
    log = EventLog(run_id, run_dir / "events.jsonl")
    log.emit("indexing", "started", {"stage": "base"})
    log.emit("indexing", "completed", {"stage": "base"})

    write_atomic(run_dir / "questions" / "base.json", json.dumps({"question_id": "q1"}))

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "git_sha": "abc123",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:01:00+00:00",
        "corpus_hash": "corpus-hash",
        "dataset_hash": "dataset-hash",
        "split_hash": "split-hash",
        "configuration_hash": "config-hash",
        "models": {},
        "strategies": ["base"],
        "execution": {},
        "artifact_root_hash": None,
    }
    write_atomic(run_dir / "manifest.json", json.dumps(manifest))

    write_checksums_file(run_dir)
    return run_dir


def test_verify_checksums_passes_clean_on_an_untampered_run(tmp_path: Path) -> None:
    """A freshly built evidence directory reports no checksum problems."""
    run_dir = _build_clean_run(tmp_path)

    assert verify_run.verify_checksums(run_dir) == []


def test_verify_checksums_detects_a_tampered_file(tmp_path: Path) -> None:
    """Modifying a file's content after checksums.sha256 was written is detected."""
    run_dir = _build_clean_run(tmp_path)

    write_atomic(run_dir / "questions" / "base.json", json.dumps({"question_id": "TAMPERED"}))

    problems = verify_run.verify_checksums(run_dir)

    assert any("checksum mismatch" in problem for problem in problems)


def test_verify_checksums_detects_a_missing_referenced_file(tmp_path: Path) -> None:
    """A file listed in checksums.sha256 that no longer exists on disk is reported missing."""
    run_dir = _build_clean_run(tmp_path)

    (run_dir / "questions" / "base.json").unlink()

    problems = verify_run.verify_checksums(run_dir)

    assert any("references missing file" in problem for problem in problems)


def test_verify_checksums_detects_an_unlisted_extra_file(tmp_path: Path) -> None:
    """A file added after checksums.sha256 was written, never recorded, is flagged."""
    run_dir = _build_clean_run(tmp_path)

    write_atomic(run_dir / "questions" / "extra.json", "{}")

    problems = verify_run.verify_checksums(run_dir)

    assert any("missing from checksums.sha256" in problem for problem in problems)


def test_verify_event_chain_passes_clean_on_an_untampered_run(tmp_path: Path) -> None:
    """A freshly built event log's hash chain verifies with no problems."""
    run_dir = _build_clean_run(tmp_path)

    assert verify_run.verify_event_chain(run_dir) == []


def test_verify_event_chain_detects_a_tampered_event(tmp_path: Path) -> None:
    """Rewriting one event's event_type without recomputing its hash breaks the chain."""
    run_dir = _build_clean_run(tmp_path)
    events_path = run_dir / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    first_event = json.loads(lines[0])
    first_event["event_type"] = "tampered"
    lines[0] = json.dumps(first_event)
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    problems = verify_run.verify_event_chain(run_dir)

    assert any("event_hash does not match content" in problem for problem in problems)


def test_verify_event_chain_detects_a_broken_previous_hash_link(tmp_path: Path) -> None:
    """Rewriting the second event's previous_event_hash without its own hash breaks the chain."""
    run_dir = _build_clean_run(tmp_path)
    events_path = run_dir / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    second_event = json.loads(lines[1])
    second_event["previous_event_hash"] = "0" * 64
    lines[1] = json.dumps(second_event)
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    problems = verify_run.verify_event_chain(run_dir)

    assert any("chain is broken" in problem for problem in problems)


def test_verify_manifest_passes_clean_when_every_strategy_has_an_artifact(tmp_path: Path) -> None:
    """A completed manifest whose declared strategies all have question artifacts is clean."""
    run_dir = _build_clean_run(tmp_path)

    assert verify_run.verify_manifest(run_dir) == []


def test_verify_manifest_reports_a_non_completed_status(tmp_path: Path) -> None:
    """A manifest still status="running" is reported rather than treated as verifiable."""
    run_dir = _build_clean_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "running"
    write_atomic(run_dir / "manifest.json", json.dumps(manifest))
    write_checksums_file(run_dir)

    problems = verify_run.verify_manifest(run_dir)

    assert any("not 'completed'" in problem for problem in problems)


def test_verify_manifest_detects_a_strategy_with_no_question_artifact(tmp_path: Path) -> None:
    """A strategy declared in the manifest with no corresponding questions/*.json is flagged."""
    run_dir = _build_clean_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["strategies"] = ["base", "raptor"]
    write_atomic(run_dir / "manifest.json", json.dumps(manifest))
    write_checksums_file(run_dir)

    problems = verify_run.verify_manifest(run_dir)

    assert any("raptor" in problem for problem in problems)


def test_main_exits_zero_and_prints_ok_for_a_clean_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() exits cleanly (no SystemExit) and prints an OK message for an untampered run."""
    _build_clean_run(tmp_path, run_id="run-clean")
    monkeypatch.setattr(verify_run, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_run.py", "run-clean"])

    verify_run.main()

    assert "OK" in capsys.readouterr().out


def test_main_exits_non_zero_when_evidence_is_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() raises SystemExit(1) and reports every problem found for a tampered run."""
    run_dir = _build_clean_run(tmp_path, run_id="run-tampered")
    write_atomic(run_dir / "questions" / "base.json", json.dumps({"question_id": "TAMPERED"}))
    monkeypatch.setattr(verify_run, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_run.py", "run-tampered"])

    with pytest.raises(SystemExit) as exc_info:
        verify_run.main()

    assert exc_info.value.code == 1
    assert "FAILED" in capsys.readouterr().out


def test_main_fails_closed_for_an_unknown_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() raises SystemExit for a run_id with no evidence directory at all."""
    monkeypatch.setattr(verify_run, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_run.py", "does-not-exist"])

    with pytest.raises(SystemExit):
        verify_run.main()
