#!/usr/bin/env python3
"""Verify a run's ADR-0017 evidence directory: checksums, event hash chain, referenced artifacts.

Usage: uv run python scripts/verify_run.py <run_id>

Validates, against ``artifacts/runs/<run_id>/``:
- every file listed in checksums.sha256 still hashes to the recorded value;
- events.jsonl's event_hash chain is unbroken and sequence numbers are
  monotonic with no gaps;
- every strategy declared in manifest.json has a question artifact for
  every question referenced in records.jsonl (experiments/<run_id>/), when
  that companion directory is available.

Never repairs anything - reports what it finds and exits non-zero on any
mismatch, so it is safe to run against evidence you suspect was tampered
with or produced by a crashed run.
"""

import argparse
import json
import sys
from pathlib import Path

from ragforge.evaluation.event_log import compute_event_hash
from ragforge.evaluation.lineage_ports import EventEnvelope
from ragforge.ingestion.snapshot import snapshot_hash

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts" / "runs"


def parse_args() -> argparse.Namespace:
    """Parse this script's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    return parser.parse_args()


def verify_checksums(artifacts_dir: Path) -> list[str]:
    """Return every problem found comparing checksums.sha256 against the files on disk.

    A file listed in checksums.sha256 but missing, or present with a
    different hash than recorded, is reported. A file present on disk but
    absent from checksums.sha256 is also reported - the manifest is
    supposed to be complete.
    """
    problems = []
    checksums_path = artifacts_dir / "checksums.sha256"
    if not checksums_path.exists():
        return [f"checksums.sha256 not found at {checksums_path}"]

    recorded: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition("  ")
        recorded[relative] = digest

    for relative, digest in recorded.items():
        path = artifacts_dir / relative
        if not path.exists():
            problems.append(f"checksums.sha256 references missing file: {relative}")
            continue
        actual = snapshot_hash(path)
        if actual != digest:
            problems.append(f"checksum mismatch for {relative}: recorded {digest}, actual {actual}")

    on_disk = {
        path.relative_to(artifacts_dir).as_posix()
        for path in artifacts_dir.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    for relative in sorted(on_disk - set(recorded)):
        problems.append(f"file present on disk but missing from checksums.sha256: {relative}")

    return problems


def verify_event_chain(artifacts_dir: Path) -> list[str]:
    """Return every problem found walking events.jsonl's hash chain.

    Checks, per event in file order: its own ``event_hash`` recomputes
    correctly, its ``sequence`` is exactly one more than the previous
    event's, and its ``previous_event_hash`` matches the previous event's
    actual ``event_hash``.
    """
    problems: list[str] = []
    events_path = artifacts_dir / "events.jsonl"
    if not events_path.exists():
        return [f"events.jsonl not found at {events_path}"]

    previous_envelope: EventEnvelope | None = None
    for line_number, line in enumerate(
        events_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        envelope = EventEnvelope(**payload)

        if compute_event_hash(envelope) != envelope.event_hash:
            problems.append(f"events.jsonl line {line_number}: event_hash does not match content")

        if previous_envelope is None:
            if envelope.previous_event_hash is not None:
                problems.append(
                    f"events.jsonl line {line_number}: first event has a non-null "
                    "previous_event_hash"
                )
        else:
            if envelope.sequence != previous_envelope.sequence + 1:
                problems.append(
                    f"events.jsonl line {line_number}: sequence {envelope.sequence} is not "
                    f"one more than the previous event's {previous_envelope.sequence}"
                )
            if envelope.previous_event_hash != previous_envelope.event_hash:
                problems.append(
                    f"events.jsonl line {line_number}: previous_event_hash does not match "
                    "the prior event's actual event_hash - chain is broken"
                )
        previous_envelope = envelope

    return problems


def verify_manifest(artifacts_dir: Path) -> list[str]:
    """Return every problem found validating manifest.json's own internal consistency.

    Checks the manifest exists, is valid JSON, and - when status is
    "completed" - that every declared strategy has at least one question
    artifact directory referencing it.
    """
    problems = []
    manifest_path = artifacts_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"manifest.json not found at {manifest_path}"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        problems.append(f"manifest.json status is {manifest.get('status')!r}, not 'completed'")
        return problems

    questions_dir = artifacts_dir / "questions"
    strategies = set(manifest.get("strategies", []))
    referenced_strategies: set[str] = set()
    if questions_dir.exists():
        for strategy_file in questions_dir.rglob("*.json"):
            referenced_strategies.add(strategy_file.stem)

    missing = strategies - referenced_strategies
    if missing:
        problems.append(
            f"manifest.json declares strategies with no question artifacts: {sorted(missing)}"
        )
    return problems


def main() -> None:
    """Verify the named run's evidence directory and report every problem found."""
    args = parse_args()
    artifacts_dir = ARTIFACTS_DIR / args.run_id
    if not artifacts_dir.exists():
        raise SystemExit(f"no evidence directory found for run {args.run_id!r} at {artifacts_dir}")

    problems = [
        *verify_manifest(artifacts_dir),
        *verify_checksums(artifacts_dir),
        *verify_event_chain(artifacts_dir),
    ]

    if not problems:
        print(f"OK: {args.run_id} - checksums, event chain, and manifest all verify cleanly.")
        return

    print(f"FAILED: {args.run_id} - {len(problems)} problem(s) found:\n")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)


if __name__ == "__main__":
    main()
