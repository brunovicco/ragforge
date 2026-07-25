"""Tests for the serialized, hash-chained event writer (ADR-0017)."""

import concurrent.futures
import dataclasses
import json
from pathlib import Path

from ragforge.evaluation.event_log import EventLog, compute_event_hash
from ragforge.evaluation.lineage_ports import EventEnvelope


def test_emit_assigns_monotonic_sequence_numbers(tmp_path: Path) -> None:
    """Consecutive emit() calls get sequence 1, 2, 3, ... with no gaps."""
    log = EventLog("run-1", tmp_path / "events.jsonl")

    first = log.emit("indexing", "started", {"stage": "base"})
    second = log.emit("indexing", "completed", {"stage": "base"})
    third = log.emit("indexing", "started", {"stage": "contextual"})

    assert (first.sequence, second.sequence, third.sequence) == (1, 2, 3)


def test_emit_links_previous_event_hash_to_the_prior_events_actual_hash(tmp_path: Path) -> None:
    """Each event's previous_event_hash equals the actual event_hash of the event before it."""
    log = EventLog("run-1", tmp_path / "events.jsonl")

    first = log.emit("indexing", "started", {"stage": "base"})
    second = log.emit("indexing", "completed", {"stage": "base"})

    assert first.previous_event_hash is None
    assert second.previous_event_hash == first.event_hash


def test_emit_writes_one_json_line_per_event(tmp_path: Path) -> None:
    """events.jsonl accumulates one JSON object per line, in emission order."""
    log = EventLog("run-1", tmp_path / "events.jsonl")

    log.emit("indexing", "started", {"stage": "base"})
    log.emit("indexing", "completed", {"stage": "base"})

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["sequence"] for line in lines] == [1, 2]


def test_emit_defaults_correlation_id_to_the_run_id(tmp_path: Path) -> None:
    """When no correlation_id is given, it defaults to the run's own run_id."""
    log = EventLog("run-42", tmp_path / "events.jsonl")

    envelope = log.emit("preflight", "started", {})

    assert envelope.correlation_id == "run-42"


def test_emit_uses_an_explicit_correlation_id_when_given(tmp_path: Path) -> None:
    """An explicit correlation_id overrides the run_id default."""
    log = EventLog("run-42", tmp_path / "events.jsonl")

    envelope = log.emit("strategy", "started", {}, correlation_id="question-7")

    assert envelope.correlation_id == "question-7"


def test_compute_event_hash_recomputes_to_the_same_value_for_an_untampered_envelope(
    tmp_path: Path,
) -> None:
    """compute_event_hash on a stored envelope reproduces its own recorded event_hash."""
    log = EventLog("run-1", tmp_path / "events.jsonl")

    envelope = log.emit("indexing", "started", {"stage": "base"})

    assert compute_event_hash(envelope) == envelope.event_hash


def test_compute_event_hash_detects_tampering_with_any_field(tmp_path: Path) -> None:
    """Modifying any field of a stored envelope changes its recomputed hash."""
    log = EventLog("run-1", tmp_path / "events.jsonl")
    envelope = log.emit("indexing", "started", {"stage": "base"})

    tampered = dataclasses.replace(envelope, event_type="completed")

    assert compute_event_hash(tampered) != envelope.event_hash


def test_emit_is_thread_safe_and_produces_a_gap_free_sequence_under_concurrency(
    tmp_path: Path,
) -> None:
    """Concurrent emit() calls from multiple threads never collide or skip a sequence number."""
    log = EventLog("run-1", tmp_path / "events.jsonl")

    def _emit(index: int) -> EventEnvelope:
        return log.emit("strategy", "completed", {"index": index})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        envelopes = list(executor.map(_emit, range(50)))

    sequences = sorted(envelope.sequence for envelope in envelopes)
    assert sequences == list(range(1, 51))
