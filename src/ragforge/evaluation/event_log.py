"""Serialized, hash-chained event writer for one run's evidence directory (ADR-0017).

Parallel workers (run_bounded's bounded thread pool - the actual concurrency
this project has, not distributed processes) SHALL NOT write directly to the
shared event stream; they submit events to a single serialized writer that
assigns monotonic sequence numbers and updates the hash chain. ``EventLog``'s
internal lock, shared by every caller through one instance per run, is that
serialized writer - a separate writer process/queue would be over-engineering
for a bounded in-process thread pool.

``event_hash`` covers the canonical serialization of every envelope field
except itself; ``previous_event_hash`` links to the prior event, forming a
local tamper-evident chain that ``scripts/verify_run.py`` walks end to end.
"""

import dataclasses
import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ragforge.evaluation.canonical_hash import canonical_json_hash
from ragforge.evaluation.lineage_ports import EventEnvelope

_SCHEMA_VERSION = 1


def compute_event_hash(envelope: EventEnvelope) -> str:
    """Return the hash ``envelope.event_hash`` should equal - recomputed from every other field.

    Shared by ``EventLog.emit()`` (computes it once, when writing) and
    ``scripts/verify_run.py`` (recomputes it from a stored envelope, to
    detect tampering) - the one place this exact computation lives, so the
    two can never drift apart.
    """
    envelope_without_hash = {
        "schema_version": envelope.schema_version,
        "sequence": envelope.sequence,
        "event_id": envelope.event_id,
        "run_id": envelope.run_id,
        "correlation_id": envelope.correlation_id,
        "stage": envelope.stage,
        "event_type": envelope.event_type,
        "occurred_at": envelope.occurred_at,
        "payload_hash": envelope.payload_hash,
        "previous_event_hash": envelope.previous_event_hash,
    }
    return canonical_json_hash(envelope_without_hash)


class EventLog:
    """Appends hash-chained events to one run's ``events.jsonl``, serialized by an internal lock."""

    def __init__(self, run_id: str, path: Path) -> None:
        """Create the event log for ``run_id``, appending to ``path`` (created if absent)."""
        self._run_id = run_id
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sequence = 0
        self._previous_event_hash: str | None = None

    def emit(
        self,
        stage: str,
        event_type: str,
        payload: dict[str, object],
        correlation_id: str | None = None,
    ) -> EventEnvelope:
        """Append one hash-chained event; return the envelope actually written.

        ``correlation_id`` defaults to this run's ``run_id`` when not given
        - most stage-level events (indexing, per-strategy evaluation) have
        no narrower correlation scope than the run itself.
        """
        with self._lock:
            self._sequence += 1
            envelope = EventEnvelope(
                schema_version=_SCHEMA_VERSION,
                sequence=self._sequence,
                event_id=str(uuid.uuid4()),
                run_id=self._run_id,
                correlation_id=correlation_id or self._run_id,
                stage=stage,
                event_type=event_type,
                occurred_at=datetime.now(UTC).isoformat(),
                payload_hash=canonical_json_hash(payload),
                previous_event_hash=self._previous_event_hash,
                event_hash="",
            )
            event_hash = compute_event_hash(envelope)
            envelope = dataclasses.replace(envelope, event_hash=event_hash)

            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dataclasses.asdict(envelope), ensure_ascii=False))
                handle.write("\n")

            self._previous_event_hash = event_hash
            return envelope
