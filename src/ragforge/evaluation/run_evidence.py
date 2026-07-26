"""ADR-0017 evidence-directory helpers for run.py: fail-closed checks, per-question artifacts.

Split out of run.py (which grew past 1200 lines). Each function here takes
``artifacts_dir`` explicitly rather than closing over it, so none of them
depend on main()'s local state.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

from ragforge.evaluation.artifact_writer import (
    compute_checksums,
    write_atomic,
    write_checksums_file,
)
from ragforge.evaluation.audit_metrics import compute_audit_report
from ragforge.evaluation.audit_ports import AuditResult
from ragforge.evaluation.canonical_hash import canonical_json_hash
from ragforge.evaluation.lineage_ports import (
    GenerationLineage,
    RetrievalCandidateLineage,
    RunManifest,
)
from ragforge.evaluation.records import QuestionRecord
from ragforge.evaluation.run_manifest import finalize_manifest

_MANIFEST_FILENAME = "manifest.json"


def _load_json_object(path: Path) -> dict[str, object]:
    """Load an existing summary object, or return an empty mapping."""
    if not path.exists():
        return {}
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary {path} must contain a JSON object")
    return {str(key): value for key, value in payload.items()}


def reject_if_evidence_dir_already_completed(artifacts_dir: Path) -> None:
    """Fail closed if ``artifacts_dir``'s manifest.json already says status="completed" (ADR-0017).

    No manifest yet (a genuinely new run_id, or one whose evidence directory
    was never started) is not an error - only an already-completed one is
    rejected, matching "a completed directory SHALL not be overwritten".

    Raises:
        SystemExit: If a manifest exists there with status "completed".
    """
    manifest_path = artifacts_dir / "manifest.json"
    if not manifest_path.exists():
        return
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("status") == "completed":
        raise SystemExit(
            f"run {previous.get('run_id')!r} already has a completed evidence directory at "
            f"{artifacts_dir} - a completed artifacts/runs/<run_id>/ is never overwritten; "
            "use a new run_id"
        )


def finalize_evidence_directory(
    artifacts_dir: Path,
    run_manifest: RunManifest,
) -> RunManifest:
    """Atomically publish a completed manifest covered by the checksum inventory.

    ``artifact_root_hash`` covers every final artifact except
    ``manifest.json`` and ``checksums.sha256``. Excluding the manifest from
    that root avoids a self-reference because the root itself is stored in
    the manifest. ``checksums.sha256`` still includes the exact final
    manifest bytes, so post-completion changes remain detectable.

    The checksum inventory is written before the final manifest. A crash
    before the last atomic rename therefore leaves the manifest in
    ``running`` state rather than exposing a completed-but-unverifiable run.
    """
    if run_manifest.status != "running":
        raise ValueError("only a running manifest can be finalized")
    artifact_checksums = compute_checksums(
        artifacts_dir,
        excluded_paths=frozenset({_MANIFEST_FILENAME}),
    )
    final_manifest = finalize_manifest(
        run_manifest,
        canonical_json_hash(artifact_checksums),
    )
    final_manifest_content = json.dumps(
        dataclasses.asdict(final_manifest),
        ensure_ascii=False,
        indent=2,
    )
    complete_checksums = {
        **artifact_checksums,
        _MANIFEST_FILENAME: hashlib.sha256(final_manifest_content.encode("utf-8")).hexdigest(),
    }
    write_checksums_file(artifacts_dir, complete_checksums)
    write_atomic(artifacts_dir / _MANIFEST_FILENAME, final_manifest_content)
    return final_manifest


def write_question_artifacts(
    artifacts_dir: Path,
    label: str,
    records: list[QuestionRecord],
    candidate_lineage: list[RetrievalCandidateLineage],
) -> None:
    """Write ``questions/<question_id>/<label>.json`` (ADR-0017): QuestionRecord + its lineage.

    Only retrieval candidate lineage is embedded here - it is reliably
    correlatable by ``question_id``. Generation/audit lineage is produced in
    worker-thread completion order (run_bounded), not canonical question
    order, so attaching it to a specific question file here would risk
    mislabeling; it is reported per-strategy in summaries/generation.json
    and summaries/audit.json instead.
    """
    lineage_by_question: dict[str, list[RetrievalCandidateLineage]] = {}
    for entry in candidate_lineage:
        lineage_by_question.setdefault(entry.query_id, []).append(entry)
    for record in records:
        payload = {
            "question_record": record.to_json_dict(),
            "candidate_lineage": [
                dataclasses.asdict(entry)
                for entry in lineage_by_question.get(record.question_id, [])
            ],
        }
        write_atomic(
            artifacts_dir / "questions" / record.question_id / f"{label}.json",
            json.dumps(payload, ensure_ascii=False, indent=2),
        )


def write_summaries(
    artifacts_dir: Path,
    run_metrics: dict[str, dict[str, float]],
    generation_lineage_by_strategy: dict[str, list[GenerationLineage]],
    audit_results_by_strategy: dict[str, list[AuditResult]],
    *,
    metric_breakdowns: dict[str, object] | None = None,
    generation_usage: dict[str, dict[str, float | int | None]] | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Write ``summaries/retrieval.json``, ``summaries/generation.json``, ``summaries/audit.json``.

    The same per-strategy aggregates already computed for
    format_results_table/format_answer_quality_table/compute_audit_report,
    persisted as JSON (ADR-0017).
    """
    write_atomic(
        artifacts_dir / "summaries" / "retrieval.json",
        json.dumps(run_metrics, ensure_ascii=False, indent=2),
    )
    generation_path = artifacts_dir / "summaries" / "generation.json"
    generation_payload = _load_json_object(generation_path)
    generation_payload.update(
        {
            label: [dataclasses.asdict(entry) for entry in entries]
            for label, entries in generation_lineage_by_strategy.items()
        }
    )
    write_atomic(
        generation_path,
        json.dumps(generation_payload, ensure_ascii=False, indent=2),
    )
    audit_path = artifacts_dir / "summaries" / "audit.json"
    audit_payload = _load_json_object(audit_path)
    audit_payload.update(
        {
            label: compute_audit_report(results)
            for label, results in audit_results_by_strategy.items()
        }
    )
    write_atomic(
        audit_path,
        json.dumps(audit_payload, ensure_ascii=False, indent=2),
    )
    write_atomic(
        artifacts_dir / "summaries" / "breakdowns.json",
        json.dumps(metric_breakdowns or {}, ensure_ascii=False, indent=2),
    )
    usage_path = artifacts_dir / "summaries" / "usage.json"
    usage_payload = _load_json_object(usage_path)
    usage_payload.update(generation_usage or {})
    write_atomic(usage_path, json.dumps(usage_payload, ensure_ascii=False, indent=2))
    merged_usage: dict[str, dict[str, float | int | None]] = {}
    for strategy, raw_summary in usage_payload.items():
        if not isinstance(raw_summary, dict):
            raise ValueError(f"usage summary for {strategy!r} must be an object")
        normalized: dict[str, float | int | None] = {}
        for key, value in raw_summary.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError(f"usage value {strategy}.{key} must be numeric or null")
            normalized[str(key)] = value
        merged_usage[strategy] = normalized
    return merged_usage
