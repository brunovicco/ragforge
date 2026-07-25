"""ADR-0017 evidence-directory helpers for run.py: fail-closed checks, per-question artifacts.

Split out of run.py (which grew past 1200 lines). Each function here takes
``artifacts_dir`` explicitly rather than closing over it, so none of them
depend on main()'s local state.
"""

import dataclasses
import json
from pathlib import Path

from ragforge.evaluation.artifact_writer import write_atomic
from ragforge.evaluation.audit_metrics import compute_audit_report
from ragforge.evaluation.audit_ports import AuditResult
from ragforge.evaluation.lineage_ports import GenerationLineage, RetrievalCandidateLineage
from ragforge.evaluation.records import QuestionRecord


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
) -> None:
    """Write ``summaries/retrieval.json``, ``summaries/generation.json``, ``summaries/audit.json``.

    The same per-strategy aggregates already computed for
    format_results_table/format_answer_quality_table/compute_audit_report,
    persisted as JSON (ADR-0017).
    """
    write_atomic(
        artifacts_dir / "summaries" / "retrieval.json",
        json.dumps(run_metrics, ensure_ascii=False, indent=2),
    )
    write_atomic(
        artifacts_dir / "summaries" / "generation.json",
        json.dumps(
            {
                label: [dataclasses.asdict(entry) for entry in entries]
                for label, entries in generation_lineage_by_strategy.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    write_atomic(
        artifacts_dir / "summaries" / "audit.json",
        json.dumps(
            {
                label: compute_audit_report(results)
                for label, results in audit_results_by_strategy.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
