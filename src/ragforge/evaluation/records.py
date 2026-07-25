"""Per-question, per-strategy result records (ADR-0012).

evaluate_strategy (harness.py) and evaluate_answer_quality (answer_harness.py)
each produce one partial record per question they process - RetrievalRecord
and AnswerRecord respectively. merge_question_records joins them by
question_id into the final immutable QuestionRecord, so every selected
question has an explicit outcome for every strategy it was run against, not
just an aggregate average that a failure could silently shrink.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class RetrievalRecord:
    """One question's retrieval outcome, produced by evaluate_strategy."""

    question_id: str
    query_class: str | None
    unanswerable: bool
    status: str
    retrieved_structural_ids: tuple[str, ...]
    metrics: dict[str, float]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerRecord:
    """One question's answer-quality outcome, produced by evaluate_answer_quality."""

    question_id: str
    status: str
    answer_text: str | None
    answer_citations: tuple[str, ...]
    metrics: dict[str, float]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    """The immutable per-question, per-strategy record (ADR-0012)."""

    question_id: str
    query_class: str | None
    strategy: str
    unanswerable: bool
    retrieval_status: str
    generation_status: str
    judge_status: str
    retrieved_structural_ids: tuple[str, ...]
    answer_text: str | None
    answer_citations: tuple[str, ...]
    metrics: dict[str, float]
    errors: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        """Render as a JSON-serializable dict (one line of records.jsonl)."""
        return {
            "question_id": self.question_id,
            "query_class": self.query_class,
            "strategy": self.strategy,
            "unanswerable": self.unanswerable,
            "retrieval_status": self.retrieval_status,
            "generation_status": self.generation_status,
            "judge_status": self.judge_status,
            "retrieved_structural_ids": list(self.retrieved_structural_ids),
            "answer_text": self.answer_text,
            "answer_citations": list(self.answer_citations),
            "metrics": self.metrics,
            "errors": list(self.errors),
        }


def merge_question_records(
    strategy: str,
    retrieval_records: list[RetrievalRecord],
    answer_records: list[AnswerRecord],
) -> list[QuestionRecord]:
    """Join retrieval and answer-quality outcomes by question_id into one record each.

    A question with no matching entry in ``answer_records`` (e.g. a strategy
    aborted before reaching it) gets "not_applicable" generation/judge status
    rather than being silently absent from the merged output. Since ADR-0018,
    unanswerable-class questions are no longer inherently absent here -
    evaluate_answer_quality scores them too (for abstention appropriateness),
    just without a citation_accuracy metric.
    """
    answer_by_id = {record.question_id: record for record in answer_records}
    merged = []
    for retrieval in retrieval_records:
        answer = answer_by_id.get(retrieval.question_id)
        errors = [error for error in (retrieval.error, answer.error if answer else None) if error]
        merged.append(
            QuestionRecord(
                question_id=retrieval.question_id,
                query_class=retrieval.query_class,
                strategy=strategy,
                unanswerable=retrieval.unanswerable,
                retrieval_status=retrieval.status,
                generation_status=answer.status if answer is not None else "not_applicable",
                judge_status=answer.status if answer is not None else "not_applicable",
                retrieved_structural_ids=retrieval.retrieved_structural_ids,
                answer_text=answer.answer_text if answer is not None else None,
                answer_citations=answer.answer_citations if answer is not None else (),
                metrics={**retrieval.metrics, **(answer.metrics if answer is not None else {})},
                errors=tuple(errors),
            )
        )
    return merged


def append_records_jsonl(path: Path, records: list[QuestionRecord]) -> None:
    """Append records not already present by strategy/question identity."""
    existing_keys = (
        {(record.strategy, record.question_id) for record in read_records_jsonl(path)}
        if path.exists()
        else set()
    )
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            if (record.strategy, record.question_id) in existing_keys:
                continue
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False))
            handle.write("\n")


def read_records_jsonl(path: Path) -> list[QuestionRecord]:
    """Load stored records, keeping the latest unique strategy/question pair."""
    if not path.exists():
        return []
    records_by_key: dict[tuple[str, str], QuestionRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = cast(dict[str, object], json.loads(line))
        metrics_payload = cast(dict[str, object], payload["metrics"])
        metrics: dict[str, float] = {}
        for key, value in metrics_payload.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"record metric {key!r} must be numeric")
            metrics[str(key)] = float(value)
        record = QuestionRecord(
            question_id=str(payload["question_id"]),
            query_class=(
                str(payload["query_class"]) if payload.get("query_class") is not None else None
            ),
            strategy=str(payload["strategy"]),
            unanswerable=bool(payload["unanswerable"]),
            retrieval_status=str(payload["retrieval_status"]),
            generation_status=str(payload["generation_status"]),
            judge_status=str(payload["judge_status"]),
            retrieved_structural_ids=tuple(
                str(value) for value in cast(list[object], payload["retrieved_structural_ids"])
            ),
            answer_text=(
                str(payload["answer_text"]) if payload.get("answer_text") is not None else None
            ),
            answer_citations=tuple(
                str(value) for value in cast(list[object], payload["answer_citations"])
            ),
            metrics=metrics,
            errors=tuple(str(value) for value in cast(list[object], payload["errors"])),
        )
        records_by_key[(record.strategy, record.question_id)] = record
    return list(records_by_key.values())
