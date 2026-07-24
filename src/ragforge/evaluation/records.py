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

    A question with no matching entry in ``answer_records`` - every
    unanswerable-class question, which evaluate_answer_quality never scores
    since Citation Accuracy has nothing to check citations against - gets
    "not_applicable" generation/judge status rather than being silently
    absent from the merged output.
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
    """Append each record as one JSON line to ``path`` (created if it does not exist)."""
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False))
            handle.write("\n")
