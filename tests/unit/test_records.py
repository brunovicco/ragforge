"""Tests for per-question record merging and JSONL persistence (ADR-0012)."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ragforge.evaluation.records import (
    AnswerRecord,
    RetrievalRecord,
    append_records_jsonl,
    merge_question_records,
)


def _retrieval(question_id: str, **overrides: Any) -> RetrievalRecord:
    base = RetrievalRecord(
        question_id=question_id,
        query_class="exact_factual",
        unanswerable=False,
        status="succeeded",
        retrieved_structural_ids=("NORM::art-1",),
        metrics={"recall_at_k": 1.0},
        error=None,
    )
    return replace(base, **overrides)


def _answer(question_id: str, **overrides: Any) -> AnswerRecord:
    base = AnswerRecord(
        question_id=question_id,
        status="succeeded",
        answer_text="the answer",
        answer_citations=("NORM::art-1",),
        metrics={"citation_accuracy": 1.0},
        error=None,
    )
    return replace(base, **overrides)


def test_merge_joins_retrieval_and_answer_records_by_question_id() -> None:
    """A question with both a retrieval and an answer record gets one merged QuestionRecord."""
    records = merge_question_records("dense", [_retrieval("q1")], [_answer("q1")])

    assert len(records) == 1
    record = records[0]
    assert record.question_id == "q1"
    assert record.strategy == "dense"
    assert record.retrieval_status == "succeeded"
    assert record.generation_status == "succeeded"
    assert record.judge_status == "succeeded"
    assert record.answer_text == "the answer"
    assert record.metrics == {"recall_at_k": 1.0, "citation_accuracy": 1.0}
    assert record.errors == ()


def test_merge_marks_generation_and_judge_not_applicable_when_no_answer_record_exists() -> None:
    """An unanswerable-class question (never scored for answer quality) still gets a record."""
    records = merge_question_records("dense", [_retrieval("q1", unanswerable=True, metrics={})], [])

    assert len(records) == 1
    record = records[0]
    assert record.unanswerable is True
    assert record.generation_status == "not_applicable"
    assert record.judge_status == "not_applicable"
    assert record.answer_text is None
    assert record.answer_citations == ()
    assert record.metrics == {}


def test_merge_collects_errors_from_both_stages() -> None:
    """Retrieval and answer-quality errors are both surfaced on the merged record."""
    records = merge_question_records(
        "dense",
        [_retrieval("q1", status="failed", error="retrieval: boom")],
        [_answer("q1", status="failed", error="judge: boom")],
    )

    assert records[0].errors == ("retrieval: boom", "judge: boom")


def test_append_records_jsonl_writes_one_json_line_per_record(tmp_path: Path) -> None:
    """Each record is serialized on its own line; a second call appends, not overwrites."""
    path = tmp_path / "records.jsonl"
    first_batch = merge_question_records("dense", [_retrieval("q1")], [_answer("q1")])
    second_batch = merge_question_records("sparse_bm25", [_retrieval("q1")], [_answer("q1")])

    append_records_jsonl(path, first_batch)
    append_records_jsonl(path, second_batch)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["strategy"] == "dense"
    assert parsed[1]["strategy"] == "sparse_bm25"
    assert parsed[0]["metrics"] == {"recall_at_k": 1.0, "citation_accuracy": 1.0}
