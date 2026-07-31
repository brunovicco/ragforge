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
    read_records_jsonl,
    replace_strategy_records_jsonl,
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
        judge_contexts=("authoritative context",),
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
    assert record.judge_contexts == ("authoritative context",)
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
    assert record.judge_contexts == ()
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
    assert parsed[0]["judge_contexts"] == ["authoritative context"]
    assert parsed[0]["metrics"] == {"recall_at_k": 1.0, "citation_accuracy": 1.0}


def test_read_records_jsonl_defaults_legacy_judge_contexts_to_empty(tmp_path: Path) -> None:
    """Records written before exact judge-context persistence remain readable."""
    path = tmp_path / "records.jsonl"
    record = merge_question_records("dense", [_retrieval("q1")], [_answer("q1")])[0]
    payload = record.to_json_dict()
    del payload["judge_contexts"]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    loaded = read_records_jsonl(path)

    assert loaded[0].judge_contexts == ()


def test_append_records_jsonl_is_idempotent_for_a_resumed_strategy(tmp_path: Path) -> None:
    """Resume cannot duplicate an already persisted strategy/question record."""
    path = tmp_path / "records.jsonl"
    records = merge_question_records("dense", [_retrieval("q1")], [_answer("q1")])

    append_records_jsonl(path, records)
    append_records_jsonl(path, records)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_replace_strategy_records_jsonl_removes_failed_resume_records(tmp_path: Path) -> None:
    """Retrying a strategy replaces stale outcomes while preserving other strategies."""
    path = tmp_path / "records.jsonl"
    stale = merge_question_records(
        "dense",
        [_retrieval("q1", status="failed", error="quota")],
        [_answer("q1", status="failed", error="quota")],
    )
    preserved = merge_question_records(
        "sparse_bm25",
        [_retrieval("q1")],
        [_answer("q1")],
    )
    replacement = merge_question_records(
        "dense",
        [_retrieval("q1")],
        [_answer("q1")],
    )
    append_records_jsonl(path, [*stale, *preserved])

    replace_strategy_records_jsonl(path, "dense", replacement)

    records = read_records_jsonl(path)
    assert len(records) == 2
    by_strategy = {record.strategy: record for record in records}
    assert by_strategy["dense"].errors == ()
    assert by_strategy["sparse_bm25"] == preserved[0]
