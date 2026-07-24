"""Tests for the aggregate evaluation harness (ADR-0002/0003), using a fake strategy."""

import pytest

from ragforge.domain.models import (
    Chunk,
    JudgedRef,
    Judgment,
    Query,
    QueryClass,
    RelevanceGrade,
    RetrievalResult,
    StructuralRef,
)
from ragforge.evaluation.harness import evaluate_strategy

ART_1 = "NORM::art-1"
ART_2 = "NORM::art-2"


def _judgment(question_id: str, *canonical_refs: str, unanswerable: bool = False) -> Judgment:
    return Judgment(
        question_id=question_id,
        query=Query(
            text=question_id,
            query_class=QueryClass.UNANSWERABLE if unanswerable else QueryClass.EXACT_FACTUAL,
        ),
        relevant_refs=tuple(
            JudgedRef(ref=StructuralRef.parse(c), grade=RelevanceGrade.RELEVANT)
            for c in canonical_refs
        ),
    )


class _FakeStrategy:
    name = "fake"

    def __init__(
        self, hits_by_question: dict[str, list[str]], fails_for: frozenset[str] = frozenset()
    ) -> None:
        self._hits_by_question = hits_by_question
        self._fails_for = fails_for

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        if query.text in self._fails_for:
            raise RuntimeError("simulated retrieval failure")
        structural_ids = self._hits_by_question.get(query.text, [])
        return [
            RetrievalResult(
                chunk=Chunk(chunk_id=ref, text=ref, structural_ids=(ref,)),
                score=1.0,
                strategy="fake",
            )
            for ref in structural_ids[:top_k]
        ]


def test_evaluate_strategy_averages_recall_across_judgments() -> None:
    """Two questions, one perfect hit and one miss, average to 0.5 recall."""
    strategy = _FakeStrategy({"q1": [ART_1], "q2": [ART_2]})
    judgments = [_judgment("q1", ART_1), _judgment("q2", ART_1)]

    result = evaluate_strategy(strategy, judgments, k=5)

    assert result.metrics["recall_at_k"] == 0.5
    assert result.metrics["n"] == 2.0


def test_evaluate_strategy_excludes_unanswerable_questions_from_the_average() -> None:
    """An unanswerable-class judgment (no relevant refs) doesn't dilute the average."""
    strategy = _FakeStrategy({"q1": [ART_1], "q2": []})
    judgments = [_judgment("q1", ART_1), _judgment("q2", unanswerable=True)]

    result = evaluate_strategy(strategy, judgments, k=5)

    assert result.metrics["recall_at_k"] == 1.0
    assert result.metrics["n"] == 1.0


def test_evaluate_strategy_still_emits_a_record_for_an_unanswerable_question() -> None:
    """An unanswerable-class question is retrieved and recorded, just excluded from the average."""
    strategy = _FakeStrategy({"q1": [ART_1], "q2": []})
    judgments = [_judgment("q1", ART_1), _judgment("q2", unanswerable=True)]

    result = evaluate_strategy(strategy, judgments, k=5)

    assert len(result.records) == 2
    unanswerable_record = next(r for r in result.records if r.question_id == "q2")
    assert unanswerable_record.unanswerable is True
    assert unanswerable_record.status == "succeeded"
    assert unanswerable_record.metrics == {}


def test_evaluate_strategy_reports_k_in_the_result() -> None:
    """The k used for the run is echoed back in the result dict."""
    result = evaluate_strategy(_FakeStrategy({"q1": [ART_1]}), [_judgment("q1", ART_1)], k=3)
    assert result.metrics["k"] == 3.0


def test_evaluate_strategy_raises_for_an_empty_judgment_list() -> None:
    """An empty golden set is a caller error, not a silently meaningless 0.0 result."""
    with pytest.raises(ValueError, match="judgments must not be empty"):
        evaluate_strategy(_FakeStrategy({}), [], k=5)


def test_evaluate_strategy_survives_a_single_retrieval_failure() -> None:
    """One question whose retrieve() raises doesn't abort the whole strategy."""
    strategy = _FakeStrategy({"q1": [ART_1], "q3": [ART_1]}, fails_for=frozenset({"q2"}))
    judgments = [_judgment("q1", ART_1), _judgment("q2", ART_1), _judgment("q3", ART_1)]

    result = evaluate_strategy(strategy, judgments, k=5)

    assert result.metrics["n"] == 2.0
    assert result.metrics["errors"] == 1.0
    assert result.metrics["recall_at_k"] == 1.0
    failed_record = next(r for r in result.records if r.question_id == "q2")
    assert failed_record.status == "failed"
    assert failed_record.error == "simulated retrieval failure"


def test_evaluate_strategy_stops_early_after_consecutive_retrieval_failures() -> None:
    """Five-in-a-row failures (a systemic problem) stop the strategy rather than retry forever."""
    strategy = _FakeStrategy({}, fails_for=frozenset({f"q{i}" for i in range(1, 8)}))
    judgments = [_judgment(f"q{i}", ART_1) for i in range(1, 8)]

    result = evaluate_strategy(strategy, judgments, k=5)

    assert result.metrics["n"] == 0.0
    assert result.metrics["errors"] == 5.0


def test_evaluate_strategy_still_records_questions_skipped_after_the_abort() -> None:
    """Questions left unattempted after the circuit breaker trips still get an explicit record."""
    strategy = _FakeStrategy({}, fails_for=frozenset({f"q{i}" for i in range(1, 8)}))
    judgments = [_judgment(f"q{i}", ART_1) for i in range(1, 8)]

    result = evaluate_strategy(strategy, judgments, k=5)

    assert len(result.records) == 7
    skipped = [r for r in result.records if r.status == "skipped"]
    assert len(skipped) == 2
    assert {r.question_id for r in skipped} == {"q6", "q7"}
