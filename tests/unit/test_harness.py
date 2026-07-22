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

    def __init__(self, hits_by_question: dict[str, list[str]]) -> None:
        self._hits_by_question = hits_by_question

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
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

    metrics = evaluate_strategy(strategy, judgments, k=5)

    assert metrics["recall_at_k"] == 0.5
    assert metrics["n"] == 2.0


def test_evaluate_strategy_excludes_unanswerable_questions_from_the_average() -> None:
    """An unanswerable-class judgment (no relevant refs) doesn't dilute the average."""
    strategy = _FakeStrategy({"q1": [ART_1], "q2": []})
    judgments = [_judgment("q1", ART_1), _judgment("q2", unanswerable=True)]

    metrics = evaluate_strategy(strategy, judgments, k=5)

    assert metrics["recall_at_k"] == 1.0
    assert metrics["n"] == 1.0


def test_evaluate_strategy_reports_k_in_the_result() -> None:
    """The k used for the run is echoed back in the result dict."""
    strategy = _FakeStrategy({"q1": [ART_1]})
    metrics = evaluate_strategy(strategy, [_judgment("q1", ART_1)], k=3)
    assert metrics["k"] == 3.0


def test_evaluate_strategy_raises_for_an_empty_judgment_list() -> None:
    """An empty golden set is a caller error, not a silently meaningless 0.0 result."""
    with pytest.raises(ValueError, match="judgments must not be empty"):
        evaluate_strategy(_FakeStrategy({}), [], k=5)
