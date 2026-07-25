"""Tests for benchmark dimension and usage reporting."""

from ragforge.domain.models import (
    JudgedRef,
    Judgment,
    Query,
    QueryClass,
    RelevanceGrade,
    StructuralRef,
)
from ragforge.evaluation.lineage_ports import GenerationLineage
from ragforge.evaluation.records import QuestionRecord
from ragforge.evaluation.run_reporting import (
    build_metric_breakdowns,
    summarize_generation_usage,
)


def _record(question_id: str, query_class: str, score: float) -> QuestionRecord:
    return QuestionRecord(
        question_id=question_id,
        query_class=query_class,
        strategy="dense",
        unanswerable=False,
        retrieval_status="succeeded",
        generation_status="succeeded",
        judge_status="succeeded",
        retrieved_structural_ids=(),
        answer_text="answer",
        answer_citations=(),
        metrics={"recall_at_k": score},
        errors=(),
    )


def test_metric_breakdowns_report_class_document_mean_and_coverage() -> None:
    """Breakdowns preserve denominators instead of publishing bare averages."""
    judgments = [
        Judgment(
            question_id="q1",
            query=Query("one", QueryClass.EXACT_FACTUAL),
            relevant_refs=(
                JudgedRef(
                    StructuralRef("NORM-1", "art-1"),
                    RelevanceGrade.RELEVANT,
                ),
            ),
        ),
        Judgment(
            question_id="q2",
            query=Query("two", QueryClass.EXACT_FACTUAL),
            relevant_refs=(
                JudgedRef(
                    StructuralRef("NORM-1", "art-2"),
                    RelevanceGrade.RELEVANT,
                ),
            ),
        ),
    ]

    breakdowns = build_metric_breakdowns(
        [
            _record("q1", "exact_factual", 1.0),
            _record("q2", "exact_factual", 0.0),
        ],
        judgments,
    )

    by_class = breakdowns["by_query_class"]
    assert isinstance(by_class, dict)
    exact = by_class["dense"]["exact_factual"]
    assert exact["selected"] == 2
    assert exact["metrics"]["recall_at_k"] == {"mean": 0.5, "n": 2}
    by_document = breakdowns["by_document"]
    assert isinstance(by_document, dict)
    assert by_document["dense"]["NORM-1"]["selected"] == 2


def test_generation_usage_computes_configured_cost() -> None:
    """Cost is calculated only from explicit per-million-token prices."""
    lineage = GenerationLineage(
        provider="gemini",
        model="model",
        prompt_hash="hash",
        input_chunk_ids=(),
        input_source_hashes=(),
        answer_hash="answer",
        parsed_citations=(),
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        total_tokens=1_500_000,
        latency_seconds=2.5,
        cache_hit=False,
    )

    usage = summarize_generation_usage(
        {"dense": [lineage]},
        input_price_per_million_usd=1.0,
        output_price_per_million_usd=2.0,
    )

    assert usage["dense"]["estimated_cost_usd"] == 2.0
    assert usage["dense"]["latency_seconds"] == 2.5
