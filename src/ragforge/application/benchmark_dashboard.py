"""Framework-independent view models for the analytical dashboard."""

from dataclasses import dataclass

from ragforge.application.benchmark_results import PublishedBenchmarkRun


@dataclass(frozen=True, slots=True)
class StrategyMetricRow:
    """Represent one display row in the benchmark comparison."""

    strategy: str
    recall_at_5: float
    precision_at_5: float
    ndcg_at_5: float
    mrr: float
    document_mismatch_at_5: float
    citation_accuracy: float
    faithfulness: float
    answer_relevancy: float
    abstention: float


def strategy_metric_rows(run: PublishedBenchmarkRun) -> tuple[StrategyMetricRow, ...]:
    """Build dashboard rows sorted by nDCG and MRR, descending."""
    rows = (
        StrategyMetricRow(
            strategy=metric.strategy,
            recall_at_5=metric.recall_at_k,
            precision_at_5=metric.precision_at_k,
            ndcg_at_5=metric.ndcg_at_k,
            mrr=metric.mrr,
            document_mismatch_at_5=metric.drm_at_k,
            citation_accuracy=metric.citation_accuracy,
            faithfulness=metric.faithfulness,
            answer_relevancy=metric.answer_relevancy,
            abstention=metric.abstention_appropriate,
        )
        for metric in run.metrics
    )
    return tuple(sorted(rows, key=lambda row: (row.ndcg_at_5, row.mrr), reverse=True))
