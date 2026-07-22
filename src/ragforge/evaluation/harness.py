"""Aggregate evaluation harness: run a strategy against a judgment set (ADR-0002/0003)."""

from statistics import mean

from ragforge.domain.models import Judgment
from ragforge.domain.protocols import RetrievalStrategy
from ragforge.evaluation.metrics.relevance import mrr, ndcg_at_k, precision_at_k, recall_at_k


def evaluate_strategy(
    strategy: RetrievalStrategy, judgments: list[Judgment], k: int = 5
) -> dict[str, float]:
    """Run ``strategy`` against every judgment's query and average each metric.

    Judgments with no relevant refs (e.g. unanswerable-class questions) are
    excluded from the average: Recall/Precision/nDCG/MRR are all trivially
    0.0 without a positive class to measure against, which would just dilute
    the comparison rather than say anything about ranking quality. ``n``
    reports how many judgments were actually scored, so an empty golden set
    (or one made entirely of unanswerable questions) is visible in the
    result rather than silently producing a perfect-looking average of zero
    contributions.

    Raises:
        ValueError: If judgments is empty.
    """
    if not judgments:
        raise ValueError("judgments must not be empty")

    scored = [judgment for judgment in judgments if judgment.relevant_refs]
    recalls: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    for judgment in scored:
        results = strategy.retrieve(judgment.query, top_k=k)
        recalls.append(recall_at_k(results, judgment, k))
        precisions.append(precision_at_k(results, judgment, k))
        ndcgs.append(ndcg_at_k(results, judgment, k))
        reciprocal_ranks.append(mrr(results, judgment))

    return {
        "recall_at_k": mean(recalls) if recalls else 0.0,
        "precision_at_k": mean(precisions) if precisions else 0.0,
        "ndcg_at_k": mean(ndcgs) if ndcgs else 0.0,
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "k": float(k),
        "n": float(len(scored)),
    }
