"""Aggregate evaluation harness: run a strategy against a judgment set (ADR-0002/0003/0017)."""

from dataclasses import dataclass, field
from statistics import mean

from ragforge.domain.models import Judgment
from ragforge.domain.protocols import RetrievalStrategy
from ragforge.evaluation.lineage_ports import RetrievalCandidateLineage
from ragforge.evaluation.metrics.drm import document_level_retrieval_mismatch
from ragforge.evaluation.metrics.relevance import mrr, ndcg_at_k, precision_at_k, recall_at_k
from ragforge.evaluation.records import RetrievalRecord

_MAX_CONSECUTIVE_ERRORS = 5


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate retrieval metrics plus one RetrievalRecord per judgment (ADR-0012).

    ``candidate_lineage`` (ADR-0017) is additive and empty by default -
    every existing caller/test from Increment 1 keeps working unchanged;
    only run.py (Increment 7) passes ``embedding_identity_hash`` to
    populate it.
    """

    metrics: dict[str, float]
    records: list[RetrievalRecord]
    candidate_lineage: list[RetrievalCandidateLineage] = field(default_factory=list)


def evaluate_strategy(
    strategy: RetrievalStrategy,
    judgments: list[Judgment],
    k: int = 5,
    embedding_identity_hash: str | None = None,
) -> EvaluationResult:
    """Run ``strategy`` against every judgment's query; average metrics and record every outcome.

    Every judgment produces exactly one RetrievalRecord, including
    unanswerable-class questions (no relevant refs) - the ADR-0012
    requirement that a selected question is never silently dropped from
    coverage. Their ranking metrics stay out of the aggregate average
    (Recall/Precision/nDCG/MRR are all trivially 0.0 without a positive
    class to measure against, which would just dilute the comparison rather
    than say anything about ranking quality) but their record still exists,
    with an empty ``metrics`` dict and an explicit "succeeded"/"failed"
    ``status``. ``n`` reports how many judgments contributed to the average,
    not how many were selected - see ``len(records)`` for the latter.

    A retrieve() failure for one question (e.g. a transient embedding-API
    error) is counted in "errors" and excluded from the averages rather than
    aborting the whole strategy - except that _MAX_CONSECUTIVE_ERRORS
    consecutive failures is treated as a systemic problem (e.g. depleted API
    credits, not one flaky question) and stops attempting the remaining
    questions rather than retrying a run that cannot succeed. Those
    unattempted questions still get a "skipped" record rather than silently
    disappearing from coverage.

    ``embedding_identity_hash`` (ADR-0017), when given, populates
    ``EvaluationResult.candidate_lineage`` with one RetrievalCandidateLineage
    per candidate a successful retrieve() call returns - rank is the
    candidate's position in that returned list. Left ``None`` (the default),
    no lineage is collected - existing callers that don't need it pay
    nothing extra.

    Raises:
        ValueError: If judgments is empty.
    """
    if not judgments:
        raise ValueError("judgments must not be empty")

    recalls: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    drms: list[float] = []
    errors = 0
    consecutive_errors = 0
    aborted = False
    records: list[RetrievalRecord] = []
    candidate_lineage: list[RetrievalCandidateLineage] = []

    for judgment in judgments:
        query_class = judgment.query.query_class.value if judgment.query.query_class else None
        unanswerable = not judgment.relevant_refs

        if aborted:
            records.append(
                RetrievalRecord(
                    question_id=judgment.question_id,
                    query_class=query_class,
                    unanswerable=unanswerable,
                    status="skipped",
                    retrieved_structural_ids=(),
                    metrics={},
                    error="not attempted: strategy aborted after consecutive failures",
                )
            )
            continue

        try:
            results = strategy.retrieve(judgment.query, top_k=k)
        except Exception as exc:
            errors += 1
            consecutive_errors += 1
            records.append(
                RetrievalRecord(
                    question_id=judgment.question_id,
                    query_class=query_class,
                    unanswerable=unanswerable,
                    status="failed",
                    retrieved_structural_ids=(),
                    metrics={},
                    error=str(exc),
                )
            )
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                aborted = True
            continue

        consecutive_errors = 0
        if embedding_identity_hash is not None:
            candidate_lineage.extend(
                RetrievalCandidateLineage(
                    query_id=judgment.question_id,
                    strategy=strategy.name,
                    embedding_identity_hash=embedding_identity_hash,
                    candidate_rank=rank,
                    chunk_id=result.chunk.chunk_id,
                    structural_ids=result.chunk.structural_ids,
                    raw_score=result.score,
                )
                for rank, result in enumerate(results)
            )
        retrieved_ids = tuple(
            dict.fromkeys(ref for result in results for ref in result.chunk.structural_ids)
        )
        question_metrics: dict[str, float] = {}
        if not unanswerable:
            question_metrics = {
                "recall_at_k": recall_at_k(results, judgment, k),
                "precision_at_k": precision_at_k(results, judgment, k),
                "ndcg_at_k": ndcg_at_k(results, judgment, k),
                "mrr": mrr(results, judgment),
                "drm_at_k": document_level_retrieval_mismatch(results, judgment, k),
            }
            recalls.append(question_metrics["recall_at_k"])
            precisions.append(question_metrics["precision_at_k"])
            ndcgs.append(question_metrics["ndcg_at_k"])
            reciprocal_ranks.append(question_metrics["mrr"])
            drms.append(question_metrics["drm_at_k"])

        records.append(
            RetrievalRecord(
                question_id=judgment.question_id,
                query_class=query_class,
                unanswerable=unanswerable,
                status="succeeded",
                retrieved_structural_ids=retrieved_ids,
                metrics=question_metrics,
            )
        )

    metrics = {
        "recall_at_k": mean(recalls) if recalls else 0.0,
        "precision_at_k": mean(precisions) if precisions else 0.0,
        "ndcg_at_k": mean(ndcgs) if ndcgs else 0.0,
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "drm_at_k": mean(drms) if drms else 0.0,
        "k": float(k),
        "n": float(len(recalls)),
        "errors": float(errors),
    }
    return EvaluationResult(metrics=metrics, records=records, candidate_lineage=candidate_lineage)
