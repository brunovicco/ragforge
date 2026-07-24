"""Aggregate answer-quality evaluation harness: generate then score (ADR-0007).

Companion to evaluate_strategy (harness.py), which only measures retrieval
ranking. This module additionally generates a cited answer per question and
scores it for Citation Accuracy (deterministic) and, via RagasJudge,
Faithfulness and Answer Relevancy.
"""

import threading
from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass
from statistics import mean
from typing import Protocol, runtime_checkable

from ragforge.domain.models import Answer, Judgment, RetrievalResult
from ragforge.domain.protocols import RetrievalStrategy
from ragforge.evaluation.metrics.citation import citation_accuracy
from ragforge.evaluation.records import AnswerRecord
from ragforge.evaluation.scheduler import run_bounded
from ragforge.generation.ports import AnswerGenerator

_MAX_CONSECUTIVE_ERRORS = 5
_DEFAULT_MAX_WORKERS = 5


@runtime_checkable
class AnswerJudge(Protocol):
    """Shape required of a RagasJudge (or a test fake) by this harness."""

    def score(self, query_text: str, contexts: list[str], answer_text: str) -> dict[str, float]:
        """Return at least ``{"faithfulness": ..., "answer_relevancy": ...}`` for the triple."""
        ...


@dataclass(frozen=True, slots=True)
class AnswerEvaluationResult:
    """Aggregate answer-quality metrics plus one AnswerRecord per answerable judgment (ADR-0012).

    Unanswerable-class judgments never appear here: Citation Accuracy has
    nothing to check citations against for them, so they are not scored -
    downstream merging (``ragforge.evaluation.records.merge_question_records``)
    treats their absence as "not_applicable", not as a dropped question.
    """

    metrics: dict[str, float]
    records: list[AnswerRecord]


def evaluate_answer_quality(
    strategy: RetrievalStrategy,
    judgments: list[Judgment],
    generator: AnswerGenerator,
    judge_factory: Callable[[], AnswerJudge],
    k: int = 5,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> AnswerEvaluationResult:
    """Generate an answer per judgment's query and average Citation Accuracy/Faithfulness/Relevancy.

    Mirrors evaluate_strategy's judgment filtering: judgments with no
    relevant refs (unanswerable-class questions) are excluded, since
    Citation Accuracy has nothing to check citations against for them.

    Retrieval runs sequentially, one judgment at a time: strategies here can
    share a store with a single non-thread-safe connection (e.g.
    DenseChunkStore's one psycopg connection), which a thread pool would hit
    concurrently and unsafely. Answer generation and judge scoring - the
    actual bottleneck, at multiple sequential LLM round-trips per question -
    then run concurrently across up to ``max_workers`` questions at once,
    since each question's Gemini calls are independent HTTP requests.

    ``judge_factory`` builds one judge, not many: ``generator`` (a plain
    synchronous HTTP client) is shared safely across worker threads, but
    RagasJudge.score() calls ragas's sync wrapper, which internally does
    ``asyncio.run(self.ascore(...))`` - a fresh event loop per call, reusing
    the same async client underneath. Calling that concurrently from
    multiple threads against one shared judge corrupts its connection pool
    (observed for real: thousands of leaked CLOSE_WAIT sockets and a stalled
    run). Each worker thread instead lazily builds and caches its own judge
    via ``judge_factory``, so no judge instance is ever touched from more
    than one thread.

    A retrieval, generation, or judge-scoring failure for one question is
    counted in "answer_errors" and excluded from the averages rather than
    aborting the whole strategy - except that _MAX_CONSECUTIVE_ERRORS
    consecutive failures is treated as a systemic problem (e.g. depleted API
    credits, not one flaky question) and stops attempting the remaining
    questions rather than retrying a run that cannot succeed. Every
    answerable question still gets exactly one AnswerRecord - "failed" for
    an actual error, "skipped" for a question left unattempted after the
    circuit breaker trips - so none silently disappear from coverage.
    "answer_n" reports how many questions were actually scored, deliberately
    separate from evaluate_strategy's "n" (retrieval ranking is scored
    independently and keeps its own count).

    Scoring uses ``ragforge.evaluation.scheduler.run_bounded`` (ADR-0014):
    workers may still finish generation+judge calls in any order, but the
    returned ``records`` are always restored to ``retrieved``'s canonical
    order before this function returns - workers=1 and workers>1 produce
    identical record order and content, never just equivalent aggregates.
    The circuit breaker (consecutive scoring failures) observes outcomes in
    completion order via ``run_bounded``'s ``on_result`` callback, since that
    is inherently about real-time failure velocity, not artifact order.

    Raises:
        ValueError: If judgments is empty.
    """
    if not judgments:
        raise ValueError("judgments must not be empty")

    scored_judgments = [judgment for judgment in judgments if judgment.relevant_refs]

    errors = 0
    consecutive_errors = 0
    retrieval_aborted = False
    records: list[AnswerRecord] = []
    retrieved: list[tuple[Judgment, list[RetrievalResult]]] = []
    for judgment in scored_judgments:
        if retrieval_aborted:
            records.append(
                AnswerRecord(
                    question_id=judgment.question_id,
                    status="skipped",
                    answer_text=None,
                    answer_citations=(),
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
                AnswerRecord(
                    question_id=judgment.question_id,
                    status="failed",
                    answer_text=None,
                    answer_citations=(),
                    metrics={},
                    error=str(exc),
                )
            )
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                retrieval_aborted = True
            continue
        consecutive_errors = 0
        retrieved.append((judgment, results))

    thread_local = threading.local()

    def _judge_for_this_thread() -> AnswerJudge:
        judge = getattr(thread_local, "judge", None)
        if judge is None:
            judge = judge_factory()
            thread_local.judge = judge
        return judge

    def _score_one(
        judgment: Judgment, results: list[RetrievalResult]
    ) -> tuple[Answer, dict[str, float]]:
        answer = generator.generate(judgment.query, results)
        judged = _judge_for_this_thread().score(
            query_text=judgment.query.text,
            contexts=[result.chunk.source_text for result in results],
            answer_text=answer.text,
        )
        question_metrics = {
            "citation_accuracy": citation_accuracy(answer, judgment),
            "faithfulness": judged["faithfulness"],
            "answer_relevancy": judged["answer_relevancy"],
        }
        return answer, question_metrics

    citation_accuracies: list[float] = []
    faithfulness_scores: list[float] = []
    answer_relevancy_scores: list[float] = []
    if retrieved and not retrieval_aborted:
        consecutive_scoring_errors = 0

        def _on_scoring_result(
            _index: int,
            _value: tuple[Answer, dict[str, float]] | None,
            exc: BaseException | None,
        ) -> bool:
            """Track consecutive scoring failures; request cancellation past the threshold.

            A fresh counter from the retrieval phase's: a scoring failure
            (generation/judge) is a different failure mode than a retrieval
            failure, so one doesn't inflate the other's circuit breaker.
            """
            nonlocal consecutive_scoring_errors
            if exc is None:
                consecutive_scoring_errors = 0
                return False
            consecutive_scoring_errors += 1
            return consecutive_scoring_errors >= _MAX_CONSECUTIVE_ERRORS

        outcomes = run_bounded(
            retrieved,
            lambda pair: _score_one(pair[0], pair[1]),
            max_workers=max_workers,
            on_result=_on_scoring_result,
        )
        for (judgment, _results), outcome in zip(retrieved, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                is_cancelled = isinstance(outcome, CancelledError)
                status = "skipped" if is_cancelled else "failed"
                if not is_cancelled:
                    errors += 1
                records.append(
                    AnswerRecord(
                        question_id=judgment.question_id,
                        status=status,
                        answer_text=None,
                        answer_citations=(),
                        metrics={},
                        error=str(outcome) or outcome.__class__.__name__,
                    )
                )
                continue
            answer, question_metrics = outcome
            citation_accuracies.append(question_metrics["citation_accuracy"])
            faithfulness_scores.append(question_metrics["faithfulness"])
            answer_relevancy_scores.append(question_metrics["answer_relevancy"])
            records.append(
                AnswerRecord(
                    question_id=judgment.question_id,
                    status="succeeded",
                    answer_text=answer.text,
                    answer_citations=answer.citations,
                    metrics=question_metrics,
                )
            )

    metrics = {
        "citation_accuracy": mean(citation_accuracies) if citation_accuracies else 0.0,
        "faithfulness": mean(faithfulness_scores) if faithfulness_scores else 0.0,
        "answer_relevancy": mean(answer_relevancy_scores) if answer_relevancy_scores else 0.0,
        "answer_n": float(len(citation_accuracies)),
        "answer_errors": float(errors),
    }
    return AnswerEvaluationResult(metrics=metrics, records=records)
