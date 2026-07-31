"""Aggregate answer-quality evaluation harness: generate then score (ADR-0007/ADR-0018).

Companion to evaluate_strategy (harness.py), which only measures retrieval
ranking. This module additionally generates a cited answer per question and
scores it for Citation Accuracy (deterministic) and, via the ADR-0018
AnswerQualityJudge port, Faithfulness, Answer Relevancy, and abstention
appropriateness.
"""

import threading
from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass
from statistics import mean
from typing import Protocol, runtime_checkable

from ragforge.domain.models import Answer, Judgment, RetrievalResult
from ragforge.domain.protocols import RetrievalStrategy
from ragforge.evaluation.judge_ports import AnswerQualityJudge, JudgeSample
from ragforge.evaluation.metrics.citation import citation_accuracy
from ragforge.evaluation.records import AnswerRecord
from ragforge.evaluation.scheduler import run_bounded
from ragforge.generation.ports import AnswerGenerator

_MAX_CONSECUTIVE_ERRORS = 5
_DEFAULT_MAX_WORKERS = 5


@runtime_checkable
class _ClosableJudge(Protocol):
    """Optional lifecycle exposed by judges that own network clients."""

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AnswerEvaluationResult:
    """Aggregate answer-quality metrics plus one AnswerRecord per attempted judgment (ADR-0012).

    Unanswerable-class questions get records too (ADR-0018: abstention needs
    them judged); only Citation Accuracy stays absent for them.
    """

    metrics: dict[str, float]
    records: list[AnswerRecord]


def evaluate_answer_quality(
    strategy: RetrievalStrategy,
    judgments: list[Judgment],
    generator: AnswerGenerator,
    judge_factory: Callable[[], AnswerQualityJudge],
    k: int = 5,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> AnswerEvaluationResult:
    """Generate an answer per judgment's query and average Citation Accuracy/Faithfulness/Relevancy.

    Every judgment - including unanswerable-class ones - goes through
    retrieve/generate/judge (ADR-0018); Citation Accuracy alone is skipped
    without relevant refs. Retrieval runs sequentially (stores may hold one
    non-thread-safe connection); generation + judging, the real bottleneck,
    run concurrently across up to ``max_workers`` questions. The generator
    is shared; each worker builds its own judge via ``judge_factory``
    (RagasJudge's event loop must not be shared), closed after the pool
    finishes. Per-question failures are counted in "answer_errors" and
    excluded from averages; _MAX_CONSECUTIVE_ERRORS consecutive failures is
    treated as systemic and stops the remaining questions.

    Raises:
        ValueError: If judgments is empty.
    """
    if not judgments:
        raise ValueError("judgments must not be empty")

    errors = 0
    consecutive_errors = 0
    retrieval_aborted = False
    records: list[AnswerRecord] = []
    retrieved: list[tuple[Judgment, list[RetrievalResult]]] = []
    for judgment in judgments:
        if retrieval_aborted:
            records.append(
                AnswerRecord(
                    question_id=judgment.question_id,
                    status="skipped",
                    answer_text=None,
                    answer_citations=(),
                    judge_contexts=(),
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
                    judge_contexts=(),
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
    judges: list[AnswerQualityJudge] = []
    judges_lock = threading.Lock()

    def _judge_for_this_thread() -> AnswerQualityJudge:
        judge = getattr(thread_local, "judge", None)
        if judge is None:
            judge = judge_factory()
            thread_local.judge = judge
            with judges_lock:
                judges.append(judge)
        return judge

    def _score_one(
        judgment: Judgment, results: list[RetrievalResult]
    ) -> tuple[Answer, dict[str, float]]:
        answer = generator.generate(judgment.query, results)
        sample = JudgeSample(
            question=judgment.query.text,
            contexts=tuple(result.chunk.source_text for result in results),
            answer=answer.text,
            query_class=judgment.query.query_class.value if judgment.query.query_class else None,
            unanswerable=not judgment.relevant_refs,
        )
        judged = _judge_for_this_thread().evaluate(sample)
        question_metrics = {
            "faithfulness": judged.faithfulness.score,
            "answer_relevancy": judged.answer_relevancy.score,
            "abstention_appropriate": 1.0 if judged.abstention.appropriate else 0.0,
        }
        if judgment.relevant_refs:
            question_metrics["citation_accuracy"] = citation_accuracy(answer, judgment)
        return answer, question_metrics

    citation_accuracies: list[float] = []
    faithfulness_scores: list[float] = []
    answer_relevancy_scores: list[float] = []
    abstention_scores: list[float] = []
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

        try:
            outcomes = run_bounded(
                retrieved,
                lambda pair: _score_one(pair[0], pair[1]),
                max_workers=max_workers,
                on_result=_on_scoring_result,
            )
        finally:
            for judge in judges:
                if isinstance(judge, _ClosableJudge):
                    judge.close()
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
                        judge_contexts=(),
                        metrics={},
                        error=str(outcome) or outcome.__class__.__name__,
                    )
                )
                continue
            answer, question_metrics = outcome
            if "citation_accuracy" in question_metrics:
                citation_accuracies.append(question_metrics["citation_accuracy"])
            faithfulness_scores.append(question_metrics["faithfulness"])
            answer_relevancy_scores.append(question_metrics["answer_relevancy"])
            abstention_scores.append(question_metrics["abstention_appropriate"])
            records.append(
                AnswerRecord(
                    question_id=judgment.question_id,
                    status="succeeded",
                    answer_text=answer.text,
                    answer_citations=answer.citations,
                    judge_contexts=tuple(result.chunk.source_text for result in _results),
                    metrics=question_metrics,
                )
            )

    metrics = {
        "citation_accuracy": mean(citation_accuracies) if citation_accuracies else 0.0,
        "faithfulness": mean(faithfulness_scores) if faithfulness_scores else 0.0,
        "answer_relevancy": mean(answer_relevancy_scores) if answer_relevancy_scores else 0.0,
        "abstention_appropriate": mean(abstention_scores) if abstention_scores else 0.0,
        "citation_n": float(len(citation_accuracies)),
        "answer_n": float(len(faithfulness_scores)),
        "answer_errors": float(errors),
    }
    return AnswerEvaluationResult(metrics=metrics, records=records)
