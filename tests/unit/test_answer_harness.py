"""Tests for the answer-quality evaluation harness (ADR-0007/ADR-0018), using fakes."""

import threading
import time

import pytest

from ragforge.domain.models import (
    Answer,
    Chunk,
    JudgedRef,
    Judgment,
    Query,
    QueryClass,
    RelevanceGrade,
    RetrievalResult,
    StructuralRef,
)
from ragforge.evaluation.answer_harness import evaluate_answer_quality
from ragforge.evaluation.judge_ports import (
    AbstentionJudgment,
    JudgeResult,
    JudgeSample,
    MetricScore,
    ModelIdentity,
)
from ragforge.generation.errors import GenerationError

ART_1 = "NORM::art-1"
ART_2 = "NORM::art-2"

_FAKE_IDENTITY = ModelIdentity(
    provider="fake", model="fake-model", reasoning_effort=None, output_schema_version=1
)


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


def _judge_result(
    faithfulness: float = 1.0, answer_relevancy: float = 1.0, *, abstention_appropriate: bool = True
) -> JudgeResult:
    return JudgeResult(
        schema_version=1,
        faithfulness=MetricScore(score=faithfulness),
        answer_relevancy=MetricScore(score=answer_relevancy),
        abstention=AbstentionJudgment(appropriate=abstention_appropriate, rationale=""),
    )


class _FakeStrategy:
    name = "fake"

    def __init__(self, fails_for: frozenset[str] = frozenset()) -> None:
        self._fails_for = fails_for

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        if query.text in self._fails_for:
            raise RuntimeError("simulated retrieval failure")
        return [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id=ART_1,
                    source_text="chunk text",
                    retrieval_text="chunk text",
                    structural_ids=(ART_1,),
                ),
                score=1.0,
                strategy="fake",
            )
        ]


class _FakeGenerator:
    name = "fake-generator"

    def __init__(self, answers_by_question: dict[str, Answer]) -> None:
        self._answers_by_question = answers_by_question

    def generate(self, query: Query, results: list[RetrievalResult]) -> Answer:
        return self._answers_by_question[query.text]


class _FakeJudge:
    identity = _FAKE_IDENTITY

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls: list[JudgeSample] = []

    def evaluate(self, sample: JudgeSample) -> JudgeResult:
        self.calls.append(sample)
        return _judge_result(self._scores["faithfulness"], self._scores["answer_relevancy"])


class _FlakyJudge:
    """Raises GenerationError for the given query texts, scores everything else normally."""

    identity = _FAKE_IDENTITY

    def __init__(self, scores: dict[str, float], fails_for: set[str]) -> None:
        self._scores = scores
        self._fails_for = fails_for

    def evaluate(self, sample: JudgeSample) -> JudgeResult:
        if sample.question in self._fails_for:
            raise GenerationError("judge model did not return a function call")
        return _judge_result(self._scores["faithfulness"], self._scores["answer_relevancy"])


class _ConcurrentTrackingJudge:
    """Records the highest number of evaluate() calls observed running at once."""

    identity = _FAKE_IDENTITY

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self._lock = threading.Lock()
        self._active = 0
        self.max_concurrent = 0

    def evaluate(self, sample: JudgeSample) -> JudgeResult:
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        time.sleep(0.05)
        with self._lock:
            self._active -= 1
        return _judge_result(self._scores["faithfulness"], self._scores["answer_relevancy"])


class _FactoryTrackingJudge:
    """Records every thread id that ever called evaluate() on this particular instance."""

    identity = _FAKE_IDENTITY

    def __init__(self) -> None:
        self.thread_ids: set[int] = set()

    def evaluate(self, sample: JudgeSample) -> JudgeResult:
        self.thread_ids.add(threading.get_ident())
        time.sleep(0.02)
        return _judge_result()


def test_evaluate_answer_quality_averages_citation_accuracy_across_judgments() -> None:
    """One perfectly-cited answer and one uncited answer average to 0.5 citation accuracy."""
    generator = _FakeGenerator(
        {
            "q1": Answer(text="answer 1", citations=(ART_1,)),
            "q2": Answer(text="answer 2", citations=()),
        }
    )
    judge = _FakeJudge({"faithfulness": 0.8, "answer_relevancy": 0.9})
    judgments = [_judgment("q1", ART_1), _judgment("q2", ART_1)]

    result = evaluate_answer_quality(_FakeStrategy(), judgments, generator, lambda: judge, k=5)

    assert result.metrics["citation_accuracy"] == 0.5
    assert result.metrics["faithfulness"] == pytest.approx(0.8)
    assert result.metrics["answer_relevancy"] == pytest.approx(0.9)
    assert result.metrics["answer_n"] == 2.0
    assert result.metrics["citation_n"] == 2.0
    assert result.metrics["answer_errors"] == 0.0
    assert len(result.records) == 2
    assert all(record.status == "succeeded" for record in result.records)


def test_evaluate_answer_quality_still_scores_unanswerable_questions() -> None:
    """An unanswerable-class judgment is generated and judged too (ADR-0018), unlike before.

    Only Citation Accuracy stays absent for it - abstention appropriateness
    needs unanswerable questions to actually reach the judge to mean
    anything.
    """
    generator = _FakeGenerator(
        {
            "q1": Answer(text="answer", citations=(ART_1,)),
            "q2": Answer(text="não há evidência suficiente", citations=()),
        }
    )
    judge = _FakeJudge({"faithfulness": 1.0, "answer_relevancy": 1.0})
    judgments = [_judgment("q1", ART_1), _judgment("q2", unanswerable=True)]

    result = evaluate_answer_quality(_FakeStrategy(), judgments, generator, lambda: judge, k=5)

    assert result.metrics["answer_n"] == 2.0
    assert result.metrics["citation_n"] == 1.0
    assert len(judge.calls) == 2
    assert len(result.records) == 2
    unanswerable_record = next(r for r in result.records if r.question_id == "q2")
    assert unanswerable_record.status == "succeeded"
    assert "citation_accuracy" not in unanswerable_record.metrics
    assert "abstention_appropriate" in unanswerable_record.metrics


def test_evaluate_answer_quality_marks_the_judge_sample_unanswerable_correctly() -> None:
    """JudgeSample.unanswerable reflects whether the judgment has a relevant-ref set."""
    generator = _FakeGenerator(
        {
            "q1": Answer(text="answer", citations=(ART_1,)),
            "q2": Answer(text="não sei", citations=()),
        }
    )
    judge = _FakeJudge({"faithfulness": 1.0, "answer_relevancy": 1.0})
    judgments = [_judgment("q1", ART_1), _judgment("q2", unanswerable=True)]

    evaluate_answer_quality(_FakeStrategy(), judgments, generator, lambda: judge, k=5)

    by_question = {sample.question: sample for sample in judge.calls}
    assert by_question["q1"].unanswerable is False
    assert by_question["q2"].unanswerable is True


def test_evaluate_answer_quality_passes_retrieved_chunk_text_as_judge_contexts() -> None:
    """The judge is called with the retrieved chunks' text, not the judgment's refs."""
    generator = _FakeGenerator({"q1": Answer(text="answer", citations=(ART_1,))})
    judge = _FakeJudge({"faithfulness": 1.0, "answer_relevancy": 1.0})

    evaluate_answer_quality(
        _FakeStrategy(), [_judgment("q1", ART_1)], generator, lambda: judge, k=5
    )

    assert judge.calls[0].question == "q1"
    assert judge.calls[0].contexts == ("chunk text",)
    assert judge.calls[0].answer == "answer"


def test_evaluate_answer_quality_raises_for_an_empty_judgment_list() -> None:
    """An empty golden set is a caller error, not a silently meaningless 0.0 result."""
    with pytest.raises(ValueError, match="judgments must not be empty"):
        evaluate_answer_quality(
            _FakeStrategy(), [], _FakeGenerator({}), lambda: _FakeJudge({}), k=5
        )


def test_evaluate_answer_quality_survives_a_single_judge_failure() -> None:
    """One question whose judge call raises GenerationError doesn't abort the whole strategy.

    Regression test: a live benchmark-v01 run crashed entirely because
    RAGAS's Faithfulness metric raised GenerationError (the judge model
    didn't return the structured output instructor expected) for one
    question, and that exception propagated uncaught through the whole
    8-strategy x 219-question run.
    """
    generator = _FakeGenerator(
        {
            "q1": Answer(text="answer 1", citations=(ART_1,)),
            "q2": Answer(text="answer 2", citations=(ART_1,)),
            "q3": Answer(text="answer 3", citations=(ART_1,)),
        }
    )
    judge = _FlakyJudge({"faithfulness": 1.0, "answer_relevancy": 1.0}, fails_for={"q2"})
    judgments = [_judgment("q1", ART_1), _judgment("q2", ART_1), _judgment("q3", ART_1)]

    result = evaluate_answer_quality(_FakeStrategy(), judgments, generator, lambda: judge, k=5)

    assert result.metrics["answer_n"] == 2.0
    assert result.metrics["answer_errors"] == 1.0
    assert result.metrics["faithfulness"] == pytest.approx(1.0)
    failed_record = next(r for r in result.records if r.question_id == "q2")
    assert failed_record.status == "failed"


def test_evaluate_answer_quality_survives_a_single_retrieval_failure() -> None:
    """One question whose retrieve() raises doesn't abort the whole strategy.

    Regression test: a live benchmark-v01 run crashed entirely because a
    dense strategy's embedding call raised EmbeddingError (API credits
    depleted) for one question inside this harness's retrieval step, and
    that exception propagated uncaught through the whole run.
    """
    generator = _FakeGenerator(
        {
            "q1": Answer(text="answer 1", citations=(ART_1,)),
            "q3": Answer(text="answer 3", citations=(ART_1,)),
        }
    )
    judge = _FakeJudge({"faithfulness": 1.0, "answer_relevancy": 1.0})
    strategy = _FakeStrategy(fails_for=frozenset({"q2"}))
    judgments = [_judgment("q1", ART_1), _judgment("q2", ART_1), _judgment("q3", ART_1)]

    result = evaluate_answer_quality(strategy, judgments, generator, lambda: judge, k=5)

    assert result.metrics["answer_n"] == 2.0
    assert result.metrics["answer_errors"] == 1.0
    failed_record = next(r for r in result.records if r.question_id == "q2")
    assert failed_record.status == "failed"
    assert failed_record.error == "simulated retrieval failure"


def test_evaluate_answer_quality_stops_early_after_consecutive_retrieval_failures() -> None:
    """Five-in-a-row failures (a systemic problem) stop the strategy rather than retry forever."""
    strategy = _FakeStrategy(fails_for=frozenset({f"q{i}" for i in range(1, 8)}))
    judgments = [_judgment(f"q{i}", ART_1) for i in range(1, 8)]

    result = evaluate_answer_quality(
        strategy, judgments, _FakeGenerator({}), lambda: _FakeJudge({}), k=5
    )

    assert result.metrics["answer_n"] == 0.0
    assert result.metrics["answer_errors"] == 5.0
    assert len(result.records) == 7
    skipped = [r for r in result.records if r.status == "skipped"]
    assert len(skipped) == 2
    assert {r.question_id for r in skipped} == {"q6", "q7"}


class _FlakyGenerator:
    """Raises GenerationError for the given query texts, generates normally otherwise."""

    name = "flaky-generator"

    def __init__(self, answers_by_question: dict[str, Answer], fails_for: frozenset[str]) -> None:
        self._answers_by_question = answers_by_question
        self._fails_for = fails_for

    def generate(self, query: Query, results: list[RetrievalResult]) -> Answer:
        if query.text in self._fails_for:
            raise GenerationError("simulated generation failure")
        return self._answers_by_question[query.text]


def test_evaluate_answer_quality_stops_early_after_consecutive_scoring_failures() -> None:
    """Five-in-a-row generation failures stop scoring without inflating answer_errors.

    Mirrors the retrieval-phase circuit breaker test: a separate counter for
    the scoring phase (generation+judge), not shared with retrieval failures.
    """
    generator = _FlakyGenerator({}, fails_for=frozenset({f"q{i}" for i in range(1, 8)}))
    judgments = [_judgment(f"q{i}", ART_1) for i in range(1, 8)]

    result = evaluate_answer_quality(
        _FakeStrategy(), judgments, generator, lambda: _FakeJudge({}), k=5, max_workers=1
    )

    assert result.metrics["answer_n"] == 0.0
    assert result.metrics["answer_errors"] == 5.0
    assert len(result.records) == 7
    skipped = [r for r in result.records if r.status == "skipped"]
    assert len(skipped) == 2
    assert {r.question_id for r in skipped} == {"q6", "q7"}


class _VariableLatencyJudge:
    """Deterministic scores; sleeps per a per-question delay map to scramble completion order."""

    identity = _FAKE_IDENTITY

    def __init__(self, delays_by_question: dict[str, float]) -> None:
        self._delays = delays_by_question

    def evaluate(self, sample: JudgeSample) -> JudgeResult:
        time.sleep(self._delays.get(sample.question, 0.0))
        return _judge_result()


def test_evaluate_answer_quality_produces_identical_records_regardless_of_worker_count() -> None:
    """workers=1 and workers>1 produce identical record order and content (ADR-0014 exit gate).

    Delays are inverted (q0 slowest, q(n-1) fastest) so concurrent completion
    order is the exact reverse of input order - a real stress test of the
    ordering fix, not one that happens to pass by accident.
    """
    count = 6
    generator = _FakeGenerator(
        {f"q{i}": Answer(text=f"answer {i}", citations=(ART_1,)) for i in range(count)}
    )
    delays = {f"q{i}": (count - i) * 0.01 for i in range(count)}
    judgments = [_judgment(f"q{i}", ART_1) for i in range(count)]

    serial_result = evaluate_answer_quality(
        _FakeStrategy(),
        judgments,
        generator,
        lambda: _VariableLatencyJudge(delays),
        k=5,
        max_workers=1,
    )
    parallel_result = evaluate_answer_quality(
        _FakeStrategy(),
        judgments,
        generator,
        lambda: _VariableLatencyJudge(delays),
        k=5,
        max_workers=count,
    )

    expected_ids = [f"q{i}" for i in range(count)]
    assert [r.question_id for r in serial_result.records] == expected_ids
    assert [r.question_id for r in parallel_result.records] == expected_ids
    assert serial_result.metrics == parallel_result.metrics


def test_evaluate_answer_quality_scores_questions_concurrently() -> None:
    """Multiple questions are scored in parallel, not strictly one at a time."""
    count = 6
    generator = _FakeGenerator(
        {f"q{i}": Answer(text=f"answer {i}", citations=(ART_1,)) for i in range(count)}
    )
    judge = _ConcurrentTrackingJudge({"faithfulness": 1.0, "answer_relevancy": 1.0})
    judgments = [_judgment(f"q{i}", ART_1) for i in range(count)]

    evaluate_answer_quality(
        _FakeStrategy(), judgments, generator, lambda: judge, k=5, max_workers=3
    )

    assert judge.max_concurrent >= 2


def test_evaluate_answer_quality_never_shares_one_judge_instance_across_threads() -> None:
    """judge_factory builds a separate judge per worker thread, never shared across threads.

    Regression test: a real live run shared one RagasJudge across concurrent
    worker threads. Each RagasJudge.evaluate() call does asyncio.run(...)
    internally (a fresh event loop per call, reusing the same async client
    underneath); calling that concurrently from multiple threads against one
    shared judge corrupted its connection pool - observed for real as
    thousands of leaked CLOSE_WAIT sockets and a stalled run.
    """
    count = 10
    generator = _FakeGenerator(
        {f"q{i}": Answer(text=f"answer {i}", citations=(ART_1,)) for i in range(count)}
    )
    built_instances: list[_FactoryTrackingJudge] = []
    lock = threading.Lock()

    def factory() -> _FactoryTrackingJudge:
        with lock:
            judge = _FactoryTrackingJudge()
            built_instances.append(judge)
        return judge

    judgments = [_judgment(f"q{i}", ART_1) for i in range(count)]

    evaluate_answer_quality(_FakeStrategy(), judgments, generator, factory, k=5, max_workers=3)

    assert built_instances
    assert len(built_instances) <= 3, "at most one judge instance per worker thread"
    for judge in built_instances:
        assert len(judge.thread_ids) == 1, "a judge instance was used from more than one thread"
