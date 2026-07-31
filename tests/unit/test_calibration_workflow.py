"""Tests for the blind judge-calibration workflow (ADR-0007/ADR-0018)."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from ragforge.domain.models import (
    JudgedRef,
    Judgment,
    Query,
    QueryClass,
    RelevanceGrade,
    StructuralRef,
)
from ragforge.evaluation.calibration_workflow import (
    HumanLabel,
    SealedCalibration,
    build_calibration_report,
    build_calibration_samples,
    build_human_labels,
    build_sealed_calibration,
    calibration_feature_coverage,
    calibration_gate_passed,
    merge_existing_human_labels,
    render_calibration_worksheet,
    scores_have_ordinal_disagreement,
    select_calibration_records,
)
from ragforge.evaluation.judge_calibration import CalibrationSample
from ragforge.evaluation.records import QuestionRecord


def _judgment(question_id: str, query_class: QueryClass) -> Judgment:
    refs = [
        JudgedRef(StructuralRef("NORM", "art-1"), RelevanceGrade.RELEVANT),
    ]
    if query_class in {QueryClass.MULTI_HOP, QueryClass.SECTION_COMPARATIVE}:
        refs.append(JudgedRef(StructuralRef("NORM", "art-2"), RelevanceGrade.RELEVANT))
    return Judgment(
        question_id=question_id,
        query=Query(
            text=f"A regra não se aplica, salvo exceção de 30% para {question_id}?",
            query_class=query_class,
        ),
        relevant_refs=tuple(refs) if query_class is not QueryClass.UNANSWERABLE else (),
        reference_answer="Resposta de referência com 30%.",
    )


def _record(
    question_id: str,
    query_class: QueryClass,
    strategy: str,
    *,
    faithfulness: float = 1.0,
    relevancy: float = 1.0,
    citation_accuracy: float = 1.0,
    judge_contexts: tuple[str, ...] = ("exact judge context",),
) -> QuestionRecord:
    unanswerable = query_class is QueryClass.UNANSWERABLE
    metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "abstention_appropriate": 1.0,
    }
    if not unanswerable:
        metrics["citation_accuracy"] = citation_accuracy
    return QuestionRecord(
        question_id=question_id,
        query_class=query_class.value,
        strategy=strategy,
        unanswerable=unanswerable,
        retrieval_status="succeeded",
        generation_status="succeeded",
        judge_status="succeeded",
        retrieved_structural_ids=("NORM::art-1",),
        answer_text="Resposta não aplicável, salvo 30% [NORM::art-1].",
        answer_citations=("NORM::art-1",),
        judge_contexts=judge_contexts,
        metrics=metrics,
        errors=(),
    )


def _candidate_set(count: int = 36) -> tuple[list[QuestionRecord], dict[str, Judgment]]:
    query_classes = list(QueryClass)
    strategies = ["dense", "graphrag"]
    records = []
    judgments = {}
    for index in range(count):
        question_id = f"q{index:03d}"
        query_class = query_classes[index % len(query_classes)]
        judgment = _judgment(question_id, query_class)
        record = _record(question_id, query_class, strategies[index % len(strategies)])
        if index == 1:
            record = replace(record, metrics={**record.metrics, "answer_relevancy": 0.5})
        if index == 2 and not record.unanswerable:
            record = replace(record, metrics={**record.metrics, "citation_accuracy": 0.0})
        judgments[question_id] = judgment
        records.append(record)
    return records, judgments


@pytest.mark.parametrize("score", [-1.0, 0.25, 1.5, float("inf"), float("nan"), "1.0"])
def test_human_label_rejects_values_outside_the_documented_scale(score: object) -> None:
    with pytest.raises(ValidationError, match="human_score"):
        HumanLabel(
            sample_id="q1-dense-faithfulness",
            dimension="faithfulness",
            human_score=score,
        )


def test_human_label_rejects_dimension_suffix_mismatch() -> None:
    with pytest.raises(ValidationError, match="suffix"):
        HumanLabel(
            sample_id="q1-dense-faithfulness",
            dimension="answer_relevancy",
            human_score=1.0,
        )


@pytest.mark.parametrize("score", [-0.1, 1.1, float("inf"), float("nan"), "1.0"])
def test_sealed_calibration_rejects_invalid_judge_scores(score: object) -> None:
    with pytest.raises(ValidationError, match="judge score"):
        SealedCalibration(
            schema_version=1,
            evidence_schema_version=1,
            run_id="run-1",
            judge_scores={"q1-dense-faithfulness": score},
        )


def test_merge_existing_human_labels_preserves_completed_scores() -> None:
    expected = [
        HumanLabel(
            sample_id="q1-dense-faithfulness",
            dimension="faithfulness",
            human_score=None,
        )
    ]
    existing = [expected[0].model_copy(update={"human_score": 0.5})]

    merged = merge_existing_human_labels(expected, existing)

    assert merged[0].human_score == 0.5


def test_merge_existing_human_labels_rejects_a_different_sample() -> None:
    expected = [
        HumanLabel(
            sample_id="q1-dense-faithfulness",
            dimension="faithfulness",
            human_score=None,
        )
    ]
    existing = [
        HumanLabel(
            sample_id="q2-dense-faithfulness",
            dimension="faithfulness",
            human_score=1.0,
        )
    ]

    with pytest.raises(ValueError, match="different calibration sample"):
        merge_existing_human_labels(expected, existing)


def test_select_calibration_records_rejects_legacy_records_without_judge_contexts() -> None:
    records, judgments = _candidate_set(30)
    legacy = [replace(record, judge_contexts=()) for record in records]

    with pytest.raises(ValueError, match="Legacy runs must be rerun"):
        select_calibration_records(legacy, judgments, 30)


def test_select_calibration_records_is_deterministic_and_covers_required_features() -> None:
    records, judgments = _candidate_set()

    first = select_calibration_records(records, judgments, 30)
    second = select_calibration_records(list(reversed(records)), judgments, 30)

    first_ids = [(record.question_id, record.strategy) for record in first]
    second_ids = [(record.question_id, record.strategy) for record in second]
    assert first_ids == second_ids
    coverage = calibration_feature_coverage(first, judgments)
    assert {
        "answer_quality:complete",
        "answer_quality:partial",
        "citation:unsupported",
        "language:negation",
        "language:exception",
        "reasoning:cross_reference",
        "claim:numerical",
    } <= coverage


def test_render_calibration_worksheet_uses_exact_context_and_separates_citation_audit() -> None:
    record = replace(
        _record("q1", QueryClass.EXACT_FACTUAL, "dense"),
        answer_text="Resposta [NORM::art-1, NORM::art-2].",
        answer_citations=(),
        judge_contexts=("context the judge actually saw",),
    )
    judgment = _judgment("q1", QueryClass.EXACT_FACTUAL)

    worksheet = render_calibration_worksheet(
        [record],
        {"q1": judgment},
        {"NORM": {"NORM::art-1": "canonical source"}},
        "run-1",
    )

    assert "### Exact judge contexts" in worksheet
    assert "> context the judge actually saw" in worksheet
    assert "### Citation audit (not judge evidence)" in worksheet
    assert "malformed or grouped citation" in worksheet


def test_build_calibration_samples_requires_exact_label_score_alignment() -> None:
    records, _ = _candidate_set(30)
    labels = build_human_labels(records)
    completed = [label.model_copy(update={"human_score": 1.0}) for label in labels]
    sealed = build_sealed_calibration(records, "run-1")
    truncated_scores = {
        key: value for key, value in sealed.judge_scores.items() if key != labels[0].sample_id
    }
    misaligned = sealed.model_copy(update={"judge_scores": truncated_scores})

    with pytest.raises(ValueError, match="labels and sealed scores differ"):
        build_calibration_samples(completed, misaligned)


def test_build_calibration_samples_rejects_incomplete_labels() -> None:
    records, _ = _candidate_set(30)
    labels = build_human_labels(records)
    sealed = build_sealed_calibration(records, "run-1")

    with pytest.raises(ValueError, match="labels are still empty"):
        build_calibration_samples(labels, sealed)


def _agreement_samples(*, invert_faithfulness: bool) -> list[CalibrationSample]:
    samples = []
    for index in range(30):
        judge_score = float(index % 2)
        faithfulness_human = 1.0 - judge_score if invert_faithfulness else judge_score
        samples.extend(
            [
                CalibrationSample(
                    sample_id=f"q{index}-dense-faithfulness",
                    dimension="faithfulness",
                    judge_score=judge_score,
                    human_score=faithfulness_human,
                ),
                CalibrationSample(
                    sample_id=f"q{index}-dense-answer_relevancy",
                    dimension="answer_relevancy",
                    judge_score=judge_score,
                    human_score=judge_score,
                ),
            ]
        )
    return samples


def test_build_calibration_report_fails_when_one_required_dimension_misses_floor() -> None:
    report = build_calibration_report(_agreement_samples(invert_faithfulness=True), "run-1")

    assert calibration_gate_passed(report) is False
    gate = report["gate"]
    assert isinstance(gate, dict)
    assert "faithfulness" in gate["failed_results"]


def test_build_calibration_report_passes_for_perfect_dimension_agreement() -> None:
    report = build_calibration_report(_agreement_samples(invert_faithfulness=False), "run-1")

    assert calibration_gate_passed(report) is True


def test_scores_have_ordinal_disagreement_uses_kappa_thirds() -> None:
    sample = CalibrationSample(
        sample_id="q1-dense-faithfulness",
        dimension="faithfulness",
        judge_score=0.66,
        human_score=0.67,
    )

    assert scores_have_ordinal_disagreement(sample) is True
