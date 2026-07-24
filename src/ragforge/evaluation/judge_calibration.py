"""Judge calibration against human evaluation (ADR-0007/ADR-0018).

Publishing a strategy ranking on an unvalidated LLM judge is the easiest
attack point on this benchmark - ADR-0007/ADR-0018 require measuring
judge-vs-human agreement (weighted Cohen's kappa >= 0.60) before treating
judge scores as more than an unvalidated caveat. This module computes that
agreement; it does not (and cannot) produce the ~30+ hand-labeled samples
the calibration exercise needs - that is genuine human curation work
(stratified by query class, answerable/unanswerable, strong/weak strategies,
negation, exceptions, cross-references, numerical claims - see ADR-0018),
which this session has no way to fabricate. Scores from RagasJudge remain
unvalidated (ragas_judge.py's own docstring) until a real calibration file
is produced and run through ``compute_calibration_report``.

Ordinal kappa needs discrete categories; this project's judge scores are
continuous (0.0-1.0), so ``_to_ordinal`` bins them into 3 categories -
disagree/partial/agree - matching the same 0/0.5/1.0 grading scale
``RelevanceGrade`` already uses for structural relevance judgments
(domain/models.py), for consistency across the project's ordinal scales.
"""

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

_ORDINAL_CATEGORIES = 3
_BINARY_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One (judge_score, human_score) pair for a single dimension of a single question."""

    sample_id: str
    dimension: str
    judge_score: float
    human_score: float


def _to_ordinal(score: float) -> int:
    """Map a continuous 0.0-1.0 score into one of 3 ordinal bins for weighted kappa."""
    if score < 1 / _ORDINAL_CATEGORIES:
        return 0
    if score < 2 / _ORDINAL_CATEGORIES:
        return 1
    return 2


def weighted_cohens_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Return quadratic-weighted Cohen's kappa between two ordinal label sequences.

    1.0 for perfect agreement (including the trivial case where every label,
    judge and human alike, falls in the same single category - nothing to
    disagree about). Standard formula: 1 - (weighted observed disagreement)
    / (weighted expected disagreement under independence).

    Raises:
        ValueError: If the two sequences differ in length or are empty.
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must be the same length")
    if not judge_labels:
        raise ValueError("labels must not be empty")

    categories = sorted(set(judge_labels) | set(human_labels))
    if len(categories) == 1:
        return 1.0
    index = {category: i for i, category in enumerate(categories)}
    n_categories = len(categories)
    n = len(judge_labels)

    confusion = [[0] * n_categories for _ in range(n_categories)]
    for judge_label, human_label in zip(judge_labels, human_labels, strict=True):
        confusion[index[judge_label]][index[human_label]] += 1

    judge_marginal = [sum(row) for row in confusion]
    human_marginal = [
        sum(confusion[row][col] for row in range(n_categories)) for col in range(n_categories)
    ]
    weights = [
        [((i - j) ** 2) / (n_categories - 1) ** 2 for j in range(n_categories)]
        for i in range(n_categories)
    ]

    observed = sum(
        weights[i][j] * confusion[i][j] for i in range(n_categories) for j in range(n_categories)
    )
    expected = sum(
        weights[i][j] * judge_marginal[i] * human_marginal[j] / n
        for i in range(n_categories)
        for j in range(n_categories)
    )
    if expected == 0:
        return 1.0
    return 1.0 - observed / expected


def spearman_correlation(judge_scores: list[float], human_scores: list[float]) -> float:
    """Return the Spearman rank correlation between two continuous score sequences.

    Raises:
        ValueError: If the sequences differ in length or have fewer than 2 samples.
    """
    if len(judge_scores) != len(human_scores):
        raise ValueError("judge_scores and human_scores must be the same length")
    if len(judge_scores) < 2:
        raise ValueError("need at least 2 samples to compute a correlation")
    return statistics.correlation(judge_scores, human_scores, method="ranked")


def false_supported_rate(samples: list[CalibrationSample]) -> float:
    """Fraction of faithfulness samples judged supported that the human called unsupported.

    0.0 when the judge never called anything supported - nothing to be
    false about.
    """
    judge_supported = [
        sample
        for sample in samples
        if sample.dimension == "faithfulness" and sample.judge_score >= _BINARY_THRESHOLD
    ]
    if not judge_supported:
        return 0.0
    false_positives = sum(1 for sample in judge_supported if sample.human_score < _BINARY_THRESHOLD)
    return false_positives / len(judge_supported)


def false_unsupported_rate(samples: list[CalibrationSample]) -> float:
    """Fraction of faithfulness samples judged unsupported that the human called supported.

    0.0 when the judge never called anything unsupported.
    """
    judge_unsupported = [
        sample
        for sample in samples
        if sample.dimension == "faithfulness" and sample.judge_score < _BINARY_THRESHOLD
    ]
    if not judge_unsupported:
        return 0.0
    false_negatives = sum(
        1 for sample in judge_unsupported if sample.human_score >= _BINARY_THRESHOLD
    )
    return false_negatives / len(judge_unsupported)


def abstention_agreement(samples: list[CalibrationSample]) -> float:
    """Fraction of abstention samples where judge and human land on the same side of 0.5.

    0.0 when there are no abstention-dimension samples.
    """
    abstention_samples = [sample for sample in samples if sample.dimension == "abstention"]
    if not abstention_samples:
        return 0.0
    agreements = sum(
        1
        for sample in abstention_samples
        if (sample.judge_score >= _BINARY_THRESHOLD) == (sample.human_score >= _BINARY_THRESHOLD)
    )
    return agreements / len(abstention_samples)


def compute_calibration_report(samples: list[CalibrationSample]) -> dict[str, float]:
    """Aggregate every ADR-0007/ADR-0018 calibration metric across all dimensions.

    ``spearman_correlation`` is omitted when fewer than 2 samples are given
    (undefined below that), rather than raising - callers publishing a
    partial report shouldn't need to special-case a tiny sample.

    Raises:
        ValueError: If samples is empty.
    """
    if not samples:
        raise ValueError("samples must not be empty")

    judge_ordinal = [_to_ordinal(sample.judge_score) for sample in samples]
    human_ordinal = [_to_ordinal(sample.human_score) for sample in samples]

    report = {
        "weighted_kappa": weighted_cohens_kappa(judge_ordinal, human_ordinal),
        "false_supported_rate": false_supported_rate(samples),
        "false_unsupported_rate": false_unsupported_rate(samples),
        "abstention_agreement": abstention_agreement(samples),
        "n": float(len(samples)),
    }
    if len(samples) >= 2:
        judge_scores = [sample.judge_score for sample in samples]
        human_scores = [sample.human_score for sample in samples]
        report["spearman_correlation"] = spearman_correlation(judge_scores, human_scores)
    return report


def load_calibration_samples(path: Path) -> list[CalibrationSample]:
    """Load calibration samples from a JSON file.

    Expected schema - a JSON array of objects::

        [
          {
            "sample_id": "q001-faithfulness",
            "dimension": "faithfulness",
            "judge_score": 0.8,
            "human_score": 1.0
          },
          ...
        ]

    No such file ships with this repository (see module docstring) - this
    loader exists so the calibration mechanism is ready to consume real
    human-labeled data the moment a human curator produces it.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        CalibrationSample(
            sample_id=entry["sample_id"],
            dimension=entry["dimension"],
            judge_score=entry["judge_score"],
            human_score=entry["human_score"],
        )
        for entry in payload
    ]
