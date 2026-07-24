"""Tests for judge calibration metrics (ADR-0007/ADR-0018)."""

import json
from pathlib import Path

import pytest

from ragforge.evaluation.judge_calibration import (
    CalibrationSample,
    abstention_agreement,
    compute_calibration_report,
    false_supported_rate,
    false_unsupported_rate,
    load_calibration_samples,
    spearman_correlation,
    weighted_cohens_kappa,
)


def _sample(
    sample_id: str, dimension: str, judge_score: float, human_score: float
) -> CalibrationSample:
    return CalibrationSample(
        sample_id=sample_id, dimension=dimension, judge_score=judge_score, human_score=human_score
    )


class TestWeightedCohensKappa:
    def test_is_one_for_perfect_agreement(self) -> None:
        """Identical ordinal labels yield maximum agreement."""
        assert weighted_cohens_kappa([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]) == 1.0

    def test_is_one_for_the_trivial_case_of_a_single_shared_category(self) -> None:
        """Every label landing in the same single category is trivial agreement, not 0/0."""
        assert weighted_cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0

    def test_is_negative_one_for_a_complete_ordinal_reversal(self) -> None:
        """Every judge/human pair maximally apart (0 vs 2) is the worst possible score."""
        assert weighted_cohens_kappa([0, 0, 0, 2, 2, 2], [2, 2, 2, 0, 0, 0]) == -1.0

    def test_is_between_the_extremes_for_partial_disagreement(self) -> None:
        """Some agreement, some disagreement lands strictly between -1.0 and 1.0."""
        kappa = weighted_cohens_kappa([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0])
        assert -1.0 < kappa < 1.0

    def test_raises_for_mismatched_lengths(self) -> None:
        """Judge and human label sequences must describe the same samples."""
        with pytest.raises(ValueError, match="same length"):
            weighted_cohens_kappa([0, 1], [0])

    def test_raises_for_empty_input(self) -> None:
        """An empty calibration set has nothing to measure agreement over."""
        with pytest.raises(ValueError, match="must not be empty"):
            weighted_cohens_kappa([], [])


class TestSpearmanCorrelation:
    def test_is_one_for_perfectly_matched_ranks(self) -> None:
        """Identical orderings correlate perfectly."""
        assert spearman_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0

    def test_is_negative_one_for_completely_reversed_ranks(self) -> None:
        """Exactly opposite orderings anti-correlate perfectly."""
        assert spearman_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0

    def test_raises_for_fewer_than_two_samples(self) -> None:
        """A single sample has no rank variation to correlate."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            spearman_correlation([1.0], [1.0])


class TestFalseSupportedAndUnsupportedRate:
    def test_false_supported_rate_counts_judge_yes_human_no(self) -> None:
        """The judge calling something supported that the human calls unsupported is a miss."""
        samples = [
            _sample("q1", "faithfulness", judge_score=0.9, human_score=0.0),
            _sample("q2", "faithfulness", judge_score=0.9, human_score=1.0),
        ]
        assert false_supported_rate(samples) == 0.5

    def test_false_supported_rate_is_zero_when_the_judge_never_calls_anything_supported(
        self,
    ) -> None:
        """No judge-supported samples means nothing to be falsely supported."""
        samples = [_sample("q1", "faithfulness", judge_score=0.1, human_score=1.0)]
        assert false_supported_rate(samples) == 0.0

    def test_false_unsupported_rate_counts_judge_no_human_yes(self) -> None:
        """The judge calling something unsupported that the human calls supported is a miss."""
        samples = [
            _sample("q1", "faithfulness", judge_score=0.1, human_score=1.0),
            _sample("q2", "faithfulness", judge_score=0.1, human_score=0.0),
        ]
        assert false_unsupported_rate(samples) == 0.5

    def test_rates_ignore_samples_from_other_dimensions(self) -> None:
        """Only faithfulness-dimension samples count toward these two rates."""
        samples = [_sample("q1", "answer_relevancy", judge_score=0.9, human_score=0.0)]
        assert false_supported_rate(samples) == 0.0
        assert false_unsupported_rate(samples) == 0.0


class TestAbstentionAgreement:
    def test_is_one_when_judge_and_human_always_agree(self) -> None:
        """Both landing on the same side of 0.5 every time is full agreement."""
        samples = [
            _sample("q1", "abstention", judge_score=1.0, human_score=1.0),
            _sample("q2", "abstention", judge_score=0.0, human_score=0.2),
        ]
        assert abstention_agreement(samples) == 1.0

    def test_is_partial_when_judge_and_human_sometimes_disagree(self) -> None:
        """One disagreement out of two abstention samples gives 0.5 agreement."""
        samples = [
            _sample("q1", "abstention", judge_score=1.0, human_score=1.0),
            _sample("q2", "abstention", judge_score=1.0, human_score=0.0),
        ]
        assert abstention_agreement(samples) == 0.5

    def test_is_zero_when_there_are_no_abstention_samples(self) -> None:
        """No abstention-dimension samples means nothing to measure agreement over."""
        samples = [_sample("q1", "faithfulness", judge_score=1.0, human_score=1.0)]
        assert abstention_agreement(samples) == 0.0


class TestComputeCalibrationReport:
    def test_aggregates_every_metric_across_dimensions(self) -> None:
        """A mixed-dimension calibration set produces every metric key, all in agreement."""
        samples = [
            _sample("q1-faithfulness", "faithfulness", judge_score=1.0, human_score=1.0),
            _sample("q1-answer_relevancy", "answer_relevancy", judge_score=0.9, human_score=0.9),
            _sample("q1-abstention", "abstention", judge_score=1.0, human_score=1.0),
        ]

        report = compute_calibration_report(samples)

        assert report["weighted_kappa"] == 1.0
        assert report["abstention_agreement"] == 1.0
        assert report["false_supported_rate"] == 0.0
        assert report["false_unsupported_rate"] == 0.0
        assert report["n"] == 3.0
        assert report["spearman_correlation"] == pytest.approx(1.0)

    def test_omits_spearman_for_a_single_sample(self) -> None:
        """A single sample can't have a rank correlation - omitted, not raised."""
        report = compute_calibration_report(
            [_sample("q1", "faithfulness", judge_score=1.0, human_score=1.0)]
        )

        assert "spearman_correlation" not in report
        assert report["n"] == 1.0

    def test_raises_for_an_empty_sample_list(self) -> None:
        """An empty calibration file is a caller error, not a silently meaningless report."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_calibration_report([])


class TestLoadCalibrationSamples:
    def test_loads_every_sample_from_a_json_file(self, tmp_path: Path) -> None:
        """Every field in the documented JSON schema round-trips into a CalibrationSample."""
        path = tmp_path / "calibration.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "sample_id": "q1-faithfulness",
                        "dimension": "faithfulness",
                        "judge_score": 0.8,
                        "human_score": 1.0,
                    }
                ]
            ),
            encoding="utf-8",
        )

        samples = load_calibration_samples(path)

        assert samples == [
            CalibrationSample(
                sample_id="q1-faithfulness",
                dimension="faithfulness",
                judge_score=0.8,
                human_score=1.0,
            )
        ]
