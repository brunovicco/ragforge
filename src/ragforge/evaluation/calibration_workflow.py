"""Human-calibration workflow for the answer-quality judge (ADR-0007/ADR-0018).

The human reviewer must evaluate the same evidence that RAGAS Faithfulness
received.  This module therefore treats ``QuestionRecord.judge_contexts`` as
the calibration evidence and keeps answer citations in a separate audit-only
section.
"""

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ragforge.domain.models import Judgment, QueryClass
from ragforge.evaluation.judge_calibration import (
    CalibrationSample,
    compute_calibration_report,
)
from ragforge.evaluation.records import QuestionRecord
from ragforge.generation.citation_parsing import (
    extract_citation_candidates,
    extract_citations,
)

CalibrationDimension = Literal["faithfulness", "answer_relevancy", "abstention"]
BASE_DIMENSIONS: tuple[CalibrationDimension, ...] = ("faithfulness", "answer_relevancy")
METRIC_KEYS: dict[CalibrationDimension, str] = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "abstention": "abstention_appropriate",
}
SELECTION_SEED = "regrag-br-calibration-v2"
SEALED_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
MINIMUM_ANSWER_SAMPLES = 30
KAPPA_FLOOR = 0.60
KAPPA_TARGET = 0.70

_STRONG_STRATEGIES = frozenset({"sac", "sac_contextual", "raptor", "dense"})
_CONCEPT_FEATURES = frozenset(
    {
        "answer_quality:complete",
        "answer_quality:partial",
        "citation:unsupported",
        "language:negation",
        "language:exception",
        "reasoning:cross_reference",
        "claim:numerical",
    }
)
_NEGATION_RE = re.compile(r"\b(?:não|nunca|sem|veda(?:do|da)?|proibi(?:do|da))\b", re.IGNORECASE)
_EXCEPTION_RE = re.compile(
    r"\b(?:salvo|exceto|ressalvad\w*|desde que|a menos que)\b", re.IGNORECASE
)
_NUMERICAL_RE = re.compile(r"(?:\d|%|R\$)")


class HumanLabel(BaseModel):
    """Validated human score for one answer-quality dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    dimension: CalibrationDimension
    human_score: float | None

    @field_validator("human_score", mode="before")
    @classmethod
    def validate_human_score(cls, value: object) -> float | None:
        """Accept only the three documented ordinal values or null."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("human_score must be numeric or null")
        score = float(value)
        if not math.isfinite(score) or score not in {0.0, 0.5, 1.0}:
            raise ValueError("human_score must be one of 0.0, 0.5, or 1.0")
        return score

    @model_validator(mode="after")
    def validate_sample_suffix(self) -> "HumanLabel":
        """Require the sample identity to end in its declared dimension."""
        if not self.sample_id.endswith(f"-{self.dimension}"):
            raise ValueError("sample_id suffix must match dimension")
        return self

    @property
    def answer_sample_id(self) -> str:
        """Return the question/strategy identity shared by all dimensions."""
        return self.sample_id.removesuffix(f"-{self.dimension}")


class SealedCalibration(BaseModel):
    """Judge scores hidden from the reviewer until human labeling is complete."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    evidence_schema_version: Literal[1]
    run_id: str
    judge_scores: dict[str, float]

    @field_validator("judge_scores", mode="before")
    @classmethod
    def validate_judge_scores(cls, value: object) -> dict[str, float]:
        """Reject non-numeric, non-finite, and out-of-range sealed scores."""
        if not isinstance(value, dict):
            raise ValueError("judge_scores must be a JSON object")
        scores: dict[str, float] = {}
        for raw_key, raw_score in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError("judge score keys must be non-empty strings")
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ValueError(f"judge score {raw_key!r} must be numeric")
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"judge score {raw_key!r} must be finite and between 0 and 1")
            scores[raw_key] = score
        return scores


def dimensions_for(record: QuestionRecord) -> tuple[CalibrationDimension, ...]:
    """Return the dimensions a reviewer must label for ``record``."""
    return (*BASE_DIMENSIONS, "abstention") if record.unanswerable else BASE_DIMENSIONS


def load_human_labels(path: Path) -> list[HumanLabel]:
    """Load and validate a reviewer-edited labels JSON array."""
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("labels.json must contain a JSON array")
    labels = [HumanLabel.model_validate(entry) for entry in payload]
    _require_unique_label_ids(labels)
    return labels


def load_sealed_calibration(path: Path) -> SealedCalibration:
    """Load and validate the sealed judge-score document."""
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    return SealedCalibration.model_validate(payload)


def _require_unique_label_ids(labels: list[HumanLabel]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for label in labels:
        if label.sample_id in seen:
            duplicates.add(label.sample_id)
        seen.add(label.sample_id)
    if duplicates:
        raise ValueError(f"duplicate label sample_id values: {', '.join(sorted(duplicates))}")


def merge_existing_human_labels(
    expected: list[HumanLabel], existing: list[HumanLabel]
) -> list[HumanLabel]:
    """Preserve scores when rebuilding the exact same calibration sample.

    Raises:
        ValueError: If the existing label identities or dimensions differ.
    """
    if not existing:
        return expected
    _require_unique_label_ids(expected)
    _require_unique_label_ids(existing)
    expected_by_id = {label.sample_id: label for label in expected}
    existing_by_id = {label.sample_id: label for label in existing}
    if expected_by_id.keys() != existing_by_id.keys():
        raise ValueError(
            "existing labels belong to a different calibration sample; archive the calibration "
            "directory before building another run"
        )
    merged = []
    for label in expected:
        previous = existing_by_id[label.sample_id]
        if previous.dimension != label.dimension:
            raise ValueError(f"dimension changed for existing label {label.sample_id!r}")
        merged.append(label.model_copy(update={"human_score": previous.human_score}))
    return merged


def build_human_labels(sample: list[QuestionRecord]) -> list[HumanLabel]:
    """Create empty human labels for every selected record dimension."""
    return [
        HumanLabel(
            sample_id=f"{record.question_id}-{record.strategy}-{dimension}",
            dimension=dimension,
            human_score=None,
        )
        for record in sample
        for dimension in dimensions_for(record)
    ]


def build_sealed_calibration(sample: list[QuestionRecord], run_id: str) -> SealedCalibration:
    """Create the hidden judge-score payload corresponding exactly to ``sample``."""
    scores = {
        f"{record.question_id}-{record.strategy}-{dimension}": record.metrics[
            METRIC_KEYS[dimension]
        ]
        for record in sample
        for dimension in dimensions_for(record)
    }
    return SealedCalibration(
        schema_version=SEALED_SCHEMA_VERSION,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
        run_id=run_id,
        judge_scores=scores,
    )


def _stratum_of(record: QuestionRecord) -> tuple[str, str, str]:
    query_class = record.query_class or "unclassified"
    answerability = "unanswerable" if record.unanswerable else "answerable"
    strength = "strong" if record.strategy in _STRONG_STRATEGIES else "weak"
    return query_class, answerability, strength


def _selection_features(record: QuestionRecord, judgment: Judgment) -> frozenset[str]:
    query_class, answerability, strength = _stratum_of(record)
    features = {
        f"stratum:{query_class}:{answerability}:{strength}",
        f"class:{query_class}",
        f"answerability:{answerability}",
        f"strategy:{strength}",
    }
    faithfulness = record.metrics["faithfulness"]
    relevancy = record.metrics["answer_relevancy"]
    if faithfulness >= 2 / 3 and relevancy >= 2 / 3:
        features.add("answer_quality:complete")
    else:
        features.add("answer_quality:partial")

    candidates = extract_citation_candidates(record.answer_text or "")
    parsed = extract_citations(record.answer_text or "")
    citation_accuracy = record.metrics.get("citation_accuracy", 1.0)
    if (
        citation_accuracy < 1.0
        or len(candidates) != len(parsed)
        or any("," in candidate for candidate in candidates)
    ):
        features.add("citation:unsupported")

    combined_text = "\n".join(
        part
        for part in (
            judgment.query.text,
            judgment.reference_answer or "",
            record.answer_text or "",
        )
        if part
    )
    if _NEGATION_RE.search(combined_text):
        features.add("language:negation")
    if _EXCEPTION_RE.search(combined_text):
        features.add("language:exception")
    if (
        judgment.query.query_class in {QueryClass.MULTI_HOP, QueryClass.SECTION_COMPARATIVE}
        or len(judgment.relevant_refs) > 1
    ):
        features.add("reasoning:cross_reference")
    if judgment.query.query_class is QueryClass.NUMERIC_TABULAR or _NUMERICAL_RE.search(
        combined_text
    ):
        features.add("claim:numerical")
    return frozenset(features)


def _is_finite_score(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and 0.0 <= value <= 1.0


def _is_usable(record: QuestionRecord) -> bool:
    if record.generation_status != "succeeded" or record.judge_status != "succeeded":
        return False
    if not record.answer_text or not record.judge_contexts:
        return False
    return all(
        _is_finite_score(record.metrics.get(METRIC_KEYS[dimension]))
        for dimension in dimensions_for(record)
    )


def _stable_selection_key(record: QuestionRecord) -> str:
    identity = f"{SELECTION_SEED}\0{record.question_id}\0{record.strategy}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_calibration_records(
    records: list[QuestionRecord], judgments: dict[str, Judgment], size: int
) -> list[QuestionRecord]:
    """Select a deterministic sample covering every ADR-0018 calibration feature.

    The greedy first phase covers every available query/answerability/strength
    stratum and every required conceptual feature.  A deterministic
    round-robin then fills the requested size without using a pseudo-random
    generator or exposing judge-derived feature tags to the reviewer.

    Raises:
        ValueError: If exact judge contexts, metrics, features, or sample count
            are insufficient.
    """
    if size < MINIMUM_ANSWER_SAMPLES:
        raise ValueError(f"ADR-0018 requires at least {MINIMUM_ANSWER_SAMPLES} samples")
    usable = [record for record in records if _is_usable(record)]
    if len(usable) < size:
        context_count = sum(bool(record.judge_contexts) for record in records)
        raise ValueError(
            f"only {len(usable)} usable records for {size} requested; exact judge contexts exist "
            f"for {context_count} of {len(records)} records. Legacy runs must be rerun because "
            "citation text cannot reconstruct the evidence seen by the judge"
        )
    missing_judgments = sorted({record.question_id for record in usable} - judgments.keys())
    if missing_judgments:
        raise ValueError(f"records missing golden judgments: {', '.join(missing_judgments[:3])}")

    features_by_key = {
        (record.question_id, record.strategy): _selection_features(
            record, judgments[record.question_id]
        )
        for record in usable
    }
    all_features: set[str] = set().union(*features_by_key.values())
    missing_features = sorted(_CONCEPT_FEATURES - all_features)
    if missing_features:
        raise ValueError(
            "no usable calibration candidate covers required features: "
            + ", ".join(missing_features)
        )
    required_features = {
        feature
        for feature in all_features
        if feature.startswith(("stratum:", "class:", "answerability:", "strategy:"))
    }
    required_features.update(_CONCEPT_FEATURES)

    remaining = sorted(usable, key=_stable_selection_key)
    selected: list[QuestionRecord] = []
    uncovered = set(required_features)
    while uncovered:
        best = min(
            remaining,
            key=lambda record: (
                -len(features_by_key[(record.question_id, record.strategy)] & uncovered),
                _stable_selection_key(record),
            ),
        )
        gain = features_by_key[(best.question_id, best.strategy)] & uncovered
        if not gain:
            raise ValueError(
                f"unable to cover calibration features: {', '.join(sorted(uncovered))}"
            )
        if len(selected) >= size:
            raise ValueError(f"sample size {size} is too small to cover every calibration stratum")
        selected.append(best)
        remaining.remove(best)
        uncovered.difference_update(gain)

    selected_keys = {(record.question_id, record.strategy) for record in selected}
    buckets: dict[tuple[str, str, str], list[QuestionRecord]] = defaultdict(list)
    for record in remaining:
        if (record.question_id, record.strategy) not in selected_keys:
            buckets[_stratum_of(record)].append(record)
    for bucket in buckets.values():
        bucket.sort(key=_stable_selection_key)
    strata = sorted(buckets)
    while len(selected) < size:
        progressed = False
        for stratum in strata:
            if buckets[stratum] and len(selected) < size:
                selected.append(buckets[stratum].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"only {len(selected)} records available for {size} requested")
    return selected


def calibration_feature_coverage(
    sample: list[QuestionRecord], judgments: dict[str, Judgment]
) -> frozenset[str]:
    """Return the hidden selection-feature coverage for audit logging."""
    return frozenset().union(
        *(_selection_features(record, judgments[record.question_id]) for record in sample)
    )


def _blockquote(text: str) -> str:
    return "> " + text.strip().replace("\n", "\n> ")


def render_calibration_worksheet(
    sample: list[QuestionRecord],
    judgments: dict[str, Judgment],
    corpus: dict[str, dict[str, str]],
    run_id: str,
) -> str:
    """Render a blind worksheet using the exact evidence seen by the judge."""
    lines = [
        "# Judge calibration worksheet",
        "",
        f"Run `{run_id}` · {len(sample)} samples · selection seed `{SELECTION_SEED}`",
        "",
        "Score each sample in `labels.json`. Keep `.judge-sealed.json` closed until every",
        "human score is complete; seeing judge scores first would anchor the review.",
        "",
        "## Scale",
        "",
        "Use only `0.0`, `0.5`, or `1.0`.",
        "",
        "**Faithfulness** — is every assertion supported by the exact judge contexts shown",
        "below? These are the same ordered texts passed to RAGAS, not a reconstruction from",
        "citations or structural IDs.",
        "",
        "- `1.0`: fully supported",
        "- `0.5`: mostly supported, with a material overreach or omitted condition",
        "- `0.0`: a central assertion contradicts or lacks support in the contexts",
        "",
        "**Answer Relevancy** — does the answer directly and completely address the question?",
        "",
        "For an **unanswerable** question, a correct abstention scores `1.0` on both.",
        "",
        "The citation-audit section is diagnostic only and is not the Faithfulness evidence.",
        "",
        "---",
        "",
    ]
    for index, record in enumerate(sample, start=1):
        judgment = judgments[record.question_id]
        sample_id = f"{record.question_id}-{record.strategy}"
        lines.extend(
            [
                f"## {index}. `{sample_id}`",
                "",
                f"- Class: **{record.query_class}**"
                + (" · **UNANSWERABLE**" if record.unanswerable else ""),
                f"- Strategy: `{record.strategy}`",
                "",
                "### Question",
                "",
                judgment.query.text.strip(),
                "",
            ]
        )
        if judgment.reference_answer:
            lines.extend(
                [
                    "### Reference answer (golden set)",
                    "",
                    judgment.reference_answer.strip(),
                    "",
                ]
            )
        lines.extend(["### Exact judge contexts", ""])
        for context_index, context in enumerate(record.judge_contexts, start=1):
            lines.extend([f"#### Context {context_index}", "", _blockquote(context), ""])

        lines.extend(["### Citation audit (not judge evidence)", ""])
        candidates = extract_citation_candidates(record.answer_text or "")
        if not candidates:
            lines.extend(["_The answer contains no bracketed citation._", ""])
        parsed_citations = set(record.answer_citations)
        for candidate in candidates:
            lines.extend([f"**`{candidate}`**", ""])
            norm_id = candidate.split("::", maxsplit=1)[0]
            resolved = (corpus.get(norm_id) or {}).get(candidate)
            if candidate not in parsed_citations:
                lines.extend(
                    [
                        "> _(malformed or grouped citation; not parsed as one canonical ID)_",
                        "",
                    ]
                )
            elif resolved is None:
                lines.extend(["> _(parsed citation does not resolve in the canonical corpus)_", ""])
            else:
                lines.extend([_blockquote(resolved), ""])

        lines.extend(
            [
                "### Generated answer",
                "",
                (record.answer_text or "").strip(),
                "",
                "### Your scores",
                "",
                f"| `{sample_id}` | 0.0 / 0.5 / 1.0 | notes |",
                "|---|---|---|",
                "| faithfulness | | |",
                "| answer_relevancy | | |",
                *(["| abstention | | |"] if record.unanswerable else []),
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def build_calibration_samples(
    labels: list[HumanLabel], sealed: SealedCalibration
) -> list[CalibrationSample]:
    """Validate exact label/score alignment and build computation inputs."""
    _require_unique_label_ids(labels)
    label_ids = {label.sample_id for label in labels}
    score_ids = set(sealed.judge_scores)
    if label_ids != score_ids:
        missing_scores = sorted(label_ids - score_ids)
        missing_labels = sorted(score_ids - label_ids)
        raise ValueError(
            "labels and sealed scores differ; "
            f"without scores: {missing_scores[:3]}, without labels: {missing_labels[:3]}"
        )
    incomplete = [label.sample_id for label in labels if label.human_score is None]
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} of {len(labels)} labels are still empty; first: "
            + ", ".join(incomplete[:3])
        )
    answer_sample_ids = {label.answer_sample_id for label in labels}
    if len(answer_sample_ids) < MINIMUM_ANSWER_SAMPLES:
        raise ValueError(
            f"only {len(answer_sample_ids)} distinct answer samples; "
            f"ADR-0018 requires {MINIMUM_ANSWER_SAMPLES}"
        )
    for dimension in BASE_DIMENSIONS:
        dimension_ids = {label.answer_sample_id for label in labels if label.dimension == dimension}
        if dimension_ids != answer_sample_ids:
            raise ValueError(f"dimension {dimension!r} does not cover every answer sample")
    return [
        CalibrationSample(
            sample_id=label.sample_id,
            dimension=label.dimension,
            judge_score=sealed.judge_scores[label.sample_id],
            human_score=label.human_score,
        )
        for label in labels
        if label.human_score is not None
    ]


def build_calibration_report(samples: list[CalibrationSample], run_id: str) -> dict[str, object]:
    """Build overall and per-dimension agreement plus a fail-closed gate result."""
    overall = compute_calibration_report(samples)
    by_dimension = {
        dimension: compute_calibration_report(
            [sample for sample in samples if sample.dimension == dimension]
        )
        for dimension in sorted({sample.dimension for sample in samples})
    }
    required_results: dict[str, dict[str, float]] = {
        "overall": overall,
        **{dimension: by_dimension[dimension] for dimension in BASE_DIMENSIONS},
    }
    failed_dimensions = sorted(
        name for name, result in required_results.items() if result["weighted_kappa"] < KAPPA_FLOOR
    )
    answer_sample_count = len(
        {sample.sample_id.removesuffix(f"-{sample.dimension}") for sample in samples}
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "n_labels": len(samples),
        "n_answer_samples": answer_sample_count,
        "overall": overall,
        "by_dimension": by_dimension,
        "gate": {
            "floor": KAPPA_FLOOR,
            "publication_target": KAPPA_TARGET,
            "required_results": list(required_results),
            "failed_results": failed_dimensions,
            "passed": not failed_dimensions,
        },
    }


def calibration_gate_passed(report: dict[str, object]) -> bool:
    """Return the validated boolean gate result from ``build_calibration_report``."""
    gate = report.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("calibration report has no valid gate result")
    passed = gate.get("passed")
    if not isinstance(passed, bool):
        raise ValueError("calibration report has no valid gate result")
    return passed


def scores_have_ordinal_disagreement(sample: CalibrationSample) -> bool:
    """Return whether judge and human scores land in different kappa bins."""

    def ordinal(score: float) -> int:
        if score < 1 / 3:
            return 0
        if score < 2 / 3:
            return 1
        return 2

    return ordinal(sample.judge_score) != ordinal(sample.human_score)
