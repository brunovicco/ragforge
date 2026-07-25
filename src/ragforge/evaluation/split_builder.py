"""Deterministic stratified split construction for RegRAG-BR (ADR-0003)."""

import hashlib
from collections import defaultdict

from ragforge.domain.models import Judgment
from ragforge.evaluation.split import Split


def build_stratified_split(
    judgments: list[Judgment],
    *,
    dataset_version: str,
    validation_ratio: float = 0.15,
    seed: str = "regrag-br-v1",
) -> Split:
    """Partition judgments into deterministic validation and test sets.

    Selection is stratified by query class. Within each class, question IDs
    are ordered by a SHA-256 score derived from the declared seed, avoiding
    dependence on Python's randomized hash or a mutable PRNG implementation.
    At least one question from every non-empty class is assigned to each
    partition.

    Args:
        judgments: Complete curated judgment collection.
        dataset_version: Version copied into the resulting split artifact.
        validation_ratio: Target fraction reserved for router development.
        seed: Versioned selection seed.

    Raises:
        ValueError: If inputs cannot produce a valid two-way split.
    """
    if not judgments:
        raise ValueError("judgments must not be empty")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    by_class: dict[str, list[str]] = defaultdict(list)
    original_order = [judgment.question_id for judgment in judgments]
    for judgment in judgments:
        if judgment.query.query_class is None:
            raise ValueError(f"judgment {judgment.question_id!r} has no query class")
        by_class[judgment.query.query_class.value].append(judgment.question_id)

    validation_ids: set[str] = set()
    for query_class, question_ids in sorted(by_class.items()):
        if len(question_ids) < 2:
            raise ValueError(f"query class {query_class!r} needs at least two questions")
        validation_count = max(1, round(len(question_ids) * validation_ratio))
        validation_count = min(validation_count, len(question_ids) - 1)
        ranked_ids = sorted(
            question_ids,
            key=lambda question_id: hashlib.sha256(
                f"{seed}:{query_class}:{question_id}".encode()
            ).hexdigest(),
        )
        validation_ids.update(ranked_ids[:validation_count])

    validation = tuple(
        question_id for question_id in original_order if question_id in validation_ids
    )
    test = tuple(question_id for question_id in original_order if question_id not in validation_ids)
    return Split(
        schema_version=1,
        dataset_version=dataset_version,
        train=(),
        validation=validation,
        test=test,
    )
