"""Deterministic stratified split construction for RegRAG-BR (ADR-0003)."""

import hashlib
import math
from collections import defaultdict

from ragforge.domain.models import Judgment
from ragforge.evaluation.split import Split

_SAMPLE_ALGORITHM_VERSION = "stratified-capacity-v1"


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


def select_stratified_sample(
    judgments: list[Judgment],
    *,
    max_questions: int,
    seed: str,
) -> list[Judgment]:
    """Select an exact-size deterministic sample while preserving every query class.

    One slot is reserved for each class, then the remaining capacity is
    apportioned proportionally using the largest-remainder method. Selection
    within each class is ranked by a versioned SHA-256 score; returned
    judgments retain their original split order.

    Args:
        judgments: Judgments already selected from one declared split.
        max_questions: Exact sample size, unless the split is smaller.
        seed: Versioned sampling seed recorded in the resolved configuration.

    Raises:
        ValueError: If the requested size cannot represent every query class.
    """
    if max_questions <= 0:
        raise ValueError("max_questions must be positive")
    if max_questions >= len(judgments):
        return list(judgments)

    by_class: dict[str, list[Judgment]] = defaultdict(list)
    for judgment in judgments:
        if judgment.query.query_class is None:
            raise ValueError(f"judgment {judgment.question_id!r} has no query class")
        by_class[judgment.query.query_class.value].append(judgment)
    if max_questions < len(by_class):
        raise ValueError(
            f"max_questions must be at least {len(by_class)} to represent every query class"
        )

    remaining = max_questions - len(by_class)
    total_capacity = len(judgments) - len(by_class)
    allocation: dict[str, int] = dict.fromkeys(by_class, 1)
    remainders: list[tuple[float, str]] = []
    allocated_extra = 0
    for query_class, class_judgments in sorted(by_class.items()):
        capacity = len(class_judgments) - 1
        ideal_extra = remaining * capacity / total_capacity
        extra = min(capacity, math.floor(ideal_extra))
        allocation[query_class] += extra
        allocated_extra += extra
        remainders.append((ideal_extra - extra, query_class))
    for _remainder, query_class in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if allocated_extra >= remaining:
            break
        if allocation[query_class] < len(by_class[query_class]):
            allocation[query_class] += 1
            allocated_extra += 1

    selected_ids: set[str] = set()
    for query_class, class_judgments in sorted(by_class.items()):
        ranked = sorted(
            class_judgments,
            key=lambda judgment: hashlib.sha256(
                (
                    f"{_SAMPLE_ALGORITHM_VERSION}:{seed}:{query_class}:{judgment.question_id}"
                ).encode()
            ).hexdigest(),
        )
        selected_ids.update(judgment.question_id for judgment in ranked[: allocation[query_class]])
    return [judgment for judgment in judgments if judgment.question_id in selected_ids]
