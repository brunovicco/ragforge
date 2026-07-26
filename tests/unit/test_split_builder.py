"""Tests for deterministic stratified RegRAG-BR split construction."""

from collections import Counter

import pytest

from ragforge.domain.models import Judgment, Query, QueryClass
from ragforge.evaluation.split_builder import build_stratified_split, select_stratified_sample


def _judgment(question_id: str, query_class: QueryClass) -> Judgment:
    return Judgment(
        question_id=question_id,
        query=Query(text=question_id, query_class=query_class),
        relevant_refs=(),
    )


def test_build_stratified_split_is_deterministic_and_disjoint() -> None:
    """The same seed yields a stable partition with complete, disjoint coverage."""
    judgments = [_judgment(f"exact-{index}", QueryClass.EXACT_FACTUAL) for index in range(10)] + [
        _judgment(f"global-{index}", QueryClass.GLOBAL) for index in range(10)
    ]

    first = build_stratified_split(judgments, dataset_version="1", seed="stable")
    second = build_stratified_split(judgments, dataset_version="1", seed="stable")

    assert first == second
    assert not set(first.validation) & set(first.test)
    assert set(first.validation) | set(first.test) == {
        judgment.question_id for judgment in judgments
    }


def test_build_stratified_split_reserves_each_class_in_both_partitions() -> None:
    """Every query class remains represented in validation and test."""
    judgments = [
        _judgment(f"{query_class.value}-{index}", query_class)
        for query_class in QueryClass
        for index in range(10)
    ]

    split = build_stratified_split(judgments, dataset_version="1")

    for query_class in QueryClass:
        prefix = f"{query_class.value}-"
        assert any(question_id.startswith(prefix) for question_id in split.validation)
        assert any(question_id.startswith(prefix) for question_id in split.test)


def test_build_stratified_split_rejects_invalid_ratio() -> None:
    """A non-fractional validation ratio is rejected."""
    judgments = [
        _judgment("q1", QueryClass.EXACT_FACTUAL),
        _judgment("q2", QueryClass.EXACT_FACTUAL),
    ]

    try:
        build_stratified_split(judgments, dataset_version="1", validation_ratio=1.0)
    except ValueError as exc:
        assert "validation_ratio" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_select_stratified_sample_is_exact_deterministic_and_preserves_order() -> None:
    """The cost cap is exact, reproducible, class-aware, and keeps split order."""
    judgments = [
        _judgment(f"{query_class.value}-{index}", query_class)
        for query_class in QueryClass
        for index in range(10)
    ]

    first = select_stratified_sample(judgments, max_questions=20, seed="stable")
    second = select_stratified_sample(judgments, max_questions=20, seed="stable")

    assert first == second
    assert len(first) == 20
    assert [judgments.index(item) for item in first] == sorted(
        judgments.index(item) for item in first
    )
    counts = Counter(item.query.query_class for item in first)
    assert set(counts) == set(QueryClass)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_select_stratified_sample_apportions_uneven_classes_proportionally() -> None:
    """Larger classes receive more of the remaining capacity without excluding small ones."""
    judgments = [
        *[_judgment(f"exact-{index}", QueryClass.EXACT_FACTUAL) for index in range(8)],
        *[_judgment(f"global-{index}", QueryClass.GLOBAL) for index in range(2)],
    ]

    sampled = select_stratified_sample(judgments, max_questions=5, seed="stable")

    counts = Counter(item.query.query_class for item in sampled)
    assert counts == {QueryClass.EXACT_FACTUAL: 4, QueryClass.GLOBAL: 1}


def test_select_stratified_sample_rejects_a_cap_smaller_than_the_class_count() -> None:
    """A cost cap may not silently remove a query class from comparison."""
    judgments = [_judgment(f"{query_class.value}-0", query_class) for query_class in QueryClass] + [
        _judgment("extra", QueryClass.EXACT_FACTUAL)
    ]

    with pytest.raises(ValueError, match="represent every query class"):
        select_stratified_sample(
            judgments,
            max_questions=len(QueryClass) - 1,
            seed="stable",
        )
