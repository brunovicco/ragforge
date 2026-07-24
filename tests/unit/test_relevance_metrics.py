"""Tests for structural-coverage relevance metrics (ADR-0002)."""

import math

from ragforge.domain.models import (
    Chunk,
    JudgedRef,
    Judgment,
    Query,
    RelevanceGrade,
    RetrievalResult,
    StructuralRef,
)
from ragforge.evaluation.metrics.relevance import (
    is_hit,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    result_gain,
)

NORM = "RES-CMN-4893/2021"


def _result(chunk_id: str, structural_ids: tuple[str, ...]) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        source_text=chunk_id,
        retrieval_text=chunk_id,
        structural_ids=structural_ids,
    )
    return RetrievalResult(chunk=chunk, score=1.0, strategy="dense")


def _judgment(*refs: tuple[str, RelevanceGrade]) -> Judgment:
    return Judgment(
        question_id="q1",
        query=Query(text="pergunta"),
        relevant_refs=tuple(
            JudgedRef(ref=StructuralRef.parse(canonical), grade=grade) for canonical, grade in refs
        ),
    )


ART_1 = f"{NORM}::art-1"
ART_2_PAR_1 = f"{NORM}::art-2::par-1"
ART_3 = f"{NORM}::art-3"


def test_is_hit_true_when_chunk_covers_a_relevant_id() -> None:
    """A chunk whose structural_ids include a judged-relevant ref is a hit."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    result = _result("c1", (ART_1,))
    assert is_hit(result, judgment) is True


def test_is_hit_false_when_chunk_covers_no_relevant_id() -> None:
    """A chunk covering only irrelevant structural IDs is not a hit."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    result = _result("c1", (ART_3,))
    assert is_hit(result, judgment) is False


def test_result_gain_takes_the_max_across_covered_relevant_ids() -> None:
    """A chunk covering both a fully and a partially relevant ref takes the higher gain."""
    judgment = _judgment(
        (ART_1, RelevanceGrade.RELEVANT), (ART_2_PAR_1, RelevanceGrade.PARTIALLY_RELEVANT)
    )
    covers_both = _result("art-2-full", (f"{NORM}::art-2", ART_2_PAR_1, ART_1))
    assert result_gain(covers_both, judgment) == 1.0


def test_result_gain_is_zero_for_an_irrelevant_chunk() -> None:
    """A chunk covering no judged ID has zero gain, not an error."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    result = _result("c1", (ART_3,))
    assert result_gain(result, judgment) == 0.0


def test_recall_at_k_counts_coverage_not_exact_chunk_match() -> None:
    """One article-level chunk covering two judged refs yields full recall by itself."""
    judgment = _judgment(
        (ART_1, RelevanceGrade.RELEVANT), (ART_2_PAR_1, RelevanceGrade.PARTIALLY_RELEVANT)
    )
    single_chunk_covering_both = [_result("art-2-full", (f"{NORM}::art-2", ART_2_PAR_1, ART_1))]
    assert recall_at_k(single_chunk_covering_both, judgment, k=1) == 1.0


def test_recall_at_k_is_partial_when_only_some_relevant_ids_are_covered() -> None:
    """Covering one of two relevant refs yields 0.5 recall."""
    judgment = _judgment(
        (ART_1, RelevanceGrade.RELEVANT), (ART_2_PAR_1, RelevanceGrade.PARTIALLY_RELEVANT)
    )
    results = [_result("c1", (ART_1,))]
    assert recall_at_k(results, judgment, k=1) == 0.5


def test_recall_at_k_is_zero_for_an_unjudged_question() -> None:
    """A judgment with no relevant refs yields recall 0.0, not a division error."""
    judgment = _judgment()
    results = [_result("c1", (ART_1,))]
    assert recall_at_k(results, judgment, k=1) == 0.0


def test_precision_at_k_is_the_hit_fraction_of_the_top_k() -> None:
    """One hit out of two top-k results is 0.5 precision."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    results = [_result("hit", (ART_1,)), _result("miss", (ART_3,))]
    assert precision_at_k(results, judgment, k=2) == 0.5


def test_precision_at_k_is_zero_for_no_results() -> None:
    """Precision on an empty result list is 0.0, not a division error."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    assert precision_at_k([], judgment, k=5) == 0.0


def test_mrr_is_the_reciprocal_rank_of_the_first_hit() -> None:
    """A hit at rank 2 gives MRR 0.5."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    results = [_result("miss", (ART_3,)), _result("hit", (ART_1,))]
    assert mrr(results, judgment) == 0.5


def test_mrr_is_zero_when_nothing_is_a_hit() -> None:
    """No relevant chunk anywhere in the ranking gives MRR 0.0."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    results = [_result("miss", (ART_3,))]
    assert mrr(results, judgment) == 0.0


def test_ndcg_at_k_is_one_for_a_perfectly_ordered_ranking() -> None:
    """Best-gain-first ranking achieves the maximum nDCG of 1.0."""
    judgment = _judgment(
        (ART_1, RelevanceGrade.RELEVANT), (ART_2_PAR_1, RelevanceGrade.PARTIALLY_RELEVANT)
    )
    results = [_result("best", (ART_1,)), _result("second", (ART_2_PAR_1,))]
    assert ndcg_at_k(results, judgment, k=2) == 1.0


def test_ndcg_at_k_penalizes_a_reversed_ranking() -> None:
    """Worst-gain-first ranking scores below the perfect ordering."""
    judgment = _judgment(
        (ART_1, RelevanceGrade.RELEVANT), (ART_2_PAR_1, RelevanceGrade.PARTIALLY_RELEVANT)
    )
    reversed_results = [_result("worst", (ART_2_PAR_1,)), _result("best", (ART_1,))]
    dcg = 0.5 / math.log2(2) + 1.0 / math.log2(3)
    idcg = 1.0 / math.log2(2) + 0.5 / math.log2(3)
    assert ndcg_at_k(reversed_results, judgment, k=2) == dcg / idcg


def test_ndcg_at_k_does_not_double_count_the_same_relevant_id_across_results() -> None:
    """Multiple results covering the same single relevant ID stay capped at nDCG 1.0.

    Reproduces a real bug found running the full benchmark (ADR-0004):
    RAPTOR's tree summary nodes carry the union of their descendants'
    structural_ids, so a leaf article's ID can appear in several returned
    chunks (the leaf itself, plus every ancestor summary). nDCG must credit
    that ID's gain only once, or DCG can exceed IDCG and nDCG can exceed 1.0.
    """
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    leaf = _result("leaf", (ART_1,))
    summary_l1 = _result("summary-l1", (ART_1,))
    summary_l2 = _result("summary-l2", (ART_1,))

    assert ndcg_at_k([leaf, summary_l1, summary_l2], judgment, k=3) == 1.0


def test_ndcg_at_k_credits_a_later_result_for_a_still_uncovered_id() -> None:
    """A second relevant ID first covered at rank 2 still contributes its own gain."""
    judgment = _judgment(
        (ART_1, RelevanceGrade.RELEVANT), (ART_2_PAR_1, RelevanceGrade.PARTIALLY_RELEVANT)
    )
    redundant_then_new = [_result("dup", (ART_1,)), _result("new", (ART_1, ART_2_PAR_1))]

    dcg = 1.0 / math.log2(2) + 0.5 / math.log2(3)
    idcg = 1.0 / math.log2(2) + 0.5 / math.log2(3)
    assert ndcg_at_k(redundant_then_new, judgment, k=2) == dcg / idcg


def test_ndcg_at_k_is_zero_for_an_unjudged_question() -> None:
    """A judgment with no relevant refs yields nDCG 0.0, not a division error."""
    judgment = _judgment()
    results = [_result("c1", (ART_1,))]
    assert ndcg_at_k(results, judgment, k=1) == 0.0
