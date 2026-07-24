"""Tests for Document-Level Retrieval Mismatch (ADR-0015)."""

from ragforge.domain.models import (
    Chunk,
    JudgedRef,
    Judgment,
    Query,
    RelevanceGrade,
    RetrievalResult,
    StructuralRef,
)
from ragforge.evaluation.metrics.drm import document_level_retrieval_mismatch

NORM = "RES-CMN-4893/2021"
OTHER_NORM = "LC-105/2001"


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
OTHER_ART_1 = f"{OTHER_NORM}::art-1"


def test_is_zero_when_every_result_belongs_to_a_relevant_norm() -> None:
    """No mismatch when every top-k result's structural_ids all point to a judged-relevant norm."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    results = [_result("c1", (ART_1,)), _result("c2", (f"{NORM}::art-2",))]

    assert document_level_retrieval_mismatch(results, judgment, k=2) == 0.0


def test_is_one_when_every_result_belongs_to_a_different_norm() -> None:
    """Every top-k result from an unrelated norm is a full mismatch."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    results = [_result("c1", (OTHER_ART_1,))]

    assert document_level_retrieval_mismatch(results, judgment, k=1) == 1.0


def test_is_partial_when_only_some_results_belong_to_a_different_norm() -> None:
    """One mismatched result out of two top-k results gives 0.5 DRM."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    results = [_result("right-norm", (ART_1,)), _result("wrong-norm", (OTHER_ART_1,))]

    assert document_level_retrieval_mismatch(results, judgment, k=2) == 0.5


def test_a_chunk_covering_a_wrong_article_but_right_norm_is_not_a_mismatch() -> None:
    """DRM checks the norm, not the exact article - a wrong-article hit is a recall gap, not DRM."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    results = [_result("c1", (f"{NORM}::art-99",))]

    assert document_level_retrieval_mismatch(results, judgment, k=1) == 0.0


def test_is_zero_for_an_unanswerable_judgment() -> None:
    """A judgment with no relevant refs yields DRM 0.0, not a division error."""
    judgment = _judgment()
    results = [_result("c1", (OTHER_ART_1,))]

    assert document_level_retrieval_mismatch(results, judgment, k=1) == 0.0


def test_is_zero_for_an_empty_result_list() -> None:
    """No results means nothing to mismatch against."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))

    assert document_level_retrieval_mismatch([], judgment, k=5) == 0.0


def test_only_considers_the_top_k_results() -> None:
    """A mismatched result beyond the k cutoff doesn't affect the score."""
    judgment = _judgment((ART_1, RelevanceGrade.RELEVANT))
    results = [_result("right-norm", (ART_1,)), _result("wrong-norm", (OTHER_ART_1,))]

    assert document_level_retrieval_mismatch(results, judgment, k=1) == 0.0
