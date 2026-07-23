"""Structural-coverage relevance metrics (ADR-0002).

Judgments are annotated at the norm's stable structural unit
(``{norm}::{art}::{fragment}``), not at the chunk level, so retrieval metrics
stay comparable across strategies whose result granularity differs
(parent-child returns whole articles, RAPTOR/GraphRAG project synthetic
summaries onto their source structural IDs). A retrieved chunk counts as a
hit if its own ``structural_ids`` (ADR-0006) cover at least one relevant ID
from the judgment - graded relevance (relevant vs. partially relevant) is
preserved as nDCG's gain, collapsed to a binary hit for Recall/Precision/MRR.
"""

import math

from ragforge.domain.models import Judgment, RelevanceGrade, RetrievalResult

_GAIN_BY_GRADE: dict[RelevanceGrade, float] = {
    RelevanceGrade.RELEVANT: 1.0,
    RelevanceGrade.PARTIALLY_RELEVANT: 0.5,
}


def _gains_by_canonical(judgment: Judgment) -> dict[str, float]:
    """Map each judged structural ID (canonical string) to its relevance gain."""
    return {judged.ref.canonical: _GAIN_BY_GRADE[judged.grade] for judged in judgment.relevant_refs}


def result_gain(result: RetrievalResult, judgment: Judgment) -> float:
    """Return the highest relevance gain among the judged IDs a result's chunk covers.

    Zero if the chunk covers no judged-relevant structural ID.
    """
    gains = _gains_by_canonical(judgment)
    covered = (gains.get(ref_id, 0.0) for ref_id in result.chunk.structural_ids)
    return max(covered, default=0.0)


def is_hit(result: RetrievalResult, judgment: Judgment) -> bool:
    """Return True if the result's chunk covers at least one relevant structural ID."""
    return result_gain(result, judgment) > 0.0


def recall_at_k(results: list[RetrievalResult], judgment: Judgment, k: int) -> float:
    """Fraction of judged-relevant structural IDs covered by the top-k results.

    Coverage, not exact match: one chunk can cover several relevant IDs (e.g.
    an article-level chunk covering all of its judged paragraphs). Returns
    0.0 for an unjudged question (no relevant refs).
    """
    gains = _gains_by_canonical(judgment)
    if not gains:
        return 0.0
    covered_ids: set[str] = set()
    for result in results[:k]:
        covered_ids.update(ref_id for ref_id in result.chunk.structural_ids if ref_id in gains)
    return len(covered_ids) / len(gains)


def precision_at_k(results: list[RetrievalResult], judgment: Judgment, k: int) -> float:
    """Fraction of the top-k results that are hits. 0.0 if there are no results."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for result in top_k if is_hit(result, judgment))
    return hits / len(top_k)


def mrr(results: list[RetrievalResult], judgment: Judgment) -> float:
    """Reciprocal rank of the first hit; 0.0 if no result is a hit."""
    for rank, result in enumerate(results, start=1):
        if is_hit(result, judgment):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[RetrievalResult], judgment: Judgment, k: int) -> float:
    """Normalized Discounted Cumulative Gain over the top-k results.

    Coverage-aware like recall_at_k: a judged-relevant ID already credited by
    an earlier, higher-ranked result contributes no further gain at a later
    rank. Without this, a strategy whose results legitimately overlap in
    structural coverage - e.g. RAPTOR, where a leaf's structural_ids
    propagate into every ancestor summary node - could count the same
    relevant ID's gain once per covering result and push nDCG above the
    theoretical maximum of 1.0, since IDCG assumes each distinct relevant ID
    contributes at most once.

    0.0 when the judgment has no relevant refs (nothing to normalize against).
    """
    gains = _gains_by_canonical(judgment)
    covered_ids: set[str] = set()
    dcg = 0.0
    for rank, result in enumerate(results[:k], start=1):
        new_ids = [
            ref_id
            for ref_id in result.chunk.structural_ids
            if ref_id in gains and ref_id not in covered_ids
        ]
        if not new_ids:
            continue
        dcg += max(gains[ref_id] for ref_id in new_ids) / math.log2(rank + 1)
        covered_ids.update(new_ids)

    ideal_gains = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return dcg / idcg if idcg > 0 else 0.0
