"""Document-Level Retrieval Mismatch (ADR-0015): a retrieved chunk from the wrong source norm.

Dense retrieval can return a semantically plausible clause from a norm other
than the one a question is actually about - locally similar legal language
(obligations, controls, deadlines) recurs across unrelated regulatory
documents. DRM measures this directly, at the document level, complementing
the structural-unit-level metrics in relevance.py, which only check whether
the right *article* was found, not whether it came from the right *norm*.
"""

from ragforge.domain.models import Judgment, RetrievalResult, StructuralRef


def document_level_retrieval_mismatch(
    results: list[RetrievalResult], judgment: Judgment, k: int
) -> float:
    """Fraction of the top-k results whose source norm isn't in the judgment's relevant set.

    A result counts as matching if any of its chunk's structural_ids belongs
    to a relevant norm - the same coverage principle relevance.py's metrics
    use, just checked at the norm level rather than the exact structural ID.
    0.0 for an unanswerable judgment (no relevant refs) or an empty result
    list - nothing to mismatch against.
    """
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant_norms = {judged.ref.norm for judged in judgment.relevant_refs}
    if not relevant_norms:
        return 0.0
    mismatched = sum(
        1
        for result in top_k
        if not any(
            StructuralRef.parse(ref_id).norm in relevant_norms
            for ref_id in result.chunk.structural_ids
        )
    )
    return mismatched / len(top_k)
