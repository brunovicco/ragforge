"""Citation Accuracy (ADR-0007): deterministic, no LLM judge.

The proportion of a generated Answer's citations whose structural IDs
(ADR-0002/0006) belong to a judgment's relevant set - the one quality metric
this project can report without depending on an LLM judge at all.
"""

from ragforge.domain.models import Answer, Judgment


def citation_accuracy(answer: Answer, judgment: Judgment) -> float:
    """Fraction of ``answer.citations`` that are judged-relevant structural IDs.

    0.0 for an answer with no citations - nothing to be accurate about is
    treated as a provenance failure, not vacuous perfection.
    """
    if not answer.citations:
        return 0.0
    relevant_ids = {judged.ref.canonical for judged in judgment.relevant_refs}
    correct = sum(1 for citation in answer.citations if citation in relevant_ids)
    return correct / len(answer.citations)
