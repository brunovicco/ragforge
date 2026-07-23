"""Tests for the deterministic Citation Accuracy metric (ADR-0007)."""

from ragforge.domain.models import (
    Answer,
    JudgedRef,
    Judgment,
    Query,
    RelevanceGrade,
    StructuralRef,
)
from ragforge.evaluation.metrics.citation import citation_accuracy

NORM = "RES-CMN-4893/2021"
ART_1 = f"{NORM}::art-1"
ART_2 = f"{NORM}::art-2"
ART_3 = f"{NORM}::art-3"


def _judgment(*refs: str) -> Judgment:
    return Judgment(
        question_id="q1",
        query=Query(text="pergunta"),
        relevant_refs=tuple(
            JudgedRef(ref=StructuralRef.parse(ref), grade=RelevanceGrade.RELEVANT) for ref in refs
        ),
    )


def test_citation_accuracy_is_one_when_every_citation_is_relevant() -> None:
    """All citations landing in the relevant set gives full accuracy."""
    judgment = _judgment(ART_1, ART_2)
    answer = Answer(text="...", citations=(ART_1, ART_2))

    assert citation_accuracy(answer, judgment) == 1.0


def test_citation_accuracy_is_partial_when_some_citations_are_irrelevant() -> None:
    """One correct and one irrelevant citation out of two gives 0.5."""
    judgment = _judgment(ART_1)
    answer = Answer(text="...", citations=(ART_1, ART_3))

    assert citation_accuracy(answer, judgment) == 0.5


def test_citation_accuracy_is_zero_when_no_citation_is_relevant() -> None:
    """Every citation missing the relevant set gives 0.0."""
    judgment = _judgment(ART_1)
    answer = Answer(text="...", citations=(ART_3,))

    assert citation_accuracy(answer, judgment) == 0.0


def test_citation_accuracy_is_zero_for_an_uncited_answer() -> None:
    """No citations at all is treated as a provenance failure, not vacuous perfection."""
    judgment = _judgment(ART_1)
    answer = Answer(text="...", citations=())

    assert citation_accuracy(answer, judgment) == 0.0
