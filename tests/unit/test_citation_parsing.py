"""Tests for shared bracket-citation parsing (ADR-0002/0006), used by the answer generator
and, per-claim, by the post-generation citation auditor (ADR-0016)."""

from ragforge.generation.citation_parsing import extract_citation_candidates, extract_citations


def test_extract_citations_returns_well_formed_structural_ids_in_first_cited_order() -> None:
    """Only valid structural IDs are extracted, deduplicated, in first-seen order."""
    text = "Regra A [LC-105/2001::art-10]. Regra B [RES-CMN-4893/2021::art-2::par-1]."

    citations = extract_citations(text)

    assert citations == ("LC-105/2001::art-10", "RES-CMN-4893/2021::art-2::par-1")


def test_extract_citations_skips_malformed_brackets() -> None:
    """A bracketed span that isn't a valid structural ID is skipped, not raised."""
    text = "See [this note] and [LC-105/2001::art-10]."

    citations = extract_citations(text)

    assert citations == ("LC-105/2001::art-10",)


def test_extract_citations_deduplicates_repeated_citations() -> None:
    """The same structural ID cited twice appears once, at its first position."""
    text = "First [LC-105/2001::art-10]. Again [LC-105/2001::art-10]."

    citations = extract_citations(text)

    assert citations == ("LC-105/2001::art-10",)


def test_extract_citations_returns_empty_tuple_for_no_citations() -> None:
    """Text with no brackets at all yields no citations, not an error."""
    assert extract_citations("Uma resposta sem citações.") == ()


def test_extract_citation_candidates_keeps_malformed_brackets() -> None:
    """Unlike extract_citations, a malformed bracket is kept, not silently dropped.

    The post-generation citation auditor (ADR-0016) needs to see and flag a
    malformed candidate, not have it disappear before its well_formed check
    ever runs.
    """
    text = "See [this note] and [LC-105/2001::art-10]."

    candidates = extract_citation_candidates(text)

    assert candidates == ("this note", "LC-105/2001::art-10")


def test_extract_citation_candidates_deduplicates_repeated_candidates() -> None:
    """The same bracket content cited twice appears once, at its first position."""
    text = "First [LC-105/2001::art-10]. Again [LC-105/2001::art-10]."

    candidates = extract_citation_candidates(text)

    assert candidates == ("LC-105/2001::art-10",)


def test_extract_citation_candidates_returns_empty_tuple_for_no_brackets() -> None:
    """Text with no brackets at all yields no candidates, not an error."""
    assert extract_citation_candidates("Uma resposta sem citações.") == ()
