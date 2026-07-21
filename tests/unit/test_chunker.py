"""Tests for structural chunk derivation (ADR-0006)."""

from ragforge.chunking.chunker import chunk_norm
from ragforge.chunking.legal_parser import parse_norm

NORM_ID = "RES-CMN-4893/2021"

SAMPLE = """
Preâmbulo do normativo.

TÍTULO I
Disposições Gerais

CAPÍTULO I
Do Objeto

Art. 1º Esta Resolução dispõe sobre o objeto.
Parágrafo único. O disposto neste artigo aplica-se a todas as instituições.

Art. 2º Este artigo possui incisos e parágrafos.
§ 1º Primeiro parágrafo do artigo 2.
I - primeiro inciso do parágrafo;
II - segundo inciso do parágrafo;
a) primeira alínea do inciso II;
b) segunda alínea do inciso II.
§ 2º Segundo parágrafo do artigo 2.

Art. 3º-A Artigo com sufixo de letra e sem filhos.
"""

NO_PREAMBLE = """
Art. 1º Único artigo, sem preâmbulo.
"""


def test_no_preamble_chunk_when_preamble_is_empty() -> None:
    """A norm with nothing before its first article yields no preamble chunk."""
    tree = parse_norm(NORM_ID, NO_PREAMBLE)
    chunks = chunk_norm(tree)
    assert all(chunk.metadata.get("role") != "preamble" for chunk in chunks)


def test_preamble_chunk_carries_its_own_structural_id() -> None:
    """A non-empty preamble becomes one chunk with a stable, distinct id."""
    tree = parse_norm(NORM_ID, SAMPLE)
    chunks = chunk_norm(tree)
    preamble_chunks = [c for c in chunks if c.metadata.get("role") == "preamble"]
    assert len(preamble_chunks) == 1
    assert preamble_chunks[0].chunk_id == f"{NORM_ID}::preamble"
    assert preamble_chunks[0].structural_ids == (f"{NORM_ID}::preamble",)


def test_one_chunk_per_article_when_under_max_chars() -> None:
    """Short articles produce exactly one chunk each, with no fragment chunks."""
    tree = parse_norm(NORM_ID, SAMPLE)
    chunks = chunk_norm(tree, max_chars=10_000)
    article_chunks = [c for c in chunks if c.metadata.get("role") == "article"]
    fragment_chunks = [c for c in chunks if c.metadata.get("role") == "fragment"]
    assert [c.chunk_id for c in article_chunks] == [
        f"{NORM_ID}::art-1",
        f"{NORM_ID}::art-2",
        f"{NORM_ID}::art-3-a",
    ]
    assert fragment_chunks == []


def test_article_ids_are_included_in_article_chunk_structural_ids() -> None:
    """An article chunk's structural_ids covers the article and every descendant."""
    tree = parse_norm(NORM_ID, SAMPLE)
    chunks = chunk_norm(tree, max_chars=10_000)
    art2 = next(c for c in chunks if c.chunk_id == f"{NORM_ID}::art-2")
    assert art2.structural_ids[0] == f"{NORM_ID}::art-2"
    assert f"{NORM_ID}::art-2::par-1-inc-ii-ali-b" in art2.structural_ids


def test_long_article_subdivides_into_fragment_chunks_with_parent_id() -> None:
    """An article over max_chars gets one fragment chunk per top-level child."""
    tree = parse_norm(NORM_ID, SAMPLE)
    chunks = chunk_norm(tree, max_chars=1)
    art2_fragments = [
        c
        for c in chunks
        if c.metadata.get("role") == "fragment" and c.parent_id == f"{NORM_ID}::art-2"
    ]
    assert [c.chunk_id for c in art2_fragments] == [
        f"{NORM_ID}::art-2::par-1",
        f"{NORM_ID}::art-2::par-2",
    ]


def test_fragment_structural_ids_cover_its_own_subtree() -> None:
    """A fragment chunk's structural_ids include the fragment and its descendants only."""
    tree = parse_norm(NORM_ID, SAMPLE)
    chunks = chunk_norm(tree, max_chars=1)
    par1 = next(c for c in chunks if c.chunk_id == f"{NORM_ID}::art-2::par-1")
    assert par1.structural_ids == (
        f"{NORM_ID}::art-2::par-1",
        f"{NORM_ID}::art-2::par-1-inc-i",
        f"{NORM_ID}::art-2::par-1-inc-ii",
        f"{NORM_ID}::art-2::par-1-inc-ii-ali-a",
        f"{NORM_ID}::art-2::par-1-inc-ii-ali-b",
    )


def test_article_without_children_never_subdivides() -> None:
    """An over-long article with no children yields only its article chunk."""
    tree = parse_norm(NORM_ID, SAMPLE)
    chunks = chunk_norm(tree, max_chars=1)
    fragments_of_art3 = [c for c in chunks if c.parent_id == f"{NORM_ID}::art-3-a"]
    assert fragments_of_art3 == []
    assert any(c.chunk_id == f"{NORM_ID}::art-3-a" for c in chunks)


def test_article_metadata_includes_heading_section_path() -> None:
    """Chunk metadata exposes the accumulated TÍTULO/CAPÍTULO context, joined."""
    tree = parse_norm(NORM_ID, SAMPLE)
    chunks = chunk_norm(tree, max_chars=10_000)
    art1 = next(c for c in chunks if c.chunk_id == f"{NORM_ID}::art-1")
    assert art1.metadata["section"] == "TÍTULO I > CAPÍTULO I"
