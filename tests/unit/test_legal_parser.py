"""Tests for the legal structural parser (ADR-0006)."""

from ragforge.chunking.legal_parser import NodeKind, parse_norm

NORM_ID = "RES-CMN-4893/2021"

SAMPLE = """
Preâmbulo do normativo.
Segunda linha do preâmbulo.

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

Art. 3º-A Artigo com sufixo de letra.
"""


def test_preamble_collects_lines_before_first_article() -> None:
    """Lines before the first Art. marker go to the preamble node."""
    tree = parse_norm(NORM_ID, SAMPLE)
    assert "Preâmbulo do normativo." in tree.preamble.text
    assert "Segunda linha do preâmbulo." in tree.preamble.text


def test_headings_are_not_nodes_but_stamp_article_context() -> None:
    """TÍTULO/CAPÍTULO lines never become tree nodes; they stamp heading_context."""
    tree = parse_norm(NORM_ID, SAMPLE)
    article = tree.articles[0]
    assert article.heading_context == ("TÍTULO I", "CAPÍTULO I")
    assert all(child.kind != NodeKind.PREAMBLE for child in tree.articles)


def test_article_with_paragrafo_unico() -> None:
    """A 'Parágrafo único' line becomes a paragraph child labeled par-unico."""
    tree = parse_norm(NORM_ID, SAMPLE)
    article = tree.articles[0]
    assert article.label == "art-1"
    assert len(article.children) == 1
    assert article.children[0].kind == NodeKind.PARAGRAPH
    assert article.children[0].label == "par-unico"


def test_article_nests_paragraphs_incisos_and_alineas() -> None:
    """§, inciso, and alínea markers attach to the correct innermost parent."""
    tree = parse_norm(NORM_ID, SAMPLE)
    article = tree.articles[1]
    assert article.label == "art-2"
    assert [p.label for p in article.children] == ["par-1", "par-2"]

    par1 = article.children[0]
    assert [i.label for i in par1.children] == ["inc-i", "inc-ii"]

    inc2 = par1.children[1]
    assert [a.label for a in inc2.children] == ["ali-a", "ali-b"]


def test_new_article_resets_open_paragraph_inciso_alinea() -> None:
    """Starting a new article closes any open paragraph/inciso/alínea from the previous one."""
    tree = parse_norm(NORM_ID, SAMPLE)
    article3 = tree.articles[2]
    assert article3.label == "art-3-a"
    assert article3.children == []


def test_text_lines_attach_to_innermost_open_node() -> None:
    """A plain continuation line attaches to the deepest currently open node."""
    tree = parse_norm(NORM_ID, SAMPLE)
    article = tree.articles[1]
    inc2 = article.children[0].children[1]
    alinea_b = inc2.children[1]
    assert "segunda alínea do inciso II." in alinea_b.text


def test_structural_id_composes_article_and_fragment() -> None:
    """structural_id renders {norm}::{art}::{fragment} for a nested path."""
    tree = parse_norm(NORM_ID, SAMPLE)
    article = tree.articles[1]
    par1 = article.children[0]
    inc2 = par1.children[1]
    assert tree.structural_id(article) == f"{NORM_ID}::art-2"
    assert tree.structural_id(article, (par1,)) == f"{NORM_ID}::art-2::par-1"
    assert tree.structural_id(article, (par1, inc2)) == f"{NORM_ID}::art-2::par-1-inc-ii"


def test_article_ids_include_all_descendants_in_order() -> None:
    """article_ids walks the whole subtree and returns one id per node."""
    tree = parse_norm(NORM_ID, SAMPLE)
    article = tree.articles[1]
    ids = tree.article_ids(article)
    assert ids[0] == f"{NORM_ID}::art-2"
    assert f"{NORM_ID}::art-2::par-1" in ids
    assert f"{NORM_ID}::art-2::par-1-inc-i" in ids
    assert f"{NORM_ID}::art-2::par-1-inc-ii-ali-a" in ids
    assert f"{NORM_ID}::art-2::par-1-inc-ii-ali-b" in ids
    assert f"{NORM_ID}::art-2::par-2" in ids


def test_full_text_includes_node_and_descendants_in_document_order() -> None:
    """full_text concatenates a node's own text with all descendants, depth-first."""
    tree = parse_norm(NORM_ID, SAMPLE)
    article = tree.articles[1]
    assert "primeira alínea do inciso II;" in article.full_text
    assert article.full_text.index("§ 1º") < article.full_text.index("I - primeiro inciso")
