"""Wire validate_article_count against real corpus documents (ADR-0006, decision 5).

`parse_norm` merges a repeated ``Art. N`` (same label already seen, outside any
annex) into the existing article node instead of creating a duplicate -
Planalto's "texto compilado" pages commonly restate an article's superseded
wording inline as amendment-history annotations. It also scopes article
numbering separately inside each "ANEXO", since annexes commonly restart their
own numbering at "Art. 1º" - without that scoping, an annex's Art. 1º would
either collide with (and pollute) the main body's real Art. 1, or inflate the
count as a spurious duplicate. See git history of this file for the pre-fix
numbers and the anomalies they exposed.

Curated expected counts, and how each was determined:

- LC-105-2001 (13): matches this law's well-known article count; the parser's
  output is a clean, gap-free art-1..art-13 sequence.
- LEI-13709-2018-LGPD (79): higher than the "65 articles" commonly cited for
  the LGPD's original 2018 enactment, because the compiled/consolidated text
  Planalto serves today also includes later insertions - Art. 55-A..55-M (13
  articles, added by Lei 13.853/2019 for ANPD sanctions) and Art. 58-A/58-B (2
  articles) - and is missing Art. 57 entirely (vetoed in 2018, never
  reinstated). 65 + 13 + 2 - 1 = 79. Confirmed by direct inspection: the raw
  page has no "Art. 57" occurrence anywhere.
- RES-CMN-4893-2021 (28) / RES-CMN-5274-2025 (3): curated from a clean,
  gap-free parser run over the real PDF/HTML - no external ground truth needed
  since the sequence is already internally consistent.
- ICVM-607-2019 (115): the instruction's main body runs art-1..art-113 (clean,
  no pollution from the annexes). Two annexes - "ANEXO 64" and "ANEXO 73" in
  this instruction's own numbering, confirmed as stable, single-occurrence
  headings cross-referenced by the same numbers elsewhere in the body text,
  not page numbers - each restart at their own "Art. 1º", scoped to labels
  `anexo-64-art-1` and `anexo-73-art-1` distinct from the main body's `art-1`.
  113 + 2 = 115.

Lei 6.385/1976 is still deliberately excluded: the merge removed its duplicate
labels, but real gaps remain (many articles missing entirely, likely genuine
repeals across this 1976 law's many amendments) that need legal research, not
a parser fix, before a curated count would mean anything.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from ragforge.chunking.legal_parser import parse_norm
from ragforge.chunking.validation import ArticleCountMismatchError, validate_article_count
from ragforge.ingestion.html_extractor import HtmlTextExtractor
from ragforge.ingestion.pymupdf_extractor import PyMuPdfExtractor

CORPUS = Path(__file__).resolve().parents[2] / "datasets/corpus"


@dataclass(frozen=True, slots=True)
class CuratedNorm:
    """One curated corpus entry: where to find it, and its expected article count."""

    norm_id: str
    path: Path
    extract: Callable[[Path], str]
    expected_count: int


CURATED: tuple[CuratedNorm, ...] = (
    CuratedNorm(
        norm_id="LC-105-2001",
        path=CORPUS / "lc-lgpd/LC-105-2001.htm",
        extract=HtmlTextExtractor().extract,
        expected_count=13,
    ),
    CuratedNorm(
        norm_id="LEI-13709-2018-LGPD",
        path=CORPUS / "lc-lgpd/LEI-13709-2018-LGPD.htm",
        extract=HtmlTextExtractor().extract,
        expected_count=79,
    ),
    CuratedNorm(
        norm_id="RES-CMN-4893-2021",
        path=CORPUS / "bacen/RES-CMN-4893-2021.pdf",
        extract=PyMuPdfExtractor().extract,
        expected_count=28,
    ),
    CuratedNorm(
        norm_id="RES-CMN-5274-2025",
        path=CORPUS / "bacen/RES-CMN-5274-2025.htm",
        extract=HtmlTextExtractor().extract,
        expected_count=3,
    ),
    CuratedNorm(
        norm_id="ICVM-607-2019",
        path=CORPUS / "cvm/ICVM-607-2019.pdf",
        extract=PyMuPdfExtractor().extract,
        expected_count=115,
    ),
)


@pytest.mark.parametrize("norm", CURATED, ids=[norm.norm_id for norm in CURATED])
def test_validate_article_count_against_real_corpus(norm: CuratedNorm) -> None:
    """The gate accepts every currently curated real norm at its reviewed count."""
    text = norm.extract(norm.path)
    tree = parse_norm(norm.norm_id, text)
    validate_article_count(tree, norm.expected_count)


def test_validate_article_count_rejects_a_real_norm_at_the_wrong_count() -> None:
    """The gate still rejects a real, correctly-parsed norm given a wrong curated count."""
    first = CURATED[0]
    text = first.extract(first.path)
    tree = parse_norm(first.norm_id, text)

    with pytest.raises(ArticleCountMismatchError):
        validate_article_count(tree, first.expected_count + 1)
