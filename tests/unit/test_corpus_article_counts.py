"""Wire validate_article_count against real corpus documents (ADR-0006, decision 5).

Curated expected counts, and how each was determined:

- LC-105-2001 (13): matches this law's well-known article count; the parser's
  output is a clean, gap-free, duplicate-free art-1..art-13 sequence.
- LEI-13709-2018-LGPD (65): the LGPD's article count is public knowledge. The
  parser currently returns 105 because Planalto's "texto compilado" page repeats
  several articles' wording inline as amendment-history annotations (e.g. Art. 20
  appears three times with slightly different wording from different redações) -
  the gate is expected to, and does, reject this document.
- RES-CMN-4893-2021 (28) / RES-CMN-5274-2025 (3): curated from a clean, gap-free,
  duplicate-free parser run over the real PDF/HTML - no external ground truth
  needed since the sequence is already internally consistent.
- ICVM-607-2019 (113): the instruction's main body runs art-1..art-113; two
  annexes ("Anexo") restart their own numbering at "Art. 1º", which the parser
  currently (incorrectly) also counts as top-level articles of the main
  instrument - the gate is expected to, and does, reject this document until the
  parser learns to scope annex numbering separately.

Lei 6.385/1976 is deliberately not included: the parser's current output on that
document is internally inconsistent (gaps and duplicate labels), a known,
unresolved long-tail case - this older Planalto page appears to use different
HTML formatting - that needs its own investigation before a curated count would
mean anything.
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
    """One curated corpus entry: where to find it, and what to expect."""

    norm_id: str
    path: Path
    extract: Callable[[Path], str]
    expected_count: int
    should_pass: bool


CURATED: tuple[CuratedNorm, ...] = (
    CuratedNorm(
        norm_id="LC-105-2001",
        path=CORPUS / "lc-lgpd/LC-105-2001.htm",
        extract=HtmlTextExtractor().extract,
        expected_count=13,
        should_pass=True,
    ),
    CuratedNorm(
        norm_id="LEI-13709-2018-LGPD",
        path=CORPUS / "lc-lgpd/LEI-13709-2018-LGPD.htm",
        extract=HtmlTextExtractor().extract,
        expected_count=65,
        should_pass=False,
    ),
    CuratedNorm(
        norm_id="RES-CMN-4893-2021",
        path=CORPUS / "bacen/RES-CMN-4893-2021.pdf",
        extract=PyMuPdfExtractor().extract,
        expected_count=28,
        should_pass=True,
    ),
    CuratedNorm(
        norm_id="RES-CMN-5274-2025",
        path=CORPUS / "bacen/RES-CMN-5274-2025.htm",
        extract=HtmlTextExtractor().extract,
        expected_count=3,
        should_pass=True,
    ),
    CuratedNorm(
        norm_id="ICVM-607-2019",
        path=CORPUS / "cvm/ICVM-607-2019.pdf",
        extract=PyMuPdfExtractor().extract,
        expected_count=113,
        should_pass=False,
    ),
)


@pytest.mark.parametrize("norm", CURATED, ids=[norm.norm_id for norm in CURATED])
def test_validate_article_count_against_real_corpus(norm: CuratedNorm) -> None:
    """The gate accepts norms whose curated count matches, rejects those that don't."""
    text = norm.extract(norm.path)
    tree = parse_norm(norm.norm_id, text)

    if norm.should_pass:
        validate_article_count(tree, norm.expected_count)
    else:
        with pytest.raises(ArticleCountMismatchError):
            validate_article_count(tree, norm.expected_count)
