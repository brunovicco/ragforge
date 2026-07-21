"""Tests for the golden-set judgments loader and data (ADR-0002).

The second test class is the real safeguard: it parses the actual corpus
documents referenced by datasets/regrag-br/judgments.json and checks every
judged structural ID genuinely exists in their real chunk_norm output - not
just that the file is well-formed JSON. This is what catches a wrong or
mistyped structural ID (e.g. a hyphen where the canonical form uses "::").
"""

from pathlib import Path

from ragforge.chunking.chunker import chunk_norm
from ragforge.chunking.legal_parser import parse_norm
from ragforge.domain.models import Judgment, QueryClass
from ragforge.evaluation.judgments import load_judgments
from ragforge.ingestion.html_extractor import HtmlTextExtractor
from ragforge.ingestion.pymupdf_extractor import PyMuPdfExtractor

CORPUS = Path(__file__).resolve().parents[2] / "datasets/corpus"
JUDGMENTS_PATH = Path(__file__).resolve().parents[2] / "datasets/regrag-br/judgments.json"

_NORM_SOURCES = {
    "LC-105/2001": (CORPUS / "lc-lgpd/LC-105-2001.htm", HtmlTextExtractor().extract),
    "RES-CMN-4893/2021": (CORPUS / "bacen/RES-CMN-4893-2021.pdf", PyMuPdfExtractor().extract),
    "RES-CMN-5274/2025": (CORPUS / "bacen/RES-CMN-5274-2025.htm", HtmlTextExtractor().extract),
    "LEI-13709/2018": (CORPUS / "lc-lgpd/LEI-13709-2018-LGPD.htm", HtmlTextExtractor().extract),
}


def _real_structural_ids_by_norm() -> dict[str, set[str]]:
    """Parse the real corpus documents and collect every structural ID they produce."""
    ids_by_norm: dict[str, set[str]] = {}
    for norm_id, (path, extract) in _NORM_SOURCES.items():
        text = extract(path)
        tree = parse_norm(norm_id, text)
        chunks = chunk_norm(tree)
        ids_by_norm[norm_id] = {ref for chunk in chunks for ref in chunk.structural_ids}
    return ids_by_norm


def test_load_judgments_returns_one_judgment_per_entry() -> None:
    """The loader parses every entry in the seed file into a Judgment."""
    judgments = load_judgments(JUDGMENTS_PATH)
    assert len(judgments) == 20
    assert all(isinstance(j, Judgment) for j in judgments)
    assert len({j.question_id for j in judgments}) == 20, "question_id must be unique"


def test_load_judgments_parses_query_class() -> None:
    """Each judgment's query is tagged with a valid RegRAG-BR query class."""
    judgments = load_judgments(JUDGMENTS_PATH)
    assert all(j.query.query_class is not None for j in judgments)
    assert all(isinstance(j.query.query_class, QueryClass) for j in judgments)


def test_load_judgments_handles_the_unanswerable_question_with_no_relevant_refs() -> None:
    """The one unanswerable-class question is loaded with an empty relevant_refs tuple."""
    judgments = load_judgments(JUDGMENTS_PATH)
    unanswerable = [j for j in judgments if j.query.query_class == QueryClass.UNANSWERABLE]
    assert len(unanswerable) == 1
    assert unanswerable[0].relevant_refs == ()


def test_every_judged_structural_id_exists_in_the_real_parsed_corpus() -> None:
    """Every structural ID in the golden set is real: it appears in chunk_norm's own output.

    This is the safeguard against a mistyped or invented reference - each ID is
    checked against the actual article/paragraph structure of the real document,
    not merely well-formed as a string.
    """
    judgments = load_judgments(JUDGMENTS_PATH)
    real_ids = _real_structural_ids_by_norm()

    missing: list[str] = []
    for judgment in judgments:
        for judged in judgment.relevant_refs:
            norm_id = judged.ref.norm
            canonical = judged.ref.canonical
            if norm_id not in real_ids or canonical not in real_ids[norm_id]:
                missing.append(f"{judgment.question_id}: {canonical}")

    assert not missing, f"structural IDs not found in the real corpus: {missing}"
