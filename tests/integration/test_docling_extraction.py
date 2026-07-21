"""Integration test for the Docling extractor against a real corpus PDF (ADR-0006).

Opt-in only: excluded from the default `pytest` run by the `-m "not integration"`
addopts, because Docling loads layout-analysis models on first use (multi-second
import, large dependency chain). Run explicitly with `uv run pytest -m integration`.
"""

from pathlib import Path

import pytest

CORPUS_PDF = Path(__file__).resolve().parents[2] / "datasets/corpus/bacen/RES-CMN-4893-2021.pdf"


@pytest.mark.integration
def test_extracts_real_resolution_text_via_docling() -> None:
    """Docling converts a real BCB resolution PDF into text carrying its structure.

    The import is deliberately inside the test body: importing docling triggers a
    slow, heavy dependency load, and it must not happen just from collecting this
    module when `-m "not integration"` deselects the test.
    """
    from ragforge.ingestion.docling_extractor import DoclingExtractor

    text = DoclingExtractor().extract(CORPUS_PDF)
    assert "4.893" in text
    assert "Art. 1" in text
