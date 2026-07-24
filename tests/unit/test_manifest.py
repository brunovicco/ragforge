"""Tests for the corpus manifest loader (ADR-0012)."""

from pathlib import Path

import pytest
import yaml

from ragforge.evaluation.manifest import load_corpus_manifest

ROOT = Path(__file__).resolve().parents[2]
REAL_MANIFEST_PATH = ROOT / "datasets/regrag-br/corpus_manifest.yaml"

_MINIMAL_ENABLED_DOC = {
    "norm_id": "NORM/2000",
    "source_path": "datasets/corpus/norm.htm",
    "extractor": "html",
    "expected_article_count": 5,
    "source_sha256": "a" * 64,
    "enabled": True,
}


def _write_manifest(tmp_path: Path, documents: list[dict[str, object]]) -> Path:
    path = tmp_path / "corpus_manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "corpus_id": "regrag-br",
                "corpus_version": "0.2",
                "documents": documents,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_corpus_manifest_parses_a_well_formed_document(tmp_path: Path) -> None:
    """Every field of a well-formed enabled entry round-trips into a CorpusDocument."""
    path = _write_manifest(tmp_path, [_MINIMAL_ENABLED_DOC])

    manifest = load_corpus_manifest(path)

    assert manifest.schema_version == 1
    assert manifest.corpus_id == "regrag-br"
    assert len(manifest.documents) == 1
    doc = manifest.documents[0]
    assert doc.norm_id == "NORM/2000"
    assert doc.extractor == "html"
    assert doc.expected_article_count == 5
    assert doc.enabled is True


def test_enabled_documents_excludes_disabled_entries(tmp_path: Path) -> None:
    """enabled_documents filters out documents not enabled for indexing."""
    disabled_doc: dict[str, object] = {
        "norm_id": "OLD/1900",
        "source_path": "datasets/corpus/old.htm",
        "extractor": "html",
        "expected_article_count": None,
        "source_sha256": None,
        "enabled": False,
    }
    path = _write_manifest(tmp_path, [_MINIMAL_ENABLED_DOC, disabled_doc])

    manifest = load_corpus_manifest(path)

    assert len(manifest.documents) == 2
    assert [doc.norm_id for doc in manifest.enabled_documents] == ["NORM/2000"]


def test_enabled_document_without_expected_article_count_is_rejected(tmp_path: Path) -> None:
    """An enabled document must carry a curated expected article count."""
    bad_doc = {**_MINIMAL_ENABLED_DOC, "expected_article_count": None}
    path = _write_manifest(tmp_path, [bad_doc])

    with pytest.raises(ValueError, match="expected_article_count"):
        load_corpus_manifest(path)


def test_enabled_document_without_source_sha256_is_rejected(tmp_path: Path) -> None:
    """An enabled document must carry a pinned source hash."""
    bad_doc = {**_MINIMAL_ENABLED_DOC, "source_sha256": None}
    path = _write_manifest(tmp_path, [bad_doc])

    with pytest.raises(ValueError, match="source_sha256"):
        load_corpus_manifest(path)


def test_load_real_corpus_manifest_has_five_enabled_documents() -> None:
    """The real manifest declares six norms; only five are curated/enabled today."""
    manifest = load_corpus_manifest(REAL_MANIFEST_PATH)

    assert len(manifest.documents) == 6
    assert len(manifest.enabled_documents) == 5
    disabled = {doc.norm_id for doc in manifest.documents} - {
        doc.norm_id for doc in manifest.enabled_documents
    }
    assert disabled == {"LEI-6385/1976"}
