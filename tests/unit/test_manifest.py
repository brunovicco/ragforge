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
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def test_content_hash_is_the_same_regardless_of_document_order(tmp_path: Path) -> None:
    """content_hash depends only on which documents are enabled, not YAML entry order."""
    second_doc = {**_MINIMAL_ENABLED_DOC, "norm_id": "NORM/2001", "source_sha256": "b" * 64}
    forward = load_corpus_manifest(
        _write_manifest(tmp_path / "a", [_MINIMAL_ENABLED_DOC, second_doc])
    )
    reversed_manifest = load_corpus_manifest(
        _write_manifest(tmp_path / "b", [second_doc, _MINIMAL_ENABLED_DOC])
    )

    assert forward.content_hash == reversed_manifest.content_hash


def test_content_hash_changes_when_a_source_hash_changes(tmp_path: Path) -> None:
    """A different pinned source hash for the same document changes content_hash."""
    original = load_corpus_manifest(_write_manifest(tmp_path / "a", [_MINIMAL_ENABLED_DOC]))
    changed_doc = {**_MINIMAL_ENABLED_DOC, "source_sha256": "c" * 64}
    changed = load_corpus_manifest(_write_manifest(tmp_path / "b", [changed_doc]))

    assert original.content_hash != changed.content_hash


def test_content_hash_ignores_disabled_documents(tmp_path: Path) -> None:
    """A disabled document's presence (or absence) doesn't affect content_hash."""
    disabled_doc: dict[str, object] = {
        "norm_id": "OLD/1900",
        "source_path": "datasets/corpus/old.htm",
        "extractor": "html",
        "expected_article_count": None,
        "source_sha256": None,
        "enabled": False,
    }
    without_disabled = load_corpus_manifest(_write_manifest(tmp_path / "a", [_MINIMAL_ENABLED_DOC]))
    with_disabled = load_corpus_manifest(
        _write_manifest(tmp_path / "b", [_MINIMAL_ENABLED_DOC, disabled_doc])
    )

    assert without_disabled.content_hash == with_disabled.content_hash


def test_load_real_corpus_manifest_has_five_enabled_documents() -> None:
    """The real manifest declares six norms; only five are curated/enabled today."""
    manifest = load_corpus_manifest(REAL_MANIFEST_PATH)

    assert len(manifest.documents) == 6
    assert len(manifest.enabled_documents) == 5
    disabled = {doc.norm_id for doc in manifest.documents} - {
        doc.norm_id for doc in manifest.enabled_documents
    }
    assert disabled == {"LEI-6385/1976"}
