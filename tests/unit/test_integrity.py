"""Tests for the preflight integrity gate (ADR-0012)."""

from pathlib import Path

import pytest

from ragforge.domain.models import (
    Chunk,
    JudgedRef,
    Judgment,
    Query,
    QueryClass,
    RelevanceGrade,
    StructuralRef,
)
from ragforge.evaluation.integrity import (
    IntegrityError,
    verify_source_integrity,
    verify_split_integrity,
    verify_structural_references,
)
from ragforge.evaluation.manifest import CorpusDocument, CorpusManifest
from ragforge.evaluation.split import Split
from ragforge.ingestion.snapshot import snapshot_hash

NORM = "NORM/2000"
ART_1 = f"{NORM}::art-1"


def _manifest(*docs: CorpusDocument) -> CorpusManifest:
    return CorpusManifest(schema_version=1, corpus_id="test", corpus_version="0.1", documents=docs)


def _doc(
    norm_id: str, source_path: str, sha256: str | None, enabled: bool = True
) -> CorpusDocument:
    return CorpusDocument(
        norm_id=norm_id,
        source_path=source_path,
        extractor="html",
        expected_article_count=1 if enabled else None,
        source_sha256=sha256,
        enabled=enabled,
    )


def _judgment(question_id: str, *canonical_refs: str, unanswerable: bool = False) -> Judgment:
    return Judgment(
        question_id=question_id,
        query=Query(
            text=question_id,
            query_class=QueryClass.UNANSWERABLE if unanswerable else QueryClass.EXACT_FACTUAL,
        ),
        relevant_refs=tuple(
            JudgedRef(ref=StructuralRef.parse(c), grade=RelevanceGrade.RELEVANT)
            for c in canonical_refs
        ),
    )


class TestVerifySourceIntegrity:
    def test_passes_when_every_enabled_source_exists_and_matches_its_hash(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "norm.htm"
        source.write_text("<html>real text</html>", encoding="utf-8")
        manifest = _manifest(_doc(NORM, "norm.htm", snapshot_hash(source)))

        verify_source_integrity(manifest, root=tmp_path)

    def test_fails_when_an_enabled_source_file_is_missing(self, tmp_path: Path) -> None:
        manifest = _manifest(_doc(NORM, "missing.htm", "a" * 64))

        with pytest.raises(IntegrityError, match="not found"):
            verify_source_integrity(manifest, root=tmp_path)

    def test_fails_when_an_enabled_source_hash_does_not_match_the_manifest(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "norm.htm"
        source.write_text("<html>real text</html>", encoding="utf-8")
        manifest = _manifest(_doc(NORM, "norm.htm", "0" * 64))

        with pytest.raises(IntegrityError, match="hash mismatch"):
            verify_source_integrity(manifest, root=tmp_path)

    def test_ignores_a_disabled_document_missing_its_source_file(self, tmp_path: Path) -> None:
        manifest = _manifest(_doc(NORM, "missing.htm", None, enabled=False))

        verify_source_integrity(manifest, root=tmp_path)


class TestVerifySplitIntegrity:
    def test_passes_when_split_and_judgments_select_exactly_the_same_questions(self) -> None:
        split = Split(
            schema_version=1, dataset_version="0.1", train=(), validation=(), test=("q1", "q2")
        )
        judgments = [_judgment("q1", ART_1), _judgment("q2", ART_1)]

        verify_split_integrity(split, judgments)

    def test_fails_when_a_question_id_is_duplicated_across_partitions(self) -> None:
        split = Split(
            schema_version=1, dataset_version="0.1", train=("q1",), validation=("q1",), test=()
        )
        judgments = [_judgment("q1", ART_1)]

        with pytest.raises(IntegrityError, match="duplicated"):
            verify_split_integrity(split, judgments)

    def test_fails_when_the_split_references_an_unknown_question_id(self) -> None:
        split = Split(
            schema_version=1, dataset_version="0.1", train=(), validation=(), test=("ghost",)
        )
        judgments = [_judgment("q1", ART_1)]

        with pytest.raises(IntegrityError, match="unknown question IDs"):
            verify_split_integrity(split, judgments)

    def test_fails_when_a_golden_set_question_is_not_covered_by_the_split(self) -> None:
        split = Split(schema_version=1, dataset_version="0.1", train=(), validation=(), test=())
        judgments = [_judgment("q1", ART_1)]

        with pytest.raises(IntegrityError, match="not selected"):
            verify_split_integrity(split, judgments)


class TestVerifyStructuralReferences:
    def _chunk(self) -> Chunk:
        return Chunk(
            chunk_id="c1", source_text="text", retrieval_text="text", structural_ids=(ART_1,)
        )

    def test_passes_when_every_relevant_ref_resolves_in_the_indexed_corpus(self) -> None:
        judgments = [_judgment("q1", ART_1)]
        documents = {NORM: ("full text", [self._chunk()])}

        verify_structural_references(judgments, documents)

    def test_passes_for_an_unanswerable_judgment_with_no_relevant_refs(self) -> None:
        judgments = [_judgment("q1", unanswerable=True)]
        documents: dict[str, tuple[str, list[Chunk]]] = {}

        verify_structural_references(judgments, documents)

    def test_fails_when_a_ref_points_to_a_document_not_in_the_indexed_corpus(self) -> None:
        judgments = [_judgment("q1", ART_1)]
        documents: dict[str, tuple[str, list[Chunk]]] = {}

        with pytest.raises(IntegrityError, match="not indexed"):
            verify_structural_references(judgments, documents)

    def test_fails_when_a_ref_does_not_resolve_to_any_real_structural_id(self) -> None:
        judgments = [_judgment("q1", f"{NORM}::art-99")]
        documents = {NORM: ("full text", [self._chunk()])}

        with pytest.raises(IntegrityError, match="does not resolve"):
            verify_structural_references(judgments, documents)
