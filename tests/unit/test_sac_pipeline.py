"""Tests for the SAC (Summary-Augmented Chunking) pipeline (ADR-0015)."""

from ragforge.domain.models import Chunk
from ragforge.retrieval.sac.pipeline import apply_document_summary


def test_apply_document_summary_prepends_summary_to_retrieval_text_only() -> None:
    """retrieval_text gains a summary prefix; source_text is untouched (ADR-0015)."""
    chunks = [
        Chunk(
            chunk_id="c1",
            source_text="Art. 1º dispõe sobre X.",
            retrieval_text="Art. 1º dispõe sobre X.",
            structural_ids=("c1",),
        ),
        Chunk(
            chunk_id="c2",
            source_text="Art. 2º dispõe sobre Y.",
            retrieval_text="Art. 2º dispõe sobre Y.",
            structural_ids=("c2",),
        ),
    ]

    result = apply_document_summary("Resumo do documento.", chunks)

    assert result[0].retrieval_text == "Resumo do documento.\n\nArt. 1º dispõe sobre X."
    assert result[1].retrieval_text == "Resumo do documento.\n\nArt. 2º dispõe sobre Y."
    assert result[0].source_text == "Art. 1º dispõe sobre X."
    assert result[1].source_text == "Art. 2º dispõe sobre Y."


def test_apply_document_summary_composes_on_top_of_an_existing_context_prefix() -> None:
    """Applied after contextualization, the summary stacks on top (ADR-0015's sac_contextual)."""
    already_contextualized = Chunk(
        chunk_id="c1",
        source_text="Art. 1º dispõe sobre X.",
        retrieval_text="Contexto do chunk.\n\nArt. 1º dispõe sobre X.",
        structural_ids=("c1",),
    )

    [result] = apply_document_summary("Resumo do documento.", [already_contextualized])

    assert result.retrieval_text == (
        "Resumo do documento.\n\nContexto do chunk.\n\nArt. 1º dispõe sobre X."
    )
    assert result.source_text == "Art. 1º dispõe sobre X."


def test_apply_document_summary_preserves_provenance_fields() -> None:
    """Only retrieval_text changes - everything else stays intact."""
    chunk = Chunk(
        chunk_id="c1",
        source_text="Art. 1º",
        retrieval_text="Art. 1º",
        structural_ids=("LEI-1/2020::art-1",),
        parent_id="p1",
        metadata={"source_hash": "abc123"},
    )

    [result] = apply_document_summary("Resumo.", [chunk])

    assert result.chunk_id == "c1"
    assert result.source_text == "Art. 1º"
    assert result.structural_ids == ("LEI-1/2020::art-1",)
    assert result.parent_id == "p1"
    assert result.metadata == {"source_hash": "abc123"}


def test_apply_document_summary_returns_empty_list_for_no_chunks() -> None:
    """An empty chunk list produces an empty result."""
    assert apply_document_summary("Resumo.", []) == []
