"""Tests for the contextual retrieval pipeline, using a fake contextualizer."""

from ragforge.domain.models import Chunk
from ragforge.retrieval.contextual.pipeline import contextualize_chunks


class _FakeContextualizer:
    name = "fake-contextualizer"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def contextualize(self, document_text: str, chunk_text: str) -> str:
        self.calls.append((document_text, chunk_text))
        return f"Context for: {chunk_text[:10]}"


def test_contextualize_chunks_prepends_context_to_retrieval_text_only() -> None:
    """retrieval_text gains a context prefix; source_text is untouched (ADR-0015)."""
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
    contextualizer = _FakeContextualizer()

    result = contextualize_chunks("documento completo", chunks, contextualizer)

    assert result[0].retrieval_text == "Context for: Art. 1º di\n\nArt. 1º dispõe sobre X."
    assert result[1].retrieval_text == "Context for: Art. 2º di\n\nArt. 2º dispõe sobre Y."
    assert result[0].source_text == "Art. 1º dispõe sobre X."
    assert result[1].source_text == "Art. 2º dispõe sobre Y."
    assert contextualizer.calls == [
        ("documento completo", "Art. 1º dispõe sobre X."),
        ("documento completo", "Art. 2º dispõe sobre Y."),
    ]


def test_contextualize_chunks_preserves_provenance_fields() -> None:
    """Only retrieval_text changes - everything else stays intact."""
    chunk = Chunk(
        chunk_id="c1",
        source_text="Art. 1º",
        retrieval_text="Art. 1º",
        structural_ids=("LEI-1/2020::art-1",),
        parent_id="p1",
        metadata={"source_hash": "abc123"},
    )

    [result] = contextualize_chunks("documento", [chunk], _FakeContextualizer())

    assert result.chunk_id == "c1"
    assert result.source_text == "Art. 1º"
    assert result.structural_ids == ("LEI-1/2020::art-1",)
    assert result.parent_id == "p1"
    assert result.metadata == {"source_hash": "abc123"}


def test_contextualize_chunks_returns_empty_list_for_no_chunks() -> None:
    """An empty chunk list produces an empty result with no contextualizer calls."""
    contextualizer = _FakeContextualizer()

    result = contextualize_chunks("documento", [], contextualizer)

    assert result == []
    assert contextualizer.calls == []
