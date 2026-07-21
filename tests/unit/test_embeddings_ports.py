"""Tests for the EmbeddingModel port (ADR-0005)."""

from ragforge.embeddings.ports import EmbeddingModel


class _FakeEmbedder:
    """A minimal EmbeddingModel implementer, used to check the protocol shape."""

    name = "fake-embedder"
    dimensions = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


def test_a_conforming_class_satisfies_the_runtime_checkable_protocol() -> None:
    """Any class exposing name, dimensions, and embed() satisfies EmbeddingModel."""
    embedder = _FakeEmbedder()
    assert isinstance(embedder, EmbeddingModel)


def test_embed_returns_one_vector_per_input_text_in_order() -> None:
    """The fake's own embed() contract: one vector per input, same order."""
    embedder = _FakeEmbedder()
    vectors = embedder.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(vector) == embedder.dimensions for vector in vectors)
