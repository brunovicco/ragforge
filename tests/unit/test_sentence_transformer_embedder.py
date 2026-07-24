"""Unit tests for the sentence-transformers adapter's construction wiring (ADR-0013).

Loading a real model is out of bounds for a unit test (network on first
download, ~17s import - see tests/integration/test_sentence_transformer_embedder.py
for that). These tests substitute a fake in place of
sentence_transformers.SentenceTransformer to verify constructor argument
plumbing only: the revision kwarg forwarding and its "main" default.
"""

from typing import Any, ClassVar

import pytest

from ragforge.embeddings.errors import EmbeddingError
from ragforge.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder


class _FakeSentenceTransformer:
    """Records the arguments it was constructed with; encodes deterministically."""

    last_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(
        self, model_name: str, device: str | None = None, revision: str | None = None
    ) -> None:
        _FakeSentenceTransformer.last_kwargs = {
            "model_name": model_name,
            "device": device,
            "revision": revision,
        }
        self._dimensions = 3

    def get_embedding_dimension(self) -> int:
        return self._dimensions

    def encode(self, texts: list[str], convert_to_numpy: bool, normalize_embeddings: bool) -> Any:
        class _FakeArray:
            def __init__(self, rows: int, cols: int) -> None:
                self._rows, self._cols = rows, cols

            def tolist(self) -> list[list[float]]:
                return [[0.0] * self._cols for _ in range(self._rows)]

        return _FakeArray(len(texts), self._dimensions)


@pytest.fixture(autouse=True)
def _patch_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ragforge.embeddings.sentence_transformer_embedder.SentenceTransformer",
        _FakeSentenceTransformer,
    )


def test_revision_defaults_to_main_when_not_specified() -> None:
    """Omitting revision resolves to "main", the same branch Hugging Face would use."""
    embedder = SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B")

    assert embedder.revision == "main"
    assert _FakeSentenceTransformer.last_kwargs["revision"] is None


def test_revision_is_forwarded_to_the_underlying_model() -> None:
    """An explicit revision is passed through to SentenceTransformer and recorded."""
    embedder = SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B", revision="abc123")

    assert embedder.revision == "abc123"
    assert _FakeSentenceTransformer.last_kwargs["revision"] == "abc123"


def test_dimensions_and_name_are_exposed_from_the_loaded_model() -> None:
    """name/dimensions come from the constructor argument and the model itself."""
    embedder = SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B")

    assert embedder.name == "Qwen/Qwen3-Embedding-0.6B"
    assert embedder.dimensions == 3


def test_embed_returns_one_vector_per_text_matching_the_model_dimension() -> None:
    """embed() delegates to the (fake) model's encode() and reshapes its output."""
    embedder = SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B")

    vectors = embedder.embed(["a", "b"])

    assert len(vectors) == 2
    assert all(len(vector) == 3 for vector in vectors)


def test_load_failure_is_translated_to_embedding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure inside SentenceTransformer's constructor becomes an EmbeddingError."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no internet")

    monkeypatch.setattr(
        "ragforge.embeddings.sentence_transformer_embedder.SentenceTransformer", _raise
    )

    with pytest.raises(EmbeddingError, match="failed to load model"):
        SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B")
