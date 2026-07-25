"""Tests for persistent per-text embedding caching."""

from pathlib import Path

import pytest

from ragforge.adapters.llm_cache import FileLLMCache
from ragforge.embeddings.caching import CachedEmbeddingModel
from ragforge.embeddings.identity import NO_QUERY_INSTRUCTION_HASH, EmbeddingIdentity


class _CountingEmbedder:
    name = "counting"
    dimensions = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]


def _identity(model: str = "model-a") -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider="local",
        model=model,
        revision="revision",
        dimensions=2,
        normalize=True,
        query_instruction_hash=NO_QUERY_INSTRUCTION_HASH,
        runtime="local",
    )


def test_embedding_cache_batches_unique_misses_and_preserves_order(tmp_path: Path) -> None:
    """Repeated texts are encoded once while output order and duplicates remain intact."""
    delegate = _CountingEmbedder()
    embedder = CachedEmbeddingModel(delegate, _identity(), FileLLMCache(tmp_path))

    vectors = embedder.embed(["aa", "bbb", "aa"])

    assert delegate.calls == [["aa", "bbb"]]
    assert vectors == [[2.0, 1.0], [3.0, 1.0], [2.0, 1.0]]


def test_embedding_cache_survives_a_new_wrapper_instance(tmp_path: Path) -> None:
    """A later run with the same identity reuses the file-backed vector."""
    first_delegate = _CountingEmbedder()
    CachedEmbeddingModel(first_delegate, _identity(), FileLLMCache(tmp_path)).embed(["text"])
    second_delegate = _CountingEmbedder()

    vectors = CachedEmbeddingModel(second_delegate, _identity(), FileLLMCache(tmp_path)).embed(
        ["text"]
    )

    assert vectors == [[4.0, 1.0]]
    assert second_delegate.calls == []


def test_embedding_cache_isolated_by_embedding_identity(tmp_path: Path) -> None:
    """Changing the model identity cannot reuse vectors from another embedding space."""
    first_delegate = _CountingEmbedder()
    CachedEmbeddingModel(first_delegate, _identity("model-a"), FileLLMCache(tmp_path)).embed(
        ["text"]
    )
    second_delegate = _CountingEmbedder()

    CachedEmbeddingModel(second_delegate, _identity("model-b"), FileLLMCache(tmp_path)).embed(
        ["text"]
    )

    assert second_delegate.calls == [["text"]]


def test_embedding_cache_rejects_a_corrupt_vector(tmp_path: Path) -> None:
    """A cached vector with the wrong dimension fails closed."""
    delegate = _CountingEmbedder()
    embedder = CachedEmbeddingModel(delegate, _identity(), FileLLMCache(tmp_path))
    embedder.embed(["text"])
    [cache_file] = tmp_path.iterdir()
    cache_file.write_text('{"value": "[1.0]"}', encoding="utf-8")

    with pytest.raises(ValueError, match="dimensions"):
        embedder.embed(["text"])
