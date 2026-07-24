"""Tests for the index-namespace derivation (ADR-0013)."""

import re
from dataclasses import replace

from ragforge.embeddings.identity import NO_QUERY_INSTRUCTION_HASH, EmbeddingIdentity
from ragforge.evaluation.index_namespace import derive_index_namespace

_IDENTITY = EmbeddingIdentity(
    provider="local",
    model="Qwen/Qwen3-Embedding-0.6B",
    revision="main",
    dimensions=1024,
    normalize=True,
    query_instruction_hash=NO_QUERY_INSTRUCTION_HASH,
    runtime="local",
)


def _namespace(**overrides: object) -> str:
    kwargs = {
        "corpus_hash": "corpus-abc",
        "chunking_config_version": "adr-0006-v1",
        "retrieval_text_schema_version": "source-text-v1",
        "embedding": _IDENTITY,
    }
    kwargs.update(overrides)
    return derive_index_namespace(**kwargs)  # type: ignore[arg-type]


def test_same_inputs_produce_the_same_namespace() -> None:
    """The derivation is a pure function: identical arguments, identical output."""
    assert _namespace() == _namespace()


def test_namespace_is_a_short_hex_token() -> None:
    """The output is safe to use as a SQL identifier suffix."""
    assert re.fullmatch(r"[0-9a-f]{16}", _namespace())


def test_namespace_changes_when_corpus_hash_changes() -> None:
    assert _namespace(corpus_hash="corpus-xyz") != _namespace()


def test_namespace_changes_when_chunking_config_version_changes() -> None:
    assert _namespace(chunking_config_version="adr-0006-v2") != _namespace()


def test_namespace_changes_when_retrieval_text_schema_version_changes() -> None:
    assert _namespace(retrieval_text_schema_version="sac-v1") != _namespace()


def test_namespace_changes_when_embedding_provider_changes() -> None:
    other = replace(_IDENTITY, provider="gemini")
    assert _namespace(embedding=other) != _namespace()


def test_namespace_changes_when_embedding_model_changes() -> None:
    other = replace(_IDENTITY, model="BAAI/bge-m3")
    assert _namespace(embedding=other) != _namespace()


def test_namespace_changes_when_dimensions_change() -> None:
    other = replace(_IDENTITY, dimensions=1536)
    assert _namespace(embedding=other) != _namespace()


def test_namespace_changes_when_normalize_changes() -> None:
    other = replace(_IDENTITY, normalize=False)
    assert _namespace(embedding=other) != _namespace()
