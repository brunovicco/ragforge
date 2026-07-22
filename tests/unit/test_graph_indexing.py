"""Tests for the LightRAG indexing helpers (ADR-0010)."""

from typing import Any

from ragforge.domain.models import Chunk
from ragforge.retrieval.graph.indexing import build_content_index, index_norm


def test_build_content_index_maps_stripped_text_to_chunk() -> None:
    """Each chunk is keyed by its stripped text, recoverable by exact match."""
    chunks = [
        Chunk(chunk_id="c1", text="  Art. 1º dispõe sobre X.  ", structural_ids=("s1",)),
        Chunk(chunk_id="c2", text="Art. 2º dispõe sobre Y.", structural_ids=("s2",)),
    ]

    index = build_content_index(chunks)

    assert index["Art. 1º dispõe sobre X."] is chunks[0]
    assert index["Art. 2º dispõe sobre Y."] is chunks[1]
    assert len(index) == 2


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())


class _FakeRag:
    def __init__(self) -> None:
        self.chunking_func: Any = None
        self.insert_calls: list[dict[str, object]] = []

    def insert(self, input: str, ids: str) -> None:
        self.insert_calls.append({"input": input, "ids": ids})


def test_index_norm_overrides_chunking_func_to_return_our_chunks_verbatim() -> None:
    """The installed chunking_func returns our chunks' text, unchanged, in order."""
    rag = _FakeRag()
    chunks = [
        Chunk(chunk_id="c1", text="Art. 1º", structural_ids=("s1",)),
        Chunk(chunk_id="c2", text="Art. 2º", structural_ids=("s2",)),
    ]

    index_norm(rag, "NORM-1", chunks)

    result = rag.chunking_func(_FakeTokenizer(), "ignored raw text")
    assert [item["content"] for item in result] == ["Art. 1º", "Art. 2º"]
    assert [item["chunk_order_index"] for item in result] == [0, 1]


def test_index_norm_calls_insert_with_the_norm_id() -> None:
    """insert() is called once, tagging the document with the norm's id."""
    rag = _FakeRag()
    chunks = [Chunk(chunk_id="c1", text="Art. 1º", structural_ids=("s1",))]

    index_norm(rag, "NORM-1", chunks)

    assert len(rag.insert_calls) == 1
    assert rag.insert_calls[0]["ids"] == "NORM-1"
