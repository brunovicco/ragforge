"""Integration test for SparseChunkStore against a real OpenSearch instance (ADR-0005).

Opt-in only: excluded from the default `pytest` run by `-m "not integration"`,
since it needs the local OpenSearch container from
`docker compose --profile search up -d` (see docs/DEVELOPMENT.md). Run with
`uv run pytest -m integration -k sparse_chunk_store`.
"""

from collections.abc import Iterator

import pytest
from opensearchpy import OpenSearch

from ragforge.domain.models import Chunk
from ragforge.retrieval.sparse.store import SparseChunkStore

_TEST_INDEX = "chunks_test_sparse_chunk_store"


@pytest.fixture
def store() -> Iterator[SparseChunkStore]:
    """A SparseChunkStore over a disposable test index, deleted after the test."""
    client = OpenSearch(hosts=["http://localhost:9200"], use_ssl=False, verify_certs=False)
    chunk_store = SparseChunkStore(client, index=_TEST_INDEX)
    chunk_store.create_index()
    try:
        yield chunk_store
    finally:
        client.indices.delete(index=_TEST_INDEX, ignore=[404])


@pytest.mark.integration
def test_search_ranks_by_bm25_relevance_with_portuguese_stemming(
    store: SparseChunkStore,
) -> None:
    """The brazilian analyzer stems Portuguese words, matching across inflections."""
    chunks = [
        Chunk(
            chunk_id="art-1",
            source_text="As instituições financeiras conservarão sigilo em suas operações.",
            retrieval_text="As instituições financeiras conservarão sigilo em suas operações.",
            structural_ids=("art-1",),
        ),
        Chunk(
            chunk_id="art-2",
            source_text="Esta Resolução dispõe sobre a política de segurança cibernética.",
            retrieval_text="Esta Resolução dispõe sobre a política de segurança cibernética.",
            structural_ids=("art-2",),
        ),
    ]
    store.index_chunks(chunks)

    results = store.search("sigilo bancário das instituições", top_k=2)

    assert results[0].chunk.chunk_id == "art-1"
    assert results[0].score > 0


@pytest.mark.integration
def test_reindexing_a_chunk_id_overwrites_instead_of_duplicating(
    store: SparseChunkStore,
) -> None:
    """Indexing the same chunk_id twice updates the document, not appends a new one."""
    original = Chunk(
        chunk_id="art-1",
        source_text="texto original",
        retrieval_text="texto original",
        structural_ids=("art-1",),
    )
    store.index_chunks([original])

    updated = Chunk(
        chunk_id="art-1",
        source_text="texto atualizado",
        retrieval_text="texto atualizado",
        structural_ids=("art-1",),
    )
    store.index_chunks([updated])

    results = store.search("atualizado", top_k=10)

    assert len(results) == 1
    assert results[0].chunk.source_text == "texto atualizado"


@pytest.mark.integration
def test_search_returns_structural_ids_and_metadata(store: SparseChunkStore) -> None:
    """Round-tripped chunks preserve structural_ids, parent_id, and metadata."""
    chunk = Chunk(
        chunk_id="art-2::par-1",
        source_text="parágrafo com conteúdo pesquisável",
        retrieval_text="parágrafo com conteúdo pesquisável",
        structural_ids=("art-2", "art-2::par-1"),
        parent_id="art-2",
        metadata={"norm": "NORM-1", "role": "fragment"},
    )
    store.index_chunks([chunk])

    [result] = store.search("conteúdo pesquisável", top_k=1)

    assert result.chunk.structural_ids == ("art-2", "art-2::par-1")
    assert result.chunk.parent_id == "art-2"
    assert result.chunk.metadata == {"norm": "NORM-1", "role": "fragment"}
