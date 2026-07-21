"""Integration test for DenseChunkStore against a real pgvector instance (ADR-0005).

Opt-in only: excluded from the default `pytest` run by `-m "not integration"`,
since it needs the local Postgres+pgvector container from
`docker compose --profile core up -d` (see docs/DEVELOPMENT.md). Run with
`uv run pytest -m integration -k dense_chunk_store`.

Uses small, fake 3-dimensional vectors, not a real embedding model: this test
verifies the store's SQL and pgvector wiring, not embedding quality.
"""

from collections.abc import Iterator

import psycopg
import pytest

from ragforge.domain.models import Chunk
from ragforge.retrieval.dense.store import DenseChunkStore

_DATABASE_URL = "postgresql://ragforge:ragforge@localhost:5432/ragforge"
_TEST_TABLE = "chunks_test_dense_chunk_store"


@pytest.fixture
def store() -> Iterator[DenseChunkStore]:
    """A DenseChunkStore over a disposable test table, dropped after the test."""
    conn = psycopg.connect(_DATABASE_URL, autocommit=False)
    chunk_store = DenseChunkStore(conn, table=_TEST_TABLE)
    chunk_store.create_schema(dimensions=3)
    try:
        yield chunk_store
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {_TEST_TABLE}")
        conn.commit()
        conn.close()


@pytest.mark.integration
def test_search_ranks_the_closest_vector_first(store: DenseChunkStore) -> None:
    """A query vector matches its nearest neighbor first, by cosine similarity."""
    chunks = [
        Chunk(chunk_id="art-1", text="Art. 1º texto", structural_ids=("art-1",)),
        Chunk(chunk_id="art-2", text="Art. 2º texto", structural_ids=("art-2",)),
        Chunk(chunk_id="art-3", text="Art. 3º texto", structural_ids=("art-3",)),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    store.upsert_chunks(chunks, embeddings)

    results = store.search([0.9, 0.1, 0.0], top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["art-1", "art-2"]
    assert results[0].score > results[1].score


@pytest.mark.integration
def test_upsert_is_idempotent_and_updates_existing_chunks(store: DenseChunkStore) -> None:
    """Re-indexing the same chunk_id updates the row instead of duplicating it."""
    original = Chunk(chunk_id="art-1", text="original", structural_ids=("art-1",))
    store.upsert_chunks([original], [[1.0, 0.0, 0.0]])

    updated = Chunk(chunk_id="art-1", text="updated", structural_ids=("art-1",))
    store.upsert_chunks([updated], [[0.0, 1.0, 0.0]])

    results = store.search([0.0, 1.0, 0.0], top_k=10)

    assert len(results) == 1
    assert results[0].chunk.text == "updated"


@pytest.mark.integration
def test_search_returns_structural_ids_and_metadata(store: DenseChunkStore) -> None:
    """Round-tripped chunks preserve structural_ids, parent_id, and metadata."""
    chunk = Chunk(
        chunk_id="art-2::par-1",
        text="§ 1º texto",
        structural_ids=("art-2", "art-2::par-1"),
        parent_id="art-2",
        metadata={"norm": "NORM-1", "role": "fragment"},
    )
    store.upsert_chunks([chunk], [[0.0, 1.0, 0.0]])

    [result] = store.search([0.0, 1.0, 0.0], top_k=1)

    assert result.chunk.structural_ids == ("art-2", "art-2::par-1")
    assert result.chunk.parent_id == "art-2"
    assert result.chunk.metadata == {"norm": "NORM-1", "role": "fragment"}


@pytest.mark.integration
def test_get_returns_a_chunk_by_id(store: DenseChunkStore) -> None:
    """get() fetches a chunk directly by chunk_id, used to expand to a parent article."""
    chunk = Chunk(chunk_id="art-2", text="full article", structural_ids=("art-2",))
    store.upsert_chunks([chunk], [[1.0, 0.0, 0.0]])

    result = store.get("art-2")

    assert result is not None
    assert result.text == "full article"


@pytest.mark.integration
def test_get_returns_none_for_an_unindexed_chunk_id(store: DenseChunkStore) -> None:
    """get() returns None rather than raising when the chunk_id isn't indexed."""
    assert store.get("does-not-exist") is None
