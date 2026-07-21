"""End-to-end integration test: real corpus -> ingest -> embed -> index -> retrieve.

Opt-in only (`pytest -m integration`). Needs the local Postgres+pgvector AND
OpenSearch containers (`docker compose --profile core --profile search up -d`,
see docs/DEVELOPMENT.md) plus a real (small) sentence-transformers model
download on first run.

Ties together every piece built this session against one real document,
rather than each piece in isolation: extraction, structural parsing, the
article-count gate, chunking, embedding, dense+sparse indexing, and all four
retrieval strategies (dense, sparse, hybrid, parent-child).
"""

from pathlib import Path

import psycopg
import pytest
from opensearchpy import OpenSearch

from ragforge.domain.models import Query
from ragforge.ingestion.pipeline import ingest_norm
from ragforge.ingestion.pymupdf_extractor import PyMuPdfExtractor
from ragforge.retrieval.dense.store import DenseChunkStore
from ragforge.retrieval.dense.strategy import DenseRetrieval
from ragforge.retrieval.hierarchical.strategy import ParentChildRetrieval
from ragforge.retrieval.hybrid.strategy import HybridRetrieval
from ragforge.retrieval.sparse.store import SparseChunkStore
from ragforge.retrieval.sparse.strategy import SparseRetrieval

_CORPUS_PDF = Path(__file__).resolve().parents[2] / "datasets/corpus/bacen/RES-CMN-4893-2021.pdf"
_NORM_ID = "RES-CMN-4893-2021"
_DATABASE_URL = "postgresql://ragforge:ragforge@localhost:5432/ragforge"
_TEST_NAME = "chunks_test_e2e_retrieval"


@pytest.mark.integration
def test_full_pipeline_indexes_and_retrieves_a_real_norm() -> None:
    """A real corpus PDF flows through the whole pipeline to relevant retrieval hits."""
    from ragforge.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder

    text = PyMuPdfExtractor().extract(_CORPUS_PDF)
    chunks = ingest_norm(_NORM_ID, _CORPUS_PDF, text, expected_article_count=28)

    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.embed([chunk.text for chunk in chunks])

    conn = psycopg.connect(_DATABASE_URL)
    dense_store = DenseChunkStore(conn, table=_TEST_NAME)
    sparse_client = OpenSearch(hosts=["http://localhost:9200"], use_ssl=False, verify_certs=False)
    sparse_store = SparseChunkStore(sparse_client, index=_TEST_NAME)
    try:
        dense_store.create_schema(dimensions=embedder.dimensions)
        dense_store.upsert_chunks(chunks, embeddings)
        sparse_store.create_index()
        sparse_store.index_chunks(chunks)

        dense = DenseRetrieval(dense_store, embedder)
        sparse = SparseRetrieval(sparse_store)
        hybrid = HybridRetrieval(dense, sparse)
        parent_child = ParentChildRetrieval(dense, dense_store)

        query = Query(text="segurança cibernética e computação em nuvem")

        for strategy in (dense, sparse, hybrid, parent_child):
            results = strategy.retrieve(query, top_k=3)
            assert results, f"{strategy.name} returned no results"
            assert all(r.chunk.metadata.get("norm") == _NORM_ID for r in results)
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {_TEST_NAME}")
        conn.commit()
        conn.close()
        sparse_client.indices.delete(index=_TEST_NAME, ignore=[404])
