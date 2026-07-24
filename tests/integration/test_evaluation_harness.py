"""Real evaluation run: index the golden set's 4 documents, score every strategy.

Opt-in only (`pytest -m integration`). Needs the local Postgres+pgvector AND
OpenSearch containers (see docs/DEVELOPMENT.md) plus a real embedding model
download on first run.

This is the capstone: real corpus documents, real extraction, real chunking,
real embeddings, real indexes, the real hand-curated golden set
(datasets/regrag-br/judgments.json), scored against Dense, Sparse, Hybrid,
and Parent-Child - not fakes anywhere in the loop. Run with:

    uv run pytest -m integration -k evaluation_harness -s

(`-s` shows the printed per-strategy metrics table.)
"""

from pathlib import Path

import psycopg
import pytest
from opensearchpy import OpenSearch

from ragforge.evaluation.harness import evaluate_strategy
from ragforge.evaluation.judgments import load_judgments
from ragforge.ingestion.html_extractor import HtmlTextExtractor
from ragforge.ingestion.pipeline import ingest_norm
from ragforge.ingestion.pymupdf_extractor import PyMuPdfExtractor
from ragforge.retrieval.dense.store import DenseChunkStore
from ragforge.retrieval.dense.strategy import DenseRetrieval
from ragforge.retrieval.hierarchical.strategy import ParentChildRetrieval
from ragforge.retrieval.hybrid.strategy import HybridRetrieval
from ragforge.retrieval.sparse.store import SparseChunkStore
from ragforge.retrieval.sparse.strategy import SparseRetrieval

CORPUS = Path(__file__).resolve().parents[2] / "datasets/corpus"
JUDGMENTS_PATH = Path(__file__).resolve().parents[2] / "datasets/regrag-br/judgments.json"
_DATABASE_URL = "postgresql://ragforge:ragforge@localhost:5432/ragforge"
_TEST_NAME = "chunks_test_evaluation_harness"

# norm_id -> (source path, extractor, curated article count from
# tests/unit/test_corpus_article_counts.py)
_DOCUMENTS = {
    "LC-105/2001": (CORPUS / "lc-lgpd/LC-105-2001.htm", HtmlTextExtractor().extract, 13),
    "RES-CMN-4893/2021": (CORPUS / "bacen/RES-CMN-4893-2021.pdf", PyMuPdfExtractor().extract, 28),
    "RES-CMN-5274/2025": (CORPUS / "bacen/RES-CMN-5274-2025.htm", HtmlTextExtractor().extract, 3),
    "LEI-13709/2018": (CORPUS / "lc-lgpd/LEI-13709-2018-LGPD.htm", HtmlTextExtractor().extract, 79),
}


@pytest.mark.integration
def test_real_strategies_score_above_zero_on_the_real_golden_set() -> None:
    """Index the real golden-set documents and score Dense/Sparse/Hybrid/Parent-Child."""
    from ragforge.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder

    chunks = []
    for norm_id, (path, extract, expected_count) in _DOCUMENTS.items():
        text = extract(path)
        chunks.extend(ingest_norm(norm_id, path, text, expected_article_count=expected_count))

    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.embed([chunk.retrieval_text for chunk in chunks])

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

        judgments = load_judgments(JUDGMENTS_PATH)

        print(
            f"\n{'strategy':<14} {'recall@5':>9} {'precision@5':>12} "
            f"{'ndcg@5':>8} {'mrr':>6} {'n':>4}"
        )
        for strategy in (dense, sparse, hybrid, parent_child):
            result = evaluate_strategy(strategy, judgments, k=5)
            metrics = result.metrics
            print(
                f"{strategy.name:<14} {metrics['recall_at_k']:>9.3f} "
                f"{metrics['precision_at_k']:>12.3f} {metrics['ndcg_at_k']:>8.3f} "
                f"{metrics['mrr']:>6.3f} {metrics['n']:>4.0f}"
            )
            # A real strategy over a real, correctly-referenced golden set must
            # find at least some relevant chunks - this isn't a tight quality
            # bar, just a sanity check that indexing and retrieval are wired
            # correctly end to end.
            assert metrics["recall_at_k"] > 0.0, f"{strategy.name} found nothing relevant"
            assert metrics["n"] == 219, "expected 219 scored questions (230 minus 11 unanswerable)"
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {_TEST_NAME}")
        conn.commit()
        conn.close()
        sparse_client.indices.delete(index=_TEST_NAME, ignore=[404])
