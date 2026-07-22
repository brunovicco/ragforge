#!/usr/bin/env python3
"""Run the ADR-0005 embedding comparison (Dense/Sparse/Hybrid) on the real golden set.

Reads candidate models from embeddings-ptbr.yaml (in this directory), indexes the
real corpus documents referenced by datasets/regrag-br/judgments.json into
disposable pgvector/OpenSearch stores, scores Dense/Sparse/Hybrid against the
golden set for the requested candidate, prints the results, and appends a
timestamped run record to experiments/embeddings-ptbr/runs.jsonl.

Needs local Postgres+pgvector and OpenSearch running
(`docker compose --profile core --profile search up -d`, see
docs/DEVELOPMENT.md). Two provider families:

- sentence-transformers (open, local, free): downloads the model on first use
  (BGE-M3 is ~2.2GB). Defaults to CPU - on this project's dev machine, the MPS
  (Apple GPU) backend ran out of memory encoding the corpus with BGE-M3 and
  crashed the process, taking Docker Desktop down as a side effect. Pass
  --device mps or --device cuda to opt into GPU acceleration on hardware that
  can handle it.
- gemini (proprietary, real metered API calls): needs GEMINI_API_KEY or
  GOOGLE_API_KEY in the environment. Requests a truncated embedding size
  (--gemini-dimensions, default 1536): pgvector's HNSW index rejects columns
  over 2000 dimensions, and gemini-embedding-001's native size is 3072.

Usage:
    uv run python configs/experiments/run_embeddings_ptbr.py
    uv run python configs/experiments/run_embeddings_ptbr.py --model BAAI/bge-m3 --k 10
    GEMINI_API_KEY=... uv run python configs/experiments/run_embeddings_ptbr.py \\
        --model gemini-embedding-001
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import psycopg
import yaml
from opensearchpy import OpenSearch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ragforge.embeddings.google_gemini_embedder import GoogleGeminiEmbedder
from ragforge.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from ragforge.evaluation.harness import evaluate_strategy
from ragforge.evaluation.judgments import load_judgments
from ragforge.ingestion.html_extractor import HtmlTextExtractor
from ragforge.ingestion.pipeline import ingest_norm
from ragforge.ingestion.pymupdf_extractor import PyMuPdfExtractor
from ragforge.retrieval.dense.store import DenseChunkStore
from ragforge.retrieval.dense.strategy import DenseRetrieval
from ragforge.retrieval.hybrid.strategy import HybridRetrieval
from ragforge.retrieval.sparse.store import SparseChunkStore
from ragforge.retrieval.sparse.strategy import SparseRetrieval

CONFIG_PATH = Path(__file__).resolve().with_name("embeddings-ptbr.yaml")
CORPUS = ROOT / "datasets/corpus"
JUDGMENTS_PATH = ROOT / "datasets/regrag-br/judgments.json"
RESULTS_PATH = ROOT / "experiments/embeddings-ptbr/runs.jsonl"
DATABASE_URL = "postgresql://ragforge:ragforge@localhost:5432/ragforge"
STORE_NAME = "chunks_experiment_embeddings_ptbr"
PROVIDERS = ("sentence-transformers", "gemini")
# pgvector's HNSW index rejects columns over 2000 dimensions; gemini-embedding
# models default to more (3072 for gemini-embedding-001). 1536 is comfortably
# under the cap and a common size other providers also use.
DEFAULT_GEMINI_DIMENSIONS = 1536

# norm_id -> (source path, extractor, curated article count from
# tests/unit/test_corpus_article_counts.py)
DOCUMENTS = {
    "LC-105/2001": (CORPUS / "lc-lgpd/LC-105-2001.htm", HtmlTextExtractor().extract, 13),
    "RES-CMN-4893/2021": (CORPUS / "bacen/RES-CMN-4893-2021.pdf", PyMuPdfExtractor().extract, 28),
    "RES-CMN-5274/2025": (CORPUS / "bacen/RES-CMN-5274-2025.htm", HtmlTextExtractor().extract, 3),
    "LEI-13709/2018": (CORPUS / "lc-lgpd/LEI-13709-2018-LGPD.htm", HtmlTextExtractor().extract, 79),
}


class EmbeddingModel(Protocol):
    """Local copy of ragforge.embeddings.ports.EmbeddingModel's shape, for typing here."""

    name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _candidates() -> list[dict[str, object]]:
    """Return the candidates list from embeddings-ptbr.yaml."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return list(config["candidates"])


def _default_model_and_provider() -> tuple[str, str]:
    """Return the first named candidate's (model, provider) from embeddings-ptbr.yaml."""
    for candidate in _candidates():
        if candidate["name"]:
            return str(candidate["name"]), str(candidate["provider"])
    raise SystemExit(
        "No candidate model with a name found in embeddings-ptbr.yaml; pass --model explicitly."
    )


def _provider_for_model(model_name: str) -> str | None:
    """Look up the declared provider for model_name in embeddings-ptbr.yaml, if listed."""
    for candidate in _candidates():
        if candidate["name"] == model_name:
            return str(candidate["provider"])
    return None


def _build_embedder(
    model_name: str, provider: str, device: str, gemini_dimensions: int
) -> EmbeddingModel:
    """Construct the right embedder adapter for the requested provider.

    Raises:
        SystemExit: If provider isn't one of the known values.
    """
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name, device=device)
    if provider == "gemini":
        return GoogleGeminiEmbedder(model_name, output_dimensionality=gemini_dimensions)
    raise SystemExit(f"Unknown provider {provider!r}; expected one of {PROVIDERS}")


def parse_args() -> argparse.Namespace:
    """Parse the experiment's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="embedding model name")
    parser.add_argument(
        "--provider",
        default=None,
        choices=PROVIDERS,
        help="defaults to the provider declared for --model in embeddings-ptbr.yaml",
    )
    parser.add_argument(
        "--device", default="cpu", help="cpu | mps | cuda (sentence-transformers only)"
    )
    parser.add_argument(
        "--gemini-dimensions",
        type=int,
        default=DEFAULT_GEMINI_DIMENSIONS,
        help="Matryoshka output size for Gemini models (must be <=2000 for pgvector HNSW)",
    )
    parser.add_argument("--k", type=int, default=5, help="top_k for every metric")
    return parser.parse_args()


def main() -> None:
    """Index the real corpus with the requested model and score it against the golden set."""
    args = parse_args()
    if args.model is None:
        model_name, provider = _default_model_and_provider()
    else:
        model_name = args.model
        provider = args.provider or _provider_for_model(model_name)
        if provider is None:
            raise SystemExit(
                f"Model {model_name!r} isn't in embeddings-ptbr.yaml; pass --provider explicitly."
            )

    print("Extracting and chunking real corpus documents...")
    chunks = []
    for norm_id, (path, extract, expected_count) in DOCUMENTS.items():
        text = extract(path)
        chunks.extend(ingest_norm(norm_id, path, text, expected_article_count=expected_count))
    print(f"{len(chunks)} chunks total")

    print(f"Loading {model_name} (provider={provider})...")
    embedder = _build_embedder(model_name, provider, args.device, args.gemini_dimensions)
    print(f"Model loaded, dimensions={embedder.dimensions}")

    print("Embedding all chunks...")
    embeddings = embedder.embed([chunk.text for chunk in chunks])

    conn = psycopg.connect(DATABASE_URL)
    dense_store = DenseChunkStore(conn, table=STORE_NAME)
    sparse_client = OpenSearch(hosts=["http://localhost:9200"], use_ssl=False, verify_certs=False)
    sparse_store = SparseChunkStore(sparse_client, index=STORE_NAME)
    try:
        print("Indexing into pgvector and OpenSearch...")
        dense_store.create_schema(dimensions=embedder.dimensions)
        dense_store.upsert_chunks(chunks, embeddings)
        sparse_store.create_index()
        sparse_store.index_chunks(chunks)

        dense = DenseRetrieval(dense_store, embedder)
        sparse = SparseRetrieval(sparse_store)
        hybrid = HybridRetrieval(dense, sparse)

        judgments = load_judgments(JUDGMENTS_PATH)
        run_metrics: dict[str, dict[str, float]] = {}
        run: dict[str, object] = {
            "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "model": model_name,
            "provider": provider,
            "dimensions": embedder.dimensions,
            "device": args.device if provider == "sentence-transformers" else None,
            "k": args.k,
            "n_chunks": len(chunks),
            "metrics": run_metrics,
        }

        header = (
            f"{'strategy':<10} {'recall@k':>9} {'precision@k':>12} {'ndcg@k':>8} "
            f"{'mrr':>6} {'n':>4}"
        )
        print(f"\n{header}")
        for strategy in (dense, sparse, hybrid):
            metrics = evaluate_strategy(strategy, judgments, k=args.k)
            print(
                f"{strategy.name:<10} {metrics['recall_at_k']:>9.3f} "
                f"{metrics['precision_at_k']:>12.3f} {metrics['ndcg_at_k']:>8.3f} "
                f"{metrics['mrr']:>6.3f} {metrics['n']:>4.0f}"
            )
            run_metrics[strategy.name] = metrics
    finally:
        print("Cleaning up disposable table/index...")
        # A prior failure inside the try block (e.g. create_schema erroring
        # partway through) can leave this connection's transaction aborted;
        # roll it back first so the cleanup statement itself doesn't also fail.
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {STORE_NAME}")
        conn.commit()
        conn.close()
        sparse_client.indices.delete(index=STORE_NAME, ignore=[404])

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run, ensure_ascii=False) + "\n")
    print(f"\nRun record appended to {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
