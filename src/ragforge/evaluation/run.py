#!/usr/bin/env python3
"""RAGForge v0.1 main benchmark runner (ADR-0004). Entry point for `make bench`/`make bench-live`.

Indexes the real corpus with the frozen embedding model (ADR-0005) and runs
every strategy declared in configs/experiments/benchmark-v01.yaml against the
real golden set (datasets/regrag-br/judgments.json), reporting
recall/precision/nDCG/MRR@k per strategy plus, per ADR-0007, a generated
answer's Citation Accuracy/Faithfulness/Answer Relevancy - and writing a
versioned run record to experiments/<run_id>/results.json.

Document discovery, expected article counts, and source hashes come only
from the corpus manifest (datasets/regrag-br/corpus_manifest.yaml) and the
question selection only from the versioned split
(datasets/regrag-br/split.json) - both ADR-0012. A preflight gate
(ragforge.evaluation.integrity) validates source hashes, split/golden-set
agreement, and structural-reference resolution before any indexing starts,
and fails the run closed (SystemExit) rather than silently indexing a
reduced or drifted corpus. RAPTOR is built once per document, never across a
document boundary, so a summary node can never blend unrelated norms. Every
selected question gets one immutable QuestionRecord per strategy - including
unanswerable-class questions, which are excluded from ranking/citation
averages but never dropped from coverage - appended to
experiments/<run_id>/records.jsonl as each strategy finishes.

Per strategy, this doubles the LLM calls made per question: one
GeminiAnswerGenerator.generate() call plus the RagasJudge's Faithfulness and
Answer Relevancy scoring calls, on top of whatever the strategy itself
already costs (contextualization, RAPTOR summarization, GraphRAG entity
extraction). Judge scores are unvalidated until the ADR-0007 human
calibration exercise happens - report them with that caveat.

Per-question retrieval/generation/judge failures are isolated and counted
(evaluate_strategy's "errors", evaluate_answer_quality's "answer_errors")
rather than aborting the strategy; answer generation and judge scoring run
concurrently (see answer_harness._DEFAULT_MAX_WORKERS) since they are the
actual bottleneck. results.json is checkpointed after every strategy, so a
crash during a later strategy or index-build phase (contextualize_chunks,
build_raptor_tree, GraphRAG indexing - none of which have per-item failure
isolation) does not lose already-computed results.

Strategy -> index mapping:
    dense, sparse_bm25, hybrid_rrf, reranked, parent_child
        Share one base index (chunks unchanged from ADR-0006 chunking).
    contextual
        A second index built from contextualize_chunks() output (per-chunk
        LLM context prepended), queried with Hybrid - Anthropic's technique
        pairs contextual embeddings with contextual BM25.
    raptor
        A third index: the base chunks plus their recursive summary tree
        (build_raptor_tree()), queried with Dense - the paper's "collapsed
        tree" retrieval is vector similarity over the flattened tree.
    graphrag
        A real LightRAG index (ADR-0010), queried in "local" mode - a
        deliberate default (entity-focused, closer to this benchmark's
        mostly single-hop legal lookups); "global" is equally supported by
        GraphRagRetrieval's mode= parameter for a future side comparison,
        the same way ADR-0005 ran embeddings as an isolated experiment.

Only --mode live is implemented: it makes real, metered Gemini API calls
(embeddings, contextualization, summarization, entity extraction - ADR-0005/
ADR-0010). --mode cache (bit-for-bit replay from a versioned LLM call cache,
ADR-0004) needs a cache-recording/replay layer that does not exist yet in
this codebase and is intentionally out of scope here.

No reranker model has been chosen via a dedicated comparison (unlike the
embedding model, ADR-0005); _RERANKER_MODEL is a placeholder, not a data-
driven winner - already used and verified in test_cross_encoder_reranker.py.
"""

import argparse
import asyncio
import json
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import psycopg
import yaml
from lightrag import LightRAG
from opensearchpy import OpenSearch

from ragforge.domain.models import Chunk, Judgment
from ragforge.domain.protocols import RetrievalStrategy
from ragforge.embeddings.google_gemini_embedder import GoogleGeminiEmbedder
from ragforge.embeddings.ports import EmbeddingModel
from ragforge.evaluation.answer_harness import AnswerJudge, evaluate_answer_quality
from ragforge.evaluation.harness import evaluate_strategy
from ragforge.evaluation.integrity import (
    IntegrityError,
    verify_source_integrity,
    verify_split_integrity,
    verify_structural_references,
)
from ragforge.evaluation.judgments import load_judgments
from ragforge.evaluation.manifest import CorpusManifest, load_corpus_manifest
from ragforge.evaluation.ragas_judge import build_gemini_ragas_judge
from ragforge.evaluation.records import QuestionRecord, append_records_jsonl, merge_question_records
from ragforge.evaluation.split import load_split
from ragforge.generation.gemini_answer_generator import GeminiAnswerGenerator
from ragforge.generation.gemini_contextualizer import GeminiContextualizer
from ragforge.generation.gemini_summarizer import GeminiSummarizer
from ragforge.generation.ports import AnswerGenerator
from ragforge.ingestion.html_extractor import HtmlTextExtractor
from ragforge.ingestion.pipeline import ingest_norm
from ragforge.ingestion.pymupdf_extractor import PyMuPdfExtractor
from ragforge.reranking.cross_encoder_reranker import CrossEncoderReranker
from ragforge.retrieval.contextual.pipeline import contextualize_chunks
from ragforge.retrieval.dense.ports import ChunkSearchStore
from ragforge.retrieval.dense.store import DenseChunkStore
from ragforge.retrieval.dense.strategy import DenseRetrieval
from ragforge.retrieval.graph.indexing import build_content_index, index_norm
from ragforge.retrieval.graph.lightrag_gemini import (
    build_gemini_embedding_func,
    build_gemini_llm_model_func,
)
from ragforge.retrieval.graph.strategy import GraphRagRetrieval
from ragforge.retrieval.hierarchical.ports import ChunkFetchStore
from ragforge.retrieval.hierarchical.strategy import ParentChildRetrieval
from ragforge.retrieval.hybrid.strategy import HybridRetrieval
from ragforge.retrieval.raptor.pipeline import build_raptor_tree
from ragforge.retrieval.reranked.ports import Reranker
from ragforge.retrieval.reranked.strategy import RerankedRetrieval
from ragforge.retrieval.sparse.ports import TextSearchStore
from ragforge.retrieval.sparse.store import SparseChunkStore
from ragforge.retrieval.sparse.strategy import SparseRetrieval

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "datasets/regrag-br/corpus_manifest.yaml"
SPLIT_PATH = ROOT / "datasets/regrag-br/split.json"
JUDGMENTS_PATH = ROOT / "datasets/regrag-br/judgments.json"
RESULTS_DIR = ROOT / "experiments"
DATABASE_URL = "postgresql://ragforge:ragforge@localhost:5432/ragforge"

_BASE_TABLE = "bench_v01_base"
_CONTEXTUAL_TABLE = "bench_v01_contextual"
_RAPTOR_TABLE = "bench_v01_raptor"

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CONTEXTUALIZER_MODEL = "gemini-3.1-flash-lite"
_SUMMARIZER_MODEL = "gemini-3.1-flash-lite"
_GRAPHRAG_LLM_MODEL = "gemini-3.1-flash-lite"
_GRAPHRAG_MODE = "local"

# Bounded concurrency for answer generation + judge scoring (the actual
# bottleneck: multiple sequential LLM round-trips per question). Not a
# data-driven pick - conservative enough to not obviously trip API rate
# limits, high enough to meaningfully shorten a multi-hour live run.
_ANSWER_QUALITY_WORKERS = 5

# Composition-root mapping from the manifest's `extractor` key to the actual
# extraction callable - kept here, not in evaluation.manifest, so that module
# never needs to import an ingestion adapter.
_EXTRACTORS: dict[str, Callable[[Path], str]] = {
    "html": HtmlTextExtractor().extract,
    "pymupdf": PyMuPdfExtractor().extract,
}


def parse_args() -> argparse.Namespace:
    """Parse the benchmark runner's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cache", "live"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _reject_cache_mode(mode: str) -> None:
    """Fail fast and explain why, rather than silently behaving like --mode live.

    Raises:
        SystemExit: If ``mode`` is "cache".
    """
    if mode == "cache":
        raise SystemExit(
            "--mode cache is not implemented yet: it needs a versioned LLM call cache "
            "(ADR-0004) that does not exist in this codebase yet. Use --mode live."
        )


def _load_documents(manifest: CorpusManifest) -> dict[str, tuple[str, list[Chunk]]]:
    """Extract and chunk every enabled manifest document; return ``{norm_id: (full_text, chunks)}``.

    ``ingest_norm`` already fails closed (ArticleCountMismatchError) on a
    parsed article count that disagrees with the manifest's curated
    ``expected_article_count``, so that gate needs no duplication here.
    """
    documents = {}
    for doc in manifest.enabled_documents:
        if doc.expected_article_count is None:
            msg = (
                f"{doc.norm_id}: enabled document loaded without an expected_article_count "
                "- load_corpus_manifest should have rejected this"
            )
            raise ValueError(msg)
        path = ROOT / doc.source_path
        extract = _EXTRACTORS[doc.extractor]
        text = extract(path)
        chunks = ingest_norm(
            doc.norm_id, path, text, expected_article_count=doc.expected_article_count
        )
        documents[doc.norm_id] = (text, chunks)
    return documents


class BaseDenseStore(ChunkSearchStore, ChunkFetchStore, Protocol):
    """A dense store usable both for vector search and parent-chunk lookup.

    DenseChunkStore satisfies this by having both methods; the base index
    needs both because parent_child expands a search hit to its parent via
    the same store it was searched from.
    """


def build_base_strategies(
    dense_store: BaseDenseStore,
    sparse_store: TextSearchStore,
    embedder: EmbeddingModel,
    reranker: Reranker,
    rerank_pool: int,
) -> dict[str, RetrievalStrategy]:
    """Wire dense/sparse/hybrid/reranked/parent_child from already-indexed stores.

    ``dense_store`` doubles as the parent-fetch source for parent_child, since
    the base index carries every chunk - fragments and their parent articles
    alike (ADR-0006).
    """
    dense = DenseRetrieval(dense_store, embedder)
    sparse = SparseRetrieval(sparse_store)
    hybrid = HybridRetrieval(dense, sparse)
    reranked = RerankedRetrieval(hybrid, reranker, pool_size=rerank_pool)
    parent_child = ParentChildRetrieval(dense, dense_store)
    return {
        "dense": dense,
        "sparse_bm25": sparse,
        "hybrid_rrf": hybrid,
        "reranked": reranked,
        "parent_child": parent_child,
    }


def build_contextual_strategy(
    dense_store: ChunkSearchStore, sparse_store: TextSearchStore, embedder: EmbeddingModel
) -> RetrievalStrategy:
    """Wire Hybrid retrieval over an already-indexed contextualized-chunk store."""
    return HybridRetrieval(DenseRetrieval(dense_store, embedder), SparseRetrieval(sparse_store))


def format_results_table(
    strategy_labels: list[str], run_metrics: dict[str, dict[str, float]]
) -> str:
    """Render a fixed-width recall/precision/nDCG/MRR@k table for the given strategy order."""
    header = (
        f"{'strategy':<14} {'recall@k':>9} {'precision@k':>12} {'ndcg@k':>8} {'mrr':>6} "
        f"{'n':>4} {'errors':>6}"
    )
    lines = [header]
    for label in strategy_labels:
        metrics = run_metrics.get(label)
        if metrics is None:
            lines.append(f"{label:<14} (not run)")
            continue
        lines.append(
            f"{label:<14} {metrics['recall_at_k']:>9.3f} {metrics['precision_at_k']:>12.3f} "
            f"{metrics['ndcg_at_k']:>8.3f} {metrics['mrr']:>6.3f} {metrics['n']:>4.0f} "
            f"{metrics['errors']:>6.0f}"
        )
    return "\n".join(lines)


def format_answer_quality_table(
    strategy_labels: list[str], run_metrics: dict[str, dict[str, float]]
) -> str:
    """Render a fixed-width Citation Accuracy/Faithfulness/Answer Relevancy table (ADR-0007)."""
    header = (
        f"{'strategy':<14} {'citation_acc':>12} {'faithfulness':>12} {'relevancy':>10} "
        f"{'n':>4} {'errors':>6}"
    )
    lines = [header]
    for label in strategy_labels:
        metrics = run_metrics.get(label)
        if metrics is None or "citation_accuracy" not in metrics:
            lines.append(f"{label:<14} (not run)")
            continue
        lines.append(
            f"{label:<14} {metrics['citation_accuracy']:>12.3f} {metrics['faithfulness']:>12.3f} "
            f"{metrics['answer_relevancy']:>10.3f} {metrics['answer_n']:>4.0f} "
            f"{metrics['answer_errors']:>6.0f}"
        )
    return "\n".join(lines)


def _evaluate(
    strategy: RetrievalStrategy,
    judgments: list[Judgment],
    generator: AnswerGenerator,
    judge_factory: Callable[[], AnswerJudge],
    top_k: int,
) -> tuple[dict[str, float], list[QuestionRecord]]:
    """Score ``strategy`` for retrieval ranking and answer quality alike (ADR-0002/0007).

    Merges evaluate_strategy's and evaluate_answer_quality's per-question
    records into one QuestionRecord per judgment (ADR-0012), alongside the
    same aggregate metrics dict this returned before.
    """
    retrieval_result = evaluate_strategy(strategy, judgments, k=top_k)
    answer_result = evaluate_answer_quality(
        strategy,
        judgments,
        generator,
        judge_factory,
        k=top_k,
        max_workers=_ANSWER_QUALITY_WORKERS,
    )
    records = merge_question_records(strategy.name, retrieval_result.records, answer_result.records)
    return {**retrieval_result.metrics, **answer_result.metrics}, records


def build_run_record(
    *,
    run_id: str,
    mode: str,
    config_path: str,
    embedding_model: str,
    embedding_dimensions: int,
    generation_model: str,
    judge_model: str,
    corpus_version: str,
    split_dataset_version: str,
    n_chunks: int,
    top_k: int,
    run_metrics: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Assemble the JSON-serializable run record written to experiments/<run_id>/results.json."""
    return {
        "run_id": run_id,
        "mode": mode,
        "config_path": config_path,
        "embedding": {"model": embedding_model, "dimensions": embedding_dimensions},
        "reranker_model": _RERANKER_MODEL,
        "contextualizer_model": _CONTEXTUALIZER_MODEL,
        "summarizer_model": _SUMMARIZER_MODEL,
        "graphrag_llm_model": _GRAPHRAG_LLM_MODEL,
        "graphrag_mode": _GRAPHRAG_MODE,
        "generation_model": generation_model,
        "judge_model": judge_model,
        "corpus_version": corpus_version,
        "split_dataset_version": split_dataset_version,
        "k": top_k,
        "n_chunks": n_chunks,
        "metrics": run_metrics,
        "records_path": "records.jsonl",
    }


def main() -> None:
    """Index the real corpus with every strategy and score each against the golden set."""
    args = parse_args()
    _reject_cache_mode(args.mode)
    args.config = args.config.resolve()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    top_k = config["retrieval"]["top_k"]
    rerank_pool = config["retrieval"]["rerank_pool"]
    embedding_model = config["embedding"]["model"]
    embedding_dimensions = config["embedding"]["dimensions"]
    generation_model = config["generation"]["model"]
    judge_model = config["judge"]["model"]

    manifest = load_corpus_manifest(MANIFEST_PATH)
    split = load_split(SPLIT_PATH)
    judgments = load_judgments(JUDGMENTS_PATH)

    print("Running preflight integrity checks (ADR-0012)...")
    try:
        verify_source_integrity(manifest, root=ROOT)
        verify_split_integrity(split, judgments)
    except IntegrityError as exc:
        raise SystemExit(f"preflight integrity check failed:\n{exc}") from exc

    print("Extracting and chunking real corpus documents...")
    documents = _load_documents(manifest)
    all_chunks = [chunk for _, chunks in documents.values() for chunk in chunks]
    print(f"{len(all_chunks)} chunks total across {len(documents)} documents")

    try:
        verify_structural_references(judgments, documents)
    except IntegrityError as exc:
        raise SystemExit(f"preflight integrity check failed:\n{exc}") from exc

    print(f"Loading embedding model {embedding_model}...")
    embedder = GoogleGeminiEmbedder(embedding_model, output_dimensionality=embedding_dimensions)

    print(f"Loading generation model {generation_model} and judge model {judge_model}...")
    generator = GeminiAnswerGenerator(generation_model)

    def judge_factory() -> AnswerJudge:
        return build_gemini_ragas_judge(judge_model, embedding_model)

    conn = psycopg.connect(DATABASE_URL)
    os_client = OpenSearch(hosts=["http://localhost:9200"], use_ssl=False, verify_certs=False)
    graphrag_dir = Path(tempfile.mkdtemp(prefix="ragforge-bench-graphrag-"))
    tables = [_BASE_TABLE, _CONTEXTUAL_TABLE, _RAPTOR_TABLE]

    run_metrics: dict[str, dict[str, float]] = {}
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"

    def _checkpoint() -> None:
        """Write the run record as computed so far - survives a later strategy crashing."""
        record = build_run_record(
            run_id=run_id,
            mode=args.mode,
            config_path=str(args.config.relative_to(ROOT)),
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            generation_model=generation_model,
            judge_model=judge_model,
            corpus_version=manifest.corpus_version,
            split_dataset_version=split.dataset_version,
            n_chunks=len(all_chunks),
            top_k=top_k,
            run_metrics=run_metrics,
        )
        (run_dir / "results.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))

    def _evaluate_and_checkpoint(label: str, strategy: RetrievalStrategy) -> None:
        """Score ``strategy``, append its records.jsonl lines, then checkpoint results.json."""
        metrics, records = _evaluate(strategy, judgments, generator, judge_factory, top_k)
        run_metrics[label] = metrics
        append_records_jsonl(records_path, records)
        _checkpoint()

    try:
        print("\n[1/4] Indexing the base chunks (dense + sparse)...")
        base_dense_store = DenseChunkStore(conn, table=_BASE_TABLE)
        base_sparse_store = SparseChunkStore(os_client, index=_BASE_TABLE)
        base_embeddings = embedder.embed([chunk.text for chunk in all_chunks])
        base_dense_store.create_schema(dimensions=embedder.dimensions)
        base_dense_store.upsert_chunks(all_chunks, base_embeddings)
        base_sparse_store.create_index()
        base_sparse_store.index_chunks(all_chunks)

        base_strategies = build_base_strategies(
            base_dense_store,
            base_sparse_store,
            embedder,
            CrossEncoderReranker(_RERANKER_MODEL),
            rerank_pool,
        )
        for label, strategy in base_strategies.items():
            print(f"  evaluating {label}...")
            _evaluate_and_checkpoint(label, strategy)

        print("\n[2/4] Building the Contextual Retrieval index (1 LLM call per chunk)...")
        contextualizer = GeminiContextualizer(_CONTEXTUALIZER_MODEL)
        contextual_chunks = [
            contextualized
            for norm_id, (full_text, chunks) in documents.items()
            for contextualized in contextualize_chunks(full_text, chunks, contextualizer)
        ]
        contextual_dense_store = DenseChunkStore(conn, table=_CONTEXTUAL_TABLE)
        contextual_sparse_store = SparseChunkStore(os_client, index=_CONTEXTUAL_TABLE)
        contextual_embeddings = embedder.embed([chunk.text for chunk in contextual_chunks])
        contextual_dense_store.create_schema(dimensions=embedder.dimensions)
        contextual_dense_store.upsert_chunks(contextual_chunks, contextual_embeddings)
        contextual_sparse_store.create_index()
        contextual_sparse_store.index_chunks(contextual_chunks)
        contextual_strategy = build_contextual_strategy(
            contextual_dense_store, contextual_sparse_store, embedder
        )
        print("  evaluating contextual...")
        _evaluate_and_checkpoint("contextual", contextual_strategy)

        print("\n[3/4] Building the RAPTOR tree (1 LLM call per group, per level, per document)...")
        summarizer = GeminiSummarizer(_SUMMARIZER_MODEL)
        raptor_chunks: list[Chunk] = []
        for _, chunks in documents.values():
            raptor_chunks.extend(build_raptor_tree(chunks, summarizer))
        raptor_dense_store = DenseChunkStore(conn, table=_RAPTOR_TABLE)
        raptor_embeddings = embedder.embed([chunk.text for chunk in raptor_chunks])
        raptor_dense_store.create_schema(dimensions=embedder.dimensions)
        raptor_dense_store.upsert_chunks(raptor_chunks, raptor_embeddings)
        raptor_strategy = DenseRetrieval(raptor_dense_store, embedder)
        print("  evaluating raptor...")
        _evaluate_and_checkpoint("raptor", raptor_strategy)

        print(
            f"\n[4/4] Building the GraphRAG (LightRAG, mode={_GRAPHRAG_MODE}) index "
            "(multiple LLM calls per chunk)..."
        )
        rag = LightRAG(
            working_dir=str(graphrag_dir),
            embedding_func=build_gemini_embedding_func(embedder),
            llm_model_func=build_gemini_llm_model_func(_GRAPHRAG_LLM_MODEL),
        )
        asyncio.run(rag.initialize_storages())
        try:
            for norm_id, (_, chunks) in documents.items():
                index_norm(rag, norm_id, chunks)
            graphrag_strategy = GraphRagRetrieval(
                rag, build_content_index(all_chunks), mode=_GRAPHRAG_MODE
            )
            print("  evaluating graphrag...")
            _evaluate_and_checkpoint("graphrag", graphrag_strategy)
        finally:
            asyncio.run(rag.finalize_storages())
    finally:
        print("\nCleaning up disposable tables/indices...")
        conn.rollback()
        with conn.cursor() as cur:
            for table in tables:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        conn.close()
        for table in tables:
            os_client.indices.delete(index=table, ignore=[404])
        shutil.rmtree(graphrag_dir, ignore_errors=True)

    print(f"\n{format_results_table(config['strategies'], run_metrics)}")
    print(f"\n{format_answer_quality_table(config['strategies'], run_metrics)}")

    _checkpoint()
    print(f"\nRun record written to {run_dir.relative_to(ROOT)}/results.json")


if __name__ == "__main__":
    main()
