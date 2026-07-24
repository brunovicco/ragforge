#!/usr/bin/env python3
"""RAGForge v0.1 main benchmark runner (ADR-0004). Entry point for `make bench`/`make bench-live`.

Indexes the real corpus with the configured embedding model (ADR-0013) and
runs every strategy declared in configs/experiments/benchmark-v01.yaml against the
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

The embedding provider is provider-neutral and config-driven (ADR-0013):
``embedding.provider: local`` (operational default, no credentials needed,
via SentenceTransformerEmbedder) or ``embedding.provider: gemini`` (optional
hosted comparator, via GoogleGeminiEmbedder) - never a silent fallback
between the two. The base/contextual/RAPTOR pgvector tables and OpenSearch
indices are named from a namespace derived (ragforge.evaluation.index_namespace)
from the corpus content hash, the chunking/retrieval-text schema versions,
and the embedding's complete identity, so an index name alone reflects
exactly what produced it - even though every run still creates and drops
these tables fresh (no persistence/caching yet; that is Increment 3's job).
Only the embedding step is provider-neutral: contextualization, RAPTOR
summarization, GraphRAG's entity-extraction LLM, answer generation, and
judge scoring all stay hard-coded to Gemini models regardless of
``embedding.provider`` - those are separate ADRs (0007/0018), not ADR-0013.

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
    sac
        A third index (ADR-0015): base chunks with one per-document summary
        prepended (apply_document_summary()), queried with Dense - the ADR's
        `sac` variant, isolating the summary's effect from Contextual
        Retrieval's per-chunk blurb.
    sac_contextual
        A fourth index: contextual's already-context-prepended chunks with
        the same per-document summary prepended on top (no re-contextualization),
        queried with Dense - the ADR's `sac_contextual` variant, composing both
        techniques.
    raptor
        A fifth index: the base chunks plus their recursive summary tree
        (build_raptor_tree()), queried with Dense - the paper's "collapsed
        tree" retrieval is vector similarity over the flattened tree.
    graphrag
        A real LightRAG index (ADR-0010), queried in "local" mode - a
        deliberate default (entity-focused, closer to this benchmark's
        mostly single-hop legal lookups); "global" is equally supported by
        GraphRagRetrieval's mode= parameter for a future side comparison,
        the same way ADR-0005 ran embeddings as an isolated experiment.

Only --mode live is implemented: with the default local embedding provider,
it still makes real, metered Gemini API calls for everything except
embeddings (contextualization, summarization, entity extraction - ADR-0010);
switching to ``embedding.provider: gemini`` meters embeddings too (ADR-0005/
ADR-0013). --mode cache (bit-for-bit replay from a versioned LLM call cache,
ADR-0004) needs a cache-recording/replay layer that does not exist yet in
this codebase and is intentionally out of scope here.

No reranker model has been chosen via a dedicated comparison (unlike the
embedding model, ADR-0005); _RERANKER_MODEL is a placeholder, not a data-
driven winner - already used and verified in test_cross_encoder_reranker.py.

Bounded parallel execution and a minimal LLM cache (ADR-0014 + ADR-0004):
a FileLLMCache under experiments/<run_id>/llm-cache/ is shared by the
embedder, GeminiAnswerGenerator, and the judge - a Gemini call already made
for the exact same (model, prompt) is never repeated. A ProviderLimiter
bounds concurrent in-flight Gemini calls process-wide
(execution.gemini_max_in_flight). Answer generation + judge scoring use
ragforge.evaluation.scheduler.run_bounded (execution.answer_quality_workers
workers), which always restores canonical question order regardless of
completion order. ``--resume <run_id>`` reuses an existing run's directory
(results.json, records.jsonl, llm-cache/), skips strategies already present
in results.json's metrics, and fails closed if the corpus/embedding/model
identity doesn't match what produced that run. Resuming still re-runs a
stage's indexing (contextualization, RAPTOR summarization) even when only
some of that stage's strategies remain unscored - contextualize_chunks and
build_raptor_tree are not cache-wired in this increment, only the embedder
and the two per-question LLM calls are; see docs/adr/0014 for the fuller,
deferred scope (RPM/TPM limits, `Retry-After`, cross-process coalescing).
"""

import argparse
import asyncio
import json
import shutil
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import psycopg
import yaml
from lightrag import LightRAG
from opensearchpy import OpenSearch

from ragforge.adapters.llm_cache import FileLLMCache, LLMCache
from ragforge.domain.models import Chunk, Judgment
from ragforge.domain.protocols import RetrievalStrategy
from ragforge.embeddings.google_gemini_embedder import GoogleGeminiEmbedder
from ragforge.embeddings.identity import NO_QUERY_INSTRUCTION_HASH, EmbeddingIdentity
from ragforge.embeddings.ports import EmbeddingModel
from ragforge.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from ragforge.evaluation.answer_harness import AnswerJudge, evaluate_answer_quality
from ragforge.evaluation.harness import evaluate_strategy
from ragforge.evaluation.index_namespace import derive_index_namespace
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
from ragforge.evaluation.scheduler import run_bounded
from ragforge.evaluation.split import load_split
from ragforge.generation.gemini_answer_generator import GeminiAnswerGenerator
from ragforge.generation.gemini_contextualizer import GeminiContextualizer
from ragforge.generation.gemini_document_summarizer import GeminiDocumentSummarizer
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
from ragforge.retrieval.sac.pipeline import apply_document_summary
from ragforge.retrieval.sac.ports import DocumentSummarizer
from ragforge.retrieval.sparse.ports import TextSearchStore
from ragforge.retrieval.sparse.store import SparseChunkStore
from ragforge.retrieval.sparse.strategy import SparseRetrieval

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "datasets/regrag-br/corpus_manifest.yaml"
SPLIT_PATH = ROOT / "datasets/regrag-br/split.json"
JUDGMENTS_PATH = ROOT / "datasets/regrag-br/judgments.json"
RESULTS_DIR = ROOT / "experiments"
DATABASE_URL = "postgresql://ragforge:ragforge@localhost:5432/ragforge"

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CONTEXTUALIZER_MODEL = "gemini-3.1-flash-lite"
_SUMMARIZER_MODEL = "gemini-3.1-flash-lite"
_DOCUMENT_SUMMARIZER_MODEL = "gemini-3.1-flash-lite"
_GRAPHRAG_LLM_MODEL = "gemini-3.1-flash-lite"
_GRAPHRAG_MODE = "local"

_EMBEDDING_PROVIDERS = ("local", "gemini")
# Bumped only when the chunking logic (ragforge.chunking) or a retrieval-text
# derivation (contextualize_chunks, sac.pipeline.apply_document_summary)
# changes meaningfully enough that a prior index must not be mistaken for
# compatible (ADR-0013). Every strategy already gets its own table-name
# prefix (base/contextual/raptor/sac/sac_contextual), so this one version
# only needs bumping when a derivation's *output* changes shape/content for
# an existing strategy - introducing SAC as a new strategy didn't (ADR-0015).
_CHUNKING_CONFIG_VERSION = "adr-0006-v1"
_RETRIEVAL_TEXT_SCHEMA_VERSION = "source-text-v1"

# Bounded concurrency for answer generation + judge scoring (the actual
# bottleneck: multiple sequential LLM round-trips per question). Not a
# data-driven pick - conservative enough to not obviously trip API rate
# limits, high enough to meaningfully shorten a multi-hour live run.
# Overridable via execution.answer_quality_workers in the config.
_DEFAULT_ANSWER_QUALITY_WORKERS = 5
# Overridable via execution.gemini_max_in_flight (ADR-0014).
_DEFAULT_GEMINI_MAX_IN_FLIGHT = 4

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
    parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_ID",
        help=(
            "reuse experiments/<run_id>/ (results.json, records.jsonl, llm-cache/) "
            "instead of starting a new run; fails closed if the corpus/embedding/model "
            "identity doesn't match what produced that run"
        ),
    )
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


def _document_versions(manifest: CorpusManifest) -> dict[str, str]:
    """Return ``{norm_id: source_sha256}`` for every enabled document (ADR-0015 cache identity).

    ``load_corpus_manifest`` already rejects an enabled document missing
    ``source_sha256``, so every value here is guaranteed non-None.
    """
    versions = {}
    for doc in manifest.enabled_documents:
        if doc.source_sha256 is None:
            msg = (
                f"{doc.norm_id}: enabled document loaded without a source_sha256 "
                "- load_corpus_manifest should have rejected this"
            )
            raise ValueError(msg)
        versions[doc.norm_id] = doc.source_sha256
    return versions


def _summarize_documents(
    documents: dict[str, tuple[str, list[Chunk]]],
    document_versions: dict[str, str],
    summarizer: DocumentSummarizer,
    max_workers: int,
) -> dict[str, str]:
    """Return ``{norm_id: summary_text}``, one Gemini call per document, bounded and cached.

    Raises the first summarization failure rather than tolerating it: like
    contextualize_chunks and build_raptor_tree, this index-building step has
    no per-item failure isolation (ADR-0014's circuit breaker is for the
    per-question evaluation loop, not index construction) - a missing
    summary means the sac/sac_contextual index for that document simply
    cannot be built.
    """
    norm_ids = list(documents)

    def _summarize_one(norm_id: str) -> str:
        full_text, _ = documents[norm_id]
        summary = summarizer.summarize_document(norm_id, document_versions[norm_id], full_text)
        return summary.summary

    outcomes = run_bounded(norm_ids, _summarize_one, max_workers=max_workers)
    summaries: dict[str, str] = {}
    for norm_id, outcome in zip(norm_ids, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            raise outcome
        summaries[norm_id] = outcome
    return summaries


def _build_embedder(
    provider: str,
    model: str,
    dimensions: int | None,
    cache: LLMCache | None,
    gemini_max_in_flight: int,
) -> tuple[EmbeddingModel, EmbeddingIdentity]:
    """Construct the configured embedding provider and its identity (ADR-0013).

    ``provider: local`` is the operational default: no credentials needed,
    via SentenceTransformerEmbedder - ``dimensions`` is ignored since the
    model reports its own, and it has no hosted-call cache/limiter to wire
    (there's no network call to skip). ``provider: gemini`` is the optional
    hosted comparator, via GoogleGeminiEmbedder with ``dimensions``
    requesting a truncated (Matryoshka) size, ``cache`` (ADR-0004) consulted
    per text, and ``gemini_max_in_flight`` bounding concurrent calls
    (ADR-0014). Never a silent fallback between the two.

    Raises:
        SystemExit: If ``provider`` isn't one of "local"/"gemini".
    """
    if provider == "local":
        local_embedder = SentenceTransformerEmbedder(model)
        identity = EmbeddingIdentity(
            provider="local",
            model=model,
            revision=local_embedder.revision,
            dimensions=local_embedder.dimensions,
            normalize=True,
            query_instruction_hash=NO_QUERY_INSTRUCTION_HASH,
            runtime="local",
        )
        return local_embedder, identity
    if provider == "gemini":
        gemini_embedder = GoogleGeminiEmbedder(
            model,
            output_dimensionality=dimensions,
            cache=cache,
            max_in_flight=gemini_max_in_flight,
        )
        identity = EmbeddingIdentity(
            provider="gemini",
            model=model,
            revision="api",
            dimensions=gemini_embedder.dimensions,
            normalize=False,
            query_instruction_hash=NO_QUERY_INSTRUCTION_HASH,
            runtime="hosted",
        )
        return gemini_embedder, identity
    raise SystemExit(
        f"unknown embedding provider {provider!r}; expected one of {_EMBEDDING_PROVIDERS}"
    )


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
    """Render a fixed-width recall/precision/nDCG/MRR/DRM@k table for the given strategy order."""
    header = (
        f"{'strategy':<14} {'recall@k':>9} {'precision@k':>12} {'ndcg@k':>8} {'mrr':>6} "
        f"{'drm@k':>7} {'n':>4} {'errors':>6}"
    )
    lines = [header]
    for label in strategy_labels:
        metrics = run_metrics.get(label)
        if metrics is None:
            lines.append(f"{label:<14} (not run)")
            continue
        lines.append(
            f"{label:<14} {metrics['recall_at_k']:>9.3f} {metrics['precision_at_k']:>12.3f} "
            f"{metrics['ndcg_at_k']:>8.3f} {metrics['mrr']:>6.3f} {metrics['drm_at_k']:>7.3f} "
            f"{metrics['n']:>4.0f} {metrics['errors']:>6.0f}"
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
    answer_quality_workers: int,
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
        max_workers=answer_quality_workers,
    )
    records = merge_question_records(strategy.name, retrieval_result.records, answer_result.records)
    return {**retrieval_result.metrics, **answer_result.metrics}, records


def build_run_record(
    *,
    run_id: str,
    mode: str,
    config_path: str,
    embedding_identity: EmbeddingIdentity,
    index_namespace: str,
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
        "embedding": {
            "provider": embedding_identity.provider,
            "model": embedding_identity.model,
            "revision": embedding_identity.revision,
            "dimensions": embedding_identity.dimensions,
            "normalize": embedding_identity.normalize,
            "runtime": embedding_identity.runtime,
        },
        "index_namespace": index_namespace,
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


def _verify_resume_identity(
    previous: Mapping[str, object],
    index_namespace: str,
    generation_model: str,
    judge_model: str,
) -> None:
    """Fail closed if a resumed run's identity doesn't match the current configuration.

    ADR-0014: "changing model, prompt, split, or strategy creates a new
    run" - ``index_namespace`` alone already encodes corpus content,
    chunking/retrieval-text schema, and the embedding's complete identity
    (ADR-0013), so checking it plus the generation/judge models covers
    everything that would make reusing cached answers/judge scores unsafe.

    Raises:
        SystemExit: If any identity component differs from the prior run.
    """
    mismatches = []
    if previous.get("index_namespace") != index_namespace:
        mismatches.append(
            f"index_namespace: {previous.get('index_namespace')!r} != {index_namespace!r}"
        )
    if previous.get("generation_model") != generation_model:
        mismatches.append(
            f"generation_model: {previous.get('generation_model')!r} != {generation_model!r}"
        )
    if previous.get("judge_model") != judge_model:
        mismatches.append(f"judge_model: {previous.get('judge_model')!r} != {judge_model!r}")
    if mismatches:
        raise SystemExit(
            "--resume identity mismatch (changing model/corpus/split creates a new run):\n"
            + "\n".join(f"- {mismatch}" for mismatch in mismatches)
        )


def main() -> None:
    """Index the real corpus with every strategy and score each against the golden set."""
    args = parse_args()
    _reject_cache_mode(args.mode)
    args.config = args.config.resolve()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    top_k = config["retrieval"]["top_k"]
    rerank_pool = config["retrieval"]["rerank_pool"]
    embedding_provider = config["embedding"]["provider"]
    embedding_model = config["embedding"]["model"]
    embedding_dimensions = config["embedding"].get("dimensions")
    generation_model = config["generation"]["model"]
    judge_model = config["judge"]["model"]
    execution_config = config.get("execution", {})
    answer_quality_workers = execution_config.get(
        "answer_quality_workers", _DEFAULT_ANSWER_QUALITY_WORKERS
    )
    gemini_max_in_flight = execution_config.get(
        "gemini_max_in_flight", _DEFAULT_GEMINI_MAX_IN_FLIGHT
    )

    manifest = load_corpus_manifest(MANIFEST_PATH)
    split = load_split(SPLIT_PATH)
    judgments = load_judgments(JUDGMENTS_PATH)

    print("Running preflight integrity checks (ADR-0012)...")
    try:
        verify_source_integrity(manifest, root=ROOT)
        verify_split_integrity(split, judgments)
    except IntegrityError as exc:
        raise SystemExit(f"preflight integrity check failed:\n{exc}") from exc

    run_id = args.resume or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"
    cache = FileLLMCache(run_dir / "llm-cache")

    print("Extracting and chunking real corpus documents...")
    documents = _load_documents(manifest)
    all_chunks = [chunk for _, chunks in documents.values() for chunk in chunks]
    print(f"{len(all_chunks)} chunks total across {len(documents)} documents")

    try:
        verify_structural_references(judgments, documents)
    except IntegrityError as exc:
        raise SystemExit(f"preflight integrity check failed:\n{exc}") from exc

    print(f"Loading embedding model {embedding_model} (provider={embedding_provider})...")
    embedder, embedding_identity = _build_embedder(
        embedding_provider, embedding_model, embedding_dimensions, cache, gemini_max_in_flight
    )
    index_namespace = derive_index_namespace(
        manifest.content_hash,
        _CHUNKING_CONFIG_VERSION,
        _RETRIEVAL_TEXT_SCHEMA_VERSION,
        embedding_identity,
    )
    base_table = f"bench_v01_base_{index_namespace}"
    contextual_table = f"bench_v01_contextual_{index_namespace}"
    sac_table = f"bench_v01_sac_{index_namespace}"
    sac_contextual_table = f"bench_v01_sac_contextual_{index_namespace}"
    raptor_table = f"bench_v01_raptor_{index_namespace}"

    run_metrics: dict[str, dict[str, float]] = {}
    results_path = run_dir / "results.json"
    if args.resume is not None and results_path.exists():
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        _verify_resume_identity(previous, index_namespace, generation_model, judge_model)
        run_metrics = previous["metrics"]
        print(f"Resuming {run_id}: {sorted(run_metrics)} already scored.")

    print(f"Loading generation model {generation_model} and judge model {judge_model}...")
    generator = GeminiAnswerGenerator(
        generation_model, cache=cache, max_in_flight=gemini_max_in_flight
    )

    def judge_factory() -> AnswerJudge:
        return build_gemini_ragas_judge(
            judge_model, embedding_model, cache=cache, max_in_flight=gemini_max_in_flight
        )

    conn = psycopg.connect(DATABASE_URL)
    os_client = OpenSearch(hosts=["http://localhost:9200"], use_ssl=False, verify_certs=False)
    graphrag_dir = Path(tempfile.mkdtemp(prefix="ragforge-bench-graphrag-"))
    tables = [base_table, contextual_table, sac_table, sac_contextual_table, raptor_table]

    def _checkpoint() -> None:
        """Write the run record as computed so far - survives a later strategy crashing."""
        record = build_run_record(
            run_id=run_id,
            mode=args.mode,
            config_path=str(args.config.relative_to(ROOT)),
            embedding_identity=embedding_identity,
            index_namespace=index_namespace,
            generation_model=generation_model,
            judge_model=judge_model,
            corpus_version=manifest.corpus_version,
            split_dataset_version=split.dataset_version,
            n_chunks=len(all_chunks),
            top_k=top_k,
            run_metrics=run_metrics,
        )
        results_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))

    def _evaluate_and_checkpoint(label: str, strategy: RetrievalStrategy) -> None:
        """Score ``strategy``, append its records.jsonl lines, then checkpoint results.json.

        A no-op when ``label`` is already in ``run_metrics`` (--resume): the
        stage's indexing above this call still runs regardless (contextual/
        RAPTOR construction isn't cache-wired in this increment), but the
        expensive per-question generation+judge calls are skipped entirely
        rather than merely cache-hit.
        """
        if label in run_metrics:
            print(f"  skipping {label} (already scored, --resume)")
            return
        metrics, records = _evaluate(
            strategy, judgments, generator, judge_factory, top_k, answer_quality_workers
        )
        run_metrics[label] = metrics
        append_records_jsonl(records_path, records)
        _checkpoint()

    try:
        print("\n[1/6] Indexing the base chunks (dense + sparse)...")
        base_dense_store = DenseChunkStore(conn, table=base_table)
        base_sparse_store = SparseChunkStore(os_client, index=base_table)
        base_embeddings = embedder.embed([chunk.retrieval_text for chunk in all_chunks])
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

        print("\n[2/6] Building the Contextual Retrieval index (1 LLM call per chunk)...")
        contextualizer = GeminiContextualizer(_CONTEXTUALIZER_MODEL)
        contextual_chunks_by_norm = {
            norm_id: contextualize_chunks(full_text, chunks, contextualizer)
            for norm_id, (full_text, chunks) in documents.items()
        }
        contextual_chunks = [
            chunk for chunks in contextual_chunks_by_norm.values() for chunk in chunks
        ]
        contextual_dense_store = DenseChunkStore(conn, table=contextual_table)
        contextual_sparse_store = SparseChunkStore(os_client, index=contextual_table)
        contextual_embeddings = embedder.embed(
            [chunk.retrieval_text for chunk in contextual_chunks]
        )
        contextual_dense_store.create_schema(dimensions=embedder.dimensions)
        contextual_dense_store.upsert_chunks(contextual_chunks, contextual_embeddings)
        contextual_sparse_store.create_index()
        contextual_sparse_store.index_chunks(contextual_chunks)
        contextual_strategy = build_contextual_strategy(
            contextual_dense_store, contextual_sparse_store, embedder
        )
        print("  evaluating contextual...")
        _evaluate_and_checkpoint("contextual", contextual_strategy)

        print("\n[3/6] Building the SAC index (1 LLM call per document)...")
        document_summarizer = GeminiDocumentSummarizer(
            _DOCUMENT_SUMMARIZER_MODEL, cache=cache, max_in_flight=gemini_max_in_flight
        )
        document_summaries = _summarize_documents(
            documents, _document_versions(manifest), document_summarizer, answer_quality_workers
        )
        sac_chunks = [
            sac_chunk
            for norm_id, (_, chunks) in documents.items()
            for sac_chunk in apply_document_summary(document_summaries[norm_id], chunks)
        ]
        sac_dense_store = DenseChunkStore(conn, table=sac_table)
        sac_embeddings = embedder.embed([chunk.retrieval_text for chunk in sac_chunks])
        sac_dense_store.create_schema(dimensions=embedder.dimensions)
        sac_dense_store.upsert_chunks(sac_chunks, sac_embeddings)
        sac_strategy = DenseRetrieval(sac_dense_store, embedder)
        print("  evaluating sac...")
        _evaluate_and_checkpoint("sac", sac_strategy)

        print(
            "\n[4/6] Building the SAC+Contextual index "
            "(document summary + per-chunk context, no extra LLM calls)..."
        )
        sac_contextual_chunks = [
            sac_chunk
            for norm_id, chunks in contextual_chunks_by_norm.items()
            for sac_chunk in apply_document_summary(document_summaries[norm_id], chunks)
        ]
        sac_contextual_dense_store = DenseChunkStore(conn, table=sac_contextual_table)
        sac_contextual_embeddings = embedder.embed(
            [chunk.retrieval_text for chunk in sac_contextual_chunks]
        )
        sac_contextual_dense_store.create_schema(dimensions=embedder.dimensions)
        sac_contextual_dense_store.upsert_chunks(sac_contextual_chunks, sac_contextual_embeddings)
        sac_contextual_strategy = DenseRetrieval(sac_contextual_dense_store, embedder)
        print("  evaluating sac_contextual...")
        _evaluate_and_checkpoint("sac_contextual", sac_contextual_strategy)

        print("\n[5/6] Building the RAPTOR tree (1 LLM call per group, per level, per document)...")
        summarizer = GeminiSummarizer(_SUMMARIZER_MODEL)
        raptor_chunks: list[Chunk] = []
        for _, chunks in documents.values():
            raptor_chunks.extend(build_raptor_tree(chunks, summarizer))
        raptor_dense_store = DenseChunkStore(conn, table=raptor_table)
        raptor_embeddings = embedder.embed([chunk.retrieval_text for chunk in raptor_chunks])
        raptor_dense_store.create_schema(dimensions=embedder.dimensions)
        raptor_dense_store.upsert_chunks(raptor_chunks, raptor_embeddings)
        raptor_strategy = DenseRetrieval(raptor_dense_store, embedder)
        print("  evaluating raptor...")
        _evaluate_and_checkpoint("raptor", raptor_strategy)

        print(
            f"\n[6/6] Building the GraphRAG (LightRAG, mode={_GRAPHRAG_MODE}) index "
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
