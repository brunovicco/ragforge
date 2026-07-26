#!/usr/bin/env python3
"""RAGForge v0.1 main benchmark runner (ADR-0004). Entry point for `make bench`/`make bench-live`.

Indexes the real corpus with the configured embedding model (ADR-0013) and
runs every strategy declared in configs/experiments/benchmark-v01.yaml against the
real golden set (datasets/regrag-br/judgments.json), reporting
recall/precision/nDCG/MRR@k per strategy plus, per ADR-0007, a generated
answer's Citation Accuracy/Faithfulness/Answer Relevancy - and writing a
versioned run record to experiments/<run_id>/results.json.

Also produces an auditable, tamper-evident evidence directory (ADR-0017) at
artifacts/runs/<run_id>/ - manifest.json (hash-identified corpus/dataset/
split/config, git SHA), events.jsonl (hash-chained stage events),
questions/<question_id>/<strategy>.json (per-question retrieval candidate
lineage), summaries/*.json (per-strategy generation/audit rollups),
checksums.sha256, and report.json/report.md - alongside, never replacing,
experiments/<run_id>/. Verify with `uv run python scripts/verify_run.py
<run_id>`. Generation lineage (token usage, latency, cache hit) is captured
only for GeminiAnswerGenerator - the ADR's own field list scopes those three
fields to the answer generator, not to the judge or auditor, whose
lineage is built entirely from already-computed AuditResult/JudgeResult
data instead (see lineage_ports.py). Per-question files carry retrieval
candidate lineage (reliably correlatable by question_id) but not per-
question generation/audit lineage - those are only captured in completion
order inside worker threads, not canonical question order, so attaching
them to a specific question file would risk mislabeling; they are
reported per-strategy in summaries/generation.json and summaries/audit.json
instead, a deliberate, documented scope boundary.

Document discovery, expected article counts, and source hashes come only
from the corpus manifest (datasets/regrag-br/corpus_manifest.yaml) and the
question selection from the versioned split
(datasets/regrag-br/split.json), optionally followed by the deterministic
stratified cost cap declared in the experiment configuration - both
ADR-0012. A preflight gate
(ragforge.evaluation.integrity) validates source hashes, split/golden-set
agreement, and structural-reference resolution before any indexing starts,
and fails the run closed (SystemExit) rather than silently indexing a
reduced or drifted corpus. RAPTOR is built once per document, never across a
document boundary, so a summary node can never blend unrelated norms. Every
selected question gets one QuestionRecord per strategy - including
unanswerable-class questions, which are excluded from ranking/citation
averages (ADR-0018: they are still generated and judged, for abstention
appropriateness) but never dropped from coverage - atomically persisted to
experiments/<run_id>/records.jsonl as each strategy finishes. A retried
incomplete strategy replaces only its own stale records.

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
The embedding step, contextualization, RAPTOR summarization, and GraphRAG's
entity-extraction LLM stay hard-coded to Gemini/local models regardless of
``judge.provider`` - only answer generation stays Gemini-hardcoded too
(GeminiAnswerGenerator). The judge is its own provider-neutral choice
(ADR-0018): ``judge.provider: openai`` (canonical for publishable results,
independent from the Gemini answer generator) or ``judge.provider: gemini``
(development fallback, labeled "exploratory_same_provider_judge" in the run
record since generation is also Gemini) - never a silent fallback between
the two.

Per strategy, this doubles the LLM calls made per question: one
GeminiAnswerGenerator.generate() call plus the judge's Faithfulness, Answer
Relevancy, and abstention scoring calls, on top of whatever the strategy
itself already costs (contextualization, RAPTOR summarization, GraphRAG
entity extraction). Judge scores are unvalidated until the ADR-0007/ADR-0018
human calibration exercise happens (see judge_calibration.py) - report them
with that caveat.

``audit.enabled: true`` (ADR-0016, off by default) wraps the generator with
AuditingAnswerGenerator: every answer is segmented into claims, checked
deterministically (existence, corpus version, retrieved-context presence),
and - only for claims that already pass every deterministic check -
semantically verified via OpenAI, with at most one bounded rewrite and
full re-audit when something fails. Off by default because the semantic
verifier and any rewrite are real, additional LLM calls the ADR itself
flags as a cost/latency trade-off; the run record always states
audit_enabled/audit_provider/audit_model regardless.

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

Bounded parallel execution and a minimal LLM cache (ADR-0014 + ADR-0004):
a FileLLMCache under experiments/<run_id>/llm-cache/ is shared by the
embedder, GeminiAnswerGenerator, and the judge (Gemini or OpenAI) - a call
already made for the exact same (model, prompt) is never repeated. A
ProviderLimiter bounds concurrent in-flight calls process-wide, per provider
(execution.gemini_max_in_flight, reused as the shared bound whichever hosted
provider is active). Answer generation + judge scoring use
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

This module is the CLI entrypoint and top-level orchestration only:
strategy/embedder/judge composition lives in run_strategies.py, table
rendering and the run-record schema in run_reporting.py, and ADR-0017
evidence-directory writes in run_evidence.py.
"""

import argparse
import asyncio
import dataclasses
import json
import shutil
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import yaml
from lightrag import LightRAG
from opensearchpy import OpenSearch

from ragforge.adapters.llm_cache import FileLLMCache
from ragforge.domain.models import Chunk, Judgment
from ragforge.domain.protocols import RetrievalStrategy
from ragforge.embeddings.caching import CachedEmbeddingModel
from ragforge.evaluation.artifact_writer import write_atomic
from ragforge.evaluation.audit_metrics import compute_audit_report
from ragforge.evaluation.audit_ports import AuditResult
from ragforge.evaluation.canonical_hash import canonical_json_hash
from ragforge.evaluation.event_log import EventLog
from ragforge.evaluation.index_cache import FileIndexRegistry, index_fingerprint
from ragforge.evaluation.index_namespace import derive_index_namespace
from ragforge.evaluation.integrity import (
    IntegrityError,
    verify_source_integrity,
    verify_split_integrity,
    verify_structural_references,
)
from ragforge.evaluation.judgments import load_judgments
from ragforge.evaluation.lineage_ports import GenerationLineage, RunManifest
from ragforge.evaluation.manifest import load_corpus_manifest
from ragforge.evaluation.records import read_records_jsonl, replace_strategy_records_jsonl
from ragforge.evaluation.run_evidence import (
    finalize_evidence_directory,
    reject_if_evidence_dir_already_completed,
    write_question_artifacts,
    write_summaries,
)
from ragforge.evaluation.run_lock import BenchmarkAlreadyRunningError, BenchmarkRunLock
from ragforge.evaluation.run_manifest import (
    build_initial_manifest,
    load_run_manifest,
    require_clean_worktree,
    resolve_git_sha,
)
from ragforge.evaluation.run_reporting import (
    build_metric_breakdowns,
    build_run_record,
    format_answer_quality_table,
    format_results_table,
    summarize_generation_usage,
)
from ragforge.evaluation.run_strategies import (
    _build_embedder,
    _build_judge_factory,
    _document_versions,
    _evaluate,
    _load_documents,
    _summarize_documents,
    build_base_strategies,
    build_contextual_strategy,
)
from ragforge.evaluation.split import Split, load_split
from ragforge.evaluation.split_builder import select_stratified_sample
from ragforge.generation.auditing_answer_generator import AuditingAnswerGenerator
from ragforge.generation.gemini_answer_generator import GeminiAnswerGenerator
from ragforge.generation.gemini_contextualizer import GeminiContextualizer
from ragforge.generation.gemini_document_summarizer import GeminiDocumentSummarizer
from ragforge.generation.gemini_summarizer import GeminiSummarizer
from ragforge.generation.openai_answer_rewriter import OpenAIAnswerRewriter
from ragforge.generation.openai_semantic_verifier import OpenAISemanticSupportVerifier
from ragforge.generation.ports import AnswerGenerator
from ragforge.ingestion.snapshot import snapshot_hash
from ragforge.reranking.cross_encoder_reranker import CrossEncoderReranker
from ragforge.retrieval.contextual.pipeline import contextualize_chunks
from ragforge.retrieval.dense.store import DenseChunkStore
from ragforge.retrieval.dense.strategy import DenseRetrieval
from ragforge.retrieval.graph.indexing import build_content_index, index_norm
from ragforge.retrieval.graph.lightrag_gemini import (
    build_gemini_embedding_func,
    build_gemini_llm_model_func,
)
from ragforge.retrieval.graph.strategy import GraphRagRetrieval
from ragforge.retrieval.raptor.pipeline import build_raptor_tree
from ragforge.retrieval.sac.pipeline import apply_document_summary
from ragforge.retrieval.sparse.store import SparseChunkStore

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "datasets/regrag-br/corpus_manifest.yaml"
SPLIT_PATH = ROOT / "datasets/regrag-br/split.json"
JUDGMENTS_PATH = ROOT / "datasets/regrag-br/judgments.json"
RESULTS_DIR = ROOT / "experiments"
# ADR-0017 evidence directory, alongside (never replacing) RESULTS_DIR -
# same run_id used for both, so a reviewer never has to reconcile two IDs
# for one run.
ARTIFACTS_DIR = ROOT / "artifacts" / "runs"
DATABASE_URL = "postgresql://ragforge:ragforge@localhost:5432/ragforge"

_GRAPHRAG_LLM_MODEL = "gemini-3.1-flash-lite"
_GRAPHRAG_MODE = "local"
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CONTEXTUALIZER_MODEL = "gemini-3.1-flash-lite"
_SUMMARIZER_MODEL = "gemini-3.1-flash-lite"
_DOCUMENT_SUMMARIZER_MODEL = "gemini-3.1-flash-lite"

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
_DEFAULT_EMBEDDING_CACHE_DIR = ".ragforge/cache/embeddings"
_DEFAULT_INDEX_CACHE_DIR = ".ragforge/cache/indexes"
_DEFAULT_SAMPLING_SEED = "regrag-br-benchmark-sample-v1"
_BASE_STRATEGY_LABELS = (
    "dense",
    "sparse_bm25",
    "hybrid_rrf",
    "reranked",
    "parent_child",
)
_SUPPORTED_STRATEGY_LABELS = frozenset(
    (*_BASE_STRATEGY_LABELS, "contextual", "sac", "sac_contextual", "raptor", "graphrag")
)


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


def _resolve_embedding_cache_dir(configured_path: str | None) -> Path:
    """Resolve a repository-local persistent embedding cache directory.

    Configuration cannot redirect cached source-derived data outside the
    repository. Absolute paths are accepted only when they still resolve
    below ``ROOT``.

    Raises:
        SystemExit: If the configured path escapes the repository.
    """
    raw_path = Path(configured_path or _DEFAULT_EMBEDDING_CACHE_DIR)
    resolved = (raw_path if raw_path.is_absolute() else ROOT / raw_path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise SystemExit("execution.embedding_cache_dir must resolve inside the repository")
    return resolved


def _resolve_index_cache_dir(configured_path: str | None) -> Path:
    """Resolve the repository-local reusable-index marker directory."""
    raw_path = Path(configured_path or _DEFAULT_INDEX_CACHE_DIR)
    resolved = (raw_path if raw_path.is_absolute() else ROOT / raw_path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise SystemExit("execution.index_cache_dir must resolve inside the repository")
    return resolved


def _validate_requested_strategies(raw_labels: object) -> tuple[str, ...]:
    """Validate and preserve the configured benchmark strategy order.

    Raises:
        SystemExit: If labels are absent, duplicated, non-string, or unsupported.
    """
    if not isinstance(raw_labels, list) or not raw_labels:
        raise SystemExit("strategies must be a non-empty list")
    if any(not isinstance(label, str) for label in raw_labels):
        raise SystemExit("every strategy label must be a string")
    labels = tuple(raw_labels)
    if len(labels) != len(set(labels)):
        raise SystemExit("strategies must not contain duplicate labels")
    unknown = sorted(set(labels) - _SUPPORTED_STRATEGY_LABELS)
    if unknown:
        raise SystemExit(f"unknown strategies: {', '.join(unknown)}")
    return labels


def _select_split_judgments(
    split: Split,
    judgments: list[Judgment],
    split_name: str,
) -> list[Judgment]:
    """Return judgments in the declared split order.

    Raises:
        SystemExit: If ``split_name`` is not a supported partition.
    """
    split_ids_by_name = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    split_ids = split_ids_by_name.get(split_name)
    if split_ids is None:
        raise SystemExit("dataset.split must be one of train, validation, or test")
    by_id = {judgment.question_id: judgment for judgment in judgments}
    return [by_id[question_id] for question_id in split_ids]


def _strategy_checkpoint_complete(metrics: Mapping[str, object], expected_answers: int) -> bool:
    """Return whether a strategy checkpoint has complete, error-free answer coverage."""
    answer_n = metrics.get("answer_n")
    answer_errors = metrics.get("answer_errors")
    retrieval_errors = metrics.get("errors")
    return (
        isinstance(answer_n, (int, float))
        and not isinstance(answer_n, bool)
        and float(answer_n) == expected_answers
        and isinstance(answer_errors, (int, float))
        and not isinstance(answer_errors, bool)
        and float(answer_errors) == 0.0
        and isinstance(retrieval_errors, (int, float))
        and not isinstance(retrieval_errors, bool)
        and float(retrieval_errors) == 0.0
    )


def _verify_resume_identity(
    previous: Mapping[str, object],
    index_namespace: str,
    generation_model: str,
    judge_provider: str,
    judge_model: str,
) -> None:
    """Fail closed if a resumed run's identity doesn't match the current configuration.

    ADR-0014: "changing model, prompt, split, or strategy creates a new
    run" - ``index_namespace`` alone already encodes corpus content,
    chunking/retrieval-text schema, and the embedding's complete identity
    (ADR-0013), so checking it plus the generation/judge identity covers
    everything that would make reusing cached answers/judge scores unsafe.
    ``judge_provider`` is checked separately from ``judge_model`` since
    switching provider (e.g. gemini -> openai) is a change of judge identity
    even if a model name happened to collide (ADR-0018).

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
    if previous.get("judge_provider") != judge_provider:
        mismatches.append(
            f"judge_provider: {previous.get('judge_provider')!r} != {judge_provider!r}"
        )
    if previous.get("judge_model") != judge_model:
        mismatches.append(f"judge_model: {previous.get('judge_model')!r} != {judge_model!r}")
    if mismatches:
        raise SystemExit(
            "--resume identity mismatch (changing model/corpus/split creates a new run):\n"
            + "\n".join(f"- {mismatch}" for mismatch in mismatches)
        )


def _verify_resume_manifest_identity(
    previous: RunManifest,
    *,
    run_id: str,
    git_sha: str,
    corpus_hash: str,
    dataset_hash: str,
    split_hash: str,
    configuration_hash: str,
    models: dict[str, str],
    strategies: tuple[str, ...],
    execution: dict[str, object],
) -> None:
    """Fail closed if resumed evidence differs from the run that started it."""
    expected: dict[str, object] = {
        "run_id": run_id,
        "status": "running",
        "git_sha": git_sha,
        "corpus_hash": corpus_hash,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "configuration_hash": configuration_hash,
        "models": models,
        "strategies": strategies,
        "execution": execution,
    }
    mismatches = [
        f"{field}: {getattr(previous, field)!r} != {value!r}"
        for field, value in expected.items()
        if getattr(previous, field) != value
    ]
    if mismatches:
        raise SystemExit(
            "--resume evidence identity mismatch; start a new run:\n"
            + "\n".join(f"- {mismatch}" for mismatch in mismatches)
        )


def _run() -> None:
    """Index the real corpus with every strategy and score each against the golden set."""
    args = parse_args()
    _reject_cache_mode(args.mode)
    args.config = args.config.resolve()
    require_clean_worktree(ROOT)
    current_git_sha = resolve_git_sha(ROOT)
    if current_git_sha == "unknown":
        raise SystemExit("auditable benchmark could not resolve the current Git commit")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    requested_strategies = _validate_requested_strategies(config.get("strategies"))
    requested_strategy_set = set(requested_strategies)
    split_name = config["dataset"]["split"]
    max_questions = config["dataset"].get("max_questions")
    sampling_seed = config["dataset"].get("sampling_seed", _DEFAULT_SAMPLING_SEED)
    if max_questions is not None and (
        isinstance(max_questions, bool) or not isinstance(max_questions, int)
    ):
        raise SystemExit("dataset.max_questions must be a positive integer")
    if max_questions is not None and max_questions <= 0:
        raise SystemExit("dataset.max_questions must be a positive integer")
    if not isinstance(sampling_seed, str) or not sampling_seed:
        raise SystemExit("dataset.sampling_seed must be a non-empty string")
    top_k = config["retrieval"]["top_k"]
    rerank_pool = config["retrieval"]["rerank_pool"]
    embedding_provider = config["embedding"]["provider"]
    embedding_model = config["embedding"]["model"]
    embedding_dimensions = config["embedding"].get("dimensions")
    embedding_device = config["embedding"].get("device")
    generation_model = config["generation"]["model"]
    judge_provider = config["judge"]["provider"]
    judge_model = config["judge"]["model"]
    judge_embedding_model = config["judge"]["embedding_model"]
    judge_reasoning_effort = config["judge"].get("reasoning_effort", "medium")
    judge_max_output_tokens = config["judge"].get("max_output_tokens", 8192)
    audit_config = config.get("audit", {})
    audit_enabled = audit_config.get("enabled", False)
    audit_provider = audit_config.get("provider", "openai")
    audit_model = audit_config.get("model")
    audit_reasoning_effort = audit_config.get("reasoning_effort", "medium")
    pricing_config = config.get("pricing", {}).get("generation", {})
    generation_input_price = pricing_config.get("input_per_million_usd")
    generation_output_price = pricing_config.get("output_per_million_usd")
    execution_config = config.get("execution", {})
    answer_quality_workers = execution_config.get(
        "answer_quality_workers", _DEFAULT_ANSWER_QUALITY_WORKERS
    )
    gemini_max_in_flight = execution_config.get(
        "gemini_max_in_flight", _DEFAULT_GEMINI_MAX_IN_FLIGHT
    )
    embedding_cache_dir = _resolve_embedding_cache_dir(execution_config.get("embedding_cache_dir"))
    index_cache_dir = _resolve_index_cache_dir(execution_config.get("index_cache_dir"))

    manifest = load_corpus_manifest(MANIFEST_PATH)
    split = load_split(SPLIT_PATH)
    judgments = load_judgments(JUDGMENTS_PATH)

    print("Running preflight integrity checks (ADR-0012)...")
    try:
        verify_source_integrity(manifest, root=ROOT)
        verify_split_integrity(split, judgments)
    except IntegrityError as exc:
        raise SystemExit(f"preflight integrity check failed:\n{exc}") from exc
    judgments = _select_split_judgments(split, judgments, split_name)
    if max_questions is not None:
        try:
            judgments = select_stratified_sample(
                judgments,
                max_questions=max_questions,
                seed=sampling_seed,
            )
        except ValueError as exc:
            raise SystemExit(f"invalid dataset sample: {exc}") from exc
        print(
            f"Selected {len(judgments)} stratified questions "
            f"(max_questions={max_questions}, seed={sampling_seed})."
        )

    run_id = args.resume or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"
    cache = FileLLMCache(run_dir / "llm-cache")

    artifacts_dir = ARTIFACTS_DIR / run_id
    reject_if_evidence_dir_already_completed(artifacts_dir)
    existing_run_manifest: RunManifest | None = None
    if args.resume is not None:
        manifest_path = artifacts_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"--resume requires an existing evidence manifest at {manifest_path}")
        try:
            existing_run_manifest = load_run_manifest(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid resume evidence manifest: {exc}") from exc

    print("Extracting and chunking real corpus documents...")
    documents = _load_documents(manifest)
    all_chunks = [chunk for _, chunks in documents.values() for chunk in chunks]
    print(f"{len(all_chunks)} chunks total across {len(documents)} documents")
    document_versions = _document_versions(manifest)
    corpus_structural_ids = {
        norm_id: {ref for chunk in chunks for ref in chunk.structural_ids}
        for norm_id, (_, chunks) in documents.items()
    }

    try:
        verify_structural_references(judgments, documents)
    except IntegrityError as exc:
        raise SystemExit(f"preflight integrity check failed:\n{exc}") from exc

    print(f"Loading embedding model {embedding_model} (provider={embedding_provider})...")
    embedder, embedding_identity = _build_embedder(
        embedding_provider,
        embedding_model,
        embedding_dimensions,
        cache,
        gemini_max_in_flight,
        device=embedding_device,
    )
    embedding_identity_hash = canonical_json_hash(dataclasses.asdict(embedding_identity))
    embedder = CachedEmbeddingModel(
        embedder,
        embedding_identity,
        FileLLMCache(embedding_cache_dir / embedding_identity_hash),
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
    index_registry = FileIndexRegistry(index_cache_dir / index_namespace)

    run_metrics: dict[str, dict[str, float]] = {}
    results_path = run_dir / "results.json"
    if args.resume is not None and results_path.exists():
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        _verify_resume_identity(
            previous, index_namespace, generation_model, judge_provider, judge_model
        )
        previous_metrics = previous["metrics"]
        run_metrics = {
            label: metrics
            for label, metrics in previous_metrics.items()
            if _strategy_checkpoint_complete(metrics, len(judgments))
        }
        incomplete_labels = sorted(set(previous_metrics) - set(run_metrics))
        print(f"Resuming {run_id}: {sorted(run_metrics)} already scored.")
        if incomplete_labels:
            print(f"  retrying incomplete checkpoints: {incomplete_labels}")
    pending_strategy_set = requested_strategy_set - set(run_metrics)

    print("Writing ADR-0017 evidence manifest and snapshots...")
    dataset_hash = snapshot_hash(JUDGMENTS_PATH)
    split_hash = snapshot_hash(SPLIT_PATH)
    configuration_hash = canonical_json_hash(config)
    model_identities = {
        "embedding": f"{embedding_provider}/{embedding_model}",
        "generation": generation_model,
        "judge": f"{judge_provider}/{judge_model}",
        "audit": f"{audit_provider}/{audit_model}" if audit_enabled else "disabled",
    }
    manifest_execution = dict(execution_config)
    if existing_run_manifest is not None:
        _verify_resume_manifest_identity(
            existing_run_manifest,
            run_id=run_id,
            git_sha=current_git_sha,
            corpus_hash=manifest.content_hash,
            dataset_hash=dataset_hash,
            split_hash=split_hash,
            configuration_hash=configuration_hash,
            models=model_identities,
            strategies=requested_strategies,
            execution=manifest_execution,
        )
        run_manifest = existing_run_manifest
    else:
        run_manifest = build_initial_manifest(
            run_id=run_id,
            git_sha=current_git_sha,
            corpus_hash=manifest.content_hash,
            dataset_hash=dataset_hash,
            split_hash=split_hash,
            configuration_hash=configuration_hash,
            models=model_identities,
            strategies=requested_strategies,
            execution=manifest_execution,
        )
        write_atomic(
            artifacts_dir / "manifest.json",
            json.dumps(dataclasses.asdict(run_manifest), ensure_ascii=False, indent=2),
        )
    write_atomic(
        artifacts_dir / "configuration.resolved.yaml", args.config.read_text(encoding="utf-8")
    )
    write_atomic(
        artifacts_dir / "corpus-manifest.snapshot.yaml", MANIFEST_PATH.read_text(encoding="utf-8")
    )
    write_atomic(artifacts_dir / "split.snapshot.json", SPLIT_PATH.read_text(encoding="utf-8"))
    write_atomic(
        artifacts_dir / "question-selection.snapshot.json",
        json.dumps(
            {
                "schema_version": 1,
                "split": split_name,
                "max_questions": max_questions,
                "sampling_seed": sampling_seed if max_questions is not None else None,
                "question_ids": [judgment.question_id for judgment in judgments],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    event_log = EventLog(run_id, artifacts_dir / "events.jsonl")

    print(
        f"Loading generation model {generation_model} and "
        f"judge model {judge_model} (provider={judge_provider})..."
    )
    base_generator = GeminiAnswerGenerator(
        generation_model, cache=cache, max_in_flight=gemini_max_in_flight
    )
    generator: AnswerGenerator = base_generator
    auditing_generator: AuditingAnswerGenerator | None = None
    if audit_enabled:
        if audit_model is None:
            raise SystemExit("audit.enabled is true but audit.model is not set in the config")
        print(f"Loading audit model {audit_model} (provider={audit_provider})...")
        if audit_provider != "openai":
            raise SystemExit(f"unknown audit provider {audit_provider!r}; expected 'openai'")
        verifier = OpenAISemanticSupportVerifier(
            audit_model,
            reasoning_effort=audit_reasoning_effort,
            cache=cache,
            max_in_flight=gemini_max_in_flight,
        )
        rewriter = OpenAIAnswerRewriter(
            audit_model,
            reasoning_effort=audit_reasoning_effort,
            cache=cache,
            max_in_flight=gemini_max_in_flight,
        )
        auditing_generator = AuditingAnswerGenerator(
            generator, verifier, rewriter, corpus_structural_ids, document_versions
        )
        generator = auditing_generator
    judge_factory = _build_judge_factory(
        judge_provider,
        judge_model,
        judge_embedding_model,
        judge_reasoning_effort,
        judge_max_output_tokens,
        cache,
        gemini_max_in_flight,
    )

    conn = psycopg.connect(DATABASE_URL)
    os_client = OpenSearch(hosts=["http://localhost:9200"], use_ssl=False, verify_certs=False)
    generation_lineage_by_strategy: dict[str, list[GenerationLineage]] = {}
    audit_results_by_strategy: dict[str, list[AuditResult]] = {}
    stage_durations_seconds: dict[str, float] = {}
    stage_started_at: dict[str, float] = {}

    def _stage_started(event_stage: str, stage: str) -> None:
        """Record and emit the start of one benchmark stage."""
        key = f"{event_stage}:{stage}"
        stage_started_at[key] = time.monotonic()
        event_log.emit(event_stage, "started", {"stage": stage})

    def _stage_completed(event_stage: str, stage: str) -> None:
        """Record and emit successful completion with monotonic duration."""
        key = f"{event_stage}:{stage}"
        duration = time.monotonic() - stage_started_at.pop(key)
        stage_durations_seconds[key] = duration
        event_log.emit(
            event_stage,
            "completed",
            {"stage": stage, "duration_seconds": duration},
        )

    def _ensure_reusable_index(
        *,
        stage: str,
        chunks: list[Chunk],
        dense_store: DenseChunkStore,
        sparse_store: SparseChunkStore | None,
        derivation_identity: str,
    ) -> None:
        """Reuse a complete exact index or rebuild and atomically mark it complete."""
        fingerprint = index_fingerprint(
            stage=stage,
            index_namespace=index_namespace,
            chunks=chunks,
            derivation_identity=derivation_identity,
        )
        expected_ids = {chunk.chunk_id for chunk in chunks}
        expects_sparse = sparse_store is not None
        marker_matches = index_registry.matches(
            stage=stage,
            fingerprint=fingerprint,
            chunk_count=len(chunks),
            dense=True,
            sparse=expects_sparse,
        )
        stores_match = dense_store.has_exact_chunk_ids(expected_ids) and (
            sparse_store is None or sparse_store.has_exact_chunk_ids(expected_ids)
        )
        if marker_matches and stores_match:
            stage_durations_seconds[f"indexing:{stage}"] = 0.0
            event_log.emit(
                "indexing",
                "reused",
                {"stage": stage, "fingerprint": fingerprint, "chunk_count": len(chunks)},
            )
            return

        index_registry.invalidate(stage)
        _stage_started("indexing", stage)
        dense_store.drop_schema()
        if sparse_store is not None:
            sparse_store.drop_index()
        embeddings = embedder.embed([chunk.retrieval_text for chunk in chunks])
        dense_store.create_schema(dimensions=embedder.dimensions)
        dense_store.upsert_chunks(chunks, embeddings)
        dense_store.create_search_index()
        if sparse_store is not None:
            sparse_store.create_index()
            sparse_store.index_chunks(chunks)
        index_registry.mark_complete(
            stage=stage,
            fingerprint=fingerprint,
            chunk_count=len(chunks),
            dense=True,
            sparse=expects_sparse,
        )
        _stage_completed("indexing", stage)

    def _checkpoint() -> None:
        """Write the run record as computed so far - survives a later strategy crashing."""
        metric_breakdowns = build_metric_breakdowns(
            read_records_jsonl(records_path),
            judgments,
        )
        generation_usage = summarize_generation_usage(
            generation_lineage_by_strategy,
            input_price_per_million_usd=generation_input_price,
            output_price_per_million_usd=generation_output_price,
        )
        record = build_run_record(
            run_id=run_id,
            mode=args.mode,
            config_path=str(args.config.relative_to(ROOT)),
            embedding_identity=embedding_identity,
            index_namespace=index_namespace,
            generation_model=generation_model,
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_reasoning_effort=judge_reasoning_effort if judge_provider == "openai" else None,
            audit_enabled=audit_enabled,
            audit_provider=audit_provider if audit_enabled else None,
            audit_model=audit_model if audit_enabled else None,
            corpus_version=manifest.corpus_version,
            split_dataset_version=split.dataset_version,
            n_chunks=len(all_chunks),
            top_k=top_k,
            run_metrics=run_metrics,
            metric_breakdowns=metric_breakdowns,
            stage_durations_seconds=stage_durations_seconds,
            generation_usage=generation_usage,
        )
        results_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))

    def _evaluate_and_checkpoint(label: str, strategy: RetrievalStrategy) -> None:
        """Score ``strategy``, replace its records.jsonl lines, then checkpoint results.json.

        A no-op when ``label`` is already in ``run_metrics`` (--resume).
        The outer orchestration also excludes completed labels from stage
        construction, so neither indexing nor per-question calls are repeated.
        """
        if label in run_metrics:
            print(f"  skipping {label} (already scored, --resume)")
            return
        started = time.monotonic()
        event_log.emit("strategy", "started", {"label": label})
        try:
            metrics, records, candidate_lineage = _evaluate(
                label,
                strategy,
                judgments,
                generator,
                judge_factory,
                top_k,
                answer_quality_workers,
                embedding_identity_hash=embedding_identity_hash,
            )
        except BaseException:
            event_log.emit(
                "strategy",
                "failed",
                {"label": label, "duration_seconds": time.monotonic() - started},
            )
            raise
        generation_lineage = base_generator.drain_generation_lineage()
        generation_lineage_by_strategy[label] = generation_lineage
        if auditing_generator is not None:
            audit_results = auditing_generator.drain_audit_results()
            audit_results_by_strategy[label] = audit_results
            metrics = {**metrics, **compute_audit_report(audit_results)}
        run_metrics[label] = metrics
        replace_strategy_records_jsonl(records_path, label, records)
        write_question_artifacts(artifacts_dir, label, records, candidate_lineage)
        write_summaries(
            artifacts_dir,
            run_metrics,
            generation_lineage_by_strategy,
            audit_results_by_strategy,
            metric_breakdowns=build_metric_breakdowns(
                read_records_jsonl(records_path),
                judgments,
            ),
            generation_usage=summarize_generation_usage(
                generation_lineage_by_strategy,
                input_price_per_million_usd=generation_input_price,
                output_price_per_million_usd=generation_output_price,
            ),
        )
        _checkpoint()
        duration = time.monotonic() - started
        stage_durations_seconds[f"strategy:{label}"] = duration
        event_log.emit(
            "strategy",
            "completed",
            {"label": label, "n": metrics.get("n", 0.0), "duration_seconds": duration},
        )

    try:
        requested_base_labels = [
            label for label in _BASE_STRATEGY_LABELS if label in pending_strategy_set
        ]
        if requested_base_labels:
            print("\n[1/6] Indexing the base chunks (dense + sparse)...")
            base_dense_store = DenseChunkStore(conn, table=base_table)
            base_sparse_store = SparseChunkStore(os_client, index=base_table)
            _ensure_reusable_index(
                stage="base",
                chunks=all_chunks,
                dense_store=base_dense_store,
                sparse_store=base_sparse_store,
                derivation_identity=_RETRIEVAL_TEXT_SCHEMA_VERSION,
            )

            base_strategies = build_base_strategies(
                base_dense_store,
                base_sparse_store,
                embedder,
                CrossEncoderReranker(_RERANKER_MODEL),
                rerank_pool,
            )
            for label in requested_base_labels:
                print(f"  evaluating {label}...")
                _evaluate_and_checkpoint(label, base_strategies[label])

        contextual_chunks_by_norm: dict[str, list[Chunk]] = {}
        needs_contextual_chunks = bool({"contextual", "sac_contextual"} & pending_strategy_set)
        if needs_contextual_chunks:
            print("\n[2/6] Building contextual retrieval text (1 LLM call per chunk)...")
            _stage_started("contextualization", "contextual")
            contextualizer = GeminiContextualizer(_CONTEXTUALIZER_MODEL)
            contextual_chunks_by_norm = {
                norm_id: contextualize_chunks(full_text, chunks, contextualizer)
                for norm_id, (full_text, chunks) in documents.items()
            }
            _stage_completed("contextualization", "contextual")

        if "contextual" in pending_strategy_set:
            contextual_chunks = [
                chunk for chunks in contextual_chunks_by_norm.values() for chunk in chunks
            ]
            contextual_dense_store = DenseChunkStore(conn, table=contextual_table)
            contextual_sparse_store = SparseChunkStore(os_client, index=contextual_table)
            _ensure_reusable_index(
                stage="contextual",
                chunks=contextual_chunks,
                dense_store=contextual_dense_store,
                sparse_store=contextual_sparse_store,
                derivation_identity=_CONTEXTUALIZER_MODEL,
            )
            contextual_strategy = build_contextual_strategy(
                contextual_dense_store, contextual_sparse_store, embedder
            )
            print("  evaluating contextual...")
            _evaluate_and_checkpoint("contextual", contextual_strategy)

        document_summaries: dict[str, str] = {}
        if {"sac", "sac_contextual"} & pending_strategy_set:
            print("\n[3/6] Summarizing documents for SAC (1 LLM call per document)...")
            _stage_started("summarization", "sac")
            document_summarizer = GeminiDocumentSummarizer(
                _DOCUMENT_SUMMARIZER_MODEL, cache=cache, max_in_flight=gemini_max_in_flight
            )
            document_summaries = _summarize_documents(
                documents, document_versions, document_summarizer, answer_quality_workers
            )
            _stage_completed("summarization", "sac")

        if "sac" in pending_strategy_set:
            sac_chunks = [
                sac_chunk
                for norm_id, (_, chunks) in documents.items()
                for sac_chunk in apply_document_summary(document_summaries[norm_id], chunks)
            ]
            sac_dense_store = DenseChunkStore(conn, table=sac_table)
            _ensure_reusable_index(
                stage="sac",
                chunks=sac_chunks,
                dense_store=sac_dense_store,
                sparse_store=None,
                derivation_identity=_DOCUMENT_SUMMARIZER_MODEL,
            )
            sac_strategy = DenseRetrieval(sac_dense_store, embedder)
            print("  evaluating sac...")
            _evaluate_and_checkpoint("sac", sac_strategy)

        if "sac_contextual" in pending_strategy_set:
            print("\n[4/6] Building the SAC+Contextual index...")
            sac_contextual_chunks = [
                sac_chunk
                for norm_id, chunks in contextual_chunks_by_norm.items()
                for sac_chunk in apply_document_summary(document_summaries[norm_id], chunks)
            ]
            sac_contextual_dense_store = DenseChunkStore(conn, table=sac_contextual_table)
            _ensure_reusable_index(
                stage="sac_contextual",
                chunks=sac_contextual_chunks,
                dense_store=sac_contextual_dense_store,
                sparse_store=None,
                derivation_identity=f"{_DOCUMENT_SUMMARIZER_MODEL}+{_CONTEXTUALIZER_MODEL}",
            )
            sac_contextual_strategy = DenseRetrieval(sac_contextual_dense_store, embedder)
            print("  evaluating sac_contextual...")
            _evaluate_and_checkpoint("sac_contextual", sac_contextual_strategy)

        if "raptor" in pending_strategy_set:
            print(
                "\n[5/6] Building the RAPTOR tree "
                "(1 LLM call per group, per level, per document)..."
            )
            summarizer = GeminiSummarizer(_SUMMARIZER_MODEL)
            raptor_chunks: list[Chunk] = []
            for _, chunks in documents.values():
                raptor_chunks.extend(build_raptor_tree(chunks, summarizer))
            raptor_dense_store = DenseChunkStore(conn, table=raptor_table)
            _ensure_reusable_index(
                stage="raptor",
                chunks=raptor_chunks,
                dense_store=raptor_dense_store,
                sparse_store=None,
                derivation_identity=_SUMMARIZER_MODEL,
            )
            raptor_strategy = DenseRetrieval(raptor_dense_store, embedder)
            print("  evaluating raptor...")
            _evaluate_and_checkpoint("raptor", raptor_strategy)

        if "graphrag" in pending_strategy_set:
            print(
                f"\n[6/6] Building the GraphRAG (LightRAG, mode={_GRAPHRAG_MODE}) index "
                "(multiple LLM calls per chunk)..."
            )
            graph_fingerprint = index_fingerprint(
                stage="graphrag",
                index_namespace=index_namespace,
                chunks=all_chunks,
                derivation_identity=f"{_GRAPHRAG_LLM_MODEL}:{_GRAPHRAG_MODE}",
            )
            graphrag_dir = index_cache_dir / index_namespace / "graphrag-data" / graph_fingerprint
            graph_reusable = (
                index_registry.matches(
                    stage="graphrag",
                    fingerprint=graph_fingerprint,
                    chunk_count=len(all_chunks),
                    dense=False,
                    sparse=False,
                )
                and graphrag_dir.is_dir()
                and any(graphrag_dir.iterdir())
            )
            if graph_reusable:
                stage_durations_seconds["indexing:graphrag"] = 0.0
                event_log.emit(
                    "indexing",
                    "reused",
                    {
                        "stage": "graphrag",
                        "fingerprint": graph_fingerprint,
                        "chunk_count": len(all_chunks),
                    },
                )
            else:
                index_registry.invalidate("graphrag")
                shutil.rmtree(graphrag_dir, ignore_errors=True)
                _stage_started("indexing", "graphrag")
            graphrag_dir.mkdir(parents=True, exist_ok=True)
            rag = LightRAG(
                working_dir=str(graphrag_dir),
                embedding_func=build_gemini_embedding_func(embedder),
                llm_model_func=build_gemini_llm_model_func(_GRAPHRAG_LLM_MODEL),
            )
            asyncio.run(rag.initialize_storages())
            try:
                if not graph_reusable:
                    for norm_id, (_, chunks) in documents.items():
                        index_norm(rag, norm_id, chunks)
                    index_registry.mark_complete(
                        stage="graphrag",
                        fingerprint=graph_fingerprint,
                        chunk_count=len(all_chunks),
                        dense=False,
                        sparse=False,
                    )
                    _stage_completed("indexing", "graphrag")
                graphrag_strategy = GraphRagRetrieval(
                    rag, build_content_index(all_chunks), mode=_GRAPHRAG_MODE
                )
                print("  evaluating graphrag...")
                _evaluate_and_checkpoint("graphrag", graphrag_strategy)
            finally:
                asyncio.run(rag.finalize_storages())
    finally:
        print("\nClosing reusable index connections...")
        conn.rollback()
        conn.close()
        os_client.close()

    results_table = format_results_table(list(requested_strategies), run_metrics)
    answer_quality_table = format_answer_quality_table(list(requested_strategies), run_metrics)
    print(f"\n{results_table}")
    print(f"\n{answer_quality_table}")

    _checkpoint()
    print(f"\nRun record written to {run_dir.relative_to(ROOT)}/results.json")

    print("\nFinalizing ADR-0017 evidence directory...")
    merged_generation_usage = write_summaries(
        artifacts_dir,
        run_metrics,
        generation_lineage_by_strategy,
        audit_results_by_strategy,
        metric_breakdowns=build_metric_breakdowns(
            read_records_jsonl(records_path),
            judgments,
        ),
        generation_usage=summarize_generation_usage(
            generation_lineage_by_strategy,
            input_price_per_million_usd=generation_input_price,
            output_price_per_million_usd=generation_output_price,
        ),
    )
    report_record = build_run_record(
        run_id=run_id,
        mode=args.mode,
        config_path=str(args.config.relative_to(ROOT)),
        embedding_identity=embedding_identity,
        index_namespace=index_namespace,
        generation_model=generation_model,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_reasoning_effort=judge_reasoning_effort if judge_provider == "openai" else None,
        audit_enabled=audit_enabled,
        audit_provider=audit_provider if audit_enabled else None,
        audit_model=audit_model if audit_enabled else None,
        corpus_version=manifest.corpus_version,
        split_dataset_version=split.dataset_version,
        n_chunks=len(all_chunks),
        top_k=top_k,
        run_metrics=run_metrics,
        metric_breakdowns=build_metric_breakdowns(
            read_records_jsonl(records_path),
            judgments,
        ),
        stage_durations_seconds=stage_durations_seconds,
        generation_usage=merged_generation_usage,
    )
    write_atomic(
        results_path,
        json.dumps(report_record, ensure_ascii=False, indent=2),
    )
    write_atomic(
        artifacts_dir / "report.json", json.dumps(report_record, ensure_ascii=False, indent=2)
    )
    write_atomic(
        artifacts_dir / "report.md",
        f"# RAGForge benchmark report - {run_id}\n\n"
        f"## Retrieval\n\n```\n{results_table}\n```\n\n"
        f"## Answer quality\n\n```\n{answer_quality_table}\n```\n",
    )

    finalize_evidence_directory(artifacts_dir, run_manifest)
    print(f"Evidence directory finalized at {artifacts_dir.relative_to(ROOT)}/")


def main() -> None:
    """Run one benchmark at a time for the repository's shared indexes."""
    try:
        with BenchmarkRunLock(ROOT / ".ragforge" / "benchmark.lock"):
            _run()
    except BenchmarkAlreadyRunningError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
