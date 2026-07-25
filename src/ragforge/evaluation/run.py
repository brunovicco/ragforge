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
question selection only from the versioned split
(datasets/regrag-br/split.json) - both ADR-0012. A preflight gate
(ragforge.evaluation.integrity) validates source hashes, split/golden-set
agreement, and structural-reference resolution before any indexing starts,
and fails the run closed (SystemExit) rather than silently indexing a
reduced or drifted corpus. RAPTOR is built once per document, never across a
document boundary, so a summary node can never blend unrelated norms. Every
selected question gets one immutable QuestionRecord per strategy - including
unanswerable-class questions, which are excluded from ranking/citation
averages (ADR-0018: they are still generated and judged, for abstention
appropriateness) but never dropped from coverage - appended to
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
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import yaml
from lightrag import LightRAG
from opensearchpy import OpenSearch

from ragforge.adapters.llm_cache import FileLLMCache
from ragforge.domain.models import Chunk
from ragforge.domain.protocols import RetrievalStrategy
from ragforge.evaluation.artifact_writer import (
    compute_checksums,
    write_atomic,
    write_checksums_file,
)
from ragforge.evaluation.audit_metrics import compute_audit_report
from ragforge.evaluation.audit_ports import AuditResult
from ragforge.evaluation.canonical_hash import canonical_json_hash
from ragforge.evaluation.event_log import EventLog
from ragforge.evaluation.index_namespace import derive_index_namespace
from ragforge.evaluation.integrity import (
    IntegrityError,
    verify_source_integrity,
    verify_split_integrity,
    verify_structural_references,
)
from ragforge.evaluation.judgments import load_judgments
from ragforge.evaluation.lineage_ports import GenerationLineage
from ragforge.evaluation.manifest import load_corpus_manifest
from ragforge.evaluation.records import append_records_jsonl
from ragforge.evaluation.run_evidence import (
    reject_if_evidence_dir_already_completed,
    write_question_artifacts,
    write_summaries,
)
from ragforge.evaluation.run_manifest import (
    build_initial_manifest,
    finalize_manifest,
    resolve_git_sha,
)
from ragforge.evaluation.run_reporting import (
    build_run_record,
    format_answer_quality_table,
    format_results_table,
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
from ragforge.evaluation.split import load_split
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
    judge_provider = config["judge"]["provider"]
    judge_model = config["judge"]["model"]
    judge_embedding_model = config["judge"]["embedding_model"]
    judge_reasoning_effort = config["judge"].get("reasoning_effort", "medium")
    audit_config = config.get("audit", {})
    audit_enabled = audit_config.get("enabled", False)
    audit_provider = audit_config.get("provider", "openai")
    audit_model = audit_config.get("model")
    audit_reasoning_effort = audit_config.get("reasoning_effort", "medium")
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

    artifacts_dir = ARTIFACTS_DIR / run_id
    reject_if_evidence_dir_already_completed(artifacts_dir)

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
        embedding_provider, embedding_model, embedding_dimensions, cache, gemini_max_in_flight
    )
    embedding_identity_hash = canonical_json_hash(dataclasses.asdict(embedding_identity))
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
        _verify_resume_identity(
            previous, index_namespace, generation_model, judge_provider, judge_model
        )
        run_metrics = previous["metrics"]
        print(f"Resuming {run_id}: {sorted(run_metrics)} already scored.")

    print("Writing ADR-0017 evidence manifest and snapshots...")
    run_manifest = build_initial_manifest(
        run_id=run_id,
        git_sha=resolve_git_sha(),
        corpus_hash=manifest.content_hash,
        dataset_hash=snapshot_hash(JUDGMENTS_PATH),
        split_hash=snapshot_hash(SPLIT_PATH),
        configuration_hash=canonical_json_hash(config),
        models={
            "embedding": f"{embedding_provider}/{embedding_model}",
            "generation": generation_model,
            "judge": f"{judge_provider}/{judge_model}",
            "audit": f"{audit_provider}/{audit_model}" if audit_enabled else "disabled",
        },
        strategies=tuple(config["strategies"]),
        execution=dict(execution_config),
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
        cache,
        gemini_max_in_flight,
    )

    conn = psycopg.connect(DATABASE_URL)
    os_client = OpenSearch(hosts=["http://localhost:9200"], use_ssl=False, verify_certs=False)
    graphrag_dir = Path(tempfile.mkdtemp(prefix="ragforge-bench-graphrag-"))
    tables = [base_table, contextual_table, sac_table, sac_contextual_table, raptor_table]
    generation_lineage_by_strategy: dict[str, list[GenerationLineage]] = {}
    audit_results_by_strategy: dict[str, list[AuditResult]] = {}

    def _checkpoint() -> None:
        """Write the run record as computed so far - survives a later strategy crashing."""
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
        event_log.emit("strategy", "started", {"label": label})
        metrics, records, candidate_lineage = _evaluate(
            strategy,
            judgments,
            generator,
            judge_factory,
            top_k,
            answer_quality_workers,
            embedding_identity_hash=embedding_identity_hash,
        )
        generation_lineage = base_generator.drain_generation_lineage()
        generation_lineage_by_strategy[label] = generation_lineage
        if auditing_generator is not None:
            audit_results = auditing_generator.drain_audit_results()
            audit_results_by_strategy[label] = audit_results
            metrics = {**metrics, **compute_audit_report(audit_results)}
        run_metrics[label] = metrics
        append_records_jsonl(records_path, records)
        write_question_artifacts(artifacts_dir, label, records, candidate_lineage)
        _checkpoint()
        event_log.emit("strategy", "completed", {"label": label, "n": metrics.get("n", 0.0)})

    try:
        print("\n[1/6] Indexing the base chunks (dense + sparse)...")
        event_log.emit("indexing", "started", {"stage": "base"})
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
        event_log.emit("indexing", "completed", {"stage": "base"})

        print("\n[2/6] Building the Contextual Retrieval index (1 LLM call per chunk)...")
        event_log.emit("indexing", "started", {"stage": "contextual"})
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
        event_log.emit("indexing", "completed", {"stage": "contextual"})

        print("\n[3/6] Building the SAC index (1 LLM call per document)...")
        event_log.emit("indexing", "started", {"stage": "sac"})
        document_summarizer = GeminiDocumentSummarizer(
            _DOCUMENT_SUMMARIZER_MODEL, cache=cache, max_in_flight=gemini_max_in_flight
        )
        document_summaries = _summarize_documents(
            documents, document_versions, document_summarizer, answer_quality_workers
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
        event_log.emit("indexing", "completed", {"stage": "sac"})

        print(
            "\n[4/6] Building the SAC+Contextual index "
            "(document summary + per-chunk context, no extra LLM calls)..."
        )
        event_log.emit("indexing", "started", {"stage": "sac_contextual"})
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
        event_log.emit("indexing", "completed", {"stage": "sac_contextual"})

        print("\n[5/6] Building the RAPTOR tree (1 LLM call per group, per level, per document)...")
        event_log.emit("indexing", "started", {"stage": "raptor"})
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
        event_log.emit("indexing", "completed", {"stage": "raptor"})

        print(
            f"\n[6/6] Building the GraphRAG (LightRAG, mode={_GRAPHRAG_MODE}) index "
            "(multiple LLM calls per chunk)..."
        )
        event_log.emit("indexing", "started", {"stage": "graphrag"})
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
            event_log.emit("indexing", "completed", {"stage": "graphrag"})
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

    results_table = format_results_table(config["strategies"], run_metrics)
    answer_quality_table = format_answer_quality_table(config["strategies"], run_metrics)
    print(f"\n{results_table}")
    print(f"\n{answer_quality_table}")

    _checkpoint()
    print(f"\nRun record written to {run_dir.relative_to(ROOT)}/results.json")

    print("\nFinalizing ADR-0017 evidence directory...")
    write_summaries(
        artifacts_dir, run_metrics, generation_lineage_by_strategy, audit_results_by_strategy
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

    artifact_checksums = compute_checksums(artifacts_dir)
    write_checksums_file(artifacts_dir)
    final_manifest = finalize_manifest(run_manifest, canonical_json_hash(artifact_checksums))
    write_atomic(
        artifacts_dir / "manifest.json",
        json.dumps(dataclasses.asdict(final_manifest), ensure_ascii=False, indent=2),
    )
    print(f"Evidence directory finalized at {artifacts_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
