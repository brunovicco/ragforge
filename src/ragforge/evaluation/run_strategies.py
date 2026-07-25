"""Composition-root helpers for building strategies/collaborators for run.py (ADR-0004).

Split out of run.py (which grew past 1200 lines): document loading/chunking,
embedder/judge provider selection, base-index strategy wiring, and the
per-strategy retrieval+answer-quality scoring glue (``_evaluate``). run.py
keeps only CLI parsing, the indexing/evidence orchestration in ``main()``,
and the pure reporting functions (run_reporting.py).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ragforge.adapters.llm_cache import LLMCache
from ragforge.domain.models import Chunk, Judgment
from ragforge.domain.protocols import RetrievalStrategy
from ragforge.embeddings.google_gemini_embedder import GoogleGeminiEmbedder
from ragforge.embeddings.identity import NO_QUERY_INSTRUCTION_HASH, EmbeddingIdentity
from ragforge.embeddings.ports import EmbeddingModel
from ragforge.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from ragforge.evaluation.answer_harness import evaluate_answer_quality
from ragforge.evaluation.harness import evaluate_strategy
from ragforge.evaluation.judge_ports import AnswerQualityJudge
from ragforge.evaluation.lineage_ports import RetrievalCandidateLineage
from ragforge.evaluation.manifest import CorpusManifest
from ragforge.evaluation.ragas_judge import build_gemini_ragas_judge, build_openai_ragas_judge
from ragforge.evaluation.records import QuestionRecord, merge_question_records
from ragforge.evaluation.scheduler import run_bounded
from ragforge.generation.ports import AnswerGenerator
from ragforge.ingestion.html_extractor import HtmlTextExtractor
from ragforge.ingestion.pipeline import ingest_norm
from ragforge.ingestion.pymupdf_extractor import PyMuPdfExtractor
from ragforge.retrieval.dense.ports import ChunkSearchStore
from ragforge.retrieval.dense.strategy import DenseRetrieval
from ragforge.retrieval.hierarchical.ports import ChunkFetchStore
from ragforge.retrieval.hierarchical.strategy import ParentChildRetrieval
from ragforge.retrieval.hybrid.strategy import HybridRetrieval
from ragforge.retrieval.reranked.ports import Reranker
from ragforge.retrieval.reranked.strategy import RerankedRetrieval
from ragforge.retrieval.sac.ports import DocumentSummarizer
from ragforge.retrieval.sparse.ports import TextSearchStore
from ragforge.retrieval.sparse.strategy import SparseRetrieval

ROOT = Path(__file__).resolve().parents[3]

_EMBEDDING_PROVIDERS = ("local", "gemini")
_JUDGE_PROVIDERS = ("openai", "gemini")

# Composition-root mapping from the manifest's `extractor` key to the actual
# extraction callable - kept here, not in evaluation.manifest, so that module
# never needs to import an ingestion adapter.
_EXTRACTORS: dict[str, Callable[[Path], str]] = {
    "html": HtmlTextExtractor().extract,
    "pymupdf": PyMuPdfExtractor().extract,
}


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


def _build_judge_factory(
    provider: str,
    model: str,
    embedding_model: str,
    reasoning_effort: str,
    cache: LLMCache | None,
    max_in_flight: int,
) -> Callable[[], AnswerQualityJudge]:
    """Return a zero-arg factory building the configured judge (ADR-0018).

    ``provider: openai`` is canonical for publishable results, independent
    from the Gemini answer generator. ``provider: gemini`` is a development
    fallback - callers should label a run using it (main() does, via
    "judge_label": "exploratory_same_provider_judge") since the answer
    generator is also Gemini-based. Never a silent fallback between the two.

    A factory, not an already-built judge: evaluate_answer_quality calls
    this once per worker thread (RagasJudge's underlying sync wrapper isn't
    safe to share across threads - see answer_harness.py).

    Raises:
        SystemExit: If ``provider`` isn't one of "openai"/"gemini".
    """
    if provider == "openai":
        return lambda: build_openai_ragas_judge(
            model,
            embedding_model,
            reasoning_effort=reasoning_effort,
            cache=cache,
            max_in_flight=max_in_flight,
        )
    if provider == "gemini":
        return lambda: build_gemini_ragas_judge(
            model, embedding_model, cache=cache, max_in_flight=max_in_flight
        )
    raise SystemExit(f"unknown judge provider {provider!r}; expected one of {_JUDGE_PROVIDERS}")


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


def _evaluate(
    strategy: RetrievalStrategy,
    judgments: list[Judgment],
    generator: AnswerGenerator,
    judge_factory: Callable[[], AnswerQualityJudge],
    top_k: int,
    answer_quality_workers: int,
    embedding_identity_hash: str | None = None,
) -> tuple[dict[str, float], list[QuestionRecord], list[RetrievalCandidateLineage]]:
    """Score ``strategy`` for retrieval ranking and answer quality alike (ADR-0002/0007).

    Merges evaluate_strategy's and evaluate_answer_quality's per-question
    records into one QuestionRecord per judgment (ADR-0012), alongside the
    same aggregate metrics dict this returned before. ``embedding_identity_hash``
    (ADR-0017), when given, is forwarded to evaluate_strategy to populate
    per-candidate retrieval lineage.
    """
    retrieval_result = evaluate_strategy(
        strategy, judgments, k=top_k, embedding_identity_hash=embedding_identity_hash
    )
    answer_result = evaluate_answer_quality(
        strategy,
        judgments,
        generator,
        judge_factory,
        k=top_k,
        max_workers=answer_quality_workers,
    )
    records = merge_question_records(strategy.name, retrieval_result.records, answer_result.records)
    metrics = {**retrieval_result.metrics, **answer_result.metrics}
    return metrics, records, retrieval_result.candidate_lineage
