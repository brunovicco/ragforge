"""Tests for the pure wiring/formatting logic in the benchmark runner (ADR-0004).

The runner's main() orchestrates real infra (Postgres, OpenSearch, Gemini,
LightRAG) end-to-end and is exercised manually via `make bench-live`, not
here. These tests cover the parts of run.py that don't need real infra:
mode validation, strategy wiring from already-indexed (fake) stores, and the
pure formatting/record-building functions.
"""

import pytest

from ragforge.domain.models import (
    Answer,
    Chunk,
    JudgedRef,
    Judgment,
    Query,
    RelevanceGrade,
    RetrievalResult,
    StructuralRef,
)
from ragforge.embeddings.identity import NO_QUERY_INSTRUCTION_HASH, EmbeddingIdentity
from ragforge.evaluation import run
from ragforge.evaluation.judge_ports import (
    AbstentionJudgment,
    JudgeResult,
    JudgeSample,
    MetricScore,
    ModelIdentity,
)
from ragforge.evaluation.manifest import load_corpus_manifest
from ragforge.evaluation.run import (
    MANIFEST_PATH,
    _evaluate,
    _load_documents,
    build_base_strategies,
    build_contextual_strategy,
    build_run_record,
    format_answer_quality_table,
    format_results_table,
)
from ragforge.retrieval.reranked.strategy import RerankedRetrieval


def test_reject_cache_mode_raises_for_cache() -> None:
    """--mode cache fails fast with an explanation, not a silent no-op."""
    import pytest

    from ragforge.evaluation.run import _reject_cache_mode

    with pytest.raises(SystemExit, match="cache"):
        _reject_cache_mode("cache")


def test_reject_cache_mode_allows_live() -> None:
    """--mode live passes through without raising."""
    from ragforge.evaluation.run import _reject_cache_mode

    _reject_cache_mode("live")  # must not raise


class _FakeEmbedder:
    name = "fake-embedder"
    dimensions = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class _FakeDenseStore:
    def __init__(self, chunks: dict[str, Chunk]) -> None:
        self._chunks = chunks

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(chunk=chunk, score=1.0, strategy="dense")
            for chunk in list(self._chunks.values())[:top_k]
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)


class _FakeSparseStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def search(self, query_text: str, top_k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(chunk=chunk, score=1.0, strategy="sparse")
            for chunk in self._chunks[:top_k]
        ]


class _FakeReranker:
    name = "fake-reranker"

    def score(self, query: Query, chunks: list[Chunk]) -> list[float]:
        return [1.0 for _ in chunks]


def test_build_base_strategies_returns_all_five_labels() -> None:
    """Every base-index strategy is present, keyed by its benchmark-v01.yaml label."""
    chunk = Chunk(
        chunk_id="c1", source_text="Art. 1º", retrieval_text="Art. 1º", structural_ids=("s1",)
    )
    dense_store = _FakeDenseStore({"c1": chunk})
    sparse_store = _FakeSparseStore([chunk])

    strategies = build_base_strategies(
        dense_store, sparse_store, _FakeEmbedder(), _FakeReranker(), rerank_pool=50
    )

    assert set(strategies) == {"dense", "sparse_bm25", "hybrid_rrf", "reranked", "parent_child"}


def test_build_base_strategies_wires_each_strategy_to_retrieve_correctly() -> None:
    """Each returned strategy actually retrieves through its wired collaborators."""
    chunk = Chunk(
        chunk_id="c1", source_text="Art. 1º", retrieval_text="Art. 1º", structural_ids=("s1",)
    )
    dense_store = _FakeDenseStore({"c1": chunk})
    sparse_store = _FakeSparseStore([chunk])

    strategies = build_base_strategies(
        dense_store, sparse_store, _FakeEmbedder(), _FakeReranker(), rerank_pool=50
    )

    query = Query(text="pergunta")
    for strategy in strategies.values():
        results = strategy.retrieve(query, top_k=5)
        assert results
        assert results[0].chunk.chunk_id == "c1"


def test_build_base_strategies_passes_rerank_pool_through() -> None:
    """The reranked strategy pools from the configured rerank_pool size."""
    chunk = Chunk(
        chunk_id="c1", source_text="Art. 1º", retrieval_text="Art. 1º", structural_ids=("s1",)
    )
    dense_store = _FakeDenseStore({"c1": chunk})
    sparse_store = _FakeSparseStore([chunk])

    strategies = build_base_strategies(
        dense_store, sparse_store, _FakeEmbedder(), _FakeReranker(), rerank_pool=7
    )

    reranked = strategies["reranked"]
    assert isinstance(reranked, RerankedRetrieval)
    assert reranked._pool_size == 7


def test_build_contextual_strategy_retrieves_through_hybrid() -> None:
    """The contextual strategy is a working Hybrid retrieval over the given stores."""
    chunk = Chunk(
        chunk_id="c1",
        source_text="Contexto: Art. 1º",
        retrieval_text="Contexto: Art. 1º",
        structural_ids=("s1",),
    )
    dense_store = _FakeDenseStore({"c1": chunk})
    sparse_store = _FakeSparseStore([chunk])

    strategy = build_contextual_strategy(dense_store, sparse_store, _FakeEmbedder())

    results = strategy.retrieve(Query(text="pergunta"), top_k=5)
    assert results[0].chunk.chunk_id == "c1"


def test_format_results_table_includes_every_configured_strategy() -> None:
    """Every strategy label from the config appears in the table, in order."""
    metrics = {
        "dense": {
            "recall_at_k": 0.8,
            "precision_at_k": 0.2,
            "ndcg_at_k": 0.7,
            "mrr": 0.6,
            "drm_at_k": 0.1,
            "n": 5,
            "errors": 0,
        },
    }

    table = format_results_table(["dense", "sparse_bm25"], metrics)

    lines = table.splitlines()
    assert "dense" in lines[1]
    assert "0.800" in lines[1]
    assert "(not run)" in lines[2]


def test_build_run_record_includes_all_run_metadata() -> None:
    """The run record carries the run id, mode, embedding identity/generation/judge, and metrics."""
    metrics = {"dense": {"recall_at_k": 0.8, "n": 5.0}}
    identity = EmbeddingIdentity(
        provider="local",
        model="Qwen/Qwen3-Embedding-0.6B",
        revision="main",
        dimensions=1024,
        normalize=True,
        query_instruction_hash=NO_QUERY_INSTRUCTION_HASH,
        runtime="local",
    )

    record = build_run_record(
        run_id="20260101T000000Z",
        mode="live",
        config_path="configs/experiments/benchmark-v01.yaml",
        embedding_identity=identity,
        index_namespace="abc123",
        generation_model="gemini-3.1-flash-lite",
        judge_provider="openai",
        judge_model="gpt-5.4-mini-2026-03-17",
        judge_reasoning_effort="medium",
        audit_enabled=False,
        audit_provider=None,
        audit_model=None,
        corpus_version="0.2",
        split_dataset_version="0.2",
        n_chunks=465,
        top_k=5,
        run_metrics=metrics,
    )

    assert record["run_id"] == "20260101T000000Z"
    assert record["mode"] == "live"
    assert record["embedding"] == {
        "provider": "local",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "main",
        "dimensions": 1024,
        "normalize": True,
        "runtime": "local",
    }
    assert record["index_namespace"] == "abc123"
    assert record["generation_model"] == "gemini-3.1-flash-lite"
    assert record["judge_provider"] == "openai"
    assert record["judge_model"] == "gpt-5.4-mini-2026-03-17"
    assert record["judge_reasoning_effort"] == "medium"
    assert record["judge_label"] is None
    assert record["audit_enabled"] is False
    assert record["audit_provider"] is None
    assert record["audit_model"] is None
    assert record["corpus_version"] == "0.2"
    assert record["split_dataset_version"] == "0.2"
    assert record["n_chunks"] == 465
    assert record["metrics"] == metrics
    assert record["records_path"] == "records.jsonl"


def test_build_run_record_labels_a_gemini_judge_as_exploratory() -> None:
    """A judge_provider of 'gemini' is labeled exploratory_same_provider_judge (ADR-0018).

    The answer generator is always Gemini-based, so a Gemini judge is never
    independent from it.
    """
    identity = EmbeddingIdentity(
        provider="local",
        model="Qwen/Qwen3-Embedding-0.6B",
        revision="main",
        dimensions=1024,
        normalize=True,
        query_instruction_hash=NO_QUERY_INSTRUCTION_HASH,
        runtime="local",
    )

    record = build_run_record(
        run_id="20260101T000000Z",
        mode="live",
        config_path="configs/experiments/benchmark-v01.yaml",
        embedding_identity=identity,
        index_namespace="abc123",
        generation_model="gemini-3.1-flash-lite",
        judge_provider="gemini",
        judge_model="gemini-3.1-flash-lite",
        judge_reasoning_effort=None,
        audit_enabled=False,
        audit_provider=None,
        audit_model=None,
        corpus_version="0.2",
        split_dataset_version="0.2",
        n_chunks=465,
        top_k=5,
        run_metrics={},
    )

    assert record["judge_label"] == "exploratory_same_provider_judge"


def test_build_run_record_states_audit_identity_when_enabled() -> None:
    """audit_enabled/provider/model are always present, populated when auditing is on (ADR-0016)."""
    identity = EmbeddingIdentity(
        provider="local",
        model="Qwen/Qwen3-Embedding-0.6B",
        revision="main",
        dimensions=1024,
        normalize=True,
        query_instruction_hash=NO_QUERY_INSTRUCTION_HASH,
        runtime="local",
    )

    record = build_run_record(
        run_id="20260101T000000Z",
        mode="live",
        config_path="configs/experiments/benchmark-v01.yaml",
        embedding_identity=identity,
        index_namespace="abc123",
        generation_model="gemini-3.1-flash-lite",
        judge_provider="openai",
        judge_model="gpt-5.4-mini-2026-03-17",
        judge_reasoning_effort="medium",
        audit_enabled=True,
        audit_provider="openai",
        audit_model="gpt-5.4-mini-2026-03-17",
        corpus_version="0.2",
        split_dataset_version="0.2",
        n_chunks=465,
        top_k=5,
        run_metrics={},
    )

    assert record["audit_enabled"] is True
    assert record["audit_provider"] == "openai"
    assert record["audit_model"] == "gpt-5.4-mini-2026-03-17"


def test_format_answer_quality_table_includes_every_configured_strategy() -> None:
    """Every strategy label from the config appears in the answer-quality table, in order."""
    metrics = {
        "dense": {
            "citation_accuracy": 0.9,
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "abstention_appropriate": 0.6,
            "answer_n": 5,
            "answer_errors": 0,
        },
    }

    table = format_answer_quality_table(["dense", "sparse_bm25"], metrics)

    lines = table.splitlines()
    assert "dense" in lines[1]
    assert "0.900" in lines[1]
    assert "(not run)" in lines[2]


def test_load_documents_extracts_and_chunks_every_enabled_manifest_document() -> None:
    """_load_documents discovers documents only from the manifest, not a hard-coded constant."""
    manifest = load_corpus_manifest(MANIFEST_PATH)

    documents = _load_documents(manifest)

    assert set(documents) == {doc.norm_id for doc in manifest.enabled_documents}
    for norm_id, (text, chunks) in documents.items():
        assert text
        assert chunks
        assert all(chunk.structural_ids[0].startswith(norm_id) for chunk in chunks)


ART_1 = "NORM/2000::art-1"


class _FakeStrategyForEvaluate:
    name = "fake-strategy"

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id=ART_1,
                    source_text="chunk text",
                    retrieval_text="chunk text",
                    structural_ids=(ART_1,),
                ),
                score=1.0,
                strategy="fake-strategy",
            )
        ]


class _FakeGeneratorForEvaluate:
    name = "fake-generator"

    def generate(self, query: Query, results: list[RetrievalResult]) -> Answer:
        return Answer(text="answer", citations=(ART_1,))


class _FakeJudgeForEvaluate:
    identity = ModelIdentity(
        provider="fake", model="fake-model", reasoning_effort=None, output_schema_version=1
    )

    def evaluate(self, sample: JudgeSample) -> JudgeResult:
        return JudgeResult(
            schema_version=1,
            faithfulness=MetricScore(score=1.0),
            answer_relevancy=MetricScore(score=1.0),
            abstention=AbstentionJudgment(appropriate=True, rationale=""),
        )


def test_evaluate_merges_retrieval_and_answer_records_tagged_with_the_strategy_name() -> None:
    """_evaluate returns metrics from both harnesses plus one merged record per question."""
    judgments = [
        Judgment(
            question_id="q1",
            query=Query(text="q1"),
            relevant_refs=(
                JudgedRef(ref=StructuralRef.parse(ART_1), grade=RelevanceGrade.RELEVANT),
            ),
        )
    ]

    metrics, records, candidate_lineage = _evaluate(
        _FakeStrategyForEvaluate(),
        judgments,
        _FakeGeneratorForEvaluate(),
        lambda: _FakeJudgeForEvaluate(),
        top_k=5,
        answer_quality_workers=1,
    )

    assert metrics["recall_at_k"] == 1.0
    assert metrics["citation_accuracy"] == 1.0
    assert len(records) == 1
    assert records[0].strategy == "fake-strategy"
    assert records[0].retrieval_status == "succeeded"
    assert records[0].generation_status == "succeeded"
    assert candidate_lineage == [], "no embedding_identity_hash was passed, so no lineage collected"


def test_evaluate_populates_candidate_lineage_when_embedding_identity_hash_is_given() -> None:
    """_evaluate forwards embedding_identity_hash through to evaluate_strategy (ADR-0017)."""
    judgments = [
        Judgment(
            question_id="q1",
            query=Query(text="q1"),
            relevant_refs=(
                JudgedRef(ref=StructuralRef.parse(ART_1), grade=RelevanceGrade.RELEVANT),
            ),
        )
    ]

    _, _, candidate_lineage = _evaluate(
        _FakeStrategyForEvaluate(),
        judgments,
        _FakeGeneratorForEvaluate(),
        lambda: _FakeJudgeForEvaluate(),
        top_k=5,
        answer_quality_workers=1,
        embedding_identity_hash="embhash123",
    )

    assert len(candidate_lineage) == 1
    assert candidate_lineage[0].query_id == "q1"
    assert candidate_lineage[0].embedding_identity_hash == "embhash123"
    assert candidate_lineage[0].chunk_id == ART_1


class _FakeSentenceTransformerEmbedder:
    """Stands in for SentenceTransformerEmbedder - no real model load."""

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self.dimensions = 1024
        self.revision = "main"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimensions for _ in texts]


class _FakeGoogleGeminiEmbedder:
    """Stands in for GoogleGeminiEmbedder - no real API key/network call."""

    def __init__(
        self,
        model_name: str,
        output_dimensionality: int | None = None,
        cache: object = None,
        max_in_flight: int = 4,
    ) -> None:
        self.name = model_name
        self.dimensions = output_dimensionality or 3072
        self.cache = cache
        self.max_in_flight = max_in_flight

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimensions for _ in texts]


def test_build_embedder_constructs_a_local_embedder_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """provider: local builds SentenceTransformerEmbedder and a matching identity."""
    monkeypatch.setattr(run, "SentenceTransformerEmbedder", _FakeSentenceTransformerEmbedder)

    embedder, identity = run._build_embedder("local", "Qwen/Qwen3-Embedding-0.6B", None, None, 4)

    assert embedder.dimensions == 1024
    assert identity.provider == "local"
    assert identity.model == "Qwen/Qwen3-Embedding-0.6B"
    assert identity.revision == "main"
    assert identity.dimensions == 1024
    assert identity.normalize is True
    assert identity.runtime == "local"


def test_build_embedder_constructs_a_gemini_embedder_with_output_dimensionality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """provider: gemini builds GoogleGeminiEmbedder, passing dimensions through as truncation."""
    monkeypatch.setattr(run, "GoogleGeminiEmbedder", _FakeGoogleGeminiEmbedder)

    embedder, identity = run._build_embedder("gemini", "gemini-embedding-001", 1536, None, 4)

    assert embedder.dimensions == 1536
    assert identity.provider == "gemini"
    assert identity.revision == "api"
    assert identity.normalize is False
    assert identity.runtime == "hosted"


def test_build_embedder_fails_closed_for_an_unknown_provider() -> None:
    """An unrecognized provider fails fast rather than silently falling back to either adapter."""
    with pytest.raises(SystemExit, match="unknown embedding provider"):
        run._build_embedder("openai", "text-embedding-3", None, None, 4)


def test_build_judge_factory_constructs_an_openai_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """provider: openai builds via build_openai_ragas_judge, forwarding reasoning_effort."""
    calls: list[tuple[object, ...]] = []

    def fake_build_openai_ragas_judge(
        model: str,
        embedding_model: str,
        reasoning_effort: str = "medium",
        api_key: str | None = None,
        cache: object = None,
        max_in_flight: int = 4,
    ) -> str:
        calls.append((model, embedding_model, reasoning_effort, cache, max_in_flight))
        return "fake-openai-judge"

    monkeypatch.setattr(run, "build_openai_ragas_judge", fake_build_openai_ragas_judge)

    factory = run._build_judge_factory(
        "openai", "gpt-5.4-mini-2026-03-17", "text-embedding-3-small", "medium", None, 4
    )
    factory()

    assert calls == [("gpt-5.4-mini-2026-03-17", "text-embedding-3-small", "medium", None, 4)]


def test_build_judge_factory_constructs_a_gemini_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """provider: gemini builds via build_gemini_ragas_judge, ignoring reasoning_effort."""
    calls: list[tuple[object, ...]] = []

    def fake_build_gemini_ragas_judge(
        model: str,
        embedding_model: str,
        api_key: str | None = None,
        cache: object = None,
        max_in_flight: int = 4,
    ) -> str:
        calls.append((model, embedding_model, cache, max_in_flight))
        return "fake-gemini-judge"

    monkeypatch.setattr(run, "build_gemini_ragas_judge", fake_build_gemini_ragas_judge)

    factory = run._build_judge_factory(
        "gemini", "gemini-3.1-flash-lite", "gemini-embedding-001", "medium", None, 4
    )
    factory()

    assert calls == [("gemini-3.1-flash-lite", "gemini-embedding-001", None, 4)]


def test_build_judge_factory_fails_closed_for_an_unknown_provider() -> None:
    """An unrecognized provider fails fast rather than silently falling back to either adapter."""
    with pytest.raises(SystemExit, match="unknown judge provider"):
        run._build_judge_factory("anthropic", "claude", "embed", "medium", None, 4)


def test_verify_resume_identity_passes_when_everything_matches() -> None:
    """Identical index_namespace/generation_model/judge identity resumes cleanly."""
    previous = {
        "index_namespace": "abc123",
        "generation_model": "gemini-3.1-flash-lite",
        "judge_provider": "openai",
        "judge_model": "gpt-5.4-mini-2026-03-17",
    }

    run._verify_resume_identity(
        previous, "abc123", "gemini-3.1-flash-lite", "openai", "gpt-5.4-mini-2026-03-17"
    )


def test_verify_resume_identity_fails_closed_on_index_namespace_mismatch() -> None:
    """A different index_namespace (corpus/chunking/embedding changed) refuses to resume."""
    previous = {
        "index_namespace": "abc123",
        "generation_model": "gemini-3.1-flash-lite",
        "judge_provider": "openai",
        "judge_model": "gpt-5.4-mini-2026-03-17",
    }

    with pytest.raises(SystemExit, match="index_namespace"):
        run._verify_resume_identity(
            previous, "xyz789", "gemini-3.1-flash-lite", "openai", "gpt-5.4-mini-2026-03-17"
        )


def test_verify_resume_identity_fails_closed_on_generation_model_mismatch() -> None:
    """A different generation_model refuses to resume - cached answers wouldn't be trustworthy."""
    previous = {
        "index_namespace": "abc123",
        "generation_model": "gemini-3.1-flash-lite",
        "judge_provider": "openai",
        "judge_model": "gpt-5.4-mini-2026-03-17",
    }

    with pytest.raises(SystemExit, match="generation_model"):
        run._verify_resume_identity(
            previous, "abc123", "gemini-3.2-pro", "openai", "gpt-5.4-mini-2026-03-17"
        )


def test_verify_resume_identity_fails_closed_on_judge_provider_mismatch() -> None:
    """A different judge_provider refuses to resume - cached judge scores wouldn't be trustworthy.

    ADR-0018: switching provider (e.g. gemini -> openai) is a change of
    judge identity even when a model name happened to collide.
    """
    previous = {
        "index_namespace": "abc123",
        "generation_model": "gemini-3.1-flash-lite",
        "judge_provider": "gemini",
        "judge_model": "gemini-3.1-flash-lite",
    }

    with pytest.raises(SystemExit, match="judge_provider"):
        run._verify_resume_identity(
            previous, "abc123", "gemini-3.1-flash-lite", "openai", "gemini-3.1-flash-lite"
        )
