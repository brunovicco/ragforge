import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ragforge.application.benchmark_dashboard import strategy_metric_rows
from ragforge.application.benchmark_results import (
    BenchmarkRecommendation,
    BenchmarkSample,
    PublishedBenchmarkRun,
    StrategyMetrics,
)
from ragforge.entrypoints.dashboard_content import (
    Language,
    dashboard_copy,
    metric_explanations,
    technique_explanations,
)

_EXPECTED_STRATEGIES = {
    "dense",
    "sparse_bm25",
    "hybrid_rrf",
    "reranked",
    "contextual",
    "parent_child",
    "sac",
    "sac_contextual",
    "raptor",
    "graphrag",
}
_EXPECTED_METRICS = {
    "recall_at_5",
    "precision_at_5",
    "ndcg_at_5",
    "mrr",
    "document_mismatch_at_5",
    "citation_accuracy",
    "faithfulness",
    "answer_relevancy",
    "abstention",
}


def _metric(strategy: str, *, ndcg: float, mrr: float) -> StrategyMetrics:
    return StrategyMetrics(
        strategy=strategy,
        recall_at_k=0.9,
        precision_at_k=0.3,
        ndcg_at_k=ndcg,
        mrr=mrr,
        drm_at_k=0.0,
        citation_accuracy=0.7,
        faithfulness=0.9,
        answer_relevancy=0.8,
        abstention_appropriate=1.0,
        retrieval_question_count=57,
        answer_question_count=60,
        errors=0,
    )


def test_orders_dashboard_rows_by_ndcg_then_mrr() -> None:
    run = PublishedBenchmarkRun(
        run_id="20260726T185553Z",
        title="Published sample",
        completed_at=datetime(2026, 7, 26, 19, 59, 59, tzinfo=UTC),
        git_sha="a" * 40,
        mode="live",
        config_path="benchmark.yaml",
        embedding_model="embedding",
        generation_model="generation",
        judge_model="judge",
        k=5,
        chunk_count=735,
        sample=BenchmarkSample(
            kind="deterministic_stratified_sample",
            question_count=60,
            source_split_question_count=194,
            sampling_seed="seed",
        ),
        recommendation=BenchmarkRecommendation(
            strategy="sac",
            rationale="Best balanced result.",
            scope="Sample only.",
        ),
        metrics=(
            _metric("dense", ndcg=0.95, mrr=0.97),
            _metric("sac", ndcg=0.96, mrr=0.99),
            _metric("contextual", ndcg=0.95, mrr=0.98),
        ),
        evidence_path="artifacts/runs/id",
        results_path="experiments/id/results.json",
    )

    rows = strategy_metric_rows(run)

    assert [row.strategy for row in rows] == ["sac", "contextual", "dense"]


def test_dashboard_wrapper_imports_from_streamlit_script_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = repository_root / "apps" / "dashboard" / "main.py"
    root_resolved = repository_root.resolve()
    isolated_path = [item for item in sys.path if Path(item or ".").resolve() != root_resolved]
    monkeypatch.setattr(sys, "path", [str(script.parent), *isolated_path])
    spec = importlib.util.spec_from_file_location("streamlit_dashboard_entrypoint", script)

    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@pytest.mark.parametrize("language", ["pt", "en"])
def test_dashboard_explains_every_technique_and_metric(
    language: Language,
) -> None:
    techniques = technique_explanations(language)
    metrics = metric_explanations(language)

    assert set(techniques) == _EXPECTED_STRATEGIES
    assert set(metrics) == _EXPECTED_METRICS
    assert all(explanation.summary for explanation in techniques.values())
    assert all(explanation.interpretation for explanation in metrics.values())


def test_dashboard_has_distinct_portuguese_and_english_copy() -> None:
    portuguese = dashboard_copy("pt")
    english = dashboard_copy("en")

    assert portuguese.title == "Benchmark RAGForge"
    assert english.title == "RAGForge benchmark"
    assert portuguese.strategy_comparison != english.strategy_comparison
