from datetime import UTC, datetime

from apps.api.main import create_app
from fastapi.testclient import TestClient

from ragforge.application.benchmark_results import (
    BenchmarkRecommendation,
    BenchmarkRunNotFoundError,
    BenchmarkSample,
    PublishedBenchmarkRun,
    StrategyMetrics,
)

RUN_ID = "20260726T185553Z"


def _published_run() -> PublishedBenchmarkRun:
    return PublishedBenchmarkRun(
        run_id=RUN_ID,
        title="Published sample",
        completed_at=datetime(2026, 7, 26, 19, 59, 59, tzinfo=UTC),
        git_sha="a" * 40,
        mode="live",
        config_path="configs/experiments/benchmark-v01.yaml",
        embedding_model="gemini/gemini-embedding-001",
        generation_model="gemini-3.1-flash-lite",
        judge_model="openai/gpt-5.4-mini-2026-03-17",
        k=5,
        chunk_count=735,
        sample=BenchmarkSample(
            kind="deterministic_stratified_sample",
            question_count=60,
            source_split_question_count=194,
            sampling_seed="seed-v1",
        ),
        recommendation=BenchmarkRecommendation(
            strategy="sac",
            rationale="Best balanced result.",
            scope="Sample only.",
        ),
        metrics=(
            StrategyMetrics(
                strategy="sac",
                recall_at_k=0.97,
                precision_at_k=0.27,
                ndcg_at_k=0.96,
                mrr=0.99,
                drm_at_k=0.0,
                citation_accuracy=0.69,
                faithfulness=0.96,
                answer_relevancy=0.84,
                abstention_appropriate=0.97,
                retrieval_question_count=57,
                answer_question_count=60,
                errors=0,
            ),
        ),
        evidence_path=f"artifacts/runs/{RUN_ID}",
        results_path=f"experiments/{RUN_ID}/results.json",
    )


class _FakeRepository:
    def list_runs(self) -> tuple[PublishedBenchmarkRun, ...]:
        return (_published_run(),)

    def get_run(self, run_id: str) -> PublishedBenchmarkRun:
        if run_id != RUN_ID:
            raise BenchmarkRunNotFoundError(run_id)
        return _published_run()


def test_lists_published_benchmark_runs() -> None:
    client = TestClient(create_app(_FakeRepository()))

    response = client.get("/api/v1/benchmark-runs")

    assert response.status_code == 200
    assert response.json()["latest_run_id"] == RUN_ID
    assert response.json()["runs"][0]["recommended_strategy"] == "sac"
    assert response.json()["runs"][0]["question_count"] == 60


def test_returns_published_benchmark_details() -> None:
    client = TestClient(create_app(_FakeRepository()))

    response = client.get(f"/api/v1/benchmark-runs/{RUN_ID}")

    assert response.status_code == 200
    assert response.json()["recommendation"]["strategy"] == "sac"
    assert response.json()["metrics"][0]["ndcg_at_k"] == 0.96


def test_returns_stable_error_for_unknown_benchmark_run() -> None:
    client = TestClient(create_app(_FakeRepository()))

    response = client.get("/api/v1/benchmark-runs/20260726T000000Z")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "benchmark_run_not_found",
            "message": "Published benchmark run not found: 20260726T000000Z",
        }
    }


def test_rejects_invalid_run_identifier_at_boundary() -> None:
    client = TestClient(create_app(_FakeRepository()))

    response = client.get("/api/v1/benchmark-runs/not-a-run")

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "invalid_request",
            "message": "The request does not match the API contract.",
        }
    }
