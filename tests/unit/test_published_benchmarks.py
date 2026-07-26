import json
from pathlib import Path

import pytest

from ragforge.adapters.published_benchmarks import (
    JsonPublishedBenchmarkRepository,
    PublishedBenchmarkFormatError,
)
from ragforge.application.benchmark_results import BenchmarkRunNotFoundError

RUN_ID = "20260726T185553Z"


def _write_publication(root: Path, *, result_run_id: str = RUN_ID) -> None:
    experiments = root / "experiments"
    artifacts = root / "artifacts" / "runs" / RUN_ID
    run_dir = experiments / RUN_ID
    run_dir.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (experiments / "published-runs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "latest_run_id": RUN_ID,
                "runs": [
                    {
                        "run_id": RUN_ID,
                        "title": "Published sample",
                        "sample": {
                            "kind": "deterministic_stratified_sample",
                            "question_count": 60,
                            "source_split_question_count": 194,
                            "sampling_seed": "seed-v1",
                        },
                        "recommendation": {
                            "strategy": "sac",
                            "rationale": "Best balanced result.",
                            "scope": "Sample only.",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "run_id": result_run_id,
                "mode": "live",
                "config_path": "configs/experiments/benchmark-v01.yaml",
                "embedding": {"provider": "gemini", "model": "embedding-model"},
                "generation_model": "generation-model",
                "judge_provider": "openai",
                "judge_model": "judge-model",
                "k": 5,
                "n_chunks": 735,
                "metrics": {
                    "sac": {
                        "recall_at_k": 0.97,
                        "precision_at_k": 0.27,
                        "ndcg_at_k": 0.96,
                        "mrr": 0.99,
                        "drm_at_k": 0.0,
                        "n": 57,
                        "errors": 0,
                        "citation_accuracy": 0.69,
                        "faithfulness": 0.96,
                        "answer_relevancy": 0.84,
                        "abstention_appropriate": 0.97,
                        "answer_n": 60,
                        "answer_errors": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "status": "completed",
                "git_sha": "a" * 40,
                "completed_at": "2026-07-26T19:59:59+00:00",
            }
        ),
        encoding="utf-8",
    )


def test_loads_only_cataloged_completed_run(tmp_path: Path) -> None:
    _write_publication(tmp_path)

    repository = JsonPublishedBenchmarkRepository(tmp_path)

    run = repository.get_run(RUN_ID)

    assert run.run_id == RUN_ID
    assert run.sample.question_count == 60
    assert run.recommendation.strategy == "sac"
    assert run.metrics[0].ndcg_at_k == pytest.approx(0.96)
    assert repository.latest_run() == run


def test_rejects_run_that_is_not_cataloged(tmp_path: Path) -> None:
    _write_publication(tmp_path)
    repository = JsonPublishedBenchmarkRepository(tmp_path)

    with pytest.raises(BenchmarkRunNotFoundError):
        repository.get_run("20260726T000000Z")


def test_rejects_result_whose_identity_differs_from_catalog(tmp_path: Path) -> None:
    _write_publication(tmp_path, result_run_id="20260726T000000Z")
    repository = JsonPublishedBenchmarkRepository(tmp_path)

    with pytest.raises(PublishedBenchmarkFormatError, match="identity"):
        repository.get_run(RUN_ID)
