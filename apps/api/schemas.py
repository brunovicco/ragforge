"""Versioned HTTP response schemas for published benchmark results."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from ragforge.application.benchmark_results import PublishedBenchmarkRun


class BenchmarkSampleResponse(BaseModel):
    """Describe the deterministic sample behind a published run."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    question_count: int
    source_split_question_count: int
    sampling_seed: str


class BenchmarkRecommendationResponse(BaseModel):
    """Expose the qualified recommendation recorded for a published run."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    rationale: str
    scope: str


class StrategyMetricsResponse(BaseModel):
    """Expose comparable retrieval and answer metrics for one strategy."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    mrr: float
    drm_at_k: float
    citation_accuracy: float
    faithfulness: float
    answer_relevancy: float
    abstention_appropriate: float
    retrieval_question_count: int
    answer_question_count: int
    errors: int


class PublishedBenchmarkRunResponse(BaseModel):
    """Represent the detailed public contract for one immutable run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    title: str
    completed_at: datetime
    git_sha: str
    mode: str
    config_path: str
    embedding_model: str
    generation_model: str
    judge_model: str
    k: int
    chunk_count: int
    sample: BenchmarkSampleResponse
    recommendation: BenchmarkRecommendationResponse
    metrics: list[StrategyMetricsResponse]
    evidence_path: str
    results_path: str

    @classmethod
    def from_application(cls, run: PublishedBenchmarkRun) -> "PublishedBenchmarkRunResponse":
        """Translate the application read model into the HTTP contract."""
        return cls(
            run_id=run.run_id,
            title=run.title,
            completed_at=run.completed_at,
            git_sha=run.git_sha,
            mode=run.mode,
            config_path=run.config_path,
            embedding_model=run.embedding_model,
            generation_model=run.generation_model,
            judge_model=run.judge_model,
            k=run.k,
            chunk_count=run.chunk_count,
            sample=BenchmarkSampleResponse(
                kind=run.sample.kind,
                question_count=run.sample.question_count,
                source_split_question_count=run.sample.source_split_question_count,
                sampling_seed=run.sample.sampling_seed,
            ),
            recommendation=BenchmarkRecommendationResponse(
                strategy=run.recommendation.strategy,
                rationale=run.recommendation.rationale,
                scope=run.recommendation.scope,
            ),
            metrics=[
                StrategyMetricsResponse(
                    strategy=metric.strategy,
                    recall_at_k=metric.recall_at_k,
                    precision_at_k=metric.precision_at_k,
                    ndcg_at_k=metric.ndcg_at_k,
                    mrr=metric.mrr,
                    drm_at_k=metric.drm_at_k,
                    citation_accuracy=metric.citation_accuracy,
                    faithfulness=metric.faithfulness,
                    answer_relevancy=metric.answer_relevancy,
                    abstention_appropriate=metric.abstention_appropriate,
                    retrieval_question_count=metric.retrieval_question_count,
                    answer_question_count=metric.answer_question_count,
                    errors=metric.errors,
                )
                for metric in run.metrics
            ],
            evidence_path=run.evidence_path,
            results_path=run.results_path,
        )


class PublishedBenchmarkRunSummaryResponse(BaseModel):
    """Represent the compact public contract used by run listings."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    title: str
    completed_at: datetime
    question_count: int
    recommended_strategy: str
    evidence_path: str

    @classmethod
    def from_application(cls, run: PublishedBenchmarkRun) -> "PublishedBenchmarkRunSummaryResponse":
        """Translate a run into its compact listing representation."""
        return cls(
            run_id=run.run_id,
            title=run.title,
            completed_at=run.completed_at,
            question_count=run.sample.question_count,
            recommended_strategy=run.recommendation.strategy,
            evidence_path=run.evidence_path,
        )


class PublishedBenchmarkRunListResponse(BaseModel):
    """List explicitly published benchmark runs."""

    model_config = ConfigDict(extra="forbid")

    latest_run_id: str
    runs: list[PublishedBenchmarkRunSummaryResponse]


class HealthResponse(BaseModel):
    """Expose a minimal liveness response."""

    model_config = ConfigDict(extra="forbid")

    status: str


class ErrorDetail(BaseModel):
    """Represent a stable machine-readable API error."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Wrap API errors without exposing internal exceptions."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail
