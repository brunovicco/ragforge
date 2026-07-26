"""Application contracts for published benchmark results."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class BenchmarkRunNotFoundError(LookupError):
    """Raised when a requested run is not present in the publication catalog."""


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """Describe the deterministic question sample used by a benchmark run."""

    kind: str
    question_count: int
    source_split_question_count: int
    sampling_seed: str


@dataclass(frozen=True, slots=True)
class BenchmarkRecommendation:
    """Record the qualified strategy recommendation published for a run."""

    strategy: str
    rationale: str
    scope: str


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    """Expose the comparable aggregate metrics for one retrieval strategy."""

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


@dataclass(frozen=True, slots=True)
class PublishedBenchmarkRun:
    """Represent one immutable benchmark run exposed to read-only consumers."""

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
    sample: BenchmarkSample
    recommendation: BenchmarkRecommendation
    metrics: tuple[StrategyMetrics, ...]
    evidence_path: str
    results_path: str


class PublishedBenchmarkRepository(Protocol):
    """Load only benchmark runs explicitly approved for publication."""

    def list_runs(self) -> tuple[PublishedBenchmarkRun, ...]:
        """Return published runs in newest-first order."""
        ...

    def get_run(self, run_id: str) -> PublishedBenchmarkRun:
        """Return one published run or raise when it is not cataloged."""
        ...
