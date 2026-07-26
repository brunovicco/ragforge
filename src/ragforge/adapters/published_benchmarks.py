"""Read validated, explicitly published benchmark results from the repository."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragforge.application.benchmark_results import (
    BenchmarkRecommendation,
    BenchmarkRunNotFoundError,
    BenchmarkSample,
    PublishedBenchmarkRun,
    StrategyMetrics,
)

_RUN_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")
_MAX_JSON_BYTES = 5 * 1024 * 1024


class PublishedBenchmarkFormatError(ValueError):
    """Raised when a publication catalog or referenced result is invalid."""


class _SampleDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["deterministic_stratified_sample"]
    question_count: int = Field(gt=0)
    source_split_question_count: int = Field(gt=0)
    sampling_seed: str


class _RecommendationDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    rationale: str
    scope: str


class _CatalogEntryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    title: str
    sample: _SampleDocument
    recommendation: _RecommendationDocument


class _CatalogDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    latest_run_id: str
    runs: list[_CatalogEntryDocument]


class _EmbeddingDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str
    model: str


class _MetricDocument(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    mrr: float
    drm_at_k: float
    errors: int
    citation_accuracy: float
    faithfulness: float
    answer_relevancy: float
    abstention_appropriate: float
    n: int
    answer_n: int
    answer_errors: int


class _ResultsDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    mode: str
    config_path: str
    embedding: _EmbeddingDocument
    generation_model: str
    judge_provider: str
    judge_model: str
    k: int = Field(gt=0)
    n_chunks: int = Field(gt=0)
    metrics: dict[str, _MetricDocument] = Field(min_length=1)


class _ManifestDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    status: str
    git_sha: str
    completed_at: datetime


class JsonPublishedBenchmarkRepository:
    """Load published benchmark runs through a repository-owned catalog."""

    def __init__(self, repository_root: Path) -> None:
        """Initialize the adapter with the checked-out repository root."""
        self._root = repository_root.resolve()
        self._catalog_path = self._root / "experiments" / "published-runs.json"

    def list_runs(self) -> tuple[PublishedBenchmarkRun, ...]:
        """Return cataloged runs in newest-first order."""
        catalog = self._load_catalog()
        return tuple(
            self._load_run(entry)
            for entry in sorted(catalog.runs, key=lambda item: item.run_id, reverse=True)
        )

    def get_run(self, run_id: str) -> PublishedBenchmarkRun:
        """Return one cataloged run without accepting arbitrary filesystem paths."""
        if _RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise BenchmarkRunNotFoundError(run_id)
        catalog = self._load_catalog()
        entry = next((item for item in catalog.runs if item.run_id == run_id), None)
        if entry is None:
            raise BenchmarkRunNotFoundError(run_id)
        return self._load_run(entry)

    def latest_run(self) -> PublishedBenchmarkRun:
        """Return the catalog-designated latest run."""
        catalog = self._load_catalog()
        return self.get_run(catalog.latest_run_id)

    def _load_catalog(self) -> _CatalogDocument:
        document = self._parse_document(self._catalog_path, _CatalogDocument)
        if document.schema_version != 1:
            raise PublishedBenchmarkFormatError(
                f"unsupported published-run catalog schema: {document.schema_version}"
            )
        run_ids = [entry.run_id for entry in document.runs]
        if len(run_ids) != len(set(run_ids)):
            raise PublishedBenchmarkFormatError("published-run catalog contains duplicate run IDs")
        if document.latest_run_id not in run_ids:
            raise PublishedBenchmarkFormatError("latest_run_id is not present in the catalog")
        if any(_RUN_ID_PATTERN.fullmatch(run_id) is None for run_id in run_ids):
            raise PublishedBenchmarkFormatError("published-run catalog contains an invalid run ID")
        return document

    def _load_run(self, entry: _CatalogEntryDocument) -> PublishedBenchmarkRun:
        results_path = self._root / "experiments" / entry.run_id / "results.json"
        manifest_path = self._root / "artifacts" / "runs" / entry.run_id / "manifest.json"
        results = self._parse_document(results_path, _ResultsDocument)
        manifest = self._parse_document(manifest_path, _ManifestDocument)
        if results.run_id != entry.run_id or manifest.run_id != entry.run_id:
            raise PublishedBenchmarkFormatError(
                f"cataloged run identity does not match its files: {entry.run_id}"
            )
        if manifest.status != "completed":
            raise PublishedBenchmarkFormatError(f"cataloged run is not completed: {entry.run_id}")
        if manifest.completed_at.tzinfo is None:
            raise PublishedBenchmarkFormatError(
                f"cataloged run completion time is not timezone-aware: {entry.run_id}"
            )
        if entry.recommendation.strategy not in results.metrics:
            raise PublishedBenchmarkFormatError(
                f"recommended strategy is absent from run metrics: {entry.run_id}"
            )
        metrics = tuple(
            self._to_strategy_metrics(strategy, metric)
            for strategy, metric in results.metrics.items()
        )
        return PublishedBenchmarkRun(
            run_id=entry.run_id,
            title=entry.title,
            completed_at=manifest.completed_at,
            git_sha=manifest.git_sha,
            mode=results.mode,
            config_path=results.config_path,
            embedding_model=f"{results.embedding.provider}/{results.embedding.model}",
            generation_model=results.generation_model,
            judge_model=f"{results.judge_provider}/{results.judge_model}",
            k=results.k,
            chunk_count=results.n_chunks,
            sample=BenchmarkSample(**entry.sample.model_dump()),
            recommendation=BenchmarkRecommendation(**entry.recommendation.model_dump()),
            metrics=metrics,
            evidence_path=f"artifacts/runs/{entry.run_id}",
            results_path=f"experiments/{entry.run_id}/results.json",
        )

    def _parse_document[DocumentT: BaseModel](
        self, path: Path, document_type: type[DocumentT]
    ) -> DocumentT:
        safe_path = path.resolve()
        if not safe_path.is_relative_to(self._root):
            raise PublishedBenchmarkFormatError("published-run path escapes the repository")
        try:
            size = safe_path.stat().st_size
            if size > _MAX_JSON_BYTES:
                raise PublishedBenchmarkFormatError(
                    f"published-run JSON exceeds {_MAX_JSON_BYTES} bytes: "
                    f"{safe_path.relative_to(self._root)}"
                )
            payload = json.loads(safe_path.read_text(encoding="utf-8"))
            return document_type.model_validate(payload)
        except PublishedBenchmarkFormatError:
            raise
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            relative = safe_path.relative_to(self._root)
            raise PublishedBenchmarkFormatError(
                f"invalid published-run document: {relative}"
            ) from exc

    @staticmethod
    def _to_strategy_metrics(strategy: str, metric: _MetricDocument) -> StrategyMetrics:
        return StrategyMetrics(
            strategy=strategy,
            recall_at_k=metric.recall_at_k,
            precision_at_k=metric.precision_at_k,
            ndcg_at_k=metric.ndcg_at_k,
            mrr=metric.mrr,
            drm_at_k=metric.drm_at_k,
            citation_accuracy=metric.citation_accuracy,
            faithfulness=metric.faithfulness,
            answer_relevancy=metric.answer_relevancy,
            abstention_appropriate=metric.abstention_appropriate,
            retrieval_question_count=metric.n,
            answer_question_count=metric.answer_n,
            errors=metric.errors + metric.answer_errors,
        )
