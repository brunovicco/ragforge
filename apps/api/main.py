"""Read-only FastAPI entrypoint for published RAGForge benchmark results."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi import Path as PathParameter
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from apps.api.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    PublishedBenchmarkRunListResponse,
    PublishedBenchmarkRunResponse,
    PublishedBenchmarkRunSummaryResponse,
)
from ragforge.adapters.published_benchmarks import JsonPublishedBenchmarkRepository
from ragforge.application.benchmark_results import (
    BenchmarkRunNotFoundError,
    PublishedBenchmarkRepository,
)
from ragforge.entrypoints.logging import configure_logging

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RunId = Annotated[
    str,
    PathParameter(
        pattern=r"^[0-9]{8}T[0-9]{6}Z$",
        description="UTC benchmark run identifier",
        examples=["20260726T185553Z"],
    ),
]


def create_app(repository: PublishedBenchmarkRepository | None = None) -> FastAPI:
    """Create the HTTP application with an injectable read-only repository."""
    result_repository = repository or JsonPublishedBenchmarkRepository(_REPOSITORY_ROOT)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(
            service="ragforge-api",
            environment=os.environ.get("RAGFORGE_ENVIRONMENT", "development"),
            version=version("ragforge"),
        )
        yield

    application = FastAPI(
        title="RAGForge Benchmark API",
        version="1.0.0",
        description="Read-only API for explicitly published benchmark evidence.",
        lifespan=lifespan,
    )

    @application.exception_handler(BenchmarkRunNotFoundError)
    async def benchmark_not_found(_: Request, exception: BenchmarkRunNotFoundError) -> JSONResponse:
        response = ErrorResponse(
            error=ErrorDetail(
                code="benchmark_run_not_found",
                message=f"Published benchmark run not found: {exception.args[0]}",
            )
        )
        return JSONResponse(status_code=404, content=response.model_dump())

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
        response = ErrorResponse(
            error=ErrorDetail(
                code="invalid_request",
                message="The request does not match the API contract.",
            )
        )
        return JSONResponse(status_code=422, content=response.model_dump())

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get(
        "/api/v1/benchmark-runs",
        response_model=PublishedBenchmarkRunListResponse,
        tags=["benchmarks"],
    )
    def list_benchmark_runs() -> PublishedBenchmarkRunListResponse:
        runs = result_repository.list_runs()
        return PublishedBenchmarkRunListResponse(
            latest_run_id=runs[0].run_id,
            runs=[PublishedBenchmarkRunSummaryResponse.from_application(run) for run in runs],
        )

    @application.get(
        "/api/v1/benchmark-runs/{run_id}",
        response_model=PublishedBenchmarkRunResponse,
        responses={404: {"model": ErrorResponse}},
        tags=["benchmarks"],
    )
    def get_benchmark_run(run_id: RunId) -> PublishedBenchmarkRunResponse:
        return PublishedBenchmarkRunResponse.from_application(result_repository.get_run(run_id))

    return application


app = create_app()
