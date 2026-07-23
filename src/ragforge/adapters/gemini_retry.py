"""Shared retry-with-backoff for Gemini API calls.

Every Gemini-calling adapter in this project (GoogleGeminiEmbedder,
GeminiContextualizer, GeminiSummarizer, the LightRAG glue functions in
retrieval.graph.lightrag_gemini) retries the same class of transient
failures: HTTP 429 rate limiting, and lower-level transport errors (a dropped
connection, a timeout) that the google-genai SDK's own internal retry
sometimes still surfaces - observed for real running the full benchmark
(configs/experiments/benchmark-v01.yaml): a `Server disconnected without
sending a response` mid-run, not a 429, which the pre-existing 429-only retry
in each adapter didn't cover. Centralized here so this retry policy - and any
future fix to it - doesn't have to be repeated (and can drift) across four
call sites, per AGENTS.md's bounded-exponential-backoff-with-jitter policy
for transient, repeatable operations.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

import httpx
from google.genai import errors

_RATE_LIMIT_STATUS_CODE = 429
_MAX_RETRIES = 5
_BASE_DELAY_SECONDS = 2.0
_MAX_DELAY_SECONDS = 60.0


def is_retryable(exc: Exception) -> bool:
    """Return True for a 429 rate limit or a transient transport-level failure."""
    if isinstance(exc, errors.ClientError):
        return exc.code == _RATE_LIMIT_STATUS_CODE
    return isinstance(exc, httpx.TransportError)


def _delay_for_attempt(attempt: int) -> float:
    base: float = min(_BASE_DELAY_SECONDS * (2**attempt), _MAX_DELAY_SECONDS)
    jitter: float = random.uniform(0, base * 0.1)  # noqa: S311  # nosec B311
    return base + jitter


def call_with_retry[T](call: Callable[[], T]) -> T:
    """Call ``call``, retrying only 429s/transport errors with bounded backoff.

    Raises:
        Exception: Whatever ``call`` raises, once retries are exhausted or the
            failure isn't retryable - callers translate it into their own
            error type.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return call()
        except Exception as exc:
            is_last_attempt = attempt == _MAX_RETRIES - 1
            if not is_retryable(exc) or is_last_attempt:
                raise
            time.sleep(_delay_for_attempt(attempt))
    raise AssertionError("unreachable: the loop above always returns or raises")


async def call_with_retry_async[T](call: Callable[[], Awaitable[T]]) -> T:
    """Async counterpart of ``call_with_retry``, for LightRAG's async llm_model_func."""
    for attempt in range(_MAX_RETRIES):
        try:
            return await call()
        except Exception as exc:
            is_last_attempt = attempt == _MAX_RETRIES - 1
            if not is_retryable(exc) or is_last_attempt:
                raise
            await asyncio.sleep(_delay_for_attempt(attempt))
    raise AssertionError("unreachable: the loop above always returns or raises")
