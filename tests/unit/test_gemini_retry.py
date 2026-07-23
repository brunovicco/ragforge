"""Tests for the shared Gemini retry helper (no network)."""

import asyncio
import time

import httpx
import pytest
from google.genai import errors

from ragforge.adapters.gemini_retry import call_with_retry, call_with_retry_async, is_retryable


def _client_error(code: int) -> errors.ClientError:
    return errors.ClientError(code, {"error": {"message": "test"}})


def test_is_retryable_true_for_429_client_error() -> None:
    """A 429 ClientError is retryable."""
    assert is_retryable(_client_error(429)) is True


def test_is_retryable_false_for_non_429_client_error() -> None:
    """A non-429 ClientError (e.g. a bad request) is not retryable."""
    assert is_retryable(_client_error(400)) is False


def test_is_retryable_true_for_transport_errors() -> None:
    """Transient transport-level failures (dropped connections, timeouts) are retryable.

    This is the real failure this module was added to cover: a
    `Server disconnected without sending a response` mid-run of the full
    benchmark (ADR-0004), which a 429-only retry did not catch.
    """
    assert is_retryable(httpx.RemoteProtocolError("disconnected")) is True
    assert is_retryable(httpx.ConnectError("connect failed")) is True
    assert is_retryable(httpx.ReadTimeout("timed out")) is True


def test_is_retryable_false_for_unrelated_exceptions() -> None:
    """A generic exception (a real bug, not a transient failure) is not retryable."""
    assert is_retryable(RuntimeError("boom")) is False
    assert is_retryable(ValueError("bad input")) is False


def test_call_with_retry_returns_the_result_on_first_success() -> None:
    """No retry happens when the call succeeds immediately."""
    result = call_with_retry(lambda: "ok")

    assert result == "ok"


def test_call_with_retry_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single 429 is retried and the call succeeds without raising."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    attempts = {"n": 0}

    def call() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _client_error(429)
        return "ok"

    result = call_with_retry(call)

    assert result == "ok"
    assert attempts["n"] == 2


def test_call_with_retry_retries_on_transport_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient transport error is retried, matching the real benchmark failure."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    attempts = {"n": 0}

    def call() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.RemoteProtocolError("disconnected")
        return "ok"

    result = call_with_retry(call)

    assert result == "ok"
    assert attempts["n"] == 2


def test_call_with_retry_raises_immediately_on_a_non_retryable_error() -> None:
    """A non-retryable error (e.g. a 400) is not retried."""
    attempts = {"n": 0}

    def call() -> str:
        attempts["n"] += 1
        raise _client_error(400)

    with pytest.raises(errors.ClientError):
        call_with_retry(call)

    assert attempts["n"] == 1


def test_call_with_retry_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent retryable failures exhaust every retry and re-raise the last one."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def call() -> str:
        raise _client_error(429)

    with pytest.raises(errors.ClientError):
        call_with_retry(call)


def test_call_with_retry_async_retries_on_transport_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async variant retries a transient transport error the same way."""

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    attempts = {"n": 0}

    async def call() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.RemoteProtocolError("disconnected")
        return "ok"

    result = asyncio.run(call_with_retry_async(call))

    assert result == "ok"
    assert attempts["n"] == 2


def test_call_with_retry_async_raises_immediately_on_a_non_retryable_error() -> None:
    """A non-retryable error is not retried in the async variant either."""

    async def call() -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(call_with_retry_async(call))
