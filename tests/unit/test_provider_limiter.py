"""Tests for the per-provider concurrency limiter (ADR-0014)."""

import threading
import time

from ragforge.adapters.provider_limiter import ProviderLimiter, get_limiter


def test_limiter_bounds_concurrent_acquisitions() -> None:
    """More threads than the limit still never exceed max_in_flight concurrently."""
    limiter = ProviderLimiter(max_in_flight=3)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal active, max_active
        with limiter:
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active == 3


def test_limiter_releases_the_slot_even_if_the_guarded_call_raises() -> None:
    """A raised exception inside the `with` block still frees the slot for the next caller."""
    limiter = ProviderLimiter(max_in_flight=1)

    try:
        with limiter:
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    acquired = False
    with limiter:
        acquired = True
    assert acquired


def test_get_limiter_returns_the_same_instance_for_the_same_provider() -> None:
    """Repeated calls for one provider name share one ProviderLimiter instance."""
    first = get_limiter("test-provider-same-instance", max_in_flight=2)
    second = get_limiter("test-provider-same-instance", max_in_flight=5)

    assert first is second


def test_get_limiter_returns_independent_instances_for_different_providers() -> None:
    """Different provider names get their own, independent limiter."""
    gemini_limiter = get_limiter("test-provider-a", max_in_flight=2)
    openai_limiter = get_limiter("test-provider-b", max_in_flight=2)

    assert gemini_limiter is not openai_limiter
