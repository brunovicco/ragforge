"""Per-provider concurrency limiter (ADR-0014).

Bounds how many in-flight requests a hosted provider sees at once, process
wide - not just within one ThreadPoolExecutor's worker count, since multiple
call sites (answer generation, judge scoring) can all target the same
provider concurrently. This is concurrency-only: it does not track requests
per minute, tokens per minute, or honor a `Retry-After` header - see
docs/adr/0014-bounded-parallel-benchmark-execution.md for that fuller scope,
deferred here.
"""

import threading
from types import TracebackType


class ProviderLimiter:
    """A semaphore bounding concurrent in-flight requests to one provider."""

    def __init__(self, max_in_flight: int) -> None:
        """Allow up to ``max_in_flight`` concurrent acquisitions."""
        self._semaphore = threading.Semaphore(max_in_flight)

    def __enter__(self) -> None:
        """Block until a slot is free, then occupy it."""
        self._semaphore.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the slot, regardless of whether the guarded call raised."""
        self._semaphore.release()


_limiters: dict[str, ProviderLimiter] = {}
_limiters_guard = threading.Lock()


def get_limiter(provider: str, max_in_flight: int) -> ProviderLimiter:
    """Return the process-wide ProviderLimiter for ``provider``, creating it on first use.

    ``max_in_flight`` only takes effect the first time a given ``provider``
    name is requested in this process; later calls with a different value
    for the same provider reuse the limiter already created (a provider has
    exactly one concurrency bound at a time, not one per caller).
    """
    with _limiters_guard:
        limiter = _limiters.get(provider)
        if limiter is None:
            limiter = ProviderLimiter(max_in_flight)
            _limiters[provider] = limiter
        return limiter
