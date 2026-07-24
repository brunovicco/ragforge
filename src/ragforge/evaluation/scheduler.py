"""Deterministic bounded-concurrency scheduler (ADR-0014).

One primitive reused by every parallel stage in this project: run
``work(item)`` for each item, bounded by ``max_workers``, but always return
results in the same order as ``items`` regardless of which unit finished
first - metrics and persisted artifacts (records.jsonl) must never depend on
completion order.

``max_workers <= 1`` runs the serial reference scheduler (ADR-0014 rollout
step 2): no thread pool at all, so cancellation-after-failure is exact - a
"cancelled" item is one that provably never started, not a race against a
worker thread that may already have grabbed the next item off the queue
before the main thread's cancellation request lands. ``max_workers > 1``
uses the bounded thread scheduler (step 3): real concurrency, where a
started task is allowed to finish even after cancellation is requested
(ADR-0014: "Workers SHALL honor cancellation between operations").
"""

from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed


def run_bounded[T, R](
    items: Sequence[T],
    work: Callable[[T], R],
    max_workers: int,
    on_result: Callable[[int, R | None, BaseException | None], bool] | None = None,
) -> list[R | BaseException]:
    """Run ``work(item)`` for every item, bounded by ``max_workers``, in canonical order.

    ``on_result`` (if given) observes each outcome in *completion* order -
    the only order in which a "N consecutive failures" circuit breaker makes
    sense - and may return ``True`` to request cancellation of any work not
    yet started. The returned list is always ordered to match ``items``: one
    slot per item, holding either ``work``'s return value or the exception
    it raised (a ``CancelledError`` for an item skipped after cancellation)
    - never re-raised here, so one item's failure never aborts the batch.
    Callers decide how to treat a per-item failure.
    """
    if max_workers <= 1:
        return _run_serial(items, work, on_result)
    return _run_threaded(items, work, max_workers, on_result)


def _run_serial[T, R](
    items: Sequence[T],
    work: Callable[[T], R],
    on_result: Callable[[int, R | None, BaseException | None], bool] | None,
) -> list[R | BaseException]:
    results: list[R | BaseException] = []
    cancelled = False
    for index, item in enumerate(items):
        if cancelled:
            results.append(CancelledError())
            continue
        try:
            value = work(item)
        except BaseException as exc:
            results.append(exc)
            if on_result is not None and on_result(index, None, exc):
                cancelled = True
            continue
        results.append(value)
        if on_result is not None and on_result(index, value, None):
            cancelled = True
    return results


def _run_threaded[T, R](
    items: Sequence[T],
    work: Callable[[T], R],
    max_workers: int,
    on_result: Callable[[int, R | None, BaseException | None], bool] | None,
) -> list[R | BaseException]:
    results: dict[int, R | BaseException] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_index: dict[Future[R], int] = {
            pool.submit(work, item): index for index, item in enumerate(items)
        }
        cancelled = False
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                value = future.result()
            except BaseException as exc:
                results[index] = exc
                if not cancelled and on_result is not None and on_result(index, None, exc):
                    cancelled = True
                    for pending in future_to_index:
                        pending.cancel()
                continue
            results[index] = value
            if not cancelled and on_result is not None and on_result(index, value, None):
                cancelled = True
                for pending in future_to_index:
                    pending.cancel()
    return [results[index] for index in range(len(items))]
