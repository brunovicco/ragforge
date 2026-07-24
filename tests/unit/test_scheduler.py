"""Tests for the deterministic bounded-concurrency scheduler (ADR-0014)."""

import time

from ragforge.evaluation.scheduler import run_bounded


def test_returns_one_result_per_item_in_canonical_order_regardless_of_completion_order() -> None:
    """Items that finish fastest-first still land in their original input slot."""
    # Inverted delay: item 0 sleeps longest, item 4 finishes first - completion
    # order is the exact reverse of input order.
    delays = [0.04, 0.03, 0.02, 0.01, 0.0]

    def work(index: int) -> int:
        time.sleep(delays[index])
        return index * 10

    results = run_bounded(list(range(5)), work, max_workers=5)

    assert results == [0, 10, 20, 30, 40]


def test_captures_a_per_item_exception_instead_of_raising() -> None:
    """A failing item's exception is returned in its slot, not propagated."""

    def work(item: int) -> int:
        if item == 1:
            raise ValueError("boom")
        return item

    results = run_bounded([0, 1, 2], work, max_workers=3)

    assert results[0] == 0
    assert isinstance(results[1], ValueError)
    assert results[2] == 2


def test_on_result_observes_outcomes_in_completion_order() -> None:
    """on_result sees (index, value, None) or (index, None, exc) as each future completes."""
    delays = [0.03, 0.0, 0.02]
    observed: list[int] = []

    def work(index: int) -> int:
        time.sleep(delays[index])
        return index

    def on_result(index: int, value: int | None, exc: BaseException | None) -> bool:
        observed.append(index)
        return False

    run_bounded(list(range(3)), work, max_workers=3, on_result=on_result)

    assert observed == [1, 2, 0]


def test_on_result_returning_true_cancels_not_yet_started_work() -> None:
    """Requesting cancellation stops queued-but-unstarted items; the rest still finish."""
    started = []

    def work(index: int) -> int:
        started.append(index)
        time.sleep(0.02)
        return index

    def cancel_after_first(index: int, value: int | None, exc: BaseException | None) -> bool:
        return True

    results = run_bounded(list(range(10)), work, max_workers=1, on_result=cancel_after_first)

    # With max_workers=1, only the first task ever starts before cancellation
    # is requested; every later slot holds a CancelledError instead of a value.
    assert results[0] == 0
    assert any(isinstance(result, BaseException) for result in results[1:])


def test_serial_path_cancels_exactly_after_the_requested_item_with_no_delay() -> None:
    """max_workers=1 with instantaneous work still cancels at the exact right item.

    A naive ThreadPoolExecutor(max_workers=1) races here: its single worker
    thread can grab and finish several more queued items before the main
    thread's cancellation request lands, when work has no delay to slow it
    down. The serial path (no thread pool at all for max_workers<=1) has no
    such race - cancellation always takes effect between two items exactly.
    """
    attempted: list[int] = []

    def work(index: int) -> int:
        attempted.append(index)
        return index

    def cancel_after_third(index: int, value: int | None, exc: BaseException | None) -> bool:
        return index == 2

    results = run_bounded(list(range(10)), work, max_workers=1, on_result=cancel_after_third)

    assert attempted == [0, 1, 2]
    assert results[:3] == [0, 1, 2]
    assert all(isinstance(result, BaseException) for result in results[3:])


def test_bounded_concurrency_is_faster_than_serial_for_a_slow_fake_provider() -> None:
    """A fake slow provider demonstrates real speedup under bounded concurrency (ADR-0014)."""
    item_count = 6
    delay = 0.05

    def work(_item: int) -> None:
        time.sleep(delay)

    serial_start = time.monotonic()
    run_bounded(list(range(item_count)), work, max_workers=1)
    serial_duration = time.monotonic() - serial_start

    parallel_start = time.monotonic()
    run_bounded(list(range(item_count)), work, max_workers=item_count)
    parallel_duration = time.monotonic() - parallel_start

    assert parallel_duration < serial_duration / 2
