"""Tests for the file-backed LLM call cache (ADR-0004)."""

import threading
import time
from pathlib import Path

from ragforge.adapters.llm_cache import FileLLMCache, cache_key, cached_call


def test_get_returns_none_for_a_missing_key(tmp_path: Path) -> None:
    """A key never written is a clean cache miss, not an error."""
    cache = FileLLMCache(tmp_path)

    assert cache.get("missing") is None


def test_put_then_get_round_trips_the_value(tmp_path: Path) -> None:
    """A value written for a key is returned verbatim on the next get."""
    cache = FileLLMCache(tmp_path)

    cache.put("k1", "hello")

    assert cache.get("k1") == "hello"


def test_put_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    """Atomic publication (temp file + rename) never leaves the .tmp artifact around."""
    cache = FileLLMCache(tmp_path)

    cache.put("k1", "hello")

    assert sorted(p.name for p in tmp_path.iterdir()) == ["k1.json"]


def test_cache_key_is_stable_regardless_of_keyword_order() -> None:
    """The same parts, given in a different order, hash to the same key."""
    key_a = cache_key(model="m", prompt="p", temperature=0)
    key_b = cache_key(temperature=0, prompt="p", model="m")

    assert key_a == key_b


def test_cache_key_changes_when_any_part_changes() -> None:
    """A different prompt (or any other part) produces a different key."""
    key_a = cache_key(model="m", prompt="p1")
    key_b = cache_key(model="m", prompt="p2")

    assert key_a != key_b


def test_cached_call_with_no_cache_always_invokes_call(tmp_path: Path) -> None:
    """cache=None is a passthrough: every call reaches the real work."""
    calls = {"n": 0}

    def call() -> str:
        calls["n"] += 1
        return "result"

    for _ in range(3):
        result = cached_call(None, "k1", call, str, str)
        assert result == "result"

    assert calls["n"] == 3


def test_cached_call_invokes_call_only_once_for_a_repeated_key(tmp_path: Path) -> None:
    """A second cached_call for the same key reuses the cached value, no second call()."""
    cache = FileLLMCache(tmp_path)
    calls = {"n": 0}

    def call() -> str:
        calls["n"] += 1
        return "result"

    first = cached_call(cache, "k1", call, str, str)
    second = cached_call(cache, "k1", call, str, str)

    assert first == "result"
    assert second == "result"
    assert calls["n"] == 1


def test_cached_call_uses_serialize_and_deserialize_for_non_string_values(tmp_path: Path) -> None:
    """A non-string result round-trips through the given serialize/deserialize pair."""
    cache = FileLLMCache(tmp_path)

    def call() -> list[float]:
        return [1.0, 2.0, 3.0]

    def serialize(value: list[float]) -> str:
        return ",".join(str(v) for v in value)

    def deserialize(value: str) -> list[float]:
        return [float(v) for v in value.split(",")]

    first = cached_call(cache, "vector-key", call, serialize, deserialize)
    second = cached_call(cache, "vector-key", call, serialize, deserialize)

    assert first == [1.0, 2.0, 3.0]
    assert second == [1.0, 2.0, 3.0]


def test_cached_call_coalesces_concurrent_calls_for_the_same_key(tmp_path: Path) -> None:
    """Concurrent threads requesting the same missing key trigger only one real call."""
    cache = FileLLMCache(tmp_path)
    calls = {"n": 0}
    calls_lock = threading.Lock()

    def slow_call() -> str:
        with calls_lock:
            calls["n"] += 1
        time.sleep(0.05)
        return "result"

    results: list[str] = []
    results_lock = threading.Lock()

    def worker() -> None:
        result = cached_call(cache, "shared-key", slow_call, str, str)
        with results_lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls["n"] == 1
    assert results == ["result"] * 8
