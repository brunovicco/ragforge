"""Minimal, file-backed LLM call cache with atomic publication (ADR-0004).

Every Gemini-calling adapter in this project can optionally cache its calls
through this module, keyed by a stable hash of whatever inputs actually
determine the response (model, prompt, parameters). This is the mechanism
ADR-0004's "make bench" replay mode needs to exist at all - not that replay
mode itself, which additionally needs a validated, declared cache identity
this module does not provide. See
docs/adr/0004-benchmark-reproducibility-policy.md.

Coalescing here is in-process only: concurrent calls sharing a cache key
from different threads of the *same* run collapse into one provider call.
Nothing here coordinates across separate processes or separate runs.
"""

import hashlib
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMCache(Protocol):
    """A cache mapping a stable string key to a stable string value."""

    def get(self, key: str) -> str | None:
        """Return the cached value for ``key``, or None on a cache miss."""
        ...

    def put(self, key: str, value: str) -> None:
        """Store ``value`` under ``key``, atomically."""
        ...


class FileLLMCache:
    """One JSON file per key, under ``cache_dir`` - published atomically."""

    def __init__(self, cache_dir: Path) -> None:
        """Create ``cache_dir`` if it doesn't already exist."""
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def get(self, key: str) -> str | None:
        """Return the cached value for ``key``, or None on a cache miss."""
        path = self._path(key)
        if not path.exists():
            return None
        payload: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        return payload["value"]

    def put(self, key: str, value: str) -> None:
        """Write ``value`` for ``key``: temp file then atomic rename, never a partial file."""
        path = self._path(key)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps({"value": value}), encoding="utf-8")
        tmp_path.replace(path)


def cache_key(**parts: object) -> str:
    """Return a stable sha256 hex digest of the given keyword parts, order-independent."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_key_locks: dict[str, threading.Lock] = {}
_key_locks_guard = threading.Lock()


def _lock_for_key(key: str) -> threading.Lock:
    """Return the same Lock instance for ``key`` across calls (in-process coalescing).

    ``_key_locks`` grows for the life of the process without eviction - a
    deliberate simplification: one benchmark run makes a bounded number of
    distinct calls (thousands, not millions), so the accumulated Lock
    objects (a few dozen bytes each) never become a real concern.
    """
    with _key_locks_guard:
        lock = _key_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _key_locks[key] = lock
        return lock


def cached_call[T](
    cache: LLMCache | None,
    key: str,
    call: Callable[[], T],
    serialize: Callable[[T], str],
    deserialize: Callable[[str], T],
) -> T:
    """Return the cached value for ``key`` if present; otherwise call, cache, and return it.

    ``cache=None`` is a passthrough - no cache configured, every call reaches
    the provider. Concurrent calls sharing ``key`` (within this process)
    coalesce: only the first caller invokes ``call()``; the rest block on
    that call's lock and then reuse its now-cached result, rather than each
    making a redundant provider call.
    """
    if cache is None:
        return call()

    cached = cache.get(key)
    if cached is not None:
        return deserialize(cached)

    with _lock_for_key(key):
        # Re-check: another thread may have populated the cache while this
        # one was waiting for the lock.
        cached = cache.get(key)
        if cached is not None:
            return deserialize(cached)
        result = call()
        cache.put(key, serialize(result))
        return result
