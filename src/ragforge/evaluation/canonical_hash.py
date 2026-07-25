"""Canonical, order-independent JSON hashing (ADR-0017).

Used for ``configuration_hash`` (the resolved benchmark config), event
``payload_hash``/``event_hash`` (event_log.py), and any other place this
project needs a stable content hash of a JSON-serializable Python object -
the exact same pattern ``adapters/llm_cache.cache_key`` already established
for LLM call cache keys (ADR-0004), reused here rather than duplicated.
"""

import hashlib
import json


def canonical_json_hash(obj: object) -> str:
    """Return a stable SHA-256 hex digest of ``obj``'s canonical JSON serialization.

    ``sort_keys=True`` makes the hash independent of dict key order;
    ``default=str`` handles values that aren't natively JSON-serializable
    (e.g. tuples become lists, dataclasses fall back to their repr) the same
    way ``cache_key`` already does, so callers never need to pre-serialize.
    """
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
