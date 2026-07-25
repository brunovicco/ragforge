"""Tests for canonical, order-independent JSON hashing (ADR-0017)."""

from ragforge.evaluation.canonical_hash import canonical_json_hash


def test_canonical_json_hash_is_stable_across_dict_key_order() -> None:
    """Two dicts with the same keys/values in different order hash identically."""
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}

    assert canonical_json_hash(a) == canonical_json_hash(b)


def test_canonical_json_hash_differs_for_different_content() -> None:
    """A genuinely different payload hashes differently."""
    assert canonical_json_hash({"x": 1}) != canonical_json_hash({"x": 2})


def test_canonical_json_hash_handles_tuples_and_nested_structures() -> None:
    """Tuples (not natively JSON-serializable in a stable way) round-trip via list conversion."""
    payload = {"ids": ("a", "b", "c"), "nested": {"score": 0.5}}

    digest = canonical_json_hash(payload)

    assert isinstance(digest, str)
    assert len(digest) == 64  # sha256 hex digest length


def test_canonical_json_hash_is_deterministic_across_calls() -> None:
    """The same payload always hashes to the same digest."""
    payload = {"a": [1, 2, 3], "b": "text"}

    assert canonical_json_hash(payload) == canonical_json_hash(payload)


def test_canonical_json_hash_handles_plain_strings_and_numbers() -> None:
    """Non-dict, non-tuple payloads (a bare string) also hash deterministically."""
    assert canonical_json_hash("hello") == canonical_json_hash("hello")
    assert canonical_json_hash("hello") != canonical_json_hash("world")
