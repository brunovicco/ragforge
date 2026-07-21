"""Tests for core domain models."""

import pytest

from ragforge.domain.models import StructuralRef


def test_structural_ref_roundtrip() -> None:
    """Canonical form parses back to an equal ref."""
    ref = StructuralRef(norm="RES-CMN-4893/2021", article="art-3", fragment="par-1")
    assert ref.canonical == "RES-CMN-4893/2021::art-3::par-1"
    assert StructuralRef.parse(ref.canonical) == ref


def test_structural_ref_without_fragment() -> None:
    """Article-level refs have no fragment."""
    ref = StructuralRef.parse("RES-CVM-175/2022::art-10")
    assert ref.fragment is None
    assert ref.canonical == "RES-CVM-175/2022::art-10"


@pytest.mark.parametrize("raw", ["", "only-norm", "a::b::c::d", "::art-1"])
def test_structural_ref_rejects_invalid(raw: str) -> None:
    """Malformed refs raise ValueError."""
    with pytest.raises(ValueError, match="invalid structural ref"):
        StructuralRef.parse(raw)
