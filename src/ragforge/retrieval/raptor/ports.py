"""Ports the RAPTOR tree-building pipeline depends on, defined near its use case (ADR-0001)."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Summarizer(Protocol):
    """Summarizes a group of chunk texts into one higher-level tree node's content."""

    name: str

    def summarize(self, texts: list[str]) -> str:
        """Return a single summary covering every text in ``texts``."""
        ...
