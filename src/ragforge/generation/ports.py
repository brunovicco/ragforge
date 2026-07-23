"""Ports the answer generation pipeline depends on, defined near their use case (ADR-0001)."""

from typing import Protocol, runtime_checkable

from ragforge.domain.models import Answer, Query, RetrievalResult


@runtime_checkable
class AnswerGenerator(Protocol):
    """Generates a grounded, cited answer from a query and its retrieved context."""

    name: str

    def generate(self, query: Query, results: list[RetrievalResult]) -> Answer:
        """Return an answer to ``query``, grounded only in ``results``."""
        ...
