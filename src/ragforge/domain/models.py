"""Core domain models. Framework-free by design (enforced by validate_architecture)."""

from dataclasses import dataclass, field
from enum import StrEnum


class QueryClass(StrEnum):
    """The seven RegRAG-BR query classes."""

    EXACT_FACTUAL = "exact_factual"
    SEMANTIC = "semantic"
    MULTI_HOP = "multi_hop"
    GLOBAL = "global"
    SECTION_COMPARATIVE = "section_comparative"
    NUMERIC_TABULAR = "numeric_tabular"
    UNANSWERABLE = "unanswerable"


@dataclass(frozen=True, slots=True)
class StructuralRef:
    """Stable structural ID of a norm unit (ADR-0002/0006).

    Canonical form: ``{norm}::{art}::{par|inc|ali}``,
    e.g. ``RES-CMN-4893/2021::art-3::par-1``.
    """

    norm: str
    article: str
    fragment: str | None = None

    @property
    def canonical(self) -> str:
        """Render the canonical string form."""
        parts = [self.norm, self.article]
        if self.fragment:
            parts.append(self.fragment)
        return "::".join(parts)

    @classmethod
    def parse(cls, raw: str) -> "StructuralRef":
        """Parse a canonical string into a StructuralRef."""
        parts = raw.split("::")
        if len(parts) not in (2, 3) or not all(parts):
            msg = f"invalid structural ref: {raw!r}"
            raise ValueError(msg)
        return cls(norm=parts[0], article=parts[1], fragment=parts[2] if len(parts) == 3 else None)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit carrying its structural provenance (ADR-0006)."""

    chunk_id: str
    text: str
    structural_ids: tuple[str, ...]
    parent_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked retrieval hit."""

    chunk: Chunk
    score: float
    strategy: str


@dataclass(frozen=True, slots=True)
class Query:
    """An analyzed user query."""

    text: str
    query_class: QueryClass | None = None
