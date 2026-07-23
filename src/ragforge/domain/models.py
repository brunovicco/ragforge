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


class RelevanceGrade(StrEnum):
    """Graded relevance of a structural unit to a judged question (ADR-0002)."""

    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"


@dataclass(frozen=True, slots=True)
class JudgedRef:
    """One structural unit judged relevant (or partially) to a question."""

    ref: StructuralRef
    grade: RelevanceGrade


@dataclass(frozen=True, slots=True)
class Judgment:
    """Golden-set relevance judgment: the structural units that answer a question (ADR-0002).

    Annotated at the norm's stable structural unit, not at the chunk level, so
    the judgment survives chunking changes and stays comparable across
    strategies whose result granularity differs.
    """

    question_id: str
    query: Query
    relevant_refs: tuple[JudgedRef, ...]
    reference_answer: str | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated response grounded in retrieved chunks, with structural-ID citations (ADR-0007).

    ``citations`` are the canonical structural IDs (``StructuralRef.canonical``)
    the answer cites, in first-cited order - the input to Citation Accuracy
    (the proportion belonging to a judgment's relevant set) and, downstream,
    to a RAGAS quality judge.
    """

    text: str
    citations: tuple[str, ...]
