"""Ports the SAC pipeline depends on, defined near its use case (ADR-0001, ADR-0015)."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    """One summary generated for a single immutable document version (ADR-0015).

    ``topics``/``schema_version`` from the ADR's illustrative structured
    output are deliberately omitted: extracting them reliably needs
    structured output (e.g. via instructor), which is out of scope for this
    increment's "mechanism only" goal.
    """

    document_id: str
    document_version: str
    summary: str
    generation_model: str
    prompt_version: str


@runtime_checkable
class DocumentSummarizer(Protocol):
    """Generates one summary covering an entire document version."""

    name: str

    def summarize_document(
        self, document_id: str, document_version: str, document_text: str
    ) -> DocumentSummary:
        """Return a ``DocumentSummary`` for ``document_text``, identified by ``document_id``.

        ``document_version`` (e.g. the source's sha256) is part of the
        summary's cache identity (ADR-0015): a changed document version must
        never reuse a stale cached summary.
        """
        ...
