"""Prefixes a per-document summary to each chunk's retrieval text (ADR-0015: SAC).

Summary-Augmented Chunking (SAC): one summary is generated per immutable
document version and prepended to every chunk's retrieval text - distinct
from Contextual Retrieval's per-chunk blurb (retrieval/contextual/pipeline.py),
which is generated separately for each individual chunk.
"""

from dataclasses import replace

from ragforge.domain.models import Chunk


def apply_document_summary(summary_text: str, chunks: list[Chunk]) -> list[Chunk]:
    """Return ``chunks`` with ``summary_text`` prepended to each one's retrieval text.

    Structural IDs, chunk IDs, and ``source_text`` are preserved untouched
    (ADR-0015) - only ``retrieval_text`` gains the prefix. This composes on
    top of whatever ``retrieval_text`` already holds: applied to baseline
    chunks it yields the ADR's ``sac`` variant (summary + source_text);
    applied after ``contextualize_chunks`` it yields ``sac_contextual``
    (summary + chunk context + source_text), with no separate code path.
    """
    return [
        replace(chunk, retrieval_text=f"{summary_text}\n\n{chunk.retrieval_text}")
        for chunk in chunks
    ]
