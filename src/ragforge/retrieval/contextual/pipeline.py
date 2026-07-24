"""Prepends an LLM-generated context to each chunk (README strategy #5: Contextual Retrieval).

Anthropic's "Contextual Retrieval" technique (2024): a short blurb situating
the chunk within its source document is generated per chunk and prepended
before embedding/indexing, which measurably improves retrieval without
changing the retrieval algorithm itself - Dense/Hybrid index the enriched
text exactly as they'd index the original chunk text.
"""

from dataclasses import replace

from ragforge.domain.models import Chunk
from ragforge.retrieval.contextual.ports import Contextualizer


def contextualize_chunks(
    document_text: str, chunks: list[Chunk], contextualizer: Contextualizer
) -> list[Chunk]:
    """Return ``chunks`` with an LLM-generated context prepended to each one's retrieval text.

    Structural IDs, chunk IDs, and ``source_text`` are preserved untouched
    (ADR-0015) - only ``retrieval_text`` gains the prefix, so relevance
    judgments (ADR-0002) stay valid and the answer generator/judge, which
    read only ``source_text``, never see this synthetic blurb.
    """
    contextualized = []
    for chunk in chunks:
        context = contextualizer.contextualize(document_text, chunk.source_text)
        contextualized.append(replace(chunk, retrieval_text=f"{context}\n\n{chunk.source_text}"))
    return contextualized
