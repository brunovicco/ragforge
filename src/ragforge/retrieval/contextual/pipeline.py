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
    """Return ``chunks`` with an LLM-generated context prepended to each one's text.

    Structural IDs, chunk IDs, and other provenance are preserved untouched -
    only ``text`` changes, so relevance judgments (ADR-0002) stay valid.
    """
    contextualized = []
    for chunk in chunks:
        context = contextualizer.contextualize(document_text, chunk.text)
        contextualized.append(replace(chunk, text=f"{context}\n\n{chunk.text}"))
    return contextualized
