"""Indexes our own ADR-0006 chunks into LightRAG, preserving chunk boundaries (ADR-0010).

LightRAG normally decides its own chunk boundaries from raw document text. To
recover structural_ids from its query results later, GraphRagRetrieval instead
needs LightRAG to index our exact chunks - so this module overrides
LightRAG's chunking_func for each document's insert to return our chunks
unchanged, verbatim, in order.
"""

from typing import Any

from lightrag import LightRAG

from ragforge.domain.models import Chunk


def index_norm(rag: LightRAG, norm_id: str, chunks: list[Chunk]) -> None:
    """Insert one norm's already-chunked text into LightRAG, using our chunk boundaries."""

    def _use_our_chunks(
        tokenizer: Any, content: str, *args: object, **kwargs: object
    ) -> list[dict[str, Any]]:
        return [
            {
                "content": chunk.text,
                "tokens": len(tokenizer.encode(chunk.text)),
                "chunk_order_index": index,
            }
            for index, chunk in enumerate(chunks)
        ]

    rag.chunking_func = _use_our_chunks
    rag.insert("\n\n".join(chunk.text for chunk in chunks), ids=norm_id)


def build_content_index(chunks: list[Chunk]) -> dict[str, Chunk]:
    """Return a ``chunk.text.strip() -> Chunk`` lookup for recovering provenance after a query.

    Assumes chunk text is unique across the indexed corpus - true for this
    project's legal chunks (each is a distinct article/paragraph/item), and
    the mapping this project's GraphRagRetrieval relies on (ADR-0010).
    """
    return {chunk.text.strip(): chunk for chunk in chunks}
