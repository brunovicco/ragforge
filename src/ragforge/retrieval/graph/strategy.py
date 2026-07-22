"""GraphRAG retrieval via LightRAG, local/global modes (README #8; scope: ADR-0010).

Queries a LightRAG index built by ``retrieval.graph.indexing.index_norm`` and
recovers each result's structural provenance by matching LightRAG's returned
chunk content back to our own Chunk objects - see ADR-0010 for why this
mapping is necessary and its disclosed limitations (coverage can silently
drop; scores are a rank-based proxy, not a native LightRAG score).
"""

from lightrag import LightRAG
from lightrag.base import QueryParam

from ragforge.domain.models import Chunk, Query, RetrievalResult

_DEFAULT_MODE = "local"


class GraphRagRetrieval:
    """Retrieves chunks via LightRAG's knowledge-graph query, mapped back to our Chunks."""

    name = "graphrag"

    def __init__(
        self, rag: LightRAG, content_index: dict[str, Chunk], mode: str = _DEFAULT_MODE
    ) -> None:
        """Wire the strategy to an already-indexed LightRAG instance and its content lookup.

        Args:
            rag: A LightRAG instance already indexed via ``indexing.index_norm``.
            content_index: The ``chunk.text.strip() -> Chunk`` lookup from
                ``indexing.build_content_index``, covering every chunk indexed into ``rag``.
            mode: LightRAG query mode - "local" or "global" per the README's GraphRAG scope
                (ADR-0010); "hybrid", "mix", and "naive" are also accepted.
        """
        self._rag = rag
        self._content_index = content_index
        self._mode = mode

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        """Return up to ``top_k`` results LightRAG's query surfaced, mapped back to our Chunks.

        Chunks LightRAG returns that don't match any indexed chunk's content
        (ADR-0010's disclosed coverage limitation) are skipped rather than
        raised, so this can return fewer than ``top_k`` results.
        """
        param = QueryParam(mode=self._mode, chunk_top_k=top_k)
        result = self._rag.query_data(query.text, param)
        if result.get("status") != "success":
            return []

        raw_chunks = result.get("data", {}).get("chunks", [])[:top_k]
        results = []
        for rank, item in enumerate(raw_chunks, start=1):
            content = (item.get("content") or "").strip()
            chunk = self._content_index.get(content)
            if chunk is None:
                continue
            results.append(RetrievalResult(chunk=chunk, score=1.0 / rank, strategy=self.name))
        return results
