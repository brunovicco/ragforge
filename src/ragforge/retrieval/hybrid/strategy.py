"""Hybrid retrieval: BM25 + dense fused with Reciprocal Rank Fusion (ADR-0005)."""

from ragforge.domain.models import Chunk, Query, RetrievalResult
from ragforge.domain.protocols import RetrievalStrategy

_DEFAULT_RRF_K = 60


class HybridRetrieval:
    """Fuses two retrieval strategies' rankings via Reciprocal Rank Fusion.

    RRF scores each chunk by summing ``1 / (k + rank)`` across every ranking it
    appears in (Cormack et al., 2009): a chunk found by both a lexical and a
    semantic strategy, even at modest ranks in each, can outrank one found
    only at rank 1 by a single strategy - the point of combining the two.
    """

    name = "hybrid"

    def __init__(
        self, dense: RetrievalStrategy, sparse: RetrievalStrategy, rrf_k: int = _DEFAULT_RRF_K
    ) -> None:
        """Wire the strategy to the two rankings it fuses."""
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        """Return up to ``top_k`` results, ranked by fused RRF score."""
        rankings = (self._dense.retrieve(query, top_k), self._sparse.retrieve(query, top_k))

        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for ranking in rankings:
            for rank, result in enumerate(ranking, start=1):
                chunk_id = result.chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)
                chunks[chunk_id] = result.chunk

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=chunks[chunk_id], score=score, strategy=self.name)
            for chunk_id, score in ordered
        ]
