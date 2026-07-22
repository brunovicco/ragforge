"""Reranked retrieval: a wide candidate pool refined by a cross-encoder (README strategy #4)."""

from ragforge.domain.models import Query, RetrievalResult
from ragforge.domain.protocols import RetrievalStrategy
from ragforge.retrieval.reranked.ports import Reranker

_DEFAULT_POOL_SIZE = 50


class RerankedRetrieval:
    """Retrieves a candidate pool from a base strategy, then reorders it with a reranker.

    A cross-encoder scores a query and a chunk jointly in one forward pass -
    more accurate than a bi-encoder's independent embeddings, but too slow to
    run over a full corpus. It only re-scores the small pool a cheaper base
    strategy (typically Hybrid) already narrowed down.
    """

    name = "reranked"

    def __init__(
        self, base: RetrievalStrategy, reranker: Reranker, pool_size: int = _DEFAULT_POOL_SIZE
    ) -> None:
        """Wire the strategy to the base retriever it pools from and the reranker."""
        self._base = base
        self._reranker = reranker
        self._pool_size = pool_size

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        """Return up to ``top_k`` results from the base strategy's pool, reranked."""
        pool_size = max(self._pool_size, top_k)
        candidates = self._base.retrieve(query, pool_size)
        chunks = [candidate.chunk for candidate in candidates]
        scores = self._reranker.score(query, chunks)

        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            RetrievalResult(chunk=chunk, score=score, strategy=self.name)
            for chunk, score in ranked[:top_k]
        ]
