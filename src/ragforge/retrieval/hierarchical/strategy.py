"""Parent-child (small-to-big) retrieval: search fine-grained, return full context.

Chunks below max_chars already carry parent_id linking a fragment (§/inciso) to
its article (see chunking.chunker.chunk_norm) - the norm's real section/article
hierarchy, not an arbitrary window (ADR-0006). This strategy searches on that
fine-grained embedding for precision, then expands each hit to its parent
article chunk for the fuller context a fragment alone often lacks.
"""

from ragforge.domain.models import Query, RetrievalResult
from ragforge.domain.protocols import RetrievalStrategy
from ragforge.retrieval.hierarchical.ports import ChunkFetchStore


class ParentChildRetrieval:
    """Wraps another strategy's ranking, expanding each hit to its parent chunk.

    Two fragments of the same article can both match a query; expanding both
    would return the same parent chunk twice, so a later duplicate (in score
    order) is dropped rather than backfilled - this can return fewer than
    top_k results when the underlying ranking is parent-concentrated.
    """

    name = "parent-child"

    def __init__(self, inner: RetrievalStrategy, store: ChunkFetchStore) -> None:
        """Wire the strategy to the ranking it expands and the store it expands from."""
        self._inner = inner
        self._store = store

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        """Return up to ``top_k`` results, each expanded to its parent when it has one."""
        expanded: list[RetrievalResult] = []
        seen_chunk_ids: set[str] = set()
        for result in self._inner.retrieve(query, top_k):
            chunk = result.chunk
            if chunk.parent_id is not None:
                parent = self._store.get(chunk.parent_id)
                if parent is not None:
                    chunk = parent
            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            expanded.append(RetrievalResult(chunk=chunk, score=result.score, strategy=self.name))
        return expanded
