"""Dense retrieval baseline over pgvector (ADR-0005)."""

from ragforge.domain.models import Query, RetrievalResult
from ragforge.embeddings.ports import EmbeddingModel
from ragforge.retrieval.dense.ports import ChunkSearchStore


class DenseRetrieval:
    """Embeds the query and searches the pgvector-backed dense chunk index."""

    name = "dense"

    def __init__(self, store: ChunkSearchStore, embedder: EmbeddingModel) -> None:
        """Wire the strategy to its chunk store and embedding model."""
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        """Return up to ``top_k`` ranked results for the query."""
        [vector] = self._embedder.embed([query.text])
        return self._store.search(vector, top_k)
