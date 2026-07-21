"""OpenSearch-backed chunk store for sparse BM25 retrieval (ADR-0005)."""

from typing import Any

from opensearchpy import OpenSearch, helpers

from ragforge.domain.models import Chunk, RetrievalResult

_DEFAULT_INDEX = "chunks"
_STRATEGY_NAME = "sparse"


class SparseChunkStore:
    """Indexes and searches chunks by BM25 over a Brazilian-Portuguese analyzer."""

    def __init__(self, client: OpenSearch, index: str = _DEFAULT_INDEX) -> None:
        """Wrap an OpenSearch client bound to a single index."""
        self._client = client
        self._index = index

    def create_index(self) -> None:
        """Create the index if it doesn't exist, analyzing text as Brazilian Portuguese."""
        if self._client.indices.exists(index=self._index):
            return
        self._client.indices.create(
            index=self._index,
            body={
                "mappings": {
                    "properties": {
                        "text": {"type": "text", "analyzer": "brazilian"},
                        "structural_ids": {"type": "keyword"},
                        "parent_id": {"type": "keyword"},
                        "metadata": {"type": "object", "enabled": True},
                    }
                }
            },
        )

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Bulk-index chunks, keyed by chunk_id so re-indexing overwrites in place."""
        actions = (
            {
                "_index": self._index,
                "_id": chunk.chunk_id,
                "_source": {
                    "text": chunk.text,
                    "structural_ids": list(chunk.structural_ids),
                    "parent_id": chunk.parent_id,
                    "metadata": chunk.metadata,
                },
            }
            for chunk in chunks
        )
        helpers.bulk(self._client, actions, refresh=True)

    def search(self, query_text: str, top_k: int) -> list[RetrievalResult]:
        """Return the top_k chunks by BM25 relevance to query_text."""
        response = self._client.search(
            index=self._index,
            body={"query": {"match": {"text": query_text}}, "size": top_k},
        )
        return [self._to_result(hit) for hit in response["hits"]["hits"]]

    @staticmethod
    def _to_result(hit: dict[str, Any]) -> RetrievalResult:
        source = hit["_source"]
        chunk = Chunk(
            chunk_id=hit["_id"],
            text=source["text"],
            structural_ids=tuple(source["structural_ids"]),
            parent_id=source["parent_id"],
            metadata=source["metadata"],
        )
        return RetrievalResult(chunk=chunk, score=hit["_score"], strategy=_STRATEGY_NAME)
