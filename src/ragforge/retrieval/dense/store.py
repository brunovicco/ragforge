"""pgvector-backed chunk store for dense retrieval (ADR-0005).

The vector column's dimension is fixed at table-creation time (a pgvector
constraint), parameterized by whichever EmbeddingModel is configured - not
hardcoded to a specific model, since ADR-0005 treats the model choice itself
as an open, data-driven comparison.
"""

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb

from ragforge.domain.models import Chunk, RetrievalResult

_DEFAULT_TABLE = "chunks"
_STRATEGY_NAME = "dense"


class DenseChunkStore:
    """Indexes and searches chunk embeddings in a pgvector table."""

    def __init__(self, conn: psycopg.Connection, table: str = _DEFAULT_TABLE) -> None:
        """Wrap an open psycopg connection, registering pgvector's type adapters."""
        register_vector(conn)
        self._conn = conn
        self._table = table

    def create_schema(self, dimensions: int) -> None:
        """Create the chunk table and its HNSW cosine index if they don't exist."""
        table = sql.Identifier(self._table)
        index = sql.Identifier(f"{self._table}_embedding_idx")
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {table} ("
                    "chunk_id TEXT PRIMARY KEY, "
                    "text TEXT NOT NULL, "
                    "structural_ids TEXT[] NOT NULL, "
                    "parent_id TEXT, "
                    "metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, "
                    "embedding VECTOR({dimensions}) NOT NULL"
                    ")"
                ).format(table=table, dimensions=sql.Literal(dimensions))
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} ON {table} "
                    "USING hnsw (embedding vector_cosine_ops)"
                ).format(index=index, table=table)
            )
        self._conn.commit()

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks and their embeddings, keyed by chunk_id.

        Raises:
            ValueError: If ``chunks`` and ``embeddings`` have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )
        table = sql.Identifier(self._table)
        statement = sql.SQL(
            "INSERT INTO {table} (chunk_id, text, structural_ids, parent_id, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (chunk_id) DO UPDATE SET "
            "text = EXCLUDED.text, "
            "structural_ids = EXCLUDED.structural_ids, "
            "parent_id = EXCLUDED.parent_id, "
            "metadata = EXCLUDED.metadata, "
            "embedding = EXCLUDED.embedding"
        ).format(table=table)
        with self._conn.cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                cur.execute(
                    statement,
                    (
                        chunk.chunk_id,
                        chunk.text,
                        list(chunk.structural_ids),
                        chunk.parent_id,
                        Jsonb(chunk.metadata),
                        embedding,
                    ),
                )
        self._conn.commit()

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """Return the top_k chunks by cosine similarity to query_embedding."""
        table = sql.Identifier(self._table)
        statement = sql.SQL(
            "SELECT chunk_id, text, structural_ids, parent_id, metadata, "
            "1 - (embedding <=> %s::vector) AS score "
            "FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s"
        ).format(table=table)
        with self._conn.cursor() as cur:
            cur.execute(statement, (query_embedding, query_embedding, top_k))
            rows = cur.fetchall()
        return [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id=row[0],
                    text=row[1],
                    structural_ids=tuple(row[2]),
                    parent_id=row[3],
                    metadata=row[4],
                ),
                score=row[5],
                strategy=_STRATEGY_NAME,
            )
            for row in rows
        ]
