"""Integration test for the GraphRAG (LightRAG) adapter end-to-end (ADR-0010).

Opt-in only: excluded from the default `pytest` run by the `-m "not integration"`
addopts. Makes several real, metered Gemini API calls (entity extraction,
keyword extraction, embeddings) - small but non-trivial cost, more than the
other Gemini integration tests in this project, since LightRAG's own pipeline
runs multiple LLM passes per chunk. Run explicitly with
`uv run pytest -m integration`, with GEMINI_API_KEY or GOOGLE_API_KEY set.
"""

import asyncio
import os
from pathlib import Path

import pytest

_TEST_LLM_MODEL = "gemini-3.1-flash-lite"
_TEST_EMBEDDING_DIMENSIONS = 768


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not _has_api_key(), reason="GEMINI_API_KEY/GOOGLE_API_KEY not set")
def test_indexes_and_retrieves_real_chunks_with_provenance_recovered(tmp_path: Path) -> None:
    """A tiny real LightRAG index recovers structural_ids for a matching query."""
    from lightrag import LightRAG

    from ragforge.domain.models import Chunk, Query
    from ragforge.embeddings.google_gemini_embedder import GoogleGeminiEmbedder
    from ragforge.retrieval.graph.indexing import build_content_index, index_norm
    from ragforge.retrieval.graph.lightrag_gemini import (
        build_gemini_embedding_func,
        build_gemini_llm_model_func,
    )
    from ragforge.retrieval.graph.strategy import GraphRagRetrieval

    embedder = GoogleGeminiEmbedder(
        "gemini-embedding-001", output_dimensionality=_TEST_EMBEDDING_DIMENSIONS
    )
    rag = LightRAG(
        working_dir=str(tmp_path),
        embedding_func=build_gemini_embedding_func(embedder),
        llm_model_func=build_gemini_llm_model_func(_TEST_LLM_MODEL),
    )
    asyncio.run(rag.initialize_storages())
    try:
        chunks = [
            Chunk(
                chunk_id="c1",
                text="Art. 1º As instituições financeiras devem adotar política de "
                "segurança cibernética.",
                structural_ids=("TEST-NORM::art-1",),
            ),
            Chunk(
                chunk_id="c2",
                text="Art. 2º É proibida a comercialização de produtos financeiros "
                "sem autorização prévia do Banco Central.",
                structural_ids=("TEST-NORM::art-2",),
            ),
        ]
        index_norm(rag, "TEST-NORM", chunks)
        content_index = build_content_index(chunks)
        strategy = GraphRagRetrieval(rag, content_index, mode="global")

        results = strategy.retrieve(Query(text="segurança cibernética"), top_k=5)

        assert len(results) > 0
        assert all(r.chunk.structural_ids for r in results)
        assert all(r.strategy == "graphrag" for r in results)
    finally:
        asyncio.run(rag.finalize_storages())
