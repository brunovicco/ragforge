"""Tests for the Gemini embedding adapter (ADR-0005)."""

import pytest

from ragforge.embeddings.errors import EmbeddingError
from ragforge.embeddings.google_gemini_embedder import GoogleGeminiEmbedder


def test_raises_when_no_api_key_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key (env or explicit) fails fast with no network call, unlike a missing model."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(EmbeddingError, match="no Gemini API key"):
        GoogleGeminiEmbedder("gemini-embedding-001")
