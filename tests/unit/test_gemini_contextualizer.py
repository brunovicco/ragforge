"""Tests for the Gemini contextualization adapter (README strategy #5)."""

import pytest

from ragforge.generation.errors import GenerationError
from ragforge.generation.gemini_contextualizer import GeminiContextualizer


def test_raises_when_no_api_key_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key (env or explicit) fails fast with no network call, unlike a missing model."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="no Gemini API key"):
        GeminiContextualizer("gemini-3.1-flash-lite")
