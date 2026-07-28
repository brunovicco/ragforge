"""Google Gemini embedding adapter (ADR-0005): the project's proprietary candidate.

Wraps the metered embed_content API; requires GEMINI_API_KEY (or
GOOGLE_API_KEY), never hardcoded or logged. Shares
ragforge.adapters.gemini_retry with every Gemini adapter; an optional
LLMCache (ADR-0004) is consulted per individual text before batching, and a
ProviderLimiter (ADR-0014) bounds concurrent calls.
"""

import json
import os
from typing import cast

from google import genai
from google.genai import types
from google.genai.types import ContentListUnion

from ragforge.adapters.gemini_retry import call_with_retry
from ragforge.adapters.llm_cache import LLMCache, cache_key
from ragforge.adapters.provider_limiter import get_limiter
from ragforge.embeddings.errors import EmbeddingError

_BATCH_SIZE = 100  # embed_content accepts a batch of contents per call
# gemini-embedding-2 was verified (2026-07-22, direct API probe) to silently
# collapse any multi-text `contents` list down to a single embedding instead
# of erroring - so it must be called one text per request. gemini-embedding-001
# batches correctly and keeps the default above.
_SINGLE_TEXT_MODELS = frozenset({"gemini-embedding-2"})
_PROBE_TEXT = "probe"
_PROVIDER = "gemini"
# Conservative default - not data-driven, matches ADR-0014's own example config.
_DEFAULT_MAX_IN_FLIGHT = 4


class GoogleGeminiEmbedder:
    """Encodes text with a Google Gemini embedding model (e.g. gemini-embedding-001)."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        output_dimensionality: int | None = None,
        cache: LLMCache | None = None,
        max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
    ) -> None:
        """Create the client and probe the model once to learn its output dimension.

        ``output_dimensionality`` requests a truncated (Matryoshka) size
        instead of the native dimension - required for pgvector HNSW, which
        rejects columns over 2000 dimensions. ``api_key`` overrides the
        environment; ``cache``/``max_in_flight`` wire ADR-0004/ADR-0014.

        Raises:
            EmbeddingError: If no API key is available, the client can't be
                created, or the probe call fails.
        """
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise EmbeddingError(
                "no Gemini API key found: set GEMINI_API_KEY or GOOGLE_API_KEY, "
                "or pass api_key explicitly"
            )
        try:
            self._client = genai.Client(api_key=key)
        except Exception as exc:
            raise EmbeddingError(f"failed to create Gemini client: {exc}") from exc

        self.name = model_name
        self._output_dimensionality = output_dimensionality
        self._batch_size = 1 if model_name in _SINGLE_TEXT_MODELS else _BATCH_SIZE
        self._cache = cache
        self._limiter = get_limiter(_PROVIDER, max_in_flight)
        [probe_vector] = self._embed_uncached([_PROBE_TEXT])
        self.dimensions = len(probe_vector)

    def _cache_key_for(self, text: str) -> str:
        return cache_key(
            provider=_PROVIDER,
            model=self.name,
            text=text,
            output_dimensionality=self._output_dimensionality,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text, batching requests to the API.

        Cache hits skip the API per text; only misses are batched, in their
        original relative order, and written back before returning.

        Raises:
            EmbeddingError: If a batch request fails, or retries are exhausted.
        """
        if self._cache is None:
            return self._embed_uncached(texts)

        keys = [self._cache_key_for(text) for text in texts]
        vectors: list[list[float] | None] = [
            json.loads(cached) if (cached := self._cache.get(key)) is not None else None
            for key in keys
        ]

        missing_indices = [index for index, vector in enumerate(vectors) if vector is None]
        if missing_indices:
            fresh_vectors = self._embed_uncached([texts[index] for index in missing_indices])
            for index, vector in zip(missing_indices, fresh_vectors, strict=True):
                vectors[index] = vector
                self._cache.put(keys[index], json.dumps(vector))

        return cast(list[list[float]], vectors)

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Batch ``texts`` through the real API, bypassing the cache entirely."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size]))
        return vectors

    def _call(self, batch: list[str]) -> types.EmbedContentResponse:
        config = None
        if self._output_dimensionality is not None:
            config = types.EmbedContentConfig(output_dimensionality=self._output_dimensionality)
        try:
            with self._limiter:
                return call_with_retry(
                    lambda: self._client.models.embed_content(
                        model=self.name, contents=cast(ContentListUnion, batch), config=config
                    )
                )
        except Exception as exc:
            raise EmbeddingError(f"Gemini embed_content failed for {self.name!r}: {exc}") from exc

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = self._call(batch)

        embeddings = response.embeddings or []
        if len(embeddings) != len(batch):
            raise EmbeddingError(f"expected {len(batch)} embeddings, got {len(embeddings)}")

        vectors: list[list[float]] = []
        for embedding in embeddings:
            if embedding.values is None:
                raise EmbeddingError(
                    f"Gemini returned an embedding with no values for {self.name!r}"
                )
            vectors.append(list(embedding.values))
        return vectors
