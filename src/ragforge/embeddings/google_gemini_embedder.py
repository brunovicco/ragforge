"""Google Gemini embedding adapter (ADR-0005): the project's proprietary candidate.

Wraps the Gemini API's embed_content endpoint. Requires GEMINI_API_KEY (or
GOOGLE_API_KEY) in the environment - never hardcoded, never logged. This is a
real, metered API: unlike SentenceTransformerEmbedder, every embed() call is a
network request that costs money, however small at this project's corpus
scale. Used only for the explicit ADR-0005 embedding comparison.

Retries 429s and transient transport errors via
ragforge.adapters.gemini_retry, shared with every other Gemini-calling
adapter in this project.
"""

import os
from typing import cast

from google import genai
from google.genai import types
from google.genai.types import ContentListUnion

from ragforge.adapters.gemini_retry import call_with_retry
from ragforge.embeddings.errors import EmbeddingError

_BATCH_SIZE = 100  # embed_content accepts a batch of contents per call
# gemini-embedding-2 was verified (2026-07-22, direct API probe) to silently
# collapse any multi-text `contents` list down to a single embedding instead
# of erroring - so it must be called one text per request. gemini-embedding-001
# batches correctly and keeps the default above.
_SINGLE_TEXT_MODELS = frozenset({"gemini-embedding-2"})
_PROBE_TEXT = "probe"


class GoogleGeminiEmbedder:
    """Encodes text with a Google Gemini embedding model (e.g. gemini-embedding-001)."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        output_dimensionality: int | None = None,
    ) -> None:
        """Create the client and probe the model once to learn its output dimension.

        Args:
            model_name: The Gemini embedding model id, e.g. "gemini-embedding-001".
            api_key: Overrides GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
            output_dimensionality: Requests a truncated (Matryoshka) embedding
                size instead of the model's native dimension (3072 for
                gemini-embedding-001). Callers indexing into pgvector's HNSW
                index need this: HNSW rejects columns over 2000 dimensions.

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
        [probe_vector] = self._embed_batch([_PROBE_TEXT])
        self.dimensions = len(probe_vector)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text, batching requests to the API.

        Raises:
            EmbeddingError: If a batch request fails, or retries are exhausted.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size]))
        return vectors

    def _call(self, batch: list[str]) -> types.EmbedContentResponse:
        config = None
        if self._output_dimensionality is not None:
            config = types.EmbedContentConfig(output_dimensionality=self._output_dimensionality)
        try:
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
