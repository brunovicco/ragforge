"""Persistent per-text embedding cache decorator."""

import dataclasses
import hashlib
import json

from ragforge.adapters.llm_cache import LLMCache, cache_key
from ragforge.embeddings.identity import EmbeddingIdentity
from ragforge.embeddings.ports import EmbeddingModel

_CACHE_SCHEMA_VERSION = 1


class CachedEmbeddingModel:
    """Cache embeddings by complete model identity and retrieval-text hash."""

    def __init__(
        self,
        delegate: EmbeddingModel,
        identity: EmbeddingIdentity,
        cache: LLMCache,
    ) -> None:
        """Wrap ``delegate`` without changing its public embedding identity."""
        self._delegate = delegate
        self._identity = identity
        self._cache = cache
        self.name = delegate.name
        self.dimensions = delegate.dimensions

    def _key(self, text: str) -> str:
        return cache_key(
            kind="embedding",
            schema_version=_CACHE_SCHEMA_VERSION,
            identity=dataclasses.asdict(self._identity),
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )

    def _deserialize(self, raw: str) -> list[float]:
        payload: object = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError("cached embedding must be a JSON array")
        if len(payload) != self.dimensions:
            raise ValueError(
                f"cached embedding has {len(payload)} dimensions; expected {self.dimensions}"
            )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in payload):
            raise ValueError("cached embedding contains a non-numeric value")
        return [float(value) for value in payload]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return cached vectors and batch only unique misses through the delegate."""
        if not texts:
            return []

        vectors_by_text: dict[str, list[float]] = {}
        missing_texts: list[str] = []
        for text in dict.fromkeys(texts):
            cached = self._cache.get(self._key(text))
            if cached is None:
                missing_texts.append(text)
            else:
                vectors_by_text[text] = self._deserialize(cached)

        if missing_texts:
            fresh_vectors = self._delegate.embed(missing_texts)
            if len(fresh_vectors) != len(missing_texts):
                raise ValueError(
                    f"embedder returned {len(fresh_vectors)} vectors for {len(missing_texts)} texts"
                )
            for text, vector in zip(missing_texts, fresh_vectors, strict=True):
                if len(vector) != self.dimensions:
                    raise ValueError(
                        f"embedder returned {len(vector)} dimensions; expected {self.dimensions}"
                    )
                vectors_by_text[text] = vector
                self._cache.put(self._key(text), json.dumps(vector))

        return [vectors_by_text[text] for text in texts]
