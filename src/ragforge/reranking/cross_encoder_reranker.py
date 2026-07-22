"""Cross-encoder reranking adapter (README strategy #4: Hybrid top-50 -> cross-encoder -> top-5).

Wraps a local sentence-transformers CrossEncoder model. Unlike a bi-encoder
(EmbeddingModel), a cross-encoder scores a (query, chunk) pair jointly through
a single forward pass, which is more accurate but too slow to run over a full
corpus - it only re-scores a small pool a cheaper strategy already narrowed
down. The specific model is a data-driven choice, not hardcoded here - pass it
explicitly at construction.
"""

from typing import cast

from sentence_transformers import CrossEncoder

from ragforge.domain.models import Chunk, Query
from ragforge.reranking.errors import RerankingError


class CrossEncoderReranker:
    """Scores (query, chunk) pairs with a local sentence-transformers cross-encoder."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        """Load ``model_name`` once; the constructor is the expensive step.

        Raises:
            RerankingError: If the model cannot be loaded.
        """
        try:
            self._model = CrossEncoder(model_name, device=device)
        except Exception as exc:
            raise RerankingError(f"failed to load model {model_name!r}: {exc}") from exc
        self.name = model_name

    def score(self, query: Query, chunks: list[Chunk]) -> list[float]:
        """Return one relevance score per chunk, same order, higher meaning more relevant.

        Raises:
            RerankingError: If scoring fails.
        """
        if not chunks:
            return []
        pairs = [(query.text, chunk.text) for chunk in chunks]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            raise RerankingError(f"failed to score {len(chunks)} pair(s): {exc}") from exc
        return cast(list[float], scores.tolist())
