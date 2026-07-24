"""sentence-transformers-based embedding adapter (ADR-0005).

Wraps a local Hugging Face model (e.g. BGE-M3, Qwen3-Embedding) via
sentence-transformers. Loading the model is the expensive step (~17s import,
network on first download, CPU/GPU inference setup) - this adapter is
deliberately not exercised by the unit test suite; see
tests/integration/test_sentence_transformer_embedder.py, run via
`pytest -m integration`.

The specific model is a data-driven choice per ADR-0005 (BGE-M3 vs
Qwen3-Embedding vs a proprietary model), not hardcoded here - pass it
explicitly at construction.
"""

from typing import cast

from sentence_transformers import SentenceTransformer

from ragforge.embeddings.errors import EmbeddingError


class SentenceTransformerEmbedder:
    """Encodes text with a local sentence-transformers model."""

    def __init__(
        self, model_name: str, device: str | None = None, revision: str | None = None
    ) -> None:
        """Load ``model_name`` once; the constructor is the expensive step.

        Args:
            model_name: The Hugging Face model id, e.g. "Qwen/Qwen3-Embedding-0.6B".
            device: cpu | mps | cuda. Defaults to sentence-transformers' own
                auto-detection when omitted.
            revision: Pinned model revision (git commit/tag), for exact
                reproducibility (ADR-0013). Defaults to the same "main" HF
                resolves to when unspecified - recorded as such via
                ``self.revision`` rather than left ambiguous.

        Raises:
            EmbeddingError: If the model cannot be loaded.
        """
        try:
            self._model = SentenceTransformer(model_name, device=device, revision=revision)
        except Exception as exc:
            # The load path spans several backends (huggingface_hub, torch,
            # safetensors) with varied exception types; translate all of them
            # at this adapter boundary.
            raise EmbeddingError(f"failed to load model {model_name!r}: {exc}") from exc

        dimensions = self._model.get_embedding_dimension()
        if dimensions is None:
            raise EmbeddingError(f"model {model_name!r} reports no embedding dimension")

        self.name = model_name
        self.dimensions = dimensions
        self.revision = revision or "main"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text, encoded in a single batched call.

        Embeddings are L2-normalized so a plain dot product is equivalent to
        cosine similarity, matching pgvector's cosine distance operator.

        Raises:
            EmbeddingError: If encoding fails.
        """
        try:
            vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        except Exception as exc:
            raise EmbeddingError(f"failed to encode {len(texts)} text(s): {exc}") from exc
        return cast(list[list[float]], vectors.tolist())
