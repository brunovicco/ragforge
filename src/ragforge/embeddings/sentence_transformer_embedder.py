"""sentence-transformers-based embedding adapter (ADR-0005).

Wraps a local Hugging Face model (e.g. BGE-M3, Qwen3-Embedding). Loading is
the expensive step (~17s import, first-run download), so this adapter is
covered by the integration suite only (`pytest -m integration`). The model
is a data-driven ADR-0005 choice, passed explicitly at construction.
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

        ``device`` (cpu/mps/cuda) defaults to sentence-transformers' own
        auto-detection. ``revision`` pins the model revision for exact
        reproducibility (ADR-0013); when unspecified, the resolved default
        is still recorded via ``self.revision`` rather than left ambiguous.

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

        Embeddings are L2-normalized so dot product equals cosine similarity,
        matching pgvector's cosine operator. A tqdm bar on stderr is the only
        progress visibility during long CPU-bound encodes.

        Raises:
            EmbeddingError: If encoding fails.
        """
        try:
            vectors = self._model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
            )
        except Exception as exc:
            raise EmbeddingError(f"failed to encode {len(texts)} text(s): {exc}") from exc
        return cast(list[list[float]], vectors.tolist())
