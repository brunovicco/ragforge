"""Gemini-backed embedding_func/llm_model_func for wiring LightRAG (README #8: GraphRAG).

LightRAG (lightrag-hku) expects two callables at construction: an async
``embedding_func(list[str]) -> np.ndarray``, and an async
``llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str``.
These adapt GoogleGeminiEmbedder's synchronous embed() and a direct Gemini
generate_content call to those contracts, reusing the GEMINI_API_KEY /
GOOGLE_API_KEY credential every other Gemini adapter in this project uses.
"""

import asyncio
import os
import random
from collections.abc import Callable, Coroutine
from typing import Any, cast

import numpy as np
from google import genai
from google.genai import errors, types
from google.genai.types import ContentListUnionDict
from lightrag.utils import EmbeddingFunc

from ragforge.embeddings.google_gemini_embedder import GoogleGeminiEmbedder
from ragforge.generation.errors import GenerationError

_RATE_LIMIT_STATUS_CODE = 429
_MAX_RETRIES = 5
_BASE_DELAY_SECONDS = 2.0
_MAX_DELAY_SECONDS = 60.0


def build_gemini_embedding_func(embedder: GoogleGeminiEmbedder) -> EmbeddingFunc:
    """Wrap an already-constructed GoogleGeminiEmbedder for LightRAG's EmbeddingFunc contract."""

    async def _embed(texts: list[str]) -> np.ndarray:
        vectors = await asyncio.to_thread(embedder.embed, texts)
        return np.array(vectors, dtype=np.float32)

    return EmbeddingFunc(embedding_dim=embedder.dimensions, func=_embed)


def build_gemini_llm_model_func(
    model_name: str, api_key: str | None = None
) -> Callable[..., Coroutine[Any, Any, str]]:
    """Return an async llm_model_func for LightRAG, backed by a Gemini generation model.

    Raises:
        GenerationError: If no API key is available or the client can't be created.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GenerationError(
            "no Gemini API key found: set GEMINI_API_KEY or GOOGLE_API_KEY, "
            "or pass api_key explicitly"
        )
    try:
        client = genai.Client(api_key=key)
    except Exception as exc:
        raise GenerationError(f"failed to create Gemini client: {exc}") from exc

    async def _llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **_kwargs: object,
    ) -> str:
        contents = [
            types.Content(
                role="model" if message.get("role") == "assistant" else "user",
                parts=[types.Part(text=message.get("content", ""))],
            )
            for message in history_messages or []
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.0)

        for attempt in range(_MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=cast(ContentListUnionDict, contents),
                    config=config,
                )
            except errors.ClientError as exc:
                is_last_attempt = attempt == _MAX_RETRIES - 1
                if exc.code != _RATE_LIMIT_STATUS_CODE or is_last_attempt:
                    raise GenerationError(
                        f"Gemini generate_content failed for {model_name!r}: {exc}"
                    ) from exc
                delay = min(_BASE_DELAY_SECONDS * (2**attempt), _MAX_DELAY_SECONDS)
                # Retry-backoff jitter, not a security use of randomness.
                await asyncio.sleep(delay + random.uniform(0, delay * 0.1))  # noqa: S311  # nosec B311
                continue
            except Exception as exc:
                raise GenerationError(
                    f"Gemini generate_content failed for {model_name!r}: {exc}"
                ) from exc

            return response.text or ""

        raise GenerationError(f"Gemini generate_content retries exhausted for {model_name!r}")

    return _llm_model_func
