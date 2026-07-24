"""LightRAG glue: provider-neutral embedding_func, Gemini llm_model_func (README #8: GraphRAG).

LightRAG (lightrag-hku) expects two callables at construction: an async
``embedding_func(list[str]) -> np.ndarray``, and an async
``llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str``.

``build_gemini_embedding_func`` just adapts any synchronous EmbeddingModel's
embed() to that async contract (ADR-0013) - it has never actually depended on
Gemini specifically, only on whichever embedder the caller already
constructed. ``build_gemini_llm_model_func`` genuinely is Gemini-specific: a
direct Gemini generate_content call, reusing the GEMINI_API_KEY /
GOOGLE_API_KEY credential every other Gemini adapter in this project uses -
GraphRAG's entity-extraction LLM is out of ADR-0013's scope (embeddings
only) and stays hard-coded to Gemini for now.

Retries 429s and transient transport errors via
ragforge.adapters.gemini_retry, shared with every other Gemini-calling
adapter in this project.
"""

import asyncio
import os
from collections.abc import Callable, Coroutine
from typing import Any, cast

import numpy as np
from google import genai
from google.genai import types
from google.genai.types import ContentListUnionDict
from lightrag.utils import EmbeddingFunc

from ragforge.adapters.gemini_retry import call_with_retry_async
from ragforge.embeddings.ports import EmbeddingModel
from ragforge.generation.errors import GenerationError


def build_gemini_embedding_func(embedder: EmbeddingModel) -> EmbeddingFunc:
    """Wrap an already-constructed EmbeddingModel for LightRAG's EmbeddingFunc contract."""

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

        try:
            response = await call_with_retry_async(
                lambda: asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=cast(ContentListUnionDict, contents),
                    config=config,
                )
            )
        except Exception as exc:
            raise GenerationError(
                f"Gemini generate_content failed for {model_name!r}: {exc}"
            ) from exc

        return response.text or ""

    return _llm_model_func
