"""Gemini-based multi-chunk summarization (README strategy #7: RAPTOR, minimal impl.).

Summarizes a group of chunk texts into one higher-level tree node, for
retrieval.raptor.pipeline.build_raptor_tree. Requires GEMINI_API_KEY (or
GOOGLE_API_KEY) in the environment - the same credential GoogleGeminiEmbedder
and GeminiContextualizer use. Real, metered API calls: one generation call per
group, per tree level.

Retries 429s and transient transport errors via
ragforge.adapters.gemini_retry, shared with every other Gemini-calling
adapter in this project.
"""

import os

from google import genai
from google.genai import types

from ragforge.adapters.gemini_retry import call_with_retry
from ragforge.generation.errors import GenerationError

_TEMPERATURE = 0.0
_MAX_OUTPUT_TOKENS = 400

_PROMPT_TEMPLATE = """Summarize the following excerpts from a Brazilian \
regulatory norm into a single coherent paragraph, preserving the specific \
legal obligations and article references they contain. Answer only with the \
summary and nothing else.

{texts}"""


class GeminiSummarizer:
    """Summarizes a group of chunk texts with a Gemini generation model."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        """Create the client for ``model_name``.

        Args:
            model_name: The Gemini generation model id, e.g. "gemini-3.1-flash-lite".
            api_key: Overrides GEMINI_API_KEY / GOOGLE_API_KEY from the environment.

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
            self._client = genai.Client(api_key=key)
        except Exception as exc:
            raise GenerationError(f"failed to create Gemini client: {exc}") from exc

        self.name = model_name

    def summarize(self, texts: list[str]) -> str:
        """Return a single summary covering every text in ``texts``.

        Raises:
            GenerationError: If the call fails, retries are exhausted, or the
                model returns no text.
        """
        prompt = _PROMPT_TEMPLATE.format(texts="\n\n---\n\n".join(texts))
        config = types.GenerateContentConfig(
            temperature=_TEMPERATURE, max_output_tokens=_MAX_OUTPUT_TOKENS
        )

        try:
            response = call_with_retry(
                lambda: self._client.models.generate_content(
                    model=self.name, contents=prompt, config=config
                )
            )
        except Exception as exc:
            raise GenerationError(
                f"Gemini generate_content failed for {self.name!r}: {exc}"
            ) from exc

        if not response.text:
            raise GenerationError(f"Gemini returned no text for {self.name!r}")
        return response.text.strip()
