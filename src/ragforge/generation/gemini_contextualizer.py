"""Gemini-based per-chunk contextualization (README strategy #5: Contextual Retrieval).

Generates a short context situating each chunk within its source document,
per Anthropic's "Contextual Retrieval" technique (2024) adapted to Gemini.
Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment - the same
credential GoogleGeminiEmbedder uses. Real, metered API calls: one generation
call per chunk.

Cost control: the full document text is placed first and identically in every
prompt for a given document, letting Gemini's automatic (implicit) context
caching reduce the cost of repeated calls against the same document, without
this adapter managing an explicit CachedContent resource's lifecycle.

The free tier's generate_content quota is tight, and Google's own error
response includes a retry-after hint, so a 429 is exactly the transient,
repeatable case AGENTS.md asks for bounded exponential backoff on - this
adapter retries those, and only those, before giving up.
"""

import os
import random
import time

from google import genai
from google.genai import errors, types

from ragforge.generation.errors import GenerationError

_RATE_LIMIT_STATUS_CODE = 429
_MAX_RETRIES = 5
_BASE_DELAY_SECONDS = 2.0
_MAX_DELAY_SECONDS = 60.0
_TEMPERATURE = 0.0
_MAX_OUTPUT_TOKENS = 200

_PROMPT_TEMPLATE = """<document>
{document_text}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short, succinct context (1-2 sentences) to situate this chunk \
within the overall document, for the purposes of improving search retrieval \
of the chunk. Answer only with the succinct context and nothing else."""


class GeminiContextualizer:
    """Generates per-chunk situating context with a Gemini generation model."""

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

    def contextualize(self, document_text: str, chunk_text: str) -> str:
        """Return a short context string for ``chunk_text``, given the full ``document_text``.

        Raises:
            GenerationError: If the call fails, rate limiting persists past
                every retry, or the model returns no text.
        """
        prompt = _PROMPT_TEMPLATE.format(document_text=document_text, chunk_text=chunk_text)
        config = types.GenerateContentConfig(
            temperature=_TEMPERATURE, max_output_tokens=_MAX_OUTPUT_TOKENS
        )

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self.name, contents=prompt, config=config
                )
            except errors.ClientError as exc:
                is_last_attempt = attempt == _MAX_RETRIES - 1
                if exc.code != _RATE_LIMIT_STATUS_CODE or is_last_attempt:
                    raise GenerationError(
                        f"Gemini generate_content failed for {self.name!r}: {exc}"
                    ) from exc
                delay = min(_BASE_DELAY_SECONDS * (2**attempt), _MAX_DELAY_SECONDS)
                # Retry-backoff jitter, not a security use of randomness.
                time.sleep(delay + random.uniform(0, delay * 0.1))  # noqa: S311  # nosec B311
                continue
            except Exception as exc:
                raise GenerationError(
                    f"Gemini generate_content failed for {self.name!r}: {exc}"
                ) from exc

            if not response.text:
                raise GenerationError(f"Gemini returned no text for {self.name!r}")
            return response.text.strip()

        raise GenerationError(f"Gemini generate_content retries exhausted for {self.name!r}")
