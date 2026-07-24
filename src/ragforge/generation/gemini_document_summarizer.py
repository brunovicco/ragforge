"""Gemini-based per-document summarization (ADR-0015: Summary-Augmented Chunking).

Generates one summary per immutable document version, prepended to every
chunk's retrieval text by retrieval.sac.pipeline.apply_document_summary.
Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment - the same
credential every other Gemini adapter in this project uses. Real, metered API
calls: one generation call per document version (not per chunk).

Retries 429s and transient transport errors via ragforge.adapters.gemini_retry,
shared with every other Gemini-calling adapter in this project. An optional
LLMCache (ADR-0004) keys on provider + model + document identity + prompt
version (ADR-0015's cache identity, minus schema_version since this adapter
returns plain text rather than a structured result), so a document version
already summarized in a prior run is never re-sent to the API. A
ProviderLimiter (ADR-0014) bounds concurrent in-flight generate_content calls
process-wide.
"""

import os

from google import genai
from google.genai import types

from ragforge.adapters.gemini_retry import call_with_retry
from ragforge.adapters.llm_cache import LLMCache, cache_key, cached_call
from ragforge.adapters.provider_limiter import get_limiter
from ragforge.generation.errors import GenerationError
from ragforge.retrieval.sac.ports import DocumentSummary

_TEMPERATURE = 0.0
_MAX_OUTPUT_TOKENS = 500
_PROVIDER = "gemini"
_PROMPT_VERSION = "sac-summary-ptbr-v1"
# Conservative default - not data-driven, matches ADR-0014's own example config.
_DEFAULT_MAX_IN_FLIGHT = 4

_PROMPT_TEMPLATE = """Leia a norma regulatória brasileira abaixo e produza um \
resumo em português, conciso e factual, cobrindo apenas: propósito da norma, \
escopo de aplicação, autoridade emissora e principais categorias de \
dispositivos (obrigações, controles, prazos, exceções, governança). Não \
interprete além do texto, não invente obrigações e não faça afirmações sem \
suporte no texto. Responda apenas com o resumo.

<documento>
{document_text}
</documento>"""


class GeminiDocumentSummarizer:
    """Generates one whole-document summary with a Gemini generation model."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        cache: LLMCache | None = None,
        max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
    ) -> None:
        """Create the client for ``model_name``.

        Args:
            model_name: The Gemini generation model id, e.g. "gemini-3.1-flash-lite".
            api_key: Overrides GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
            cache: Optional LLMCache (ADR-0004). None (the default) disables
                caching - every summarize_document() call reaches the real API.
            max_in_flight: Bounds concurrent generate_content calls to this
                provider, process-wide (ADR-0014).

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
        self._cache = cache
        self._limiter = get_limiter(_PROVIDER, max_in_flight)

    def summarize_document(
        self, document_id: str, document_version: str, document_text: str
    ) -> DocumentSummary:
        """Return a ``DocumentSummary`` for ``document_text``.

        Raises:
            GenerationError: If the call fails, retries are exhausted, or the
                model returns no text.
        """
        key = cache_key(
            provider=_PROVIDER,
            model=self.name,
            document_id=document_id,
            document_version=document_version,
            prompt_version=_PROMPT_VERSION,
        )
        summary = cached_call(
            self._cache,
            key,
            lambda: self._summarize_uncached(document_text),
            serialize=lambda text: text,
            deserialize=lambda raw: raw,
        )
        return DocumentSummary(
            document_id=document_id,
            document_version=document_version,
            summary=summary,
            generation_model=self.name,
            prompt_version=_PROMPT_VERSION,
        )

    def _summarize_uncached(self, document_text: str) -> str:
        prompt = _PROMPT_TEMPLATE.format(document_text=document_text)
        config = types.GenerateContentConfig(
            temperature=_TEMPERATURE, max_output_tokens=_MAX_OUTPUT_TOKENS
        )

        try:
            with self._limiter:
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
