"""Gemini-based answer generation with structural-ID citations (ADR-0007 prerequisite).

Generates a grounded answer from a query and its retrieved chunks, citing the
structural ID(s) (ADR-0002/0006) each part of the answer draws from - the
input Citation Accuracy (ADR-0007) checks against a judgment's relevant set,
and downstream a RAGAS quality judge scores.

No generation model has been chosen via a dedicated comparison (unlike the
embedding model, ADR-0005); the model name is a constructor argument, not
hardcoded, matching every other Gemini adapter in this project.

Retries 429s and transient transport errors via
ragforge.adapters.gemini_retry, shared with every other Gemini-calling
adapter in this project. An optional LLMCache (ADR-0004) keys on model +
prompt, so an identical (query, context) pair reuses its prior answer
instead of calling the API again; a ProviderLimiter (ADR-0014) bounds
concurrent in-flight generate_content calls process-wide.

Also captures ADR-0017 "Generation lineage" (token usage, latency,
cache hit, prompt/answer hashes) via ``drain_generation_lineage()`` - a
lock-guarded buffer, same pattern as
``generation.auditing_answer_generator.AuditingAnswerGenerator``'s audit
buffer. This is the one adapter that needs this: the ADR's own field list
scopes token usage/latency to the answer generator, not to the judge or
auditor (see lineage_ports.py's module docstring).
"""

import json
import os
import threading
import time

from google import genai
from google.genai import types
from google.genai.types import GenerateContentResponseUsageMetadata

from ragforge.adapters.gemini_retry import call_with_retry
from ragforge.adapters.llm_cache import LLMCache, cache_key, cached_call
from ragforge.adapters.provider_limiter import get_limiter
from ragforge.domain.models import Answer, Query, RetrievalResult
from ragforge.evaluation.canonical_hash import canonical_json_hash
from ragforge.evaluation.lineage_ports import GenerationLineage
from ragforge.generation.citation_parsing import extract_citations
from ragforge.generation.errors import GenerationError

_TEMPERATURE = 0.0
_MAX_OUTPUT_TOKENS = 800
_PROVIDER = "gemini"
# Conservative default - not data-driven, matches ADR-0014's own example config.
_DEFAULT_MAX_IN_FLIGHT = 4

_SYSTEM_PROMPT = """You are a legal assistant answering questions about \
Brazilian financial and regulatory norms, using ONLY the context provided \
below - never outside knowledge. For every claim, cite the structural ID of \
the context chunk it comes from, inline, in square brackets immediately \
after the claim - e.g. "Instituições devem adotar controles de segurança \
[RES-CMN-4893/2021::art-2]." If the context does not contain enough \
information to answer, say so explicitly instead of guessing. Answer in the \
same language as the question."""

_USER_PROMPT_TEMPLATE = """<context>
{context}
</context>

<question>
{question}
</question>"""


def _format_context(results: list[RetrievalResult]) -> str:
    blocks = []
    for result in results:
        ids = ", ".join(result.chunk.structural_ids)
        blocks.append(f"[structural_ids: {ids}]\n{result.chunk.source_text}")
    return "\n\n".join(blocks)


class _CallState:
    """Per-generate()-call mutable state for lineage capture (ADR-0017): never shared or reused."""

    def __init__(self) -> None:
        self.cache_hit = True
        self.usage: GenerateContentResponseUsageMetadata | None = None


class GeminiAnswerGenerator:
    """Generates a cited answer with a Gemini generation model."""

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
                caching - every generate() call reaches the real API.
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
        self._lineage_lock = threading.Lock()
        self._generation_lineage: list[GenerationLineage] = []

    def generate(self, query: Query, results: list[RetrievalResult]) -> Answer:
        """Return an answer to ``query``, grounded only in ``results``.

        Raises:
            GenerationError: If the call fails, retries are exhausted, or the
                model returns no text.
        """
        prompt = _USER_PROMPT_TEMPLATE.format(context=_format_context(results), question=query.text)
        key = cache_key(
            provider=_PROVIDER, model=self.name, system_prompt=_SYSTEM_PROMPT, prompt=prompt
        )

        # Populated by _generate_uncached only on a real cache miss; stays
        # at its "hit" defaults otherwise - a fresh local instance per call,
        # so concurrent generate() calls from different worker threads never
        # share this state (unlike an instance attribute would).
        call_state = _CallState()

        def _generate_and_record() -> Answer:
            call_state.cache_hit = False
            answer, usage = self._generate_uncached(prompt)
            call_state.usage = usage
            return answer

        started = time.monotonic()
        answer = cached_call(
            self._cache,
            key,
            _generate_and_record,
            serialize=lambda answer: json.dumps(
                {"text": answer.text, "citations": list(answer.citations)}
            ),
            deserialize=lambda raw: Answer(
                text=json.loads(raw)["text"], citations=tuple(json.loads(raw)["citations"])
            ),
        )
        latency_seconds = time.monotonic() - started

        usage = call_state.usage
        lineage = GenerationLineage(
            provider=_PROVIDER,
            model=self.name,
            prompt_hash=key,
            input_chunk_ids=tuple(result.chunk.chunk_id for result in results),
            input_source_hashes=tuple(
                canonical_json_hash(result.chunk.source_text) for result in results
            ),
            answer_hash=canonical_json_hash(answer.text),
            parsed_citations=answer.citations,
            prompt_tokens=usage.prompt_token_count if usage is not None else None,
            completion_tokens=usage.candidates_token_count if usage is not None else None,
            total_tokens=usage.total_token_count if usage is not None else None,
            latency_seconds=latency_seconds,
            cache_hit=call_state.cache_hit,
        )
        with self._lineage_lock:
            self._generation_lineage.append(lineage)

        return answer

    def drain_generation_lineage(self) -> list[GenerationLineage]:
        """Return every GenerationLineage produced since the last drain, then clear the buffer.

        Same swap-and-clear pattern as
        ``AuditingAnswerGenerator.drain_audit_results()`` - callers drain
        after each strategy finishes, before the next one reuses this
        instance.
        """
        with self._lineage_lock:
            lineage, self._generation_lineage = self._generation_lineage, []
        return lineage

    def _generate_uncached(
        self, prompt: str
    ) -> tuple[Answer, GenerateContentResponseUsageMetadata | None]:
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=_TEMPERATURE,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
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
        text = response.text.strip()
        return Answer(text=text, citations=extract_citations(text)), response.usage_metadata
