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
adapter in this project.
"""

import os
import re

from google import genai
from google.genai import types

from ragforge.adapters.gemini_retry import call_with_retry
from ragforge.domain.models import Answer, Query, RetrievalResult, StructuralRef
from ragforge.generation.errors import GenerationError

_TEMPERATURE = 0.0
_MAX_OUTPUT_TOKENS = 800
_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")

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
        blocks.append(f"[structural_ids: {ids}]\n{result.chunk.text}")
    return "\n\n".join(blocks)


def _extract_citations(text: str) -> tuple[str, ...]:
    """Return every well-formed structural ID cited in ``text``, in first-cited order.

    A bracketed span that isn't a valid structural ID (a hallucinated or
    unrelated bracket) is skipped rather than raised - this parses untrusted
    model output, not a contract the model is guaranteed to honor.
    """
    seen: dict[str, None] = {}
    for candidate in _CITATION_RE.findall(text):
        try:
            ref = StructuralRef.parse(candidate)
        except ValueError:
            continue
        seen.setdefault(ref.canonical, None)
    return tuple(seen)


class GeminiAnswerGenerator:
    """Generates a cited answer with a Gemini generation model."""

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

    def generate(self, query: Query, results: list[RetrievalResult]) -> Answer:
        """Return an answer to ``query``, grounded only in ``results``.

        Raises:
            GenerationError: If the call fails, retries are exhausted, or the
                model returns no text.
        """
        prompt = _USER_PROMPT_TEMPLATE.format(context=_format_context(results), question=query.text)
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=_TEMPERATURE,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
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
        text = response.text.strip()
        return Answer(text=text, citations=_extract_citations(text))
