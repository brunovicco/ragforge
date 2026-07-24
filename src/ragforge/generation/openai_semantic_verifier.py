"""OpenAI-based semantic support verification for the post-generation citation audit (ADR-0016).

Judges whether a claim's cited authoritative source text actually supports
it - the one LLM-backed check the audit pipeline (citation_audit.py) runs,
and only for claims whose citations have already passed every deterministic
check ("prefer deterministic checks before LLM verification"). Independent
from the Gemini answer generator, same spirit as the ADR-0018 judge -
avoids correlating the generator's and the auditor's errors.

Reuses ragas.llms.InstructorLLM purely as a structured-completion wrapper
(not any RAGAS metric class): it already handles OpenAI reasoning models'
parameter quirks (temperature=1.0, max_completion_tokens, no top_p) for
free, the same way ragas_judge.py's abstention check does. "openai" and
"ragas" are both already direct project dependencies since Increment 5 -
no new dependency here.

An optional LLMCache (ADR-0004) keys on the question, claim, and cited
citations, so an identical (question, claim, citations) triple is never
re-verified; a ProviderLimiter (ADR-0014) bounds concurrent in-flight calls
process-wide, the same pattern every other adapter in this project follows.
"""

import json
import os
from typing import Protocol, runtime_checkable

import instructor
from pydantic import BaseModel
from ragas.llms import InstructorLLM

from ragforge.adapters.llm_cache import LLMCache, cache_key, cached_call
from ragforge.adapters.provider_limiter import get_limiter
from ragforge.evaluation.audit_ports import SemanticSupportResult, SupportVerdict
from ragforge.evaluation.judge_ports import ModelIdentity
from ragforge.generation.errors import GenerationError

_PROVIDER = "openai"
_OUTPUT_SCHEMA_VERSION = 1
_PROMPT_VERSION = "semantic-support-ptbr-v1"
_DEFAULT_MAX_IN_FLIGHT = 4

_SYSTEM_PROMPT = """Você avalia se um texto-fonte autoritativo sustenta uma afirmação legal ou \
regulatória em português. Use apenas o texto-fonte fornecido - nunca conhecimento externo ou \
busca. Interprete negações e exceções normativas de forma conservadora. Classifique "supported" \
só quando o texto-fonte sustenta integralmente a afirmação; "partially_supported" quando \
sustenta só parte dela; "unsupported" quando contradiz ou não tem relação; "indeterminate" \
quando o texto-fonte é insuficiente para julgar com confiança. Atribua cada ID de citação a \
supported_citation_ids ou unsupported_citation_ids conforme o texto-fonte dele sustente ou não \
a afirmação."""

_PROMPT_TEMPLATE = """<pergunta>
{question}
</pergunta>

<afirmacao>
{claim_text}
</afirmacao>

<citacoes>
{citations}
</citacoes>

Essa afirmação é sustentada pelas citações acima? Responda apenas com o julgamento \
estruturado."""


class _SemanticSupportOutput(BaseModel):
    """Pydantic shape instructor validates the verifier's output against (adapter boundary)."""

    verdict: SupportVerdict
    rationale: str
    supported_citation_ids: list[str] = []
    unsupported_citation_ids: list[str] = []
    missing_evidence: list[str] = []


@runtime_checkable
class _GeneratingLLM(Protocol):
    """Shape of ragas.llms.InstructorLLM this module calls (avoids its unresolved generic type)."""

    def generate(
        self, prompt: str, response_model: type[_SemanticSupportOutput]
    ) -> _SemanticSupportOutput: ...


def _format_citations(cited_citations: tuple[tuple[str, str], ...]) -> str:
    return "\n\n".join(
        f"[{structural_id}]\n{source_text}" for structural_id, source_text in cited_citations
    )


class OpenAISemanticSupportVerifier:
    """Scores one claim's semantic support against its cited source text via OpenAI."""

    def __init__(
        self,
        model_name: str,
        reasoning_effort: str = "medium",
        api_key: str | None = None,
        cache: LLMCache | None = None,
        max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
    ) -> None:
        """Create the client for ``model_name``.

        Args:
            model_name: The dated OpenAI generation model snapshot used as the verifier.
            reasoning_effort: Threaded into every underlying chat.completions.create() call.
            api_key: Overrides OPENAI_API_KEY from the environment.
            cache: Optional LLMCache (ADR-0004). None (the default) disables
                caching - every verify() call reaches the real API.
            max_in_flight: Bounds concurrent verify() calls to this provider,
                process-wide (ADR-0014).

        Raises:
            GenerationError: If no API key is available or the client can't be created.
        """
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise GenerationError(
                "no OpenAI API key found: set OPENAI_API_KEY, or pass api_key explicitly"
            )
        try:
            instructor_client = instructor.from_provider(
                f"openai/{model_name}", async_client=True, api_key=key
            )
        except Exception as exc:
            raise GenerationError(f"failed to create OpenAI verifier client: {exc}") from exc

        self._llm: _GeneratingLLM = InstructorLLM(
            client=instructor_client,
            model=model_name,
            provider="openai",
            reasoning_effort=reasoning_effort,
            system_prompt=_SYSTEM_PROMPT,
        )
        self._cache = cache
        self._limiter = get_limiter(_PROVIDER, max_in_flight)
        self._identity = ModelIdentity(
            provider=_PROVIDER,
            model=model_name,
            reasoning_effort=reasoning_effort,
            output_schema_version=_OUTPUT_SCHEMA_VERSION,
        )

    @property
    def identity(self) -> ModelIdentity:
        """Exact verifier configuration used by verify() - recorded in the run manifest."""
        return self._identity

    def verify(
        self, question: str, claim_text: str, cited_citations: tuple[tuple[str, str], ...]
    ) -> SemanticSupportResult:
        """Return the structured support judgment for ``claim_text``.

        Raises:
            GenerationError: If the call fails or retries are exhausted.
        """
        key = cache_key(
            provider=_PROVIDER,
            model=self._identity.model,
            reasoning_effort=self._identity.reasoning_effort,
            output_schema_version=_OUTPUT_SCHEMA_VERSION,
            prompt_version=_PROMPT_VERSION,
            question=question,
            claim_text=claim_text,
            cited_citations=cited_citations,
        )
        return cached_call(
            self._cache,
            key,
            lambda: self._verify_uncached(question, claim_text, cited_citations),
            serialize=_serialize_result,
            deserialize=_deserialize_result,
        )

    def _verify_uncached(
        self, question: str, claim_text: str, cited_citations: tuple[tuple[str, str], ...]
    ) -> SemanticSupportResult:
        prompt = _PROMPT_TEMPLATE.format(
            question=question, claim_text=claim_text, citations=_format_citations(cited_citations)
        )
        try:
            with self._limiter:
                output = self._llm.generate(prompt, response_model=_SemanticSupportOutput)
        except Exception as exc:
            raise GenerationError(f"OpenAI semantic verification failed: {exc}") from exc

        return SemanticSupportResult(
            verdict=output.verdict,
            rationale=output.rationale,
            supported_citation_ids=tuple(output.supported_citation_ids),
            unsupported_citation_ids=tuple(output.unsupported_citation_ids),
            missing_evidence=tuple(output.missing_evidence),
        )


def _serialize_result(result: SemanticSupportResult) -> str:
    return json.dumps(
        {
            "verdict": result.verdict.value,
            "rationale": result.rationale,
            "supported_citation_ids": list(result.supported_citation_ids),
            "unsupported_citation_ids": list(result.unsupported_citation_ids),
            "missing_evidence": list(result.missing_evidence),
        }
    )


def _deserialize_result(raw: str) -> SemanticSupportResult:
    payload = json.loads(raw)
    return SemanticSupportResult(
        verdict=SupportVerdict(payload["verdict"]),
        rationale=payload["rationale"],
        supported_citation_ids=tuple(payload["supported_citation_ids"]),
        unsupported_citation_ids=tuple(payload["unsupported_citation_ids"]),
        missing_evidence=tuple(payload["missing_evidence"]),
    )
