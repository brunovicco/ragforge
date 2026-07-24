"""OpenAI-based bounded answer rewrite for the post-generation citation audit (ADR-0016).

Produces the single corrected answer the audit pipeline (citation_audit.py)
allows when a claim's citations fail deterministic checks or lack semantic
support - never a second attempt ("no recursive correction"). Independent
from the Gemini answer generator, same spirit as the ADR-0018 judge and the
semantic verifier (openai_semantic_verifier.py).

Reuses ragas.llms.InstructorLLM purely as a structured-completion wrapper,
same as the semantic verifier - "openai" and "ragas" are both already direct
project dependencies since Increment 5, no new dependency here.

An optional LLMCache (ADR-0004) keys on the question, original answer, valid
source texts, and the audit findings driving the rewrite - a changed answer
or a changed set of findings is always a cache miss; a ProviderLimiter
(ADR-0014) bounds concurrent in-flight calls process-wide.
"""

import os
from typing import Protocol, runtime_checkable

import instructor
from pydantic import BaseModel
from ragas.llms import InstructorLLM

from ragforge.adapters.llm_cache import LLMCache, cache_key, cached_call
from ragforge.adapters.provider_limiter import get_limiter
from ragforge.evaluation.audit_ports import ClaimAudit
from ragforge.generation.errors import GenerationError

_PROVIDER = "openai"
_PROMPT_VERSION = "answer-rewrite-ptbr-v1"
_DEFAULT_MAX_IN_FLIGHT = 4

_SYSTEM_PROMPT = """Você reescreve uma resposta legal/regulatória em português que uma auditoria \
de citações apontou como parcial ou totalmente não sustentada. Regras: remova toda afirmação não \
sustentada pelas fontes válidas fornecidas; nunca invente um novo ID de citação além dos já \
presentes nas fontes válidas; se, depois de remover o não sustentado, não restar nada que \
responda à pergunta, declare explicitamente que a evidência disponível é insuficiente, em vez de \
inventar conteúdo. Responda apenas com o texto da resposta reescrita."""

_PROMPT_TEMPLATE = """<pergunta>
{question}
</pergunta>

<resposta_original>
{original_answer}
</resposta_original>

<fontes_validas>
{valid_source_texts}
</fontes_validas>

<achados_da_auditoria>
{findings}
</achados_da_auditoria>

Reescreva a resposta seguindo as regras do sistema."""


class _RewriteOutput(BaseModel):
    """Pydantic shape instructor validates the rewrite output against (adapter boundary)."""

    rewritten_answer: str


@runtime_checkable
class _GeneratingLLM(Protocol):
    """Shape of ragas.llms.InstructorLLM this module calls (avoids its unresolved generic type)."""

    def generate(self, prompt: str, response_model: type[_RewriteOutput]) -> _RewriteOutput: ...


def _format_valid_source_texts(valid_source_texts: tuple[str, ...]) -> str:
    return "\n\n".join(valid_source_texts) if valid_source_texts else "(nenhuma)"


def _format_findings(findings: tuple[ClaimAudit, ...]) -> str:
    lines = []
    for claim_audit in findings:
        verdict = (
            claim_audit.semantic_support.verdict.value
            if claim_audit.semantic_support is not None
            else "deterministic_check_failed"
        )
        lines.append(f'- "{claim_audit.claim.text}" -> {verdict}')
    return "\n".join(lines) if lines else "(nenhum)"


class OpenAIAnswerRewriter:
    """Produces at most one corrected answer from an audit's findings via OpenAI."""

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
            model_name: The dated OpenAI generation model snapshot used for rewriting.
            reasoning_effort: Threaded into every underlying chat.completions.create() call.
            api_key: Overrides OPENAI_API_KEY from the environment.
            cache: Optional LLMCache (ADR-0004). None (the default) disables
                caching - every rewrite() call reaches the real API.
            max_in_flight: Bounds concurrent rewrite() calls to this provider,
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
            raise GenerationError(f"failed to create OpenAI rewriter client: {exc}") from exc

        self._llm: _GeneratingLLM = InstructorLLM(
            client=instructor_client,
            model=model_name,
            provider="openai",
            reasoning_effort=reasoning_effort,
            system_prompt=_SYSTEM_PROMPT,
        )
        self._model_name = model_name
        self._reasoning_effort = reasoning_effort
        self._cache = cache
        self._limiter = get_limiter(_PROVIDER, max_in_flight)

    def rewrite(
        self,
        question: str,
        original_answer: str,
        valid_source_texts: tuple[str, ...],
        findings: tuple[ClaimAudit, ...],
    ) -> str:
        """Return a rewritten answer removing unsupported material, introducing no new citations.

        Raises:
            GenerationError: If the call fails or retries are exhausted.
        """
        findings_text = _format_findings(findings)
        key = cache_key(
            provider=_PROVIDER,
            model=self._model_name,
            reasoning_effort=self._reasoning_effort,
            prompt_version=_PROMPT_VERSION,
            question=question,
            original_answer=original_answer,
            valid_source_texts=valid_source_texts,
            findings=findings_text,
        )
        return cached_call(
            self._cache,
            key,
            lambda: self._rewrite_uncached(
                question, original_answer, valid_source_texts, findings_text
            ),
            serialize=lambda text: text,
            deserialize=lambda raw: raw,
        )

    def _rewrite_uncached(
        self,
        question: str,
        original_answer: str,
        valid_source_texts: tuple[str, ...],
        findings_text: str,
    ) -> str:
        prompt = _PROMPT_TEMPLATE.format(
            question=question,
            original_answer=original_answer,
            valid_source_texts=_format_valid_source_texts(valid_source_texts),
            findings=findings_text,
        )
        try:
            with self._limiter:
                output = self._llm.generate(prompt, response_model=_RewriteOutput)
        except Exception as exc:
            raise GenerationError(f"OpenAI answer rewrite failed: {exc}") from exc

        return output.rewritten_answer.strip()
