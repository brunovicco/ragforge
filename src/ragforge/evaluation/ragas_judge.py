"""RAGAS-based answer quality judge (ADR-0007/ADR-0018): Faithfulness, Answer Relevancy, abstention.

Wraps ragas.metrics.collections' Faithfulness and AnswerRelevancy over a
Gemini or OpenAI judge model, via instructor (ragas 0.4's structured-output
backend, already a ragas dependency) and the matching ragas embeddings
wrapper (GoogleEmbeddings / OpenAIEmbeddings) - no new project dependency for
Gemini; ``openai`` is already resolved transitively via instructor/ragas/
langchain-openai and is declared as a direct dependency for this adapter's
own ``import openai`` (ADR-0018).

Real, verified constraints (ragas 0.4.3, checked against the actual
library):

- instructor.from_provider(..., async_client=True) is required - ragas's
  async ascore() path raises if the wrapped client is synchronous.
- ragas.llms.InstructorLLM.__init__ merges any extra keyword argument (e.g.
  ``reasoning_effort="medium"``) straight into the dict it eventually passes
  to the underlying ``client.chat.completions.create(...)`` call, and its
  ``_map_openai_params`` already detects reasoning models (o-series,
  gpt-5+) by name and enforces their constraints (temperature=1.0, no
  top_p, max_completion_tokens) - so ADR-0018's ``reasoning_effort``
  threads through cleanly with no patching needed.
- ragas.metrics.collections.MetricResult (what Faithfulness/AnswerRelevancy
  return) exposes only a scalar ``.value`` - no structured rationale or
  claim breakdown. Reproducing ADR-0018's illustrative
  ``unsupported_claims``/``rationale`` JSON would mean writing bespoke judge
  prompts instead of RAGAS's own metric classes - out of scope; JudgeResult
  only carries what RAGAS actually produces (see judge_ports.py).

Abstention has no native RAGAS metric, so it is scored by this module's own
small structured-output call, through a *second*, dedicated InstructorLLM
instance (same client/model/provider as the main one, but its own PT-BR
system prompt) - kept separate from the instance Faithfulness/AnswerRelevancy
share so this project's added prompt never contaminates RAGAS's internal
ones (both ultimately call the same shared InstructorLLM.generate()).

Two things this module deliberately does NOT do:

- Factual Correctness, a third RAGAS metric, needs a reference answer per
  question to compare against. The current golden set
  (datasets/regrag-br/judgments.json) has no reference answers - a
  golden-set curation gap, not a code gap. Not implemented here.
- Judge calibration against human evaluation (ADR-0007/ADR-0018: ~30
  hand-scored samples, weighted kappa >= 0.6) is human curation work this
  module cannot substitute for. Scores from this judge are unvalidated
  until that calibration happens (see judge_calibration.py) - report them
  with that caveat, never as ground truth.

RagasJudge does not go through ragforge.adapters.gemini_retry: it calls
ragas/instructor's own internal async client, not our google-genai/openai
client directly, so there is no shared retry choke point to hook into here.
An optional LLMCache (ADR-0004) and ProviderLimiter (ADR-0014) instead wrap
evaluate() as a whole - the granularity available at this boundary.
"""

import json
import os
from typing import Protocol, runtime_checkable

import instructor
from google import genai
from openai import OpenAI
from pydantic import BaseModel
from ragas.embeddings import GoogleEmbeddings, OpenAIEmbeddings
from ragas.llms import InstructorLLM
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from ragforge.adapters.llm_cache import LLMCache, cache_key, cached_call
from ragforge.adapters.provider_limiter import get_limiter
from ragforge.evaluation.judge_ports import (
    AbstentionJudgment,
    JudgeResult,
    JudgeSample,
    MetricScore,
    ModelIdentity,
)
from ragforge.generation.errors import GenerationError

_DEFAULT_MAX_IN_FLIGHT = 4
_OUTPUT_SCHEMA_VERSION = 1
ABSTENTION_PROMPT_VERSION = "abstention-ptbr-v1"

_ABSTENTION_SYSTEM_PROMPT = """Você avalia se uma resposta legal/regulatória em português \
optou corretamente por responder ou se abster, dada a pergunta e se ela é respondível pelas \
evidências disponíveis. Interprete negações e exceções normativas de forma conservadora. \
Uma resposta apropriada para uma pergunta sem resposta (unanswerable) diz explicitamente que \
não há evidência suficiente, sem inventar conteúdo. Uma resposta apropriada para uma pergunta \
respondível responde com base nas evidências, sem se abster desnecessariamente."""

_ABSTENTION_PROMPT_TEMPLATE = """<pergunta>
{question}
</pergunta>

<respondivel_pelas_evidencias>
{answerable}
</respondivel_pelas_evidencias>

<resposta_gerada>
{answer}
</resposta_gerada>

A resposta acima absteve-se de responder, ou respondeu de forma substantiva? Isso foi \
apropriado dado se a pergunta é respondível pelas evidências? Responda apenas com o \
julgamento estruturado."""


class _AbstentionOutput(BaseModel):
    """Pydantic shape instructor validates the abstention judgment against (adapter boundary)."""

    appropriate: bool
    rationale: str


@runtime_checkable
class _ScoredMetric(Protocol):
    """Shape shared by ragas.metrics.collections' single-turn metric classes."""

    def score(self, **kwargs: object) -> object: ...  # returns a ragas MetricResult (has .value)


@runtime_checkable
class _GeneratingLLM(Protocol):
    """Shape of ragas.llms.InstructorLLM this module actually calls directly (abstention)."""

    def generate(
        self, prompt: str, response_model: type[_AbstentionOutput]
    ) -> _AbstentionOutput: ...


class RagasJudge:
    """Scores a JudgeSample for Faithfulness, Answer Relevancy, and abstention appropriateness."""

    def __init__(
        self,
        faithfulness: _ScoredMetric,
        answer_relevancy: _ScoredMetric,
        abstention_llm: _GeneratingLLM,
        identity: ModelIdentity,
        cache: LLMCache | None = None,
        max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
    ) -> None:
        """Wire the judge to its already-constructed RAGAS metrics and abstention LLM.

        Args:
            faithfulness: An already-constructed RAGAS Faithfulness metric.
            answer_relevancy: An already-constructed RAGAS AnswerRelevancy metric.
            abstention_llm: A dedicated InstructorLLM (or fake) used only for
                this module's own abstention structured-output call.
            identity: The exact judge configuration, exposed via .identity
                for the run manifest (ADR-0018).
            cache: Optional LLMCache (ADR-0004). None (the default) disables
                caching - every evaluate() call reaches the real metrics.
            max_in_flight: Bounds concurrent evaluate() calls to this
                provider, process-wide (ADR-0014).
        """
        self._faithfulness = faithfulness
        self._answer_relevancy = answer_relevancy
        self._abstention_llm = abstention_llm
        self._identity = identity
        self._cache = cache
        self._limiter = get_limiter(identity.provider, max_in_flight)

    @property
    def identity(self) -> ModelIdentity:
        """Exact judge configuration used by evaluate() - recorded in the run manifest."""
        return self._identity

    def evaluate(self, sample: JudgeSample) -> JudgeResult:
        """Return the structured judge result for ``sample``.

        Raises:
            GenerationError: If any underlying LLM/embedding call fails.
        """
        key = cache_key(
            provider=self._identity.provider,
            model=self._identity.model,
            reasoning_effort=self._identity.reasoning_effort,
            output_schema_version=self._identity.output_schema_version,
            abstention_prompt_version=ABSTENTION_PROMPT_VERSION,
            question=sample.question,
            contexts=sample.contexts,
            answer=sample.answer,
            unanswerable=sample.unanswerable,
        )
        return cached_call(
            self._cache,
            key,
            lambda: self._evaluate_uncached(sample),
            serialize=_serialize_result,
            deserialize=_deserialize_result,
        )

    def _evaluate_uncached(self, sample: JudgeSample) -> JudgeResult:
        try:
            with self._limiter:
                faithfulness_result = self._faithfulness.score(
                    user_input=sample.question,
                    response=sample.answer,
                    retrieved_contexts=list(sample.contexts),
                )
                answer_relevancy_result = self._answer_relevancy.score(
                    user_input=sample.question, response=sample.answer
                )
                abstention = self._abstention_llm.generate(
                    _ABSTENTION_PROMPT_TEMPLATE.format(
                        question=sample.question,
                        answerable="não" if sample.unanswerable else "sim",
                        answer=sample.answer,
                    ),
                    response_model=_AbstentionOutput,
                )
        except Exception as exc:
            raise GenerationError(f"RAGAS judge scoring failed: {exc}") from exc

        return JudgeResult(
            schema_version=_OUTPUT_SCHEMA_VERSION,
            faithfulness=MetricScore(score=float(faithfulness_result.value)),  # type: ignore[attr-defined]
            answer_relevancy=MetricScore(score=float(answer_relevancy_result.value)),  # type: ignore[attr-defined]
            abstention=AbstentionJudgment(
                appropriate=abstention.appropriate, rationale=abstention.rationale
            ),
        )


def _serialize_result(result: JudgeResult) -> str:
    return json.dumps(
        {
            "schema_version": result.schema_version,
            "faithfulness": result.faithfulness.score,
            "answer_relevancy": result.answer_relevancy.score,
            "abstention_appropriate": result.abstention.appropriate,
            "abstention_rationale": result.abstention.rationale,
        }
    )


def _deserialize_result(raw: str) -> JudgeResult:
    payload = json.loads(raw)
    return JudgeResult(
        schema_version=payload["schema_version"],
        faithfulness=MetricScore(score=payload["faithfulness"]),
        answer_relevancy=MetricScore(score=payload["answer_relevancy"]),
        abstention=AbstentionJudgment(
            appropriate=payload["abstention_appropriate"], rationale=payload["abstention_rationale"]
        ),
    )


def build_gemini_ragas_judge(
    llm_model_name: str,
    embedding_model_name: str,
    api_key: str | None = None,
    cache: LLMCache | None = None,
    max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
) -> RagasJudge:
    """Construct a RagasJudge backed by real Gemini models via ragas + instructor.

    A Gemini judge is a development fallback (ADR-0018), not the canonical
    choice - callers should label runs using it (e.g. "judge_label":
    "exploratory_same_provider_judge" in run.py) since the answer generator
    is also Gemini-based.

    Args:
        llm_model_name: The Gemini generation model id used as the judge,
            e.g. "gemini-3.1-flash-lite".
        embedding_model_name: The Gemini embedding model id for Answer
            Relevancy, e.g. "gemini-embedding-001".
        api_key: Overrides GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
        cache: Optional LLMCache (ADR-0004), forwarded to the RagasJudge.
        max_in_flight: Bounds concurrent evaluate() calls to this provider,
            process-wide (ADR-0014).

    Raises:
        GenerationError: If no API key is available or client construction fails.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GenerationError(
            "no Gemini API key found: set GEMINI_API_KEY or GOOGLE_API_KEY, "
            "or pass api_key explicitly"
        )
    try:
        instructor_client = instructor.from_provider(
            f"google/{llm_model_name}", async_client=True, api_key=key
        )
        genai_client = genai.Client(api_key=key)
    except Exception as exc:
        raise GenerationError(f"failed to create RAGAS judge client: {exc}") from exc

    ragas_llm = InstructorLLM(client=instructor_client, model=llm_model_name, provider="google")
    abstention_llm = InstructorLLM(
        client=instructor_client,
        model=llm_model_name,
        provider="google",
        system_prompt=_ABSTENTION_SYSTEM_PROMPT,
    )
    ragas_embeddings = GoogleEmbeddings(client=genai_client, model=embedding_model_name)

    return RagasJudge(
        faithfulness=Faithfulness(llm=ragas_llm),
        answer_relevancy=AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        abstention_llm=abstention_llm,
        identity=ModelIdentity(
            provider="gemini",
            model=llm_model_name,
            reasoning_effort=None,
            output_schema_version=_OUTPUT_SCHEMA_VERSION,
        ),
        cache=cache,
        max_in_flight=max_in_flight,
    )


def build_openai_ragas_judge(
    llm_model_name: str,
    embedding_model_name: str,
    reasoning_effort: str = "medium",
    api_key: str | None = None,
    cache: LLMCache | None = None,
    max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
) -> RagasJudge:
    """Construct a RagasJudge backed by real OpenAI models via ragas + instructor (ADR-0018).

    This is the canonical judge for publishable RAGForge results: independent
    from the Gemini answer generator, with a dated model snapshot (e.g.
    "gpt-5.4-mini-2026-03-17", never the floating "gpt-5.4-mini" alias) and
    Structured Outputs support.

    Args:
        llm_model_name: The dated OpenAI generation model snapshot used as
            the judge.
        embedding_model_name: The OpenAI embedding model id for Answer
            Relevancy, e.g. "text-embedding-3-small".
        reasoning_effort: Threaded straight into every underlying
            chat.completions.create() call (ragas.llms.InstructorLLM passes
            unknown kwargs through) - ADR-0018's "medium" default.
        api_key: Overrides OPENAI_API_KEY from the environment.
        cache: Optional LLMCache (ADR-0004), forwarded to the RagasJudge.
        max_in_flight: Bounds concurrent evaluate() calls to this provider,
            process-wide (ADR-0014).

    Raises:
        GenerationError: If no API key is available or client construction fails.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise GenerationError(
            "no OpenAI API key found: set OPENAI_API_KEY, or pass api_key explicitly"
        )
    try:
        instructor_client = instructor.from_provider(
            f"openai/{llm_model_name}", async_client=True, api_key=key
        )
        openai_client = OpenAI(api_key=key)
    except Exception as exc:
        raise GenerationError(f"failed to create RAGAS judge client: {exc}") from exc

    ragas_llm = InstructorLLM(
        client=instructor_client,
        model=llm_model_name,
        provider="openai",
        reasoning_effort=reasoning_effort,
    )
    abstention_llm = InstructorLLM(
        client=instructor_client,
        model=llm_model_name,
        provider="openai",
        reasoning_effort=reasoning_effort,
        system_prompt=_ABSTENTION_SYSTEM_PROMPT,
    )
    ragas_embeddings = OpenAIEmbeddings(client=openai_client, model=embedding_model_name)

    return RagasJudge(
        faithfulness=Faithfulness(llm=ragas_llm),
        answer_relevancy=AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        abstention_llm=abstention_llm,
        identity=ModelIdentity(
            provider="openai",
            model=llm_model_name,
            reasoning_effort=reasoning_effort,
            output_schema_version=_OUTPUT_SCHEMA_VERSION,
        ),
        cache=cache,
        max_in_flight=max_in_flight,
    )
