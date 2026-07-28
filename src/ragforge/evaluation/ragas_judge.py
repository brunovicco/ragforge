"""RAGAS-based answer quality judge (ADR-0007/ADR-0018): Faithfulness, Answer Relevancy, abstention.

Wraps ragas.metrics.collections' Faithfulness and AnswerRelevancy over a
Gemini or OpenAI judge model via instructor (ragas 0.4's structured-output
backend). Abstention has no native RAGAS metric and is scored by this
module's own structured-output call, through a second, dedicated
InstructorLLM so its PT-BR prompt never contaminates RAGAS's internal ones.

Verified quirks of ragas 0.4.3 this adapter works around: instructor must be
built with ``async_client=True``; InstructorLLM forwards unknown kwargs (e.g.
``reasoning_effort``) but mis-maps decimal model snapshots, so the OpenAI
path uses Instructor's Responses API transport with explicit token/reasoning
parameters; MetricResult exposes only a scalar ``.value``, so JudgeResult
carries no per-claim rationale (see judge_ports.py).

Out of scope here: RAGAS Factual Correctness (not wired in this increment)
and human calibration (judge_calibration.py) - judge scores stay unvalidated
until the ADR-0007 kappa exercise runs; report them with that caveat.

RagasJudge bypasses ragforge.adapters.gemini_retry (ragas/instructor own the
client); an optional LLMCache (ADR-0004) and ProviderLimiter (ADR-0014) wrap
``evaluate()`` as a whole instead.
"""

import asyncio
import json
import math
import os
from typing import Protocol, cast, runtime_checkable

import instructor
from google import genai
from openai import AsyncOpenAI
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
_METRIC_SCORE_ATTEMPTS = 2
_OUTPUT_SCHEMA_VERSION = 2
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


class _OpenAIResponsesInstructorLLM(InstructorLLM):
    """Map RAGAS model arguments to the OpenAI Responses API contract.

    RAGAS still models OpenAI arguments as Chat Completions parameters. The
    Instructor client beneath this class is configured with
    ``RESPONSES_TOOLS``, so token and reasoning fields must use the Responses
    API names before the patched client forwards them.
    """

    def _map_openai_params(self) -> dict[str, object]:
        mapped = super()._map_openai_params()
        corrected: dict[str, object] = dict(mapped)
        token_limit = corrected.pop(
            "max_completion_tokens",
            corrected.pop("max_tokens", None),
        )
        if token_limit is not None:
            corrected["max_output_tokens"] = token_limit
        reasoning_effort = corrected.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            corrected["reasoning"] = {"effort": reasoning_effort}
        corrected.pop("temperature", None)
        corrected.pop("top_p", None)
        return corrected


def _build_async_openai_embeddings(
    api_key: str,
    model: str,
) -> OpenAIEmbeddings:
    """Build the async embedding adapter required by AnswerRelevancy.ascore()."""
    return OpenAIEmbeddings(client=AsyncOpenAI(api_key=api_key), model=model)


@runtime_checkable
class _ScoredMetric(Protocol):
    """Shape shared by ragas.metrics.collections' single-turn metric classes."""

    async def ascore(self, **kwargs: object) -> object: ...


class _MetricResult(Protocol):
    """Minimal result shape exposed by RAGAS collection metrics."""

    value: float


@runtime_checkable
class _GeneratingLLM(Protocol):
    """Shape of ragas.llms.InstructorLLM this module actually calls directly (abstention)."""

    async def agenerate(
        self, prompt: str, response_model: type[_AbstentionOutput]
    ) -> _AbstentionOutput: ...


class _AsyncCloseable(Protocol):
    """Minimal asynchronous resource lifecycle used by provider clients."""

    async def close(self) -> None: ...


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
        closeables: tuple[_AsyncCloseable, ...] = (),
    ) -> None:
        """Wire the judge to its already-constructed RAGAS metrics and abstention LLM.

        ``identity`` feeds the run manifest (ADR-0018); ``cache=None``
        disables ADR-0004 caching; ``max_in_flight`` bounds concurrent
        evaluate() calls (ADR-0014); ``closeables`` are provider clients to
        close on this judge's event loop before shutdown.
        """
        self._faithfulness = faithfulness
        self._answer_relevancy = answer_relevancy
        self._abstention_llm = abstention_llm
        self._identity = identity
        self._cache = cache
        self._limiter = get_limiter(identity.provider, max_in_flight)
        self._closeables = closeables
        self._runner = asyncio.Runner()
        self._closed = False

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
            max_output_tokens=self._identity.max_output_tokens,
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
        if self._closed:
            raise GenerationError("RAGAS judge is closed")
        try:
            with self._limiter:
                return self._runner.run(self._evaluate_async(sample))
        except Exception as exc:
            raise GenerationError(f"RAGAS judge scoring failed: {exc}") from exc

    async def _evaluate_async(self, sample: JudgeSample) -> JudgeResult:
        """Score all dimensions on one persistent event loop owned by this worker."""
        faithfulness_score = await _score_metric(
            self._faithfulness,
            "faithfulness",
            user_input=sample.question,
            response=sample.answer,
            retrieved_contexts=list(sample.contexts),
        )
        answer_relevancy_score = await _score_metric(
            self._answer_relevancy,
            "answer_relevancy",
            user_input=sample.question,
            response=sample.answer,
        )
        abstention = await self._abstention_llm.agenerate(
            _ABSTENTION_PROMPT_TEMPLATE.format(
                question=sample.question,
                answerable="não" if sample.unanswerable else "sim",
                answer=sample.answer,
            ),
            response_model=_AbstentionOutput,
        )
        return JudgeResult(
            schema_version=_OUTPUT_SCHEMA_VERSION,
            faithfulness=MetricScore(score=faithfulness_score),
            answer_relevancy=MetricScore(score=answer_relevancy_score),
            abstention=AbstentionJudgment(
                appropriate=abstention.appropriate, rationale=abstention.rationale
            ),
        )

    def close(self) -> None:
        """Close provider clients on their owning loop, then release the loop."""
        if self._closed:
            return
        try:
            for closeable in self._closeables:
                self._runner.run(closeable.close())
        finally:
            self._runner.close()
            self._closed = True


async def _score_metric(metric: _ScoredMetric, name: str, **kwargs: object) -> float:
    """Return one valid bounded score, retrying a semantically invalid result once."""
    invalid_value: float | None = None
    for _attempt in range(_METRIC_SCORE_ATTEMPTS):
        result = cast(_MetricResult, await metric.ascore(**kwargs))
        value = float(result.value)
        if math.isfinite(value) and 0.0 <= value <= 1.0:
            return value
        invalid_value = value
    raise ValueError(
        f"{name} returned invalid score {invalid_value!r} after {_METRIC_SCORE_ATTEMPTS} attempts"
    )


def _serialize_result(result: JudgeResult) -> str:
    return json.dumps(
        {
            "schema_version": result.schema_version,
            "faithfulness": result.faithfulness.score,
            "answer_relevancy": result.answer_relevancy.score,
            "abstention_appropriate": result.abstention.appropriate,
            "abstention_rationale": result.abstention.rationale,
        },
        allow_nan=False,
    )


def _deserialize_result(raw: str) -> JudgeResult:
    payload = json.loads(raw)
    return JudgeResult(
        schema_version=payload["schema_version"],
        faithfulness=MetricScore(score=float(payload["faithfulness"])),
        answer_relevancy=MetricScore(score=float(payload["answer_relevancy"])),
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

    A development fallback, not the canonical judge (ADR-0018) - callers
    should label runs using it, since the answer generator is also Gemini.
    ``cache``/``max_in_flight`` wire ADR-0004 caching and ADR-0014 bounds.

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
        faithfulness=cast(_ScoredMetric, Faithfulness(llm=ragas_llm)),
        answer_relevancy=cast(
            _ScoredMetric,
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ),
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
    max_output_tokens: int = 8192,
) -> RagasJudge:
    """Construct a RagasJudge backed by real OpenAI models via ragas + instructor (ADR-0018).

    The canonical judge for publishable results: independent from the Gemini
    answer generator, pinned to a dated model snapshot (never a floating
    alias). ``reasoning_effort`` is forwarded to every underlying call;
    ``max_output_tokens`` budgets hidden reasoning plus structured output
    (RAGAS's 1024 default is too small for long NLI statement lists).
    ``cache``/``max_in_flight`` wire ADR-0004 caching and ADR-0014 bounds.

    Raises:
        GenerationError: If no API key is available or client construction fails.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise GenerationError(
            "no OpenAI API key found: set OPENAI_API_KEY, or pass api_key explicitly"
        )
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    try:
        openai_client = AsyncOpenAI(api_key=key)
        instructor_client = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.RESPONSES_TOOLS,
        )
        embeddings_client = AsyncOpenAI(api_key=key)
        ragas_embeddings = OpenAIEmbeddings(
            client=embeddings_client,
            model=embedding_model_name,
        )
    except Exception as exc:
        raise GenerationError(f"failed to create RAGAS judge client: {exc}") from exc

    ragas_llm = _OpenAIResponsesInstructorLLM(
        client=instructor_client,
        model=llm_model_name,
        provider="openai",
        reasoning_effort=reasoning_effort,
        max_tokens=max_output_tokens,
    )
    abstention_llm = _OpenAIResponsesInstructorLLM(
        client=instructor_client,
        model=llm_model_name,
        provider="openai",
        reasoning_effort=reasoning_effort,
        max_tokens=max_output_tokens,
        system_prompt=_ABSTENTION_SYSTEM_PROMPT,
    )
    return RagasJudge(
        faithfulness=cast(_ScoredMetric, Faithfulness(llm=ragas_llm)),
        answer_relevancy=cast(
            _ScoredMetric,
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ),
        abstention_llm=abstention_llm,
        identity=ModelIdentity(
            provider="openai",
            model=llm_model_name,
            reasoning_effort=reasoning_effort,
            output_schema_version=_OUTPUT_SCHEMA_VERSION,
            max_output_tokens=max_output_tokens,
        ),
        cache=cache,
        max_in_flight=max_in_flight,
        closeables=(openai_client, embeddings_client),
    )
