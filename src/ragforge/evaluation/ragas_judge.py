"""RAGAS-based answer quality judge (ADR-0007): Faithfulness and Answer Relevancy.

Wraps ragas.metrics.collections' Faithfulness and AnswerRelevancy over a
Gemini judge model, via instructor (ragas 0.4's structured-output backend,
already a ragas dependency) and GoogleEmbeddings (ragas's native Gemini
embeddings wrapper) - no new project dependency, reusing the existing
google-genai client this project already depends on.

Real, verified constraint (ragas 0.4.3, checked against the actual
library): instructor.from_provider(..., async_client=True) is required -
ragas's async ascore() path raises if the wrapped client is synchronous.

Two things this module deliberately does NOT do:

- Factual Correctness, the third metric ADR-0007 names, needs a reference
  answer per question to compare against. The current golden set
  (datasets/regrag-br/judgments.json) has no reference answers - a
  golden-set curation gap, not a code gap. Not implemented here.
- Judge calibration against human evaluation (ADR-0007: ~30 hand-scored
  triples, kappa >= 0.6) is human curation work this module cannot
  substitute for. Scores from this judge are unvalidated until that
  calibration happens - report them with that caveat, never as ground truth.

RagasJudge does not go through ragforge.adapters.gemini_retry: it calls
ragas/instructor's own internal async client, not our google-genai client
directly, so there is no shared retry choke point to hook into here. An
optional LLMCache (ADR-0004) and ProviderLimiter (ADR-0014) instead wrap
score() as a whole - the granularity available at this boundary.
"""

import json
import os
from typing import Protocol, runtime_checkable

import instructor
from google import genai
from ragas.embeddings import GoogleEmbeddings
from ragas.llms import InstructorLLM
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from ragforge.adapters.llm_cache import LLMCache, cache_key, cached_call
from ragforge.adapters.provider_limiter import get_limiter
from ragforge.generation.errors import GenerationError

_PROVIDER = "gemini"
# Conservative default - not data-driven, matches ADR-0014's own example config.
_DEFAULT_MAX_IN_FLIGHT = 4


@runtime_checkable
class _ScoredMetric(Protocol):
    """Shape shared by ragas.metrics.collections' single-turn metric classes."""

    def score(self, **kwargs: object) -> object: ...  # returns a ragas MetricResult (has .value)


class RagasJudge:
    """Scores a (question, contexts, answer) triple for Faithfulness and Answer Relevancy."""

    def __init__(
        self,
        faithfulness: _ScoredMetric,
        answer_relevancy: _ScoredMetric,
        model_name: str = "unknown",
        cache: LLMCache | None = None,
        max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
    ) -> None:
        """Wire the judge to its two already-constructed RAGAS metric instances.

        Args:
            faithfulness: An already-constructed RAGAS Faithfulness metric.
            answer_relevancy: An already-constructed RAGAS AnswerRelevancy metric.
            model_name: Judge model identity, used only for the cache key -
                defaults to "unknown" so existing callers that don't pass it
                still work, just without a meaningful cache key.
            cache: Optional LLMCache (ADR-0004). None (the default) disables
                caching - every score() call reaches the real metrics.
            max_in_flight: Bounds concurrent score() calls to this provider,
                process-wide (ADR-0014).
        """
        self._faithfulness = faithfulness
        self._answer_relevancy = answer_relevancy
        self._model_name = model_name
        self._cache = cache
        self._limiter = get_limiter(_PROVIDER, max_in_flight)

    def score(self, query_text: str, contexts: list[str], answer_text: str) -> dict[str, float]:
        """Return ``{"faithfulness": ..., "answer_relevancy": ...}`` for the given triple.

        Raises:
            GenerationError: If either metric's underlying LLM/embedding call fails.
        """
        key = cache_key(
            provider=_PROVIDER,
            model=self._model_name,
            query=query_text,
            contexts=contexts,
            answer=answer_text,
        )
        return cached_call(
            self._cache,
            key,
            lambda: self._score_uncached(query_text, contexts, answer_text),
            serialize=json.dumps,
            deserialize=json.loads,
        )

    def _score_uncached(
        self, query_text: str, contexts: list[str], answer_text: str
    ) -> dict[str, float]:
        try:
            with self._limiter:
                faithfulness_result = self._faithfulness.score(
                    user_input=query_text, response=answer_text, retrieved_contexts=contexts
                )
                answer_relevancy_result = self._answer_relevancy.score(
                    user_input=query_text, response=answer_text
                )
        except Exception as exc:
            raise GenerationError(f"RAGAS judge scoring failed: {exc}") from exc

        return {
            "faithfulness": float(faithfulness_result.value),  # type: ignore[attr-defined]
            "answer_relevancy": float(answer_relevancy_result.value),  # type: ignore[attr-defined]
        }


def build_gemini_ragas_judge(
    llm_model_name: str,
    embedding_model_name: str,
    api_key: str | None = None,
    cache: LLMCache | None = None,
    max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
) -> RagasJudge:
    """Construct a RagasJudge backed by real Gemini models via ragas + instructor.

    Args:
        llm_model_name: The Gemini generation model id used as the judge,
            e.g. "gemini-3.1-flash-lite".
        embedding_model_name: The Gemini embedding model id for Answer
            Relevancy, e.g. "gemini-embedding-001".
        api_key: Overrides GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
        cache: Optional LLMCache (ADR-0004), forwarded to the RagasJudge.
        max_in_flight: Bounds concurrent score() calls to this provider,
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
    ragas_embeddings = GoogleEmbeddings(client=genai_client, model=embedding_model_name)

    return RagasJudge(
        faithfulness=Faithfulness(llm=ragas_llm),
        answer_relevancy=AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        model_name=llm_model_name,
        cache=cache,
        max_in_flight=max_in_flight,
    )
