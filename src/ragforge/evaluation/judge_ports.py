"""Provider-neutral answer-quality judge contract (ADR-0018).

Decouples ``evaluate_answer_quality`` (answer_harness.py) from any specific
judge implementation - RAGAS, OpenAI, Gemini - so the harness depends only on
this Protocol and its schemas, never on ragas/instructor/openai/google-genai
directly (ADR-0009's adapter boundary).

``JudgeResult`` deliberately carries only the fields this project can
actually produce: RAGAS's off-the-shelf Faithfulness/AnswerRelevancy metric
classes expose a bare scalar (``MetricResult.value``), not the structured
``unsupported_claims``/``rationale`` breakdown ADR-0018's illustrative JSON
schema shows - reproducing that exact shape would mean abandoning RAGAS's
own metric classes for hand-written judge prompts, a separate, much larger
effort this increment does not take on. ``AbstentionJudgment.rationale`` is
the one exception: abstention has no native RAGAS metric at all, so it is
scored by this project's own structured-output call and genuinely has a
rationale to report.

``JudgeSample`` has no ``reference_answer`` field: the golden set
(datasets/regrag-br/judgments.json) carries no reference answers, a
golden-set curation gap already documented in ragas_judge.py (ADR-0007) and
unchanged here - Factual Correctness, which needs one, stays out of scope.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Exact identity of one judge configuration (ADR-0018): provider, model, generation params."""

    provider: str
    model: str
    reasoning_effort: str | None
    output_schema_version: int


@dataclass(frozen=True, slots=True)
class JudgeSample:
    """One (question, contexts, answer) triple to be scored, plus its answerability label."""

    question: str
    contexts: tuple[str, ...]
    answer: str
    query_class: str | None
    unanswerable: bool


@dataclass(frozen=True, slots=True)
class MetricScore:
    """A single scalar RAGAS metric result."""

    score: float


@dataclass(frozen=True, slots=True)
class AbstentionJudgment:
    """Whether abstaining (or not) was the appropriate call, given the sample's answerability."""

    appropriate: bool
    rationale: str


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """Structured judge output for one sample (ADR-0018)."""

    schema_version: int
    faithfulness: MetricScore
    answer_relevancy: MetricScore
    abstention: AbstentionJudgment


@runtime_checkable
class AnswerQualityJudge(Protocol):
    """Scores one JudgeSample and reports the exact model identity that produced the score."""

    @property
    def identity(self) -> ModelIdentity:
        """Exact judge configuration used by evaluate() - recorded in the run manifest."""
        ...

    def evaluate(self, sample: JudgeSample) -> JudgeResult:
        """Return the structured judge result for ``sample``."""
        ...
