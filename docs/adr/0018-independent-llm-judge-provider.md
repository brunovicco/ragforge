# ADR-0018: Use an independent and calibrated OpenAI LLM judge

- Status: Accepted
- Date: 2026-07-24
- Target: Current release
- Related: ADR-0004, ADR-0007, ADR-0012, ADR-0014, ADR-0016, ADR-0017

## Context

RAGForge currently generates answers through a Gemini adapter. Using the same provider family for answer generation and benchmark judging increases the risk of correlated preferences and shared failure modes.

The judge must evaluate Brazilian Portuguese legal-regulatory answers and support:

- schema-constrained output;
- an immutable model snapshot;
- bounded parallel calls;
- RAGAS integration;
- versioned prompts;
- caching;
- cost-compatible evaluation of the complete strategy matrix;
- calibration against human reviewers.

No LLM provider should be assumed reliable for this domain without calibration. Normative negation, exceptions, cross-references, partial support, and version-sensitive language can mislead an otherwise capable general-purpose judge.

## Decision drivers

- Independence from the Gemini answer generator.
- Exact model versioning.
- Structured outputs.
- Cost suitable for repeated full-dataset evaluation.
- Direct integration with RAGAS or a provider-neutral wrapper.
- Human alignment in PT-BR.
- No silent provider fallback.
- Support for parallel, resumable execution.

## Decision

The canonical LLM judge for publishable RAGForge results SHALL use OpenAI with this snapshot:

```yaml
judge:
  provider: openai
  model: gpt-5.4-mini-2026-03-17
  reasoning_effort: medium
  cache_mode: required
  max_retries: 3
  output_schema_version: 1
```

The floating alias `gpt-5.4-mini` SHALL NOT be used in a publishable run.

This selection is provisional until it passes the ADR-0007 human-calibration gate. Failure to meet the gate does not permit silent replacement; it triggers an explicit judge-comparison experiment.

## Rationale

OpenAI GPT-5.4 mini is selected because:

1. the answer generator currently uses Gemini, creating provider independence;
2. a dated snapshot is available;
3. Structured Outputs are supported;
4. the model targets high-volume, cost-sensitive workloads;
5. RAGAS supports OpenAI and structured-output integrations;
6. the price is suitable for hundreds or thousands of judge calls;
7. a stable snapshot plus cache provides reproducible report regeneration.

This is not a claim that OpenAI is universally the best legal judge. It is a controlled methodological choice that must be validated.

## Judge abstraction

RAGForge SHALL own a provider-neutral contract:

```python
class AnswerQualityJudge(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    def evaluate(self, sample: JudgeSample) -> JudgeResult: ...
```

RAGAS classes SHALL remain in an adapter layer. Domain and application code SHALL not depend directly on RAGAS or an OpenAI SDK.

## Judge input

A judge sample SHALL explicitly distinguish:

```text
question
retrieved authoritative contexts
candidate answer
reference answer, when available
structural judgments
query class
answerability label
```

The judge SHALL not retrieve external evidence or use web search. It evaluates only the provided experiment evidence.

## Structured output

The adapter SHALL require a strict schema.

Example:

```json
{
  "schema_version": 1,
  "faithfulness": {
    "score": 0.0,
    "unsupported_claims": [],
    "rationale": ""
  },
  "answer_relevancy": {
    "score": 0.0,
    "rationale": ""
  },
  "factual_correctness": {
    "score": 0.0,
    "missing_facts": [],
    "incorrect_facts": [],
    "rationale": ""
  },
  "abstention": {
    "appropriate": true,
    "rationale": ""
  }
}
```

Scores SHALL be bounded and validated before persistence.

If native RAGAS metrics require separate prompts, each call SHALL keep an independent prompt and cache key. Reports SHALL state whether dimensions were evaluated jointly or separately.

## Prompt policy

The judge system prompt SHALL:

- use Brazilian Portuguese evaluation criteria;
- distinguish faithful from merely plausible statements;
- interpret legal negations and exceptions conservatively;
- require evidence for every material assertion;
- avoid rewarding verbosity;
- treat partially supported answers separately from fully supported answers;
- evaluate abstention according to available evidence;
- avoid using outside knowledge.

Prompt templates and few-shot examples SHALL be versioned and hashed.

## Human calibration

ADR-0007 remains mandatory.

The calibration dataset SHALL contain at least 30 manually labeled samples and SHOULD grow to 50 before a public benchmark release.

Stratify by:

- all query classes;
- answerable and unanswerable questions;
- strong and weak strategies;
- complete and partial answers;
- unsupported citations;
- negation;
- exceptions;
- cross-references;
- numerical claims.

Measure:

- weighted Cohen's kappa for ordinal judgments;
- Spearman correlation for continuous scores;
- false-supported rate;
- false-unsupported rate;
- abstention agreement.

Minimum acceptance:

```text
weighted kappa >= 0.60
```

Recommended publication target:

```text
weighted kappa >= 0.70
```

If the threshold is not met:

1. inspect disagreement categories;
2. revise PT-BR legal criteria and few-shot examples;
3. rerun calibration;
4. compare a stronger model or independent provider;
5. retain human review as the final adjudicator;
6. publish the limitation if no candidate meets the target.

## Provider-free judge option

A provider-free judge MAY be implemented with a local open-weight model exposed through an OpenAI-compatible endpoint, such as `gpt-oss-20b`, subject to hardware availability.

It SHALL be classified as experimental until it independently satisfies:

- the same human-agreement threshold;
- stable structured-output validity;
- acceptable latency;
- exact weight and runtime pinning;
- recorded quantization and device configuration.

Local execution is not sufficient evidence of judge quality.

## Gemini development fallback

A Gemini judge MAY be configured for development when no OpenAI credential is available.

When the answer generator also uses Gemini, the run SHALL be labeled:

```text
exploratory_same_provider_judge
```

Such a run SHALL not replace the canonical published leaderboard without independent calibration and explicit discussion of provider correlation.

## No silent fallback

A publishable run SHALL use one judge identity.

On provider failure:

- retry using ADR-0014;
- resume later with the same provider and snapshot;
- fail the judge stage after terminal failure.

The runner SHALL not switch to Gemini, a local model, or another OpenAI model inside the same run.

## Concurrency

Judge calls are independent by question and strategy and MAY execute concurrently under ADR-0014.

The judge stage SHALL use:

- bounded worker count;
- per-provider in-flight semaphore;
- requests-per-minute and tokens-per-minute limits;
- `Retry-After` handling;
- deterministic output ordering;
- atomic result writes.

Recommended initial configuration:

```yaml
execution:
  stages:
    judge:
      workers: 4
      max_in_flight_per_provider: 4
```

Actual limits SHALL be resolved from the account tier and recorded.

## Cache identity

Cache keys SHALL include:

```text
question hash
+ context source hashes
+ candidate answer hash
+ reference answer hash
+ metric name and version
+ prompt hash
+ provider
+ model snapshot
+ reasoning effort
+ output schema version
```

Published report regeneration SHALL use cache-only mode.

## Relationship with the runtime auditor

The auditor and judge have different responsibilities:

- auditor: determines whether an answer may be delivered;
- judge: scores the final answer for benchmark analysis.

They MAY share provider infrastructure but SHALL use:

- distinct prompts;
- distinct schemas;
- distinct cache namespaces;
- distinct lineage records.

The final benchmark SHOULD report quality for both the original answer and the audited answer.

## Cost controls

The judge stage SHALL record:

- input tokens;
- output tokens;
- cached tokens where reported;
- estimated cost;
- retries;
- cache hits;
- calls by metric.

For offline experiments, a future implementation MAY evaluate provider Batch APIs, but the current synchronous runner uses bounded threaded calls to preserve straightforward stage progress and resume behavior.

## Consequences

### Positive

- Reduces generator/judge provider correlation.
- Uses a dated reproducible snapshot.
- Supports strict structured output.
- Keeps full-matrix evaluation economically practical.
- Works with the existing RAGAS direction.
- Preserves an experimental provider-free path.

### Negative

- Canonical judging requires a paid OpenAI API key.
- Calibration still requires human effort.
- Provider pricing and availability can change.
- A snapshot may eventually be deprecated.
- LLM judgment remains probabilistic even with caching and strict schemas.

## Alternatives considered

### Gemini as canonical judge

Rejected for the current architecture because Gemini also generates answers. Retained as a development fallback.

### GPT-5.6 Terra as canonical judge

Deferred. It may provide stronger judgment, but GPT-5.4 mini has a dated snapshot and lower cost. Terra is a candidate only if calibration demonstrates that mini is insufficient.

### Provider-free local judge as canonical immediately

Rejected. Local execution does not replace human-alignment evidence.

### Human-only evaluation

Rejected as the sole full-matrix method because it does not scale, but retained as calibration and final adjudication.

### Multi-judge majority vote

Deferred. It multiplies cost and complicates interpretation. Reconsider if no single calibrated judge is stable.

## Acceptance criteria

- [ ] Adapter uses `gpt-5.4-mini-2026-03-17`.
- [ ] Output is strict-schema validated.
- [ ] RAGAS is isolated behind a judge port.
- [ ] At least 30 PT-BR legal samples are manually calibrated.
- [ ] Kappa, correlation, and error rates are published.
- [ ] The full run uses one judge identity.
- [ ] No silent fallback exists.
- [ ] Calls are cached and concurrency bounded.
- [ ] Gemini fallback runs are labeled exploratory.
- [ ] Provider-free judge remains experimental until calibrated.
- [ ] Judge identity and prompt hashes appear in the run manifest.

## Rollout

1. Define judge port and domain schemas.
2. Implement OpenAI snapshot adapter.
3. Integrate RAGAS behind the adapter.
4. Build calibration sample.
5. Calibrate and tune PT-BR prompts.
6. Parallelize under ADR-0014.
7. Add cache-only report regeneration.
8. Add Gemini development fallback.
9. Add an experimental local judge.
10. Publish judge reliability with benchmark results.

## References

- OpenAI model catalog and GPT-5.4 mini model documentation.
- OpenAI Structured Outputs documentation.
- RAGAS model customization and judge-alignment documentation.
