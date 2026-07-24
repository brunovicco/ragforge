# LLM judge research note - 2026-07-24

Canonical choice:

```yaml
provider: openai
model: gpt-5.4-mini-2026-03-17
reasoning_effort: medium
```

Reasons:

- independent from the Gemini answer generator;
- dated snapshot for reproducibility;
- strict Structured Outputs;
- cost suitable for a large evaluation matrix;
- RAGAS supports OpenAI, Anthropic, Google, local, and OpenAI-compatible endpoints;
- human alignment remains mandatory.

Current official pricing observed on 2026-07-24:

- input: USD 0.75 per million tokens;
- cached input: USD 0.075 per million tokens;
- output: USD 4.50 per million tokens.

Provider-free experiment:

- local `gpt-oss-20b` or another open-weight model;
- only after exact weight/runtime pinning and human calibration;
- not canonical merely because it runs locally.

Sources consulted:

- https://developers.openai.com/api/docs/models/gpt-5.4-mini
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://docs.ragas.io/en/stable/howtos/customizations/customize_models/
- https://docs.ragas.io/en/v0.3.8/howtos/applications/align-llm-as-judge/
