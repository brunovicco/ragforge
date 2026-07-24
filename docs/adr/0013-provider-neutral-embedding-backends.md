# ADR-0013: Adopt provider-neutral embedding backends with a local default

- Status: Accepted
- Date: 2026-07-24
- Target: Current release
- Related: ADR-0001, ADR-0004, ADR-0005, ADR-0012

## Context

A benchmark that requires a hosted embedding API is harder to reproduce, may transmit source content to a third party, and can become unavailable when quotas or aliases change.

A free-tier API is not provider-free:

- free tier still depends on a vendor, network access, quotas, and provider terms;
- provider-free execution uses local pinned weights and requires no API key.

RAGForge should demonstrate both a reproducible local baseline and optional hosted comparisons.

## Decision drivers

- Execution without credentials.
- Provider neutrality in the domain and application layers.
- Privacy and data-control clarity.
- Exact model and index identity.
- Controlled comparison across embedding families.
- Compatibility with Dense, Hybrid, Contextual Retrieval, and SAC.

## Decision

RAGForge SHALL own a provider-neutral embedding port. The operational default SHALL be local. Hosted providers SHALL be optional adapters and SHALL never be selected through a silent fallback.

### Embedding port

```python
from collections.abc import Sequence
from typing import Protocol

class Embedder(Protocol):
    @property
    def identity(self) -> "EmbeddingIdentity": ...

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
```

```python
@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    provider: str
    model: str
    revision: str
    dimensions: int
    normalize: bool
    query_instruction_hash: str | None
    runtime: str
```

Provider SDK types SHALL remain in adapters.

### Current model matrix

| Role | Model | Execution |
|---|---|---|
| Local operational default | `Qwen/Qwen3-Embedding-0.6B` | Local |
| Local comparator | `BAAI/bge-m3` | Local |
| Local control | `intfloat/multilingual-e5-large-instruct` | Local |
| Hosted comparator | configured stable Gemini embedding model | Gemini API |

The local default is not a predeclared quality winner. The full RegRAG-BR experiment determines relative quality.

### Exact model revision

A publishable run SHALL record:

- model ID;
- immutable model revision;
- tokenizer revision;
- dimensions;
- normalization;
- truncation;
- query instruction and hash;
- package/runtime versions;
- device;
- numerical precision;
- effective batch size.

Unpinned remote aliases make a run exploratory.

### Index isolation

Each vector index SHALL be namespaced by:

```text
corpus_hash
+ chunking_config_hash
+ retrieval_text_schema_version
+ provider
+ model
+ revision
+ dimensions
+ normalization
+ query_instruction_hash
```

An index SHALL never be reused after any identity component changes.

### No silent fallback

If a provider fails, the run SHALL fail or resume later with the same provider. It SHALL NOT switch embedding spaces during a run.

### Query instructions

Instruction-aware models SHALL receive versioned query instructions. The instruction SHALL be included in the run and cache identity.

Example:

```text
Retrieve provisions from Brazilian financial regulations that provide
authoritative evidence for the question.
```

Document text SHALL not receive the query instruction unless required by the model contract.

### Batching and concurrency

Local embedding inference SHALL use model-native batching. It SHALL NOT create one Python thread per chunk.

Hosted embedding batches MAY run concurrently under ADR-0014 with bounded in-flight requests.

### Privacy modes

Configuration SHALL declare:

```yaml
data_policy:
  public_corpus: true
  external_embedding_allowed: true
```

External embedding of private content requires explicit opt-in.

### Benchmark protocol

Keep constant:

- corpus;
- split;
- source and retrieval text;
- chunking;
- strategy;
- candidate depth;
- top-k;
- judgments;
- metrics.

Vary only the embedding model and its required instruction.

Report:

- Recall@k;
- nDCG@k;
- MRR;
- document precision;
- Document-Level Retrieval Mismatch;
- indexing throughput;
- query latency;
- peak memory;
- index size;
- provider cost where applicable.

## Consequences

### Positive

- Benchmark runs without vendor credentials.
- Local and hosted adapters share one contract.
- Incompatible indexes cannot be accidentally reused.
- Privacy and cost behavior are explicit.

### Negative

- Local models require weight downloads.
- CPU-only runs may be slower.
- Multiple index namespaces require storage.
- Model-specific instruction handling adds configuration.

## Alternatives considered

- **Gemini only:** rejected because it prevents provider-free execution.
- **One local model only:** rejected because RAGForge is a comparative benchmark.
- **Automatic fastest-model selection:** rejected because hardware-dependent choice harms reproducibility.
- **Reuse by vector dimension:** rejected because equal dimensions do not imply compatible embedding spaces.

## Acceptance criteria

- [ ] Retrieval benchmark runs without API keys.
- [ ] Provider imports are isolated to adapters.
- [ ] Exact model identity appears in the run manifest.
- [ ] Every embedding configuration receives an isolated index.
- [ ] No provider fallback occurs inside a run.
- [ ] Local inference uses bounded batches.
- [ ] Hosted calls obey ADR-0014.
- [ ] Three local families and one hosted comparator are configurable.
- [ ] Results are reported by model and query class.

## Rollout

1. Introduce port and identity.
2. Wrap the current Gemini adapter.
3. Implement local adapter.
4. Add index namespace derivation.
5. Add model-specific instruction policies.
6. Run the full comparison.
