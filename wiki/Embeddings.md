# Embeddings in RAGForge

[Português](Embeddings-pt-BR) · [Home](Home) · [Retrieval strategies](Retrieval-Strategies)

> Documentation snapshot: 2026-07-25. Model status comes from the repository
> configuration, not from a claim that every candidate has completed the current
> RegRAG-BR experiment.

## What an embedding is

An embedding is a numerical vector that represents the semantic content of an input.
RAGForge embeds each chunk's `retrieval_text` and stores the vector in PostgreSQL with
pgvector. At query time, the same model embeds the question, and pgvector orders chunks
by cosine distance.

This is a **bi-encoder** design: documents and queries are encoded separately. It scales
to a full corpus because document vectors are computed before query time. It is different
from the project's cross-encoder reranker, which reads each query–chunk pair jointly and
is therefore applied only to a small candidate pool.

```text
document retrieval_text ──embedding model──> vector ──┐
                                                      ├─ cosine similarity ─> ranking
question text ────────────embedding model──> vector ──┘
```

The project currently uses one `embed()` operation for both documents and queries. A
versioned, model-specific query instruction is planned but not implemented.

## Dense, sparse, and multimodal: different meanings

| Term | Meaning | RAGForge behavior |
|---|---|---|
| Dense embedding | A fixed-width floating-point vector in which meaning is distributed across dimensions | Implemented through Gemini or Sentence Transformers |
| Sparse representation | Mostly-zero term or feature weights | RAGForge does **not** generate sparse model embeddings; its sparse strategy is BM25 |
| Multimodal embedding | Different media share one vector space | `gemini-embedding-2` supports it, but RAGForge currently sends text only |
| Matryoshka/MRL embedding | A model trained so useful shorter vectors can be obtained by reducing output dimensions | Gemini is requested at 1,536 dimensions; local models currently use their reported width |

BM25 is often grouped with “sparse embeddings” in broad RAG discussions, but that would
be inaccurate for this implementation. The `sparse_bm25` strategy sends query text
directly to OpenSearch and performs no embedding call.

## Model inventory

### Retrieval embedding candidates

| Model | Runtime | Width used or declared | Repository status | How RAGForge uses it |
|---|---:|---:|---|---|
| `gemini-embedding-001` | Hosted Gemini API | 1,536 | Canonical selection, provisional pending revalidation | Text-only dense vectors; configured for the publishable matrix |
| `gemini-embedding-2` | Hosted Gemini API | 1,536 | Candidate pending revalidation | Text input only, despite the model's multimodal capability |
| `Qwen/Qwen3-Embedding-0.6B` | Local Sentence Transformers | 1,024 | Operational local default; isolated experiment not yet evaluated | Credential-free dense vectors for `make bench-live-local` |
| `BAAI/bge-m3` | Local Sentence Transformers | 1,024 | Candidate pending revalidation | Dense output only; BGE-M3's sparse and ColBERT-style outputs are not used |
| `intfloat/multilingual-e5-large-instruct` | Local Sentence Transformers | 1,024 | Control candidate, not yet evaluated | Dense output without the model-specific query instruction |

### `gemini-embedding-001`

- The canonical benchmark requests 1,536 output dimensions. The model supports flexible
  output dimensions up to 3,072.
- The reduction keeps the vector below pgvector HNSW's 2,000-dimension limit for the
  `vector` type.
- Calls are hosted, metered, and require `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- The adapter batches up to 100 texts, retries transient failures, limits concurrent
  provider calls, and can cache vectors per text.
- The adapter records `normalize=false`; pgvector still compares vectors with cosine
  distance.

The repository's isolated PT-BR comparison selected this model for the canonical matrix,
but the current experiment file marks that selection as provisional pending revalidation.
It should not be presented as a permanent winner.

### `gemini-embedding-2`

- The provider describes it as a unified text, image, video, audio, and PDF embedding
  model. RAGForge's embedding port is text-only, so none of the non-text modalities are
  exercised.
- The project requests 1,536 dimensions, just as it does for
  `gemini-embedding-001`.
- A direct project probe found that a multi-text `embed_content` request returned one
  embedding instead of one per input. The adapter therefore sends one text per request
  for this model. This is slower and creates many more requests during indexing.
- It is a comparison candidate, not the canonical model.

### `Qwen/Qwen3-Embedding-0.6B`

- This is the provider-free operational alternative in
  `benchmark-local-v01.yaml`.
- It runs through `SentenceTransformerEmbedder`, reports 1,024 dimensions, and returns
  L2-normalized vectors.
- The model card describes it as multilingual, instruction-aware, and capable of
  Matryoshka dimension reduction. RAGForge currently uses the full reported width and
  does not send a query instruction.
- The configuration does not pin an immutable Hugging Face revision. The adapter records
  the unresolved revision as `main`, which is adequate for exploration but weaker than
  the reproducibility target in ADR-0013.
- “Local” only applies to the embedding stage. The full benchmark still calls Gemini for
  contextualization, summarization, GraphRAG extraction, and answer generation, and
  OpenAI for the canonical judge.

### `BAAI/bge-m3`

- The model can produce dense, learned sparse, and ColBERT-style multi-vector
  representations. The RAGForge adapter calls Sentence Transformers' standard
  `encode()`, so only the 1,024-dimensional dense representation participates in the
  comparison.
- Vectors are L2-normalized before indexing.
- The repository records CPU as the stable execution path on the current development
  machine. An MPS run exhausted memory; that is a machine-specific observation, not a
  general model restriction.
- Its PT-BR comparison result must be revalidated.

### `intfloat/multilingual-e5-large-instruct`

- This is the local control candidate.
- Its model card requires a task instruction on the query and warns of degraded
  performance without one.
- RAGForge currently embeds query and document text through the same `embed()` method and
  records an empty query-instruction hash. A score produced today would therefore measure
  the project's incomplete integration as well as the model.
- It is declared in the experiment matrix but has not yet been evaluated there.

### `text-embedding-3-small`: evaluation only

`text-embedding-3-small` is configured for RAGAS Answer Relevancy in the canonical
OpenAI judge. It does **not** create the retrieval index, retrieve chunks, or participate
in the Dense/Hybrid embedding comparison.

Keeping this distinction matters:

```text
retrieval embedding -> finds evidence
judge embedding     -> helps score whether the generated answer addresses the question
```

The Gemini judge fallback similarly uses `gemini-embedding-001` for Answer Relevancy,
but that fallback is labeled exploratory because the answer generator is also Gemini.

## Normalization, dimensions, and similarity

The local adapter calls:

```python
model.encode(..., normalize_embeddings=True)
```

For unit-normalized vectors, dot product and cosine similarity produce the same ordering.
The Gemini adapter does not normalize client-side. The storage layer consistently uses
pgvector's cosine operator (`vector_cosine_ops` and `<=>`), which accounts for vector
magnitudes when comparing them.

Dimensions are part of the vector-space contract:

- the pgvector column width is fixed when the table is created;
- equal widths do not make two models compatible;
- changing model, revision, dimensions, normalization, or instruction requires a
  different index;
- Gemini's 3,072 native width is reduced to 1,536 because pgvector HNSW indexes over the
  `vector` type support up to 2,000 dimensions.

## Identity, cache, and index isolation

RAGForge identifies an embedding space with:

```text
provider
+ model
+ revision
+ dimensions
+ normalization
+ query-instruction hash
+ runtime
```

The index namespace also includes the corpus hash, chunking configuration, and retrieval
text schema. A separate index fingerprint hashes every chunk's source text, retrieval
text, structural IDs, metadata, and the synthetic-text producer identity.

Consequences:

- vectors are never reused merely because two models have the same width;
- cached embeddings are keyed by complete identity and the input-text hash;
- contextual, SAC, RAPTOR, and base indexes are isolated;
- partial indexes do not receive a reusable completion marker;
- a provider failure does not silently switch the run to another embedding model.

Current limitation: local model revisions are not pinned by the YAML configuration, and
all query instructions currently share the hash of an empty string. Those gaps must be
closed before claiming exact cross-machine reproducibility for instruction-aware models.

## Configuration

Canonical hosted embedding:

```yaml
embedding:
  provider: gemini
  model: gemini-embedding-001
  dimensions: 1536
```

Local embedding alternative:

```yaml
embedding:
  provider: local
  model: Qwen/Qwen3-Embedding-0.6B
  dimensions: 1024
  device: cpu
```

The runner supports only `local` and `gemini` retrieval providers. There is no automatic
fallback between them.

## Choosing and comparing a model

RAGForge's comparison protocol holds corpus, split, chunking, retrieval text, top-k,
judgments, and metrics constant. It varies the embedding and measures Dense and
Hybrid+RRF, the strategies most directly affected by vector quality.

Run one candidate with:

```bash
uv run python configs/experiments/run_embeddings_ptbr.py \
  --model Qwen/Qwen3-Embedding-0.6B
```

Hosted example:

```bash
GEMINI_API_KEY=... uv run python \
  configs/experiments/run_embeddings_ptbr.py \
  --model gemini-embedding-001
```

The comparison reports Recall@k, Precision@k, nDCG@k, and MRR. Before promoting a new
winner, also record immutable model revisions, the effective query instruction, device,
precision, throughput, latency, memory, index size, and provider cost.

## Privacy and operational impact

- Local embedding keeps chunk and query text on the machine for the embedding stage.
- Hosted Gemini embedding sends retrieval text and questions to an external provider.
- The current corpus contains public official acts, but this does not imply authorization
  to send future private documents externally.
- A private-data run needs an explicit data-processing decision; the repository's general
  privacy inventory is not yet complete.
- Neither API keys nor content should be logged. Credentials are read from environment
  variables.

## Sources

### RAGForge

- [Embedding port and adapters](https://github.com/brunovicco/ragforge/tree/main/src/ragforge/embeddings)
- [ADR-0005: embedding comparison scope](https://github.com/brunovicco/ragforge/blob/main/docs/adr/0005-embedding-comparison-scope.md)
- [ADR-0013: provider-neutral backends](https://github.com/brunovicco/ragforge/blob/main/docs/adr/0013-provider-neutral-embedding-backends.md)
- [PT-BR experiment configuration](https://github.com/brunovicco/ragforge/blob/main/configs/experiments/embeddings-ptbr.yaml)

### Primary model and storage documentation

- [Google: `gemini-embedding-001`](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-001)
- [Google: `gemini-embedding-2`](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2)
- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [Multilingual E5 Large Instruct model card](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- [OpenAI: `text-embedding-3-small`](https://developers.openai.com/api/docs/models/text-embedding-3-small)
- [pgvector: vector types, dimensions, and HNSW](https://github.com/pgvector/pgvector)
