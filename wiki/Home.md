# RAGForge Wiki

> Documentation snapshot: 2026-07-25 · RAGForge v0.1 in development

[Português](#português) · [English](#english)

## Português

O RAGForge é uma plataforma experimental para comparar estratégias de
Retrieval-Augmented Generation (RAG) sobre documentos financeiros e regulatórios
brasileiros. Esta wiki explica o que o projeto **implementa hoje**, separando três
conceitos que costumam ser confundidos:

- **modelo de embedding**: transforma texto em um vetor denso;
- **método de recuperação**: decide como os candidatos são encontrados e ordenados;
- **enriquecimento de texto do Contextual/SAC**: altera apenas o texto usado para
  indexação, sem transformar esse conteúdo sintético em evidência jurídica.

### Comece por aqui

- [Embeddings (PT-BR)](Embeddings-pt-BR): conceitos, modelos candidatos, configuração,
  identidade, cache, privacidade e limitações atuais.
- [Estratégias de recuperação (PT-BR)](Estrategias-de-Recuperacao): Dense, BM25,
  Hybrid+RRF, Reranked, Contextual Retrieval, Parent-child, SAC, RAPTOR e GraphRAG,
  além dos modelos auxiliares.
- [Embeddings (English)](Embeddings)
- [Retrieval strategies (English)](Retrieval-Strategies)

### Estado resumido

| Item | Estado atual |
|---|---|
| Embedding da matriz canônica | `gemini-embedding-001`, 1.536 dimensões |
| Alternativa local sem credencial de embedding | `Qwen/Qwen3-Embedding-0.6B`, 1.024 dimensões |
| Seleção do modelo canônico | Provisória, aguardando revalidação do experimento PT-BR |
| Busca lexical | BM25 no OpenSearch com analisador `brazilian` |
| Busca vetorial | Distância cosseno em índice HNSW do pgvector |
| Estratégias configuradas | 10 rótulos, de `dense` a `graphrag` |
| Geração e enriquecimentos | `gemini-3.1-flash-lite` |
| Judge canônico | OpenAI `gpt-5.4-mini-2026-03-17`; ainda não calibrado com humanos |

> “Canônico” significa “configuração escolhida para a matriz publicável”. Não significa que
> a comparação esteja definitivamente encerrada. O arquivo
> `configs/experiments/embeddings-ptbr.yaml` marca a seleção como
> `provisional_pending_revalidation`.

## English

RAGForge is an experimental platform for comparing Retrieval-Augmented Generation
(RAG) strategies over Brazilian financial and regulatory documents. This wiki
documents what the project **implements today**, keeping three often-confused
concepts separate:

- **embedding model**: turns text into a dense vector;
- **retrieval method**: determines how candidates are found and ranked;
- **Contextual/SAC retrieval-text enrichment**: changes only the text used for
  indexing, without turning that synthetic content into legal evidence.

### Start here

- [Embeddings](Embeddings): concepts, candidate models, configuration, identity,
  caching, privacy, and current limitations.
- [Retrieval strategies](Retrieval-Strategies): Dense, BM25, Hybrid+RRF, Reranked,
  Contextual Retrieval, Parent-child, SAC, RAPTOR, and GraphRAG, plus the supporting
  models.
- [Embeddings (Português)](Embeddings-pt-BR)
- [Estratégias de recuperação (Português)](Estrategias-de-Recuperacao)

### Status at a glance

| Item | Current state |
|---|---|
| Canonical-matrix embedding | `gemini-embedding-001`, 1,536 dimensions |
| Local embedding alternative | `Qwen/Qwen3-Embedding-0.6B`, 1,024 dimensions |
| Canonical model selection | Provisional, pending PT-BR experiment revalidation |
| Lexical retrieval | OpenSearch BM25 with the `brazilian` analyzer |
| Vector retrieval | Cosine distance over a pgvector HNSW index |
| Configured strategies | 10 labels, from `dense` through `graphrag` |
| Generation and enrichment | `gemini-3.1-flash-lite` |
| Canonical judge | OpenAI `gpt-5.4-mini-2026-03-17`; not yet human-calibrated |

> “Canonical” means “selected for the publishable matrix.” It does not mean the
> comparison is permanently closed. `configs/experiments/embeddings-ptbr.yaml`
> records the selection as `provisional_pending_revalidation`.

## Project sources

- [Main benchmark configuration](https://github.com/brunovicco/ragforge/blob/main/configs/experiments/benchmark-v01.yaml)
- [Local benchmark configuration](https://github.com/brunovicco/ragforge/blob/main/configs/experiments/benchmark-local-v01.yaml)
- [PT-BR embedding experiment](https://github.com/brunovicco/ragforge/blob/main/configs/experiments/embeddings-ptbr.yaml)
- [Architecture Decision Records](https://github.com/brunovicco/ragforge/tree/main/docs/adr)
