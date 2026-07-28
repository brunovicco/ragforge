# RAGForge

*Read this in [English](README.md).*

**Plataforma adaptativa de benchmark de RAG para documentos financeiros e regulatórios brasileiros.**

O RAGForge está sendo construído para comparar estratégias sparse, dense, hybrid, contextual, hierárquica (RAPTOR), grafo (GraphRAG) e corretiva - medindo qualidade de resposta, precisão de recuperação, latência e custo sobre o **RegRAG-BR**, um golden dataset de 230 perguntas sobre normas do CMN/BCB e da CVM.

> **v0.1** - run de benchmark [`20260726T185553Z`](docs/BENCHMARK-RESULTS.md) publicado com
> evidência verificável. A tabela de [Status](#status) acompanha o que está implementado hoje
> versus o que é planejado.

## Por que isso existe

A maioria das comparações de RAG é anedótica. O RAGForge trata a pergunta "*qual estratégia de RAG eu deveria usar?*" como um experimento: 10 configurações de estratégia × 7 classes de pergunta, com um roteador adaptativo pensado para ser avaliado contra um **oráculo empírico**, e todo número publicado reproduzível bit a bit a partir de um cache versionado de chamadas de LLM.

## Estratégias avaliadas

| # | Estratégia | Abordagem | Status |
| --- | ---------- | ---------- | -------- |
| 1 | Dense (baseline) | pgvector, top-k fixo | Implementado |
| 2 | Sparse BM25 | OpenSearch, analisador `brazilian` | Implementado |
| 3 | Hybrid + RRF | BM25 + dense + Reciprocal Rank Fusion | Implementado |
| 4 | Reranked | Hybrid top-50 → cross-encoder → top-5 | Implementado |
| 5 | Contextual Retrieval | Contexto via LLM por chunk + prompt caching | Implementado |
| 6 | Parent-child / multi-vector | Busca em chunks pequenos, entrega a seção | Implementado |
| 7 | Summary-Augmented Chunking (SAC) | Resumo do documento + texto autoritativo do chunk | Implementado |
| 8 | SAC + Contextual | Resumo do documento + contexto por chunk + texto autoritativo | Implementado |
| 9 | RAPTOR | Árvore recursiva de resumos (impl. mínima) | Implementado |
| 10 | GraphRAG | Adapter LightRAG (modo local avaliado; global planejado) | Implementado |

Transversais: **Roteador Adaptativo** (regras + few-shot, planejado), **Fluxo corretivo** (avaliador de evidência com retry / reformulação / declaração de evidência insuficiente, LangGraph, planejado), **governança** (rastreamento resposta → chunk → artigo via Citation Accuracy, mais uma auditoria de suporte semântico pós-geração com reescrita limitada e uma trilha de evidência à prova de adulteração por execução, implementado), **observabilidade** (tracing do Langfuse apenas com metadados implementado; OpenTelemetry planejado).

## Status

O RAGForge está em desenvolvimento ativo. Esta seção acompanha o que
está de fato rodando hoje versus o que é meta de design - veja o [histórico de PRs](../../pulls?q=is%3Apr) para como cada linha chegou lá.

| Componente | Status |
| --- | --- |
| Chunker estrutural jurídico (ADR-0006) | Implementado |
| Pipeline de ingestão (extração, hash de snapshot) | Implementado |
| As 10 configurações de recuperação avaliadas (Dense até GraphRAG) | Implementado |
| Harness de avaliação + julgamentos de cobertura estrutural (ADR-0002) | Implementado |
| Observabilidade (Langfuse, apenas metadados) | Implementado |
| Geração de resposta + Citation Accuracy | Implementado |
| Judge de LLM independente - Faithfulness/Answer Relevancy + abstenção (ADR-0018) | Implementado - não calibrado (exercício de kappa da ADR-0007 pendente) |
| Auditoria de citação/suporte semântico pós-geração + reescrita limitada (ADR-0016) | Implementado - desligado por padrão (`audit.enabled: false`) |
| Diretório de evidência auditável e à prova de adulteração por execução (ADR-0017) | Implementado - `artifacts/runs/<run_id>/`, verificado via `scripts/verify_run.py` |
| Runner principal do benchmark (`make bench-live`, 10 estratégias + qualidade de resposta) | Implementado - apenas modo live |
| Roteador Adaptativo, Fluxo corretivo | Planejado |
| Golden set RegRAG-BR | 230 perguntas: 36 validation/dev + 194 test |
| Apps de API / dashboard | API de resultados publicados e dashboard analítico implementados; Arena ao vivo planejada |
| `make bench` (replay determinístico, bit a bit, ADR-0004) | Planejado - precisa de um cache versionado de chamadas de LLM, ainda não construído |

## Resultado do benchmark v0.1

O run [`20260726T185553Z`](experiments/20260726T185553Z/results.json) avalia uma
**amostra determinística e estratificada por classe de 60 perguntas** do split
de teste com 194 perguntas. A seed é
`regrag-br-benchmark-sample-v1`; este é um resultado da v0.1 com custo
controlado, não uma afirmação sobre o split de teste completo.

**SAC é a estratégia recomendada para a v0.1** pelo melhor equilíbrio: maior
nDCG@5 (`0,963`), MRR (`0,991`) e Citation Accuracy (`0,689`) neste run, com
Document-Level Retrieval Mismatch igual a zero. RAPTOR obteve o maior Recall@5
(`1,000`) e Precision@5 (`0,611`), mas seus nós de resumo gerados apresentam
outro compromisso de qualidade da evidência.

O scorecard completo, metodologia, limitações e instruções de verificação estão
em [Resultados do benchmark](docs/BENCHMARK-RESULTS.md).

## Início rápido

```bash
uv sync --all-groups
make infra-up                                      # Postgres+pgvector, OpenSearch
GEMINI_API_KEY=... OPENAI_API_KEY=... make bench-live
make bench-live-local                              # mesma matriz, embeddings Qwen locais
make api                                           # API read-only de resultados publicados
make dashboard                                     # dashboard analítico offline
```

`make bench-live` chama provedores reais (embeddings, contextualização, sumarização do RAPTOR, extração de entidades do GraphRAG - ver a tabela de estratégias acima). `make bench` (replay determinístico e sem custo a partir de um cache versionado de LLM) é o design-alvo segundo a [ADR-0004](docs/adr/0004-benchmark-reproducibility-policy.md), mas essa camada de cache ainda não existe - apenas o modo live está implementado. A camada de replay e sua verificação no CI são entregues juntas ([ADR-0020](docs/adr/0020-replay-cache-ci-gate.md)).

A matriz canônica e publicável usa `gemini-embedding-001`, seleção provisória da
comparação isolada de embeddings em PT-BR (ADR-0005) - marcada como
`pending_revalidation` em `configs/experiments/embeddings-ptbr.yaml`; ainda não é um
vencedor definitivo. `make bench-live-local` usa `Qwen/Qwen3-Embedding-0.6B` como
alternativa operacional sem credenciais de embedding prevista pela ADR-0013; é uma
execução identificada separadamente, não um fallback silencioso nem uma troca do
vencedor de qualidade.

Os dois comandos live persistem embeddings por texto e índices completos no
diretório ignorado `.ragforge/cache/`. O fingerprint inclui o texto derivado do
corpus, a identidade do embedding e o produtor de texto sintético; índices
parciais nunca são marcados como reutilizáveis. `--resume <run-id>` pula
estratégias já concluídas, e um lock do repositório impede benchmarks simultâneos.

## Decisões de design relevantes

Todas as escolhas não óbvias são registradas como [ADRs](docs/adr/README.md). As mais estruturantes:

- [ADR-0002](docs/adr/0002-article-level-relevance-judgments.md) - julgamentos de relevância no **nível de artigo da norma**, para que as métricas de recuperação continuem comparáveis entre estratégias que fragmentam o texto de formas diferentes (ou nem retornam chunks).
- [ADR-0003](docs/adr/0003-empirical-router-oracle.md) - o roteador é avaliado contra um **oráculo empírico por pergunta** (melhor estratégia medida, não presumida), com uma divisão dev/test que evita vazamento de few-shot.
- [ADR-0004](docs/adr/0004-benchmark-reproducibility-policy.md) - o `make bench` é especificado para reproduzir a partir de um cache versionado de LLM, bit a bit e com custo zero de API; a camada de replay e sua verificação no CI são entregues juntas ([ADR-0020](docs/adr/0020-replay-cache-ci-gate.md)) e ainda não existem.
- [ADR-0006](docs/adr/0006-legal-structural-chunker.md) - chunking sensível ao domínio pela hierarquia jurídica (Art./§/inciso), com IDs estruturais estáveis.
- [ADR-0007](docs/adr/0007-llm-judge-calibration-ptbr.md) - o judge de LLM precisa ser calibrado contra avaliação humana em PT-BR, com a concordância publicada, antes que suas notas contem como validadas; até lá, toda métrica do judge carrega essa ressalva.
- [ADR-0011](docs/adr/0011-structural-id-collision-in-amended-norms.md) - IDs estruturais que colidem entre histórico de emendas/anexos anexados são excluídos das citações do golden set, não corrigidos no nível do chunker.
- [ADR-0016](docs/adr/0016-post-generation-citation-audit.md) - um verificador de suporte semântico e no máximo uma reescrita limitada capturam alegações sem suporte que uma checagem de mera existência da citação deixaria passar.
- [ADR-0017](docs/adr/0017-auditable-evidence-lineage.md) - todo score publicado é rastreável até um diretório de evidência encadeado por hash e à prova de adulteração, por execução - entradas exatas, identidades de modelo e candidatos de recuperação, não só a métrica agregada.

## Estrutura do repositório

```
apps/            # api/ (FastAPI) e dashboard/ (Streamlit: benchmark + Arena)
src/ragforge/    # domain/ (núcleo livre de framework) · ingestion/ chunking/ embeddings/
                 # retrieval/ reranking/ routing/ generation/ evaluation/ governance/
datasets/        # corpus/ (snapshot versionado) + regrag-br/ (golden set, CC-BY-4.0)
experiments/     # resultados versionados + cache de LLM por run-id
configs/         # configs declarativas de experimentos - todo número do README nasce aqui
docs/adr/        # architecture decision records
```

O núcleo é livre de framework: `RetrievalStrategy` é um Protocol; SDKs de LLM são banidos dos pacotes centrais por uma guarda de arquitetura no CI (`scripts/validate_architecture.py`, limites em `pyproject.toml`).

## Dataset - RegRAG-BR

230 perguntas (7 classes de consulta) sobre resoluções selecionadas do CMN/BCB (4.893, gestão de risco, Open Finance, PLD/FT) e normas da CVM/CMN, com julgamentos de relevância no nível de artigo e respostas de referência, publicadas sob CC-BY-4.0 com um [datasheet](datasets/regrag-br/DATASHEET.md). O split determinístico e estratificado reserva 36 perguntas para desenvolvimento/validação do roteador e 194 para as métricas oficiais de teste. As normas são atos oficiais (art. 8º, I, Lei 9.610/98 - não protegidos por direito autoral).

Publicado (`datasets/regrag-br/judgments.json`): 230 perguntas curadas manualmente, cada uma com uma resposta de referência, verificadas contra o texto real extraído de 5 documentos do corpus (LC-105/2001, RES-CMN-4893/2021, RES-CMN-5274/2025, LEI-13709/2018-LGPD, ICVM-607/2019). Um 6º documento do corpus, a LEI-6385/1976, ainda não foi curado. IDs estruturais conhecidos por serem ambíguos no texto-fonte real - artefatos de histórico de emendas e anexos anexados em 3 dos 5 documentos - são excluídos de citação; ver [ADR-0011](docs/adr/0011-structural-id-collision-in-amended-norms.md).

## Desenvolvimento

```bash
uv sync --all-groups
uv run pytest
uv run python scripts/quality_gate.py   # ruff, mypy, pytest (≥80% do core), bandit, pip-audit, guarda de arquitetura
```

Estruturado com [claude-python-engineering-harness](https://github.com/brunovicco/claude-python-engineering-harness) ([ADR-0009](docs/adr/0009-scaffold-via-engineering-harness.md)).

## Licença

Código: [MIT](LICENSE) · Dataset (RegRAG-BR): [CC-BY-4.0](datasets/regrag-br/LICENSE)
