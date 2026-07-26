# Embeddings no RAGForge

[English](Embeddings) · [Início](Home) · [Estratégias de recuperação](Estrategias-de-Recuperacao)

> Retrato da documentação em 2026-07-25. O estado dos modelos vem da
> configuração do repositório; ele não significa que todos os candidatos já
> concluíram o experimento atual sobre o RegRAG-BR.

## O que é um embedding

Um embedding é um vetor numérico que representa o conteúdo semântico de uma
entrada. O RAGForge gera um vetor para o `retrieval_text` de cada chunk e o
armazena no PostgreSQL com pgvector. Durante uma consulta, o mesmo modelo
vetoriza a pergunta e o pgvector ordena os chunks por distância cosseno.

Esse é um desenho de **bi-encoder**: documentos e perguntas são codificados
separadamente. Ele escala para o corpus porque os vetores dos documentos são
calculados antes das consultas. É diferente do cross-encoder usado no reranking,
que lê cada par pergunta–chunk em conjunto e, por isso, só é aplicado a um
conjunto pequeno de candidatos.

```text
retrieval_text do documento ──modelo de embedding──> vetor ──┐
                                                             ├─ similaridade cosseno ─> ranking
texto da pergunta ────────────modelo de embedding──> vetor ──┘
```

O projeto usa hoje uma única operação `embed()` para documentos e perguntas. Uma
instrução de consulta específica por modelo e versionada está planejada, mas não
implementada.

## Dense, sparse e multimodal: significados diferentes

| Termo | Significado | Comportamento no RAGForge |
|---|---|---|
| Embedding denso | Vetor de ponto flutuante com largura fixa, cujo significado é distribuído entre as dimensões | Implementado com Gemini ou Sentence Transformers |
| Representação esparsa | Pesos de termos ou atributos em que a maioria dos valores é zero | O RAGForge **não** gera embeddings esparsos por modelo; sua estratégia sparse é BM25 |
| Embedding multimodal | Mídias diferentes compartilham o mesmo espaço vetorial | `gemini-embedding-2` oferece a capacidade, mas o RAGForge envia apenas texto |
| Embedding Matryoshka/MRL | Modelo treinado para manter informação útil quando a dimensão de saída é reduzida | Gemini é solicitado com 1.536 dimensões; os modelos locais usam a largura informada pelo modelo |

BM25 é às vezes agrupado com “embeddings esparsos” em explicações genéricas de
RAG, mas isso seria impreciso para esta implementação. A estratégia
`sparse_bm25` envia o texto da pergunta diretamente ao OpenSearch e não chama
nenhum modelo de embedding.

## Inventário de modelos

### Candidatos de embedding para recuperação

| Modelo | Execução | Largura usada ou declarada | Estado no repositório | Uso no RAGForge |
|---|---:|---:|---|---|
| `gemini-embedding-001` | API Gemini hospedada | 1.536 | Seleção canônica, provisória e aguardando revalidação | Vetores densos de texto; configurado na matriz publicável |
| `gemini-embedding-2` | API Gemini hospedada | 1.536 | Candidato aguardando revalidação | Apenas texto, apesar da capacidade multimodal do modelo |
| `Qwen/Qwen3-Embedding-0.6B` | Sentence Transformers local | 1.024 | Default operacional local; ainda não avaliado no experimento isolado | Vetores densos sem credencial de embedding em `make bench-live-local` |
| `BAAI/bge-m3` | Sentence Transformers local | 1.024 | Candidato aguardando revalidação | Somente saída densa; as saídas sparse e no estilo ColBERT não são usadas |
| `intfloat/multilingual-e5-large-instruct` | Sentence Transformers local | 1.024 | Candidato de controle ainda não avaliado | Saída densa sem a instrução específica de consulta |

### `gemini-embedding-001`

- O benchmark canônico solicita 1.536 dimensões. O modelo oferece dimensões
  flexíveis até 3.072.
- A redução mantém o vetor abaixo do limite de 2.000 dimensões do índice HNSW
  do pgvector para o tipo `vector`.
- As chamadas são hospedadas, tarifadas e exigem `GEMINI_API_KEY` ou
  `GOOGLE_API_KEY`.
- O adaptador processa lotes de até 100 textos, repete falhas transitórias,
  limita chamadas concorrentes e pode manter cache por texto.
- O adaptador registra `normalize=false`; a comparação no pgvector continua
  usando distância cosseno.

O experimento isolado de PT-BR selecionou este modelo para a matriz canônica,
mas o arquivo atual do experimento marca a seleção como provisória e pendente de
revalidação. Ela não deve ser apresentada como um vencedor permanente.

### `gemini-embedding-2`

- O provedor o descreve como um modelo unificado para texto, imagem, vídeo,
  áudio e PDF. A porta de embedding do RAGForge aceita apenas texto; as outras
  modalidades não são exercitadas.
- O projeto solicita 1.536 dimensões, assim como no `gemini-embedding-001`.
- Uma sondagem direta do projeto observou que uma requisição `embed_content`
  com vários textos retornava somente um embedding. Por isso, o adaptador envia
  um texto por requisição para este modelo. A indexação fica mais lenta e
  produz muito mais chamadas.
- Ele é candidato de comparação, não o modelo canônico.

### `Qwen/Qwen3-Embedding-0.6B`

- É a alternativa operacional sem provedor de embedding em
  `benchmark-local-v01.yaml`.
- Executa por `SentenceTransformerEmbedder`, informa 1.024 dimensões e devolve
  vetores normalizados por L2.
- O model card o descreve como multilíngue, sensível a instruções e compatível
  com redução de dimensão Matryoshka. O RAGForge usa hoje a largura completa e
  não envia instrução de consulta.
- A configuração não fixa uma revisão imutável do Hugging Face. O adaptador
  registra a revisão não resolvida como `main`, suficiente para exploração,
  mas abaixo da meta de reprodutibilidade da ADR-0013.
- “Local” vale somente para a etapa de embedding. O benchmark completo ainda
  chama Gemini para contextualização, resumos, extração do GraphRAG e geração
  de respostas, e OpenAI para o judge canônico.

### `BAAI/bge-m3`

- O modelo pode produzir representações densas, sparse aprendidas e multi-vector
  no estilo ColBERT. O adaptador do RAGForge chama o `encode()` padrão do
  Sentence Transformers; portanto, somente a representação densa de 1.024
  dimensões participa da comparação.
- Os vetores são normalizados por L2 antes da indexação.
- O repositório registra CPU como caminho estável na máquina de desenvolvimento
  atual. Uma execução em MPS esgotou a memória; essa é uma observação daquela
  máquina, não uma restrição geral do modelo.
- O resultado da comparação em PT-BR precisa ser revalidado.

### `intfloat/multilingual-e5-large-instruct`

- É o candidato local de controle.
- Seu model card exige uma instrução de tarefa na pergunta e alerta para perda
  de desempenho sem ela.
- O RAGForge vetoriza hoje perguntas e documentos pelo mesmo método `embed()` e
  registra o hash de uma instrução vazia. Um resultado obtido agora mediria
  tanto o modelo quanto a integração incompleta do projeto.
- Está declarado na matriz experimental, mas ainda não foi avaliado nela.

### `text-embedding-3-small`: somente avaliação

O `text-embedding-3-small` está configurado no judge OpenAI canônico para a
métrica Answer Relevancy do RAGAS. Ele **não** cria o índice de recuperação,
não recupera chunks e não participa da comparação de embeddings Dense/Hybrid.

Essa distinção é importante:

```text
embedding de recuperação -> encontra evidências
embedding do judge        -> ajuda a medir se a resposta atende à pergunta
```

O fallback de judge Gemini usa de forma semelhante o
`gemini-embedding-001` para Answer Relevancy, mas essa alternativa é rotulada
como exploratória porque o gerador de respostas também usa Gemini.

## Normalização, dimensões e similaridade

O adaptador local chama:

```python
model.encode(..., normalize_embeddings=True)
```

Para vetores de norma unitária, produto escalar e similaridade cosseno geram a
mesma ordenação. O adaptador Gemini não normaliza no cliente. A camada de
armazenamento usa sempre o operador de cosseno do pgvector
(`vector_cosine_ops` e `<=>`), que considera as magnitudes ao comparar os
vetores.

As dimensões fazem parte do contrato do espaço vetorial:

- a largura da coluna pgvector é fixa quando a tabela é criada;
- duas larguras iguais não tornam modelos diferentes compatíveis;
- mudar modelo, revisão, dimensões, normalização ou instrução exige outro
  índice;
- a largura nativa de 3.072 do Gemini é reduzida para 1.536 porque índices HNSW
  do pgvector sobre o tipo `vector` aceitam até 2.000 dimensões.

## Identidade, cache e isolamento de índices

O RAGForge identifica um espaço de embedding por:

```text
provedor
+ modelo
+ revisão
+ dimensões
+ normalização
+ hash da instrução de consulta
+ runtime
```

O namespace do índice também inclui o hash do corpus, a configuração de
chunking e o schema do texto de recuperação. Um fingerprint separado inclui
todo `source_text`, `retrieval_text`, IDs estruturais, metadados e a identidade
do produtor de texto sintético.

Consequências:

- vetores não são reutilizados só porque dois modelos têm a mesma largura;
- o cache de embeddings usa a identidade completa e o hash do texto;
- índices base, contextual, SAC e RAPTOR ficam isolados;
- índices parciais não recebem marcador de conclusão reutilizável;
- falha em um provedor não troca silenciosamente o embedding durante a
  execução.

Limitação atual: as revisões dos modelos locais não são fixadas no YAML, e
todas as instruções de consulta compartilham hoje o hash de uma string vazia.
Essas lacunas precisam ser fechadas antes de alegar reprodutibilidade exata
entre máquinas para modelos sensíveis a instruções.

## Configuração

Embedding hospedado canônico:

```yaml
embedding:
  provider: gemini
  model: gemini-embedding-001
  dimensions: 1536
```

Alternativa local:

```yaml
embedding:
  provider: local
  model: Qwen/Qwen3-Embedding-0.6B
  dimensions: 1024
  device: cpu
```

O runner aceita somente os provedores de recuperação `local` e `gemini`. Não
há fallback automático entre eles.

## Como comparar e escolher um modelo

O protocolo do RAGForge mantém constantes corpus, split, chunking, texto de
recuperação, top-k, julgamentos e métricas. Ele varia o embedding e mede Dense e
Hybrid+RRF, as estratégias afetadas mais diretamente pela qualidade vetorial.

Execução de um candidato:

```bash
uv run python configs/experiments/run_embeddings_ptbr.py \
  --model Qwen/Qwen3-Embedding-0.6B
```

Exemplo hospedado:

```bash
GEMINI_API_KEY=... uv run python \
  configs/experiments/run_embeddings_ptbr.py \
  --model gemini-embedding-001
```

A comparação produz Recall@k, Precision@k, nDCG@k e MRR. Antes de promover um
novo vencedor, também devem ser registrados revisão imutável, instrução efetiva,
dispositivo, precisão numérica, throughput, latência, memória, tamanho do índice
e custo do provedor.

## Privacidade e impacto operacional

- Embedding local mantém os textos de chunks e perguntas na máquina durante
  essa etapa.
- Embedding Gemini hospedado envia texto de recuperação e perguntas a um
  provedor externo.
- O corpus atual contém atos oficiais públicos; isso não representa autorização
  automática para enviar futuros documentos privados.
- Uma execução com dados privados exige decisão explícita de processamento. O
  inventário geral de privacidade do repositório ainda não está completo.
- Chaves de API e conteúdo não devem ser registrados em logs. As credenciais
  são lidas de variáveis de ambiente.

## Fontes

### RAGForge

- [Porta e adaptadores de embedding](https://github.com/brunovicco/ragforge/tree/main/src/ragforge/embeddings)
- [ADR-0005: escopo da comparação](https://github.com/brunovicco/ragforge/blob/main/docs/adr/0005-embedding-comparison-scope.md)
- [ADR-0013: backends neutros de provedor](https://github.com/brunovicco/ragforge/blob/main/docs/adr/0013-provider-neutral-embedding-backends.md)
- [Configuração do experimento PT-BR](https://github.com/brunovicco/ragforge/blob/main/configs/experiments/embeddings-ptbr.yaml)

### Documentação primária dos modelos e armazenamento

- [Google: `gemini-embedding-001`](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-001)
- [Google: `gemini-embedding-2`](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2?hl=pt-br)
- [Model card do Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Model card do BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Model card do Multilingual E5 Large Instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- [OpenAI: `text-embedding-3-small`](https://developers.openai.com/api/docs/models/text-embedding-3-small)
- [pgvector: tipos, dimensões e HNSW](https://github.com/pgvector/pgvector)
