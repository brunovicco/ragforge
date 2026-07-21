# Corpus sources

Raw source documents for the legal structural chunker (ADR-0006) and the RegRAG-BR
corpus (`datasets/regrag-br/`). These are public-domain Brazilian government norms;
files here are unmodified as retrieved. Retrieved 2026-07-21.

| File | Norm | Source | Notes |
|---|---|---|---|
| `bacen/RES-CMN-4893-2021.pdf` | Resolução CMN nº 4.893, de 26/02/2021 | Mirror of the official BCB "exibenormativo" page: https://www.ancord.org.br/wp-content/uploads/2021/03/Resolucao-CMN-n-4.893-de-26_2_2021.pdf | Header confirms it is a printed snapshot of `bcb.gov.br/estabilidadefinanceira/exibenormativo?...numero=4893`; original 2021 text, not the amended-consolidated version. |
| `bacen/RES-CMN-5274-2025.htm` | Resolução CMN nº 5.274, de 18/12/2025 (altera a Resolução CMN nº 4.893/2021) | https://www.legisweb.com.br/legislacao/?id=488277 | The official `bcb.gov.br`/`normativos.bcb.gov.br` page for this norm is a JS-rendered SPA with no discoverable static PDF at time of retrieval; used a third-party legal database instead. Amending resolution: body only shows the altered articles/paragraphs (ellipsis for unchanged text), which is the norm's real structure, not a truncation. |
| `lc-lgpd/LC-105-2001.htm` | Lei Complementar nº 105, de 10/01/2001 (sigilo bancário) | https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp105.htm | Official Planalto compiled text. |
| `lc-lgpd/LEI-13709-2018-LGPD.htm` | Lei nº 13.709, de 14/08/2018 (LGPD) | http://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm | Official Planalto compiled text. |
| `cvm/LEI-6385-1976.htm` | Lei nº 6.385, de 07/12/1976 (mercado de valores mobiliários / cria a CVM) | http://www.planalto.gov.br/ccivil_03/leis/l6385.htm | Official Planalto compiled text. |
| `cvm/ICVM-607-2019.pdf` | Instrução CVM nº 607, de 17/06/2019 | https://conteudo.cvm.gov.br/export/sites/cvm/legislacao/instrucoes/anexos/600/Inst607.pdf | Official CVM PDF. Revoked by Resolução CVM 45/2021 but kept for historical/regulatory reference per the user's request. |

These are raw inputs to the extraction/chunking pipeline, not the curated
`datasets/regrag-br/` benchmark itself.
