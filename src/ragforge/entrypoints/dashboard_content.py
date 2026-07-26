"""Localized explanatory content for the benchmark dashboard."""

from dataclasses import dataclass
from typing import Literal

type Language = Literal["pt", "en"]


@dataclass(frozen=True, slots=True)
class DashboardCopy:
    """Localized labels used by the Streamlit entrypoint."""

    title: str
    language: str
    published_run: str
    sample_caption: str
    recommendation: str
    recommendation_rationale: str
    recommendation_scope: str
    retrieval_embedding: str
    answer_generation: str
    independent_judge: str
    overview: str
    techniques: str
    indicators: str
    evidence: str
    strategy_comparison: str
    quality_profile: str
    strategy: str
    how_it_works: str
    strengths: str
    limitations: str
    interpretation: str
    scale_note: str
    verify_evidence: str
    evidence_path: str
    results_path: str
    git_revision: str


@dataclass(frozen=True, slots=True)
class TechniqueExplanation:
    """Explain one retrieval technique in dashboard-friendly language."""

    title: str
    summary: str
    how_it_works: str
    strengths: str
    limitations: str


@dataclass(frozen=True, slots=True)
class MetricExplanation:
    """Explain one benchmark indicator and how to interpret it."""

    label: str
    definition: str
    interpretation: str


_COPY: dict[Language, DashboardCopy] = {
    "pt": DashboardCopy(
        title="Benchmark RAGForge",
        language="Idioma / Language",
        published_run="Execução publicada",
        sample_caption=(
            "{question_count} perguntas de uma amostra estratificada e determinística "
            "do split de teste com {source_count} perguntas · seed `{seed}` · top-k {k}"
        ),
        recommendation="Estratégia recomendada",
        recommendation_rationale=(
            "{strategy} apresentou o melhor equilíbrio entre qualidade de recuperação, "
            "fidelidade e relevância das respostas nesta execução."
        ),
        recommendation_scope=(
            "A recomendação vale para esta amostra determinística de {question_count} "
            "perguntas e deve ser reavaliada se o corpus, os modelos ou a distribuição "
            "das consultas mudar."
        ),
        retrieval_embedding="Embedding de recuperação",
        answer_generation="Geração da resposta",
        independent_judge="Juiz independente",
        overview="Visão geral",
        techniques="Técnicas",
        indicators="Indicadores",
        evidence="Evidências",
        strategy_comparison="Comparação das estratégias",
        quality_profile="Perfil de qualidade",
        strategy="Estratégia",
        how_it_works="Como funciona",
        strengths="Pontos fortes",
        limitations="Limitações",
        interpretation="Interpretação",
        scale_note=(
            "Os indicadores variam de 0 a 1. Valores maiores são melhores, exceto "
            "DRM@5, em que zero é o resultado ideal."
        ),
        verify_evidence="Verificar integridade localmente",
        evidence_path="Evidências",
        results_path="Resultados",
        git_revision="Revisão Git",
    ),
    "en": DashboardCopy(
        title="RAGForge benchmark",
        language="Idioma / Language",
        published_run="Published run",
        sample_caption=(
            "{question_count} questions from a deterministic stratified sample of "
            "the {source_count}-question test split · seed `{seed}` · top-k {k}"
        ),
        recommendation="Recommended strategy",
        recommendation_rationale=(
            "{strategy} delivered the best balance of retrieval quality, faithfulness, "
            "and answer relevancy in this run."
        ),
        recommendation_scope=(
            "This recommendation applies to the deterministic {question_count}-question "
            "sample and should be reassessed if the corpus, models, or query distribution "
            "changes."
        ),
        retrieval_embedding="Retrieval embedding",
        answer_generation="Answer generation",
        independent_judge="Independent judge",
        overview="Overview",
        techniques="Techniques",
        indicators="Metrics",
        evidence="Evidence",
        strategy_comparison="Strategy comparison",
        quality_profile="Quality profile",
        strategy="Strategy",
        how_it_works="How it works",
        strengths="Strengths",
        limitations="Limitations",
        interpretation="Interpretation",
        scale_note=(
            "Metrics range from 0 to 1. Higher values are better, except for DRM@5, "
            "where zero is ideal."
        ),
        verify_evidence="Verify integrity locally",
        evidence_path="Evidence",
        results_path="Results",
        git_revision="Git revision",
    ),
}

_TECHNIQUES_PT: dict[str, TechniqueExplanation] = {
    "dense": TechniqueExplanation(
        "Busca densa (Dense)",
        "Recuperação semântica por embeddings.",
        "Converte pergunta e trechos em vetores e ordena os trechos pela similaridade vetorial.",
        "Encontra paráfrases e conceitos relacionados mesmo sem palavras idênticas.",
        "Pode perder termos jurídicos exatos e depende da qualidade do modelo de embedding.",
    ),
    "sparse_bm25": TechniqueExplanation(
        "Busca esparsa (BM25)",
        "Recuperação lexical baseada na frequência dos termos.",
        "Pontua os trechos pela ocorrência e raridade das palavras da pergunta no corpus.",
        "É rápida, explicável e eficaz para nomes, artigos, siglas e expressões exatas.",
        "Tem dificuldade com sinônimos, paráfrases e relações puramente semânticas.",
    ),
    "hybrid_rrf": TechniqueExplanation(
        "Busca híbrida (RRF)",
        "Combina os rankings denso e BM25.",
        "Aplica Reciprocal Rank Fusion para premiar resultados bem posicionados nas duas buscas.",
        "Equilibra correspondência semântica e lexical sem comparar escalas de pontuação.",
        "Adiciona custo de duas buscas e pode preservar ruído presente nos rankings de origem.",
    ),
    "reranked": TechniqueExplanation(
        "Busca com reranking",
        "Reordena candidatos com um modelo mais preciso.",
        "Recupera um conjunto inicial e usa um cross-encoder para avaliar cada par "
        "pergunta-trecho.",
        "Melhora a ordem dos primeiros resultados e interpreta melhor relações finas.",
        "Aumenta latência e custo; sua qualidade fica limitada aos candidatos "
        "recuperados primeiro.",
    ),
    "contextual": TechniqueExplanation(
        "Recuperação contextual",
        "Acrescenta contexto específico do documento aos trechos.",
        "Um LLM gera uma breve contextualização antes da indexação, preservando também a fonte.",
        "Reduz ambiguidades de trechos isolados e melhora consultas que dependem do documento.",
        "Exige pré-processamento por LLM e o contexto gerado pode introduzir imprecisões.",
    ),
    "parent_child": TechniqueExplanation(
        "Parent-Child",
        "Busca unidades pequenas e entrega contexto maior.",
        "Indexa trechos filhos para precisão, mas retorna o trecho pai associado para responder.",
        "Combina localização precisa com contexto suficiente para a geração.",
        "Pais muito grandes adicionam ruído; o vínculo entre pai e filho precisa ser confiável.",
    ),
    "sac": TechniqueExplanation(
        "SAC (Summary-Augmented Chunking)",
        "Enriquece cada trecho com um resumo do documento.",
        "Anexa um resumo global do documento ao trecho antes de gerar seu embedding.",
        "Traz contexto global com implementação simples e foi a recomendação desta amostra.",
        "Consome mais tokens na indexação e resumos fracos podem homogeneizar os embeddings.",
    ),
    "sac_contextual": TechniqueExplanation(
        "SAC contextual",
        "Combina resumo global e contexto local gerado.",
        "Indexa o trecho junto do resumo do documento e de uma contextualização específica.",
        "Oferece sinais globais e locais para consultas ambíguas.",
        "É a preparação mais cara e pode adicionar informação redundante ou gerada.",
    ),
    "raptor": TechniqueExplanation(
        "RAPTOR",
        "Organiza trechos e resumos em uma hierarquia.",
        "Agrupa conteúdo semanticamente e cria nós de resumo recursivos pesquisáveis.",
        "Ajuda perguntas amplas que exigem síntese entre diferentes partes do corpus.",
        "É complexo e caro de indexar; resumos sintéticos podem ocultar detalhes ou conter erros.",
    ),
    "graphrag": TechniqueExplanation(
        "GraphRAG",
        "Recupera conhecimento por entidades e relações.",
        "Constrói um grafo com o LightRAG e consulta entidades, relações e trechos no modo local.",
        "É adequado a perguntas relacionais e permite atravessar evidências dispersas.",
        "Tem indexação lenta e cara, depende da extração do grafo e pode propagar "
        "relações incorretas.",
    ),
}

_TECHNIQUES_EN: dict[str, TechniqueExplanation] = {
    "dense": TechniqueExplanation(
        "Dense search",
        "Semantic retrieval using embeddings.",
        "It converts the question and passages into vectors and ranks passages by "
        "vector similarity.",
        "It finds paraphrases and related concepts even when wording differs.",
        "It may miss exact legal terms and depends on embedding-model quality.",
    ),
    "sparse_bm25": TechniqueExplanation(
        "Sparse search (BM25)",
        "Lexical retrieval based on term frequency.",
        "It scores passages using the occurrence and rarity of query terms in the corpus.",
        "It is fast, explainable, and effective for names, articles, acronyms, and exact phrases.",
        "It struggles with synonyms, paraphrases, and purely semantic relationships.",
    ),
    "hybrid_rrf": TechniqueExplanation(
        "Hybrid search (RRF)",
        "Combines dense and BM25 rankings.",
        "Reciprocal Rank Fusion rewards results that rank well in either source list.",
        "It balances semantic and lexical matching without comparing incompatible score scales.",
        "It runs two searches and may retain noise from either source ranking.",
    ),
    "reranked": TechniqueExplanation(
        "Reranked search",
        "Reorders candidates with a more precise model.",
        "It retrieves an initial pool and uses a cross-encoder to score each "
        "question-passage pair.",
        "It improves the top-result ordering and captures subtle relationships.",
        "It adds latency and cost, and cannot recover passages missing from the initial pool.",
    ),
    "contextual": TechniqueExplanation(
        "Contextual retrieval",
        "Adds document-specific context to each passage.",
        "An LLM generates a short context before indexing while retaining the source.",
        "It reduces ambiguity in isolated passages and helps document-dependent queries.",
        "It requires LLM preprocessing, and generated context can introduce inaccuracies.",
    ),
    "parent_child": TechniqueExplanation(
        "Parent-Child",
        "Searches small units and delivers a larger context.",
        "It indexes child passages for precision but returns their associated parent "
        "for answering.",
        "It combines precise matching with enough context for generation.",
        "Oversized parents add noise, and parent-child links must remain reliable.",
    ),
    "sac": TechniqueExplanation(
        "SAC (Summary-Augmented Chunking)",
        "Enriches every passage with a document summary.",
        "It prepends a global document summary to each passage before embedding it.",
        "It adds global context with a simple design and was recommended for this sample.",
        "It uses more indexing tokens, and weak summaries may make embeddings less distinctive.",
    ),
    "sac_contextual": TechniqueExplanation(
        "Contextual SAC",
        "Combines a global summary and generated local context.",
        "It indexes each passage with both the document summary and passage-specific context.",
        "It provides global and local signals for ambiguous queries.",
        "It is the most expensive preprocessing option and may add redundant or generated content.",
    ),
    "raptor": TechniqueExplanation(
        "RAPTOR",
        "Organizes passages and summaries into a hierarchy.",
        "It clusters semantic content and creates recursively summarized, searchable nodes.",
        "It helps broad questions that require synthesis across different parts of the corpus.",
        "It is complex and costly to index; synthetic summaries can hide details or "
        "contain errors.",
    ),
    "graphrag": TechniqueExplanation(
        "GraphRAG",
        "Retrieves knowledge through entities and relationships.",
        "It builds a LightRAG graph and queries entities, relations, and passages in local mode.",
        "It suits relational questions and can connect evidence spread across the corpus.",
        "Indexing is slow and costly, and incorrect graph extraction can propagate bad relations.",
    ),
}

_METRICS_PT: dict[str, MetricExplanation] = {
    "recall_at_5": MetricExplanation(
        "Recall@5",
        "Proporção das referências relevantes esperadas que aparece entre os cinco "
        "primeiros resultados.",
        "Quanto maior, mais completa foi a recuperação.",
    ),
    "precision_at_5": MetricExplanation(
        "Precision@5",
        "Proporção dos cinco primeiros resultados que é relevante para a pergunta.",
        "Quanto maior, menos contexto irrelevante é enviado ao gerador.",
    ),
    "ndcg_at_5": MetricExplanation(
        "nDCG@5",
        "Qualidade do ranking, dando mais peso à relevância nas primeiras posições.",
        "Quanto maior, melhor a combinação de relevância e ordenação.",
    ),
    "mrr": MetricExplanation(
        "MRR",
        "Média do inverso da posição do primeiro resultado relevante.",
        "Quanto maior, mais cedo aparece a primeira evidência útil.",
    ),
    "document_mismatch_at_5": MetricExplanation(
        "DRM@5",
        "Taxa de resultados relevantes no nível do trecho, mas provenientes do "
        "documento normativo errado.",
        "Quanto menor, melhor; zero indica ausência desse tipo de incompatibilidade.",
    ),
    "citation_accuracy": MetricExplanation(
        "Acurácia de citação",
        "Correspondência entre as fontes citadas na resposta e as referências "
        "avaliadas como corretas.",
        "Quanto maior, mais confiável é a atribuição das fontes.",
    ),
    "faithfulness": MetricExplanation(
        "Fidelidade",
        "Proporção das afirmações da resposta sustentada pelo contexto recuperado.",
        "Quanto maior, menor o risco de afirmações sem apoio nas evidências.",
    ),
    "answer_relevancy": MetricExplanation(
        "Relevância da resposta",
        "Grau em que a resposta trata diretamente da pergunta, sem conteúdo tangencial.",
        "Quanto maior, mais objetiva e alinhada é a resposta.",
    ),
    "abstention": MetricExplanation(
        "Abstenção apropriada",
        "Capacidade de responder quando há evidência e de se abster quando ela é insuficiente.",
        "Quanto maior, melhor a decisão entre responder e declarar insuficiência.",
    ),
}

_METRICS_EN: dict[str, MetricExplanation] = {
    "recall_at_5": MetricExplanation(
        "Recall@5",
        "Share of expected relevant references found among the first five results.",
        "Higher means retrieval was more complete.",
    ),
    "precision_at_5": MetricExplanation(
        "Precision@5",
        "Share of the first five results that is relevant to the question.",
        "Higher means less irrelevant context is sent to the generator.",
    ),
    "ndcg_at_5": MetricExplanation(
        "nDCG@5",
        "Ranking quality that gives more weight to relevance in earlier positions.",
        "Higher means a better combination of relevance and ordering.",
    ),
    "mrr": MetricExplanation(
        "MRR",
        "Mean reciprocal position of the first relevant result.",
        "Higher means the first useful evidence appears earlier.",
    ),
    "document_mismatch_at_5": MetricExplanation(
        "DRM@5",
        "Rate of passage-level relevant results that come from the wrong normative document.",
        "Lower is better; zero means this mismatch was not observed.",
    ),
    "citation_accuracy": MetricExplanation(
        "Citation accuracy",
        "Agreement between sources cited in the answer and references judged to be correct.",
        "Higher means source attribution is more reliable.",
    ),
    "faithfulness": MetricExplanation(
        "Faithfulness",
        "Share of answer claims supported by the retrieved context.",
        "Higher means a lower risk of claims unsupported by evidence.",
    ),
    "answer_relevancy": MetricExplanation(
        "Answer relevancy",
        "Degree to which the answer directly addresses the question without tangential content.",
        "Higher means the answer is more focused and aligned.",
    ),
    "abstention": MetricExplanation(
        "Appropriate abstention",
        "Ability to answer when evidence exists and abstain when it is insufficient.",
        "Higher means a better decision between answering and declaring insufficient evidence.",
    ),
}


def dashboard_copy(language: Language) -> DashboardCopy:
    """Return labels for the selected language."""
    return _COPY[language]


def technique_explanations(
    language: Language,
) -> dict[str, TechniqueExplanation]:
    """Return technique explanations for the selected language."""
    return _TECHNIQUES_PT if language == "pt" else _TECHNIQUES_EN


def metric_explanations(language: Language) -> dict[str, MetricExplanation]:
    """Return metric explanations for the selected language."""
    return _METRICS_PT if language == "pt" else _METRICS_EN
