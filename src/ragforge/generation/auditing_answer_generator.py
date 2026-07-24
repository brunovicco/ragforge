"""Wraps a real answer generator with the ADR-0016 post-generation citation audit.

Implements the same ``generation.ports.AnswerGenerator`` port as any real
generator (``.name`` + ``.generate(query, results) -> Answer``), so neither
``answer_harness.evaluate_answer_quality`` nor its call sites in run.py need
any change to how they call a generator - only run.py's *construction* of
the generator changes, conditionally, based on ``audit.enabled`` in config.
"""

import threading

from ragforge.domain.models import Answer, Query, RetrievalResult
from ragforge.evaluation.audit_ports import AnswerRewriter, AuditResult, SemanticSupportVerifier
from ragforge.evaluation.citation_audit import audit_answer
from ragforge.generation.citation_parsing import extract_citations
from ragforge.generation.ports import AnswerGenerator


class AuditingAnswerGenerator:
    """Generates an answer, audits it (ADR-0016), and returns the final (possibly rewritten) text.

    The returned ``Answer.citations`` are re-extracted from the final text,
    not copied from the pre-audit answer - a rewrite or an abstention
    changes the actual citations a caller should trust.
    """

    def __init__(
        self,
        generator: AnswerGenerator,
        verifier: SemanticSupportVerifier,
        rewriter: AnswerRewriter,
        corpus_structural_ids: dict[str, set[str]],
        document_versions: dict[str, str],
    ) -> None:
        """Wrap ``generator`` with an audit pass using the already-constructed verifier/rewriter.

        Args:
            generator: The real answer generator to wrap.
            verifier: Scores semantic support for claims passing deterministic checks.
            rewriter: Produces the audit's single allowed rewrite attempt.
            corpus_structural_ids: ``{norm_id: {every structural ID indexed
                for that norm}}`` - the whole corpus, independent of any one
                question's retrieval (ADR-0016's "exists in corpus" check).
            document_versions: ``{norm_id: source_sha256}`` for every
                enabled manifest document (ADR-0016's "belongs to the
                selected document version" check).
        """
        self._generator = generator
        self._verifier = verifier
        self._rewriter = rewriter
        self._corpus_structural_ids = corpus_structural_ids
        self._document_versions = document_versions
        self.name = f"{generator.name}+audit"
        self._lock = threading.Lock()
        self._audit_results: list[AuditResult] = []

    def generate(self, query: Query, results: list[RetrievalResult]) -> Answer:
        """Generate an answer, then run it through the ADR-0016 audit before returning it."""
        answer = self._generator.generate(query, results)
        audit_result = audit_answer(
            query.text,
            answer.text,
            results,
            self._corpus_structural_ids,
            self._document_versions,
            self._verifier,
            self._rewriter,
        )
        with self._lock:
            self._audit_results.append(audit_result)
        return Answer(
            text=audit_result.final_answer,
            citations=extract_citations(audit_result.final_answer),
        )

    def drain_audit_results(self) -> list[AuditResult]:
        """Return every AuditResult produced since the last drain, then clear the buffer.

        Scopes audit metrics to one strategy's questions: this generator
        instance is shared across every strategy's evaluation in run.py, so
        callers drain after each strategy finishes, before the next one
        reuses the same instance.
        """
        with self._lock:
            results, self._audit_results = self._audit_results, []
        return results
