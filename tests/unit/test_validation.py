"""Tests for the per-norm article-count validation gate (ADR-0006, decision 5)."""

import pytest

from ragforge.chunking.legal_parser import parse_norm
from ragforge.chunking.validation import ArticleCountMismatchError, validate_article_count

NORM_ID = "RES-CMN-4893/2021"

TWO_ARTICLES = """
Art. 1º Primeiro artigo.

Art. 2º Segundo artigo.
"""


def test_passes_silently_when_article_count_matches_expected() -> None:
    """No exception is raised when the parsed count equals the curated count."""
    tree = parse_norm(NORM_ID, TWO_ARTICLES)
    validate_article_count(tree, expected_count=2)


def test_raises_when_parser_finds_fewer_articles_than_expected() -> None:
    """A parse that silently drops an article (e.g. a formatting long-tail) is caught."""
    tree = parse_norm(NORM_ID, TWO_ARTICLES)
    with pytest.raises(ArticleCountMismatchError) as exc_info:
        validate_article_count(tree, expected_count=3)
    assert exc_info.value.norm_id == NORM_ID
    assert exc_info.value.expected == 3
    assert exc_info.value.actual == 2


def test_raises_when_parser_finds_more_articles_than_expected() -> None:
    """A parse that splits one article into two is also caught, not just under-counts."""
    tree = parse_norm(NORM_ID, TWO_ARTICLES)
    with pytest.raises(ArticleCountMismatchError):
        validate_article_count(tree, expected_count=1)


def test_error_message_names_norm_expected_and_actual_counts() -> None:
    """The exception message is actionable on its own, without inspecting attributes."""
    tree = parse_norm(NORM_ID, TWO_ARTICLES)
    with pytest.raises(ArticleCountMismatchError, match=f"{NORM_ID}.*expected 5.*parsed 2"):
        validate_article_count(tree, expected_count=5)
