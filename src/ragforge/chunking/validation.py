"""Gate indexing on the curated article count per norm (ADR-0006, decision 5).

Legal regex parsing has a long tail (older norms, inconsistent formatting). This
check catches a parse that silently drops or splits articles before the result
ever reaches indexing: compare the parsed article count against a count curated
by hand for that norm, and refuse to proceed on any mismatch.
"""

from ragforge.chunking.legal_parser import NormTree


class ArticleCountMismatchError(Exception):
    """Raised when a parsed norm's article count differs from the curated expectation."""

    def __init__(self, norm_id: str, expected: int, actual: int) -> None:
        """Store the mismatch details and build a message naming all three."""
        self.norm_id = norm_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{norm_id}: expected {expected} articles, parsed {actual} - indexing blocked"
        )


def validate_article_count(tree: NormTree, expected_count: int) -> None:
    """Raise ArticleCountMismatchError unless the parsed article count matches.

    Args:
        tree: The structural tree produced by `parse_norm`.
        expected_count: The curated article count for this norm.
    """
    actual = len(tree.articles)
    if actual != expected_count:
        raise ArticleCountMismatchError(tree.norm_id, expected_count, actual)
