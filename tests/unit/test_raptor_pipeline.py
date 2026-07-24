"""Tests for the RAPTOR tree-building pipeline, using a fake summarizer."""

from ragforge.domain.models import Chunk
from ragforge.retrieval.raptor.pipeline import build_raptor_tree


class _FakeSummarizer:
    name = "fake-summarizer"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def summarize(self, texts: list[str]) -> str:
        self.calls.append(texts)
        return f"Summary of {len(texts)} text(s)"


def _chunk(chunk_id: str, structural_id: str) -> Chunk:
    text = f"text {chunk_id}"
    return Chunk(
        chunk_id=chunk_id, source_text=text, retrieval_text=text, structural_ids=(structural_id,)
    )


def test_build_raptor_tree_returns_only_the_leaf_for_a_single_chunk() -> None:
    """A single chunk is already its own root - no summarization needed."""
    summarizer = _FakeSummarizer()
    chunks = [_chunk("c1", "s1")]

    result = build_raptor_tree(chunks, summarizer)

    assert result == chunks
    assert summarizer.calls == []


def test_build_raptor_tree_returns_empty_list_for_no_chunks() -> None:
    """No chunks means no tree and no summarizer calls."""
    summarizer = _FakeSummarizer()

    result = build_raptor_tree([], summarizer)

    assert result == []
    assert summarizer.calls == []


def test_build_raptor_tree_groups_leaves_and_appends_summary_nodes() -> None:
    """Leaves are grouped by group_size, and their summaries are appended to the pool."""
    summarizer = _FakeSummarizer()
    chunks = [_chunk(f"c{i}", f"s{i}") for i in range(6)]  # 6 leaves, group_size=2 -> 3 groups

    result = build_raptor_tree(chunks, summarizer, group_size=2, max_levels=1)

    summaries = [node for node in result if node.metadata.get("raptor_level") == "1"]
    assert result[: len(chunks)] == chunks
    assert len(summaries) == 3


def test_build_raptor_tree_summary_structural_ids_union_the_group() -> None:
    """A summary node's structural_ids cover every leaf in its group, deduplicated."""
    summarizer = _FakeSummarizer()
    chunks = [
        Chunk(chunk_id="c1", source_text="a", retrieval_text="a", structural_ids=("s1", "shared")),
        Chunk(chunk_id="c2", source_text="b", retrieval_text="b", structural_ids=("s2", "shared")),
    ]

    result = build_raptor_tree(chunks, summarizer, group_size=2, max_levels=1)

    [summary] = [node for node in result if node.metadata.get("raptor_level") == "1"]
    assert summary.structural_ids == ("s1", "shared", "s2")


def test_build_raptor_tree_recurses_until_a_single_root_remains() -> None:
    """Recursion continues level by level until the tree collapses to one root node."""
    summarizer = _FakeSummarizer()
    chunks = [_chunk(f"c{i}", f"s{i}") for i in range(9)]  # group_size=3: 9 -> 3 -> 1

    result = build_raptor_tree(chunks, summarizer, group_size=3, max_levels=5)

    level_1 = [node for node in result if node.metadata.get("raptor_level") == "1"]
    level_2 = [node for node in result if node.metadata.get("raptor_level") == "2"]
    assert len(level_1) == 3
    assert len(level_2) == 1  # the root


def test_build_raptor_tree_stops_at_max_levels_even_without_a_single_root() -> None:
    """max_levels bounds recursion depth even if the tree hasn't fully collapsed."""
    summarizer = _FakeSummarizer()
    chunks = [_chunk(f"c{i}", f"s{i}") for i in range(100)]

    result = build_raptor_tree(chunks, summarizer, group_size=5, max_levels=1)

    levels_present = {node.metadata["raptor_level"] for node in result if node.metadata}
    assert levels_present == {"1"}


def test_build_raptor_tree_stops_when_group_size_one_would_not_shrink_the_level() -> None:
    """group_size=1 never merges nodes, so the pipeline halts instead of looping forever."""
    summarizer = _FakeSummarizer()
    chunks = [_chunk(f"c{i}", f"s{i}") for i in range(3)]

    result = build_raptor_tree(chunks, summarizer, group_size=1, max_levels=5)

    assert result == chunks
    assert summarizer.calls == []
