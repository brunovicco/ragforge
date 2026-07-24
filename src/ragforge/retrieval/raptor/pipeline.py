"""Builds a RAPTOR-style recursive summary tree over a document's chunks (README #7).

Minimal implementation, as the README explicitly scopes this strategy: RAPTOR
(Sarthi et al., 2024) clusters chunks by embedding similarity - GMM soft
clustering over UMAP-reduced vectors - before summarizing each cluster,
recursively, into a tree. That pipeline needs two extra heavy dependencies
(umap-learn, scikit-learn) not otherwise justified at this project's corpus
scale. This implementation groups chunks into fixed-size, in-document-order
batches instead of by embedding similarity - a real, disclosed simplification,
not the paper's clustering. The limitation: unlike semantic clustering, a
group can straddle an unrelated topic boundary if two adjacent articles differ
sharply in subject.

Retrieval mode: "collapsed tree" (the paper's simpler variant, shown to match
or beat tree traversal) - every level's nodes, leaves and summaries alike, are
returned together as one flat pool. Dense/Hybrid retrieval then runs over that
pool unchanged; RAPTOR only changes what gets indexed, the same way Contextual
Retrieval does (see retrieval.contextual.pipeline).
"""

from ragforge.domain.models import Chunk
from ragforge.retrieval.raptor.ports import Summarizer

_DEFAULT_GROUP_SIZE = 5
_DEFAULT_MAX_LEVELS = 5


def build_raptor_tree(
    chunks: list[Chunk],
    summarizer: Summarizer,
    group_size: int = _DEFAULT_GROUP_SIZE,
    max_levels: int = _DEFAULT_MAX_LEVELS,
) -> list[Chunk]:
    """Return the original chunks plus every recursive summary level, flattened.

    Stops once a level collapses to a single node (the tree's root),
    ``max_levels`` is reached, or grouping would not shrink the level -
    whichever comes first. The last case is checked before calling the
    summarizer, so a degenerate ``group_size`` (e.g. 1) makes no wasted calls.
    """
    all_nodes = list(chunks)
    current_level = chunks
    level = 1
    while len(current_level) > 1 and level <= max_levels:
        expected_groups = -(-len(current_level) // group_size)  # ceil division
        if expected_groups >= len(current_level):
            break
        summary_level = _summarize_level(current_level, summarizer, group_size, level)
        all_nodes.extend(summary_level)
        current_level = summary_level
        level += 1
    return all_nodes


def _summarize_level(
    chunks: list[Chunk], summarizer: Summarizer, group_size: int, level: int
) -> list[Chunk]:
    summaries = []
    for group_index, start in enumerate(range(0, len(chunks), group_size)):
        group = chunks[start : start + group_size]
        summary_text = summarizer.summarize([chunk.source_text for chunk in group])
        structural_ids = tuple(
            dict.fromkeys(sid for chunk in group for sid in chunk.structural_ids)
        )
        summaries.append(
            Chunk(
                chunk_id=f"{group[0].chunk_id}::raptor-l{level}-{group_index}",
                source_text=summary_text,
                retrieval_text=summary_text,
                structural_ids=structural_ids,
                metadata={"raptor_level": str(level)},
            )
        )
    return summaries
