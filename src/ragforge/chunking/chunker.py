"""Derive retrievable chunks from a norm's structural tree (ADR-0006).

The target unit is the article. Articles longer than ``max_chars`` are
subdivided into their top-level fragments (paragraphs/incisos), each carrying
``parent_id`` pointing at the article - which is exactly the hierarchy the
parent-child retrieval strategy needs. Every chunk carries ``structural_ids``,
the provenance mapping required by evaluation (ADR-0002) and governance.
"""

from ragforge.chunking.legal_parser import NormTree, StructuralNode
from ragforge.domain.models import Chunk

DEFAULT_MAX_CHARS = 1800

_ROLE_ARTICLE = "article"
_ROLE_FRAGMENT = "fragment"
_ROLE_PREAMBLE = "preamble"


def _base_metadata(tree: NormTree, article: StructuralNode, role: str) -> dict[str, str]:
    """Build shared chunk metadata for one article of a norm."""
    metadata = {"norm": tree.norm_id, "role": role}
    if article.heading_context:
        metadata["section"] = " > ".join(article.heading_context)
    return metadata


def _subtree_ids(
    tree: NormTree,
    article: StructuralNode,
    root: StructuralNode,
    root_path: tuple[StructuralNode, ...],
) -> list[str]:
    """Collect the canonical IDs of ``root`` and all its descendants."""
    ids = [tree.structural_id(article, root_path)]
    for child in root.children:
        ids.extend(_subtree_ids(tree, article, child, (*root_path, child)))
    return ids


def chunk_norm(tree: NormTree, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """Turn a parsed norm into chunks with structural provenance.

    Emits one chunk per article (role ``article``). When an article's full text
    exceeds ``max_chars``, additionally emits one chunk per top-level fragment
    (role ``fragment``, ``parent_id`` set to the article chunk). A non-empty
    preamble becomes a single ``preamble`` chunk.
    """
    chunks: list[Chunk] = []

    if tree.preamble.text:
        preamble_id = f"{tree.norm_id}::preamble"
        chunks.append(
            Chunk(
                chunk_id=preamble_id,
                source_text=tree.preamble.text,
                retrieval_text=tree.preamble.text,
                structural_ids=(preamble_id,),
                metadata={"norm": tree.norm_id, "role": _ROLE_PREAMBLE},
            )
        )

    for article in tree.articles:
        article_id = tree.structural_id(article)
        chunks.append(
            Chunk(
                chunk_id=article_id,
                source_text=article.full_text,
                retrieval_text=article.full_text,
                structural_ids=tuple(tree.article_ids(article)),
                metadata=_base_metadata(tree, article, _ROLE_ARTICLE),
            )
        )
        if len(article.full_text) <= max_chars:
            continue

        for child in article.children:
            chunks.append(
                Chunk(
                    chunk_id=tree.structural_id(article, (child,)),
                    source_text=child.full_text,
                    retrieval_text=child.full_text,
                    structural_ids=tuple(_subtree_ids(tree, article, child, (child,))),
                    parent_id=article_id,
                    metadata=_base_metadata(tree, article, _ROLE_FRAGMENT),
                )
            )

    return chunks
