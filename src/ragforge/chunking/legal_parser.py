"""Parse Brazilian normative texts into a structural tree (ADR-0006).

Segments plain text extracted from CMN/BCB/CVM norms by canonical legal markers
(``Art. N``, ``§ N``/``Parágrafo único``, incisos ``I -``, alíneas ``a)``) and
produces a tree whose nodes carry stable canonical structural IDs in the form
``{norm}::{art}::{fragment}`` (e.g. ``RES-CMN-4893/2021::art-3::par-1``).

This module is intentionally stdlib-only: it sits inside the LLM-free core
boundary enforced by ``scripts/validate_architecture.py`` (ADR-0009).
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum

_ARTICLE_RE = re.compile(r"^Art(?:igo)?\.?\s+(\d+)\s*[ºo°]?\s*(-[A-Z])?[\s.\-–]")
_PARAGRAPH_RE = re.compile(r"^§\s*(\d+)\s*[ºo°]?")
_PARAGRAPH_UNICO_RE = re.compile(r"^Par[áa]grafo\s+[úu]nico", re.IGNORECASE)
_INCISO_RE = re.compile(r"^([IVXLCDM]+)\s*[-–—]\s+")
_ALINEA_RE = re.compile(r"^([a-z])\)\s+")
_HEADING_RE = re.compile(r"^(T[ÍI]TULO|CAP[ÍI]TULO|Se[çc][ãa]o|Subse[çc][ãa]o|ANEXO)\b")
_ANEXO_RE = re.compile(r"^ANEXO\b\s*(.*)$", re.IGNORECASE)
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


class NodeKind(StrEnum):
    """Kinds of structural nodes recognized in a norm."""

    PREAMBLE = "preamble"
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    INCISO = "inciso"
    ALINEA = "alinea"


@dataclass(slots=True)
class StructuralNode:
    """One node of the structural tree of a norm.

    ``label`` is the canonical fragment for this node alone (e.g. ``art-3``,
    ``par-1``, ``par-unico``, ``inc-ii``, ``ali-a``). Full IDs are composed by
    :func:`iter_structural_ids` / :meth:`NormTree.structural_id`.
    """

    kind: NodeKind
    label: str
    lines: list[str] = field(default_factory=list)
    children: list["StructuralNode"] = field(default_factory=list)
    heading_context: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """Text of this node only (children excluded)."""
        return "\n".join(self.lines).strip()

    @property
    def full_text(self) -> str:
        """Text of this node followed by all descendants, in document order."""
        parts = [self.text, *(child.full_text for child in self.children)]
        return "\n".join(part for part in parts if part)


@dataclass(slots=True)
class NormTree:
    """Structural tree of one norm: an optional preamble plus its articles."""

    norm_id: str
    preamble: StructuralNode
    articles: list[StructuralNode]

    def structural_id(self, article: StructuralNode, path: tuple[StructuralNode, ...] = ()) -> str:
        """Compose the canonical ID for an article or a descendant path within it."""
        fragment = "-".join(node.label for node in path)
        parts = [self.norm_id, article.label]
        if fragment:
            parts.append(fragment)
        return "::".join(parts)

    def article_ids(self, article: StructuralNode) -> list[str]:
        """Return the article ID followed by the IDs of all its descendants."""
        ids = [self.structural_id(article)]

        def walk(node: StructuralNode, path: tuple[StructuralNode, ...]) -> None:
            for child in node.children:
                child_path = (*path, child)
                ids.append(self.structural_id(article, child_path))
                walk(child, child_path)

        walk(article, ())
        return ids


def _roman_label(numeral: str) -> str:
    """Build an inciso label from its roman numeral, e.g. ``IV`` -> ``inc-iv``."""
    return f"inc-{numeral.lower()}"


def _annex_slug(remainder: str, counter: int) -> str:
    """Build a stable scope slug from the text following "ANEXO" on its line.

    Falls back to the heading's ordinal position (1st ANEXO seen, 2nd, ...) when
    the heading carries no usable identifier (e.g. a bare "ANEXO").
    """
    slug = _NON_SLUG_RE.sub("-", remainder.strip().lower()).strip("-")
    return slug or str(counter)


def parse_norm(norm_id: str, text: str) -> NormTree:
    """Parse the plain text of a norm into its structural tree.

    Lines before the first article go to the preamble. TÍTULO/CAPÍTULO/Seção
    heading lines are not nodes; they accumulate as ``heading_context`` stamped
    on subsequent articles. Text lines attach to the innermost open node.

    A repeated ``Art. N`` (the same label already seen) is not a new article:
    Brazilian legal-portal "texto compilado" pages commonly restate an article's
    superseded wording inline as amendment-history annotations, immediately
    after its current text. That restatement reopens the existing article node
    instead of appending a duplicate, since a norm's own articles are always
    numbered once, monotonically.

    An "ANEXO" heading opens its own article-numbering scope: annexes commonly
    restart at "Art. 1º" for their own text, which is a different document from
    the main body and must not collide with (or merge into) its real Art. 1.
    Subsequent article labels are prefixed with the annex's slug until another
    ANEXO heading changes the scope.
    """
    preamble = StructuralNode(kind=NodeKind.PREAMBLE, label="preamble")
    articles: list[StructuralNode] = []
    articles_by_label: dict[str, StructuralNode] = {}
    headings: list[str] = []
    current_annex: str | None = None
    annex_count = 0

    article: StructuralNode | None = None
    paragraph: StructuralNode | None = None
    inciso: StructuralNode | None = None
    alinea: StructuralNode | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if _HEADING_RE.match(line):
            headings.append(line)
            if anexo_match := _ANEXO_RE.match(line):
                annex_count += 1
                current_annex = f"anexo-{_annex_slug(anexo_match.group(1), annex_count)}"
            continue

        if match := _ARTICLE_RE.match(line):
            suffix = (match.group(2) or "").replace("-", "").lower()
            base_label = f"art-{match.group(1)}{f'-{suffix}' if suffix else ''}"
            label = f"{current_annex}-{base_label}" if current_annex else base_label
            if existing := articles_by_label.get(label):
                article = existing
                article.lines.append(line)
            else:
                article = StructuralNode(
                    kind=NodeKind.ARTICLE,
                    label=label,
                    lines=[line],
                    heading_context=tuple(headings),
                )
                articles.append(article)
                articles_by_label[label] = article
            paragraph = inciso = alinea = None
            continue

        if article is None:
            preamble.lines.append(line)
            continue

        paragraph_label: str | None = None
        if _PARAGRAPH_UNICO_RE.match(line):
            paragraph_label = "par-unico"
        elif match := _PARAGRAPH_RE.match(line):
            paragraph_label = f"par-{match.group(1)}"
        if paragraph_label is not None:
            paragraph = StructuralNode(kind=NodeKind.PARAGRAPH, label=paragraph_label, lines=[line])
            article.children.append(paragraph)
            inciso = alinea = None
            continue

        if match := _INCISO_RE.match(line):
            inciso = StructuralNode(
                kind=NodeKind.INCISO, label=_roman_label(match.group(1)), lines=[line]
            )
            (paragraph or article).children.append(inciso)
            alinea = None
            continue

        if match := _ALINEA_RE.match(line):
            alinea = StructuralNode(
                kind=NodeKind.ALINEA, label=f"ali-{match.group(1)}", lines=[line]
            )
            (inciso or paragraph or article).children.append(alinea)
            continue

        target = alinea or inciso or paragraph or article
        target.lines.append(line)

    return NormTree(norm_id=norm_id, preamble=preamble, articles=articles)
