"""Shared bracket-citation parsing (ADR-0002/0006).

Both the answer generator (which reports a flat, de-duplicated, filtered
citation list) and the post-generation citation auditor (ADR-0016, which
needs the same extraction repeated per claim/sentence) parse structural-ID
citations out of raw generated text the same way - this module is the one
place that regex and candidate-extraction logic lives, so the two never
drift apart.
"""

import re

from ragforge.domain.models import StructuralRef

CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def extract_citation_candidates(text: str) -> tuple[str, ...]:
    """Return every bracketed span in ``text``, well-formed or not, in first-seen order.

    Unlike ``extract_citations``, nothing is filtered out here - the
    post-generation citation auditor (ADR-0016) needs to see and flag a
    malformed candidate as such, not have it silently disappear before its
    ``well_formed`` check ever runs.
    """
    seen: dict[str, None] = {}
    for candidate in CITATION_RE.findall(text):
        seen.setdefault(candidate, None)
    return tuple(seen)


def extract_citations(text: str) -> tuple[str, ...]:
    """Return every well-formed structural ID cited in ``text``, in first-cited order.

    A bracketed span that isn't a valid structural ID (a hallucinated or
    unrelated bracket) is skipped rather than raised - this parses untrusted
    model output, not a contract the model is guaranteed to honor.
    """
    seen: dict[str, None] = {}
    for candidate in CITATION_RE.findall(text):
        try:
            ref = StructuralRef.parse(candidate)
        except ValueError:
            continue
        seen.setdefault(ref.canonical, None)
    return tuple(seen)
