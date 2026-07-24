"""Provider-neutral embedding identity (ADR-0013).

Describes exactly which embedding configuration produced a vector space, so
an index can be namespaced by complete identity and a run record can name
precisely what it used - never just a bare model name, which alone doesn't
distinguish e.g. a hosted API's default normalization from a local model's.
"""

import hashlib
from dataclasses import dataclass

# Placeholder until embed_documents/embed_query with a real, versioned query
# instruction exists (deferred - see ADR-0013 rollout step 5). Every current
# adapter embeds queries and documents identically, so every identity today
# carries this same "no instruction" marker rather than a fabricated one.
NO_QUERY_INSTRUCTION_HASH = hashlib.sha256(b"").hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Exact identity of one embedding configuration (ADR-0013)."""

    provider: str
    model: str
    revision: str
    dimensions: int
    normalize: bool
    query_instruction_hash: str
    runtime: str
