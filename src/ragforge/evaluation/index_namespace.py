"""Derives a deterministic index namespace from complete identity (ADR-0013).

An index must never be reused after any identity component changes - corpus
content, chunking logic, the retrieval-text schema, or the embedding
configuration itself. This module turns that full identity into one short,
deterministic token suitable as a table/index name suffix.
"""

import hashlib

from ragforge.embeddings.identity import EmbeddingIdentity

_NAMESPACE_LENGTH = 16


def derive_index_namespace(
    corpus_hash: str,
    chunking_config_version: str,
    retrieval_text_schema_version: str,
    embedding: EmbeddingIdentity,
) -> str:
    """Return a short deterministic hex token identifying this exact configuration.

    Two calls with identical arguments always return the same token; any
    single differing component (including an embedding identity field)
    changes it.
    """
    payload = "|".join(
        [
            corpus_hash,
            chunking_config_version,
            retrieval_text_schema_version,
            embedding.provider,
            embedding.model,
            embedding.revision,
            str(embedding.dimensions),
            str(embedding.normalize),
            embedding.query_instruction_hash,
            embedding.runtime,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_NAMESPACE_LENGTH]
