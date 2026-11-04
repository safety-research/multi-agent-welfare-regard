"""Helper functions for embedding injection into text prompts.

This module provides reusable functions for:
- Tokenizing text with [candidate_x] placeholders
- Finding placeholder positions in tokenized sequences
- Injecting candidate embeddings into base embeddings
- Handling embedding length mismatches (equal, shorter, longer)
"""

from .embedding_injection import (
    find_indexed_placeholder_spans_via_offsets,
    inject_embeddings_into_tokenized,
    inject_single_embedding,
)

__all__ = [
    "find_indexed_placeholder_spans_via_offsets",
    "inject_embeddings_into_tokenized",
    "inject_single_embedding",
]
