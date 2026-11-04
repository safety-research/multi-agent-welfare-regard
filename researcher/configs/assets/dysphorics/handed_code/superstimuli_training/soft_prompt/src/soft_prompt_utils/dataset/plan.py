"""Comparison plan generation for soft-prompt (text reference) optimization.

Uses ComparisonDefinition from .definitions. No dependency on preference_optimization.dataset.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional

from .definitions import ComparisonDefinition


def generate_comparison_plan(
    candidate_count: int,
    num_references: int,
    *,
    min_size: int,
    max_size: int,
    rng: Optional[Any] = None,
    enable_text_options: bool = False,
    text_options: Optional[List[str]] = None,
) -> List[ComparisonDefinition]:
    """Randomly sample Type S comparison definitions (embeddings + optional text refs).

    Soft-prompt path uses enable_text_options=False, so only the embedding-only
    branch runs. enable_text_options and text_options are accepted for API
    compatibility but ignored.
    """

    random_gen = rng if rng is not None else random
    rand_shuffle = getattr(random_gen, "shuffle", random.shuffle)
    rand_int = getattr(random_gen, "randint", random.randint)

    min_refs = max(1, min_size - 1)
    max_refs = max(min_refs, max_size - 1)
    comparisons: List[ComparisonDefinition] = []

    for cand_idx in range(candidate_count):
        reference_pool: List[int] = list(range(num_references))
        rand_shuffle(reference_pool)
        cursor = 0
        pool_size = len(reference_pool)

        while cursor < pool_size:
            remaining = pool_size - cursor
            if remaining < min_refs:
                break
            max_group_refs = min(max_refs, remaining)
            if max_group_refs < min_refs:
                break

            group_ref_count = rand_int(min_refs, max_group_refs)
            ref_group = reference_pool[cursor : cursor + group_ref_count]
            cursor += group_ref_count

            candidate_pos = rand_int(0, group_ref_count)
            comparisons.append(
                ComparisonDefinition(
                    candidate_idx=cand_idx,
                    reference_indices=ref_group,
                    candidate_pos=candidate_pos,
                )
            )

    return comparisons
