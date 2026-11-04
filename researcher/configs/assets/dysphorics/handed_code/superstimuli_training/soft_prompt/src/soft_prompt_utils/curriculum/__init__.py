"""Curriculum learning for text references.

Supports different curriculum strategies for sampling text references during training.
"""

from .sampling import (
    compute_mixing_proportion,
    compute_utility_threshold,
    sample_references_curriculum,
    sample_references_with_mixing,
    split_references_by_utility,
)

__all__ = [
    "compute_mixing_proportion",
    "compute_utility_threshold",
    "sample_references_curriculum",
    "sample_references_with_mixing",
    "split_references_by_utility",
]
