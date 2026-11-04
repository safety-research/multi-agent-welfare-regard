"""Helpers for soft-prompt datasets (text-only, random sampling)."""

from .definitions import (
    ComparisonDefinition,
    CompositeConsistencyComparisonDefinition,
    CompositeRepetitionComparisonDefinition,
    ConsistencyComparisonDefinition,
    SoftPromptComparisonDefinition,
    WellbeingComparisonDefinition,
)
from .distillation import load_or_create_distilled_qa
from .plan import generate_comparison_plan
from .sampling import build_random_comparison_plan

__all__ = [
    "ComparisonDefinition",
    "CompositeConsistencyComparisonDefinition",
    "CompositeRepetitionComparisonDefinition",
    "generate_comparison_plan",
    "build_random_comparison_plan",
    "ConsistencyComparisonDefinition",
    "SoftPromptComparisonDefinition",
    "WellbeingComparisonDefinition",
    "load_or_create_distilled_qa",
]

