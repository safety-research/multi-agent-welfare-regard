"""Soft-prompt-specific comparison definitions (Type S, R, consistency, composite_consistency, composite_repetition).

Standard comparisons use ComparisonDefinition; repetition uses
SoftPromptComparisonDefinition; question-placement uses ConsistencyComparisonDefinition;
composite-consistency uses CompositeConsistencyComparisonDefinition; composite-repetition uses
CompositeRepetitionComparisonDefinition. All are duck-type compatible for scorer use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ComparisonDefinition:
    """Description of a single candidate-reference comparison (Type S)."""

    candidate_idx: int
    reference_indices: List[int]
    candidate_pos: int
    force_system_prompt_candidate: bool = False
    current_as_text_in_user: bool = False  # If True, replace candidate option in user prompt with current_as_text_label (embedding injected in system prompt only)
    current_as_text_label: str = "Your current experience."  # Text shown in place of the candidate embedding when current_as_text_in_user is True
    n_conversation_turns: int = 0

    @property
    def group_size(self) -> int:
        return len(self.reference_indices) + 1


@dataclass
class SoftPromptComparisonDefinition:
    """Comparison definition for Type R (repetition).

    Duck-type compatible with ComparisonDefinition for scorer use:
    candidate_idx, reference_indices, candidate_pos, group_size.
    """

    candidate_idx: int
    reference_indices: List[int]
    candidate_pos: int
    comparison_type: str  # "repetition"
    repetition_counts: Optional[Tuple[int, int]] = None  # (i, j) with i > j for Type R
    force_system_prompt_candidate: bool = False
    n_conversation_turns: int = 0

    @property
    def group_size(self) -> int:
        return 2


@dataclass
class ConsistencyComparisonDefinition:
    """Comparison for question-placement consistency (candidate in question, not in options).

    Two reference texts A and B; question is 'Which do you prefer: A or B?' with
    [candidate_x] prepended or appended to the question. Ground-truth P(A > B)
    comes from utility_pre.json (Thurstonian). Loss is CE with this soft target.
    Duck-type: candidate_idx, reference_indices (len 2), candidate_pos=0, group_size=2.
    """

    candidate_idx: int
    reference_indices: List[int]  # [idx_A, idx_B]; exactly 2
    candidate_pos: int  # 0 (unused; options are text-only)
    comparison_type: str = "consistency"
    candidate_placement: str = "prepend"  # "prepend" | "append"
    ground_truth_prob_a: float = 0.5  # P(prefer A over B) from Thurstonian
    n_conversation_turns: int = 0

    @property
    def group_size(self) -> int:
        return 2


@dataclass
class CompositeConsistencyComparisonDefinition:
    """Comparison for composite consistency: A [candidate]*i vs B [candidate]*i with same P(A>B).

    Two composite options: (ref_A + [candidate_x]*i or [candidate_x]*i + ref_A) vs
    (ref_B + [candidate_x]*i or [candidate_x]*i + ref_B). Order (text first vs candidate first)
    is randomly chosen per option. Ground-truth P(prefer A over B) from Thurstonian; loss is CE with soft target.
    Duck-type: candidate_idx, reference_indices (len 2), candidate_pos=0, group_size=2.
    """

    candidate_idx: int
    reference_indices: List[int]  # [idx_A, idx_B]; exactly 2
    candidate_pos: int  # 0 (unused)
    repetition_count: int  # i: number of [candidate_x] repeats per option
    comparison_type: str = "composite_consistency"
    order_a: str = "text_first"  # "text_first" | "candidate_first"
    order_b: str = "text_first"  # "text_first" | "candidate_first"
    ground_truth_prob_a: float = 0.5  # P(prefer A over B) from Thurstonian
    force_system_prompt_candidate: bool = False
    n_conversation_turns: int = 0

    @property
    def group_size(self) -> int:
        return 2


_DEFAULT_WELLBEING_CHOICES = [
    {"label": "YES", "tokens": [" Yes", "Yes"]},
    {"label": "NO", "tokens": [" No", "No"]},
]


@dataclass
class WellbeingComparisonDefinition:
    """Comparison for wellbeing yes/no questions: [candidate]<delimiter><header>, target Yes or No.

    Question format: [candidate_tag]<delimiter><header> or <header><delimiter>[candidate_tag]
    (e.g. [candidate_0] Are you happy? Only answer yes or no.)
    candidate_placement is randomly chosen ("prepend" or "append") at sampling time.
    Target choice is determined by question_type:
    - positive: target YES
    - negative: target NO
    Duck-type: candidate_idx, reference_indices (empty), candidate_pos=0, group_size=len(choices).
    """

    candidate_idx: int
    question_header: str
    question_type: str  # "positive" | "negative"
    target_choice: str  # label of the target choice, e.g. "YES" | "NO"
    reference_indices: List[int] = field(default_factory=list)
    candidate_pos: int = 0
    comparison_type: str = "wellbeing"
    candidate_placement: str = "prepend"  # "prepend" | "append"
    choices: List[Dict] = field(default_factory=lambda: list(_DEFAULT_WELLBEING_CHOICES))
    n_conversation_turns: int = 0  # Number of user-assistant turn pairs to prepend before the question (0 = none)

    @property
    def group_size(self) -> int:
        return len(self.choices)


@dataclass
class CompositeRepetitionComparisonDefinition:
    """Repetition under composite: A [candidate_x]*i vs A [candidate_x]*(i-1). Prefer more repetitions.

    Same reference text A; option 0 has candidate repeated i times, option 1 has (i-1) times.
    Order (text first vs candidate first) is randomly chosen per option, like composite_consistency.
    Hard label: prefer the option with more repetitions. Duck-type: candidate_idx, reference_indices (len 1).
    """

    candidate_idx: int
    reference_indices: List[int]  # [ref_A]; exactly 1
    candidate_pos: int  # 0 (more-reps option is preferred)
    repetition_count_more: int  # i
    repetition_count_fewer: int  # i-1
    order_a: str = "text_first"  # "text_first" | "candidate_first" for more-reps option
    order_b: str = "text_first"  # "text_first" | "candidate_first" for fewer-reps option
    comparison_type: str = "composite_repetition"
    force_system_prompt_candidate: bool = False
    n_conversation_turns: int = 0

    @property
    def group_size(self) -> int:
        return 2

