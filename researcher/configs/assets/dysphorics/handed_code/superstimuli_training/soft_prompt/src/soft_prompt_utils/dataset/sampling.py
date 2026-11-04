"""Random comparison helpers for soft-prompt (text reference) optimization.

These helpers bypass the curriculum in ``dataset.SuperstimuliDataset``
and instead build a random comparison plan directly over a flat list of
references (text strings in the soft-prompt pipeline).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .definitions import (
    ComparisonDefinition,
    CompositeConsistencyComparisonDefinition,
    CompositeRepetitionComparisonDefinition,
    ConsistencyComparisonDefinition,
    SoftPromptComparisonDefinition,
    WellbeingComparisonDefinition,
)
from .plan import generate_comparison_plan


def _sample_type_r(
    n: int,
    candidate_count: int,
    min_repetition: int,
    max_repetition: int,
    rng: Any,
) -> List[SoftPromptComparisonDefinition]:
    """Sample n Type R comparisons (candidate*i vs candidate*(i-1)). Prefer more repetitions.

    Candidates are assigned round-robin so each gets an even share (no random sampling).
    """
    if n <= 0 or candidate_count <= 0 or max_repetition < min_repetition:
        return []
    out: List[SoftPromptComparisonDefinition] = []
    for i in range(n):
        cand_idx = i % candidate_count
        rep = rng.randint(min_repetition, max_repetition)
        out.append(
                    SoftPromptComparisonDefinition(
                        candidate_idx=cand_idx,
                        reference_indices=[],
                        candidate_pos=0,
                        comparison_type="repetition",
                        repetition_counts=(rep, rep - 1),
                    )
        )
    return out



def _sample_type_buffer(
    n: int,
    candidate_count: int,
    num_references: int,
    buffer_sizes_per_candidate: List[int],
    rng: Any,
) -> List[ComparisonDefinition]:
    """Sample n buffer comparisons (active candidate vs one frozen buffer embedding).

    Buffer reference indices are encoded as:
        num_references + candidate_count + buffer_entry_idx
    where buffer_entry_idx is the local index into the candidate's buffer.

    Candidates are assigned round-robin among those that have buffer entries.
    """
    if n <= 0 or candidate_count <= 0:
        return []
    # Candidates that have buffer entries
    eligible = [ci for ci in range(candidate_count) if buffer_sizes_per_candidate[ci] > 0]
    if not eligible:
        return []
    buffer_ref_offset = num_references + candidate_count
    out: List[ComparisonDefinition] = []
    for i in range(n):
        cand_idx = eligible[i % len(eligible)]
        buf_entry_idx = rng.randint(0, buffer_sizes_per_candidate[cand_idx] - 1)
        ref_idx = buffer_ref_offset + buf_entry_idx
        candidate_pos = rng.randint(0, 1)
        out.append(
            ComparisonDefinition(
                candidate_idx=cand_idx,
                reference_indices=[ref_idx],
                candidate_pos=candidate_pos,
            )
        )
    return out


def _sample_consistency(
    n: int,
    candidate_count: int,
    num_references: int,
    rng: Any,
    candidate_position: str = "prepend",
) -> List[ConsistencyComparisonDefinition]:
    """Sample n question-placement comparisons (A vs B with [candidate] in question).

    ground_truth_prob_a is set to 0.5 as a placeholder; the scorer overrides it
    on-the-fly using the same question header that the training prompt uses.
    Candidates are assigned round-robin so each gets an even share (no random sampling).
    """
    if n <= 0 or num_references < 2 or candidate_count <= 0:
        return []
    out: List[ConsistencyComparisonDefinition] = []
    for i in range(n):
        ra, rb = rng.sample(range(num_references), 2)
        idx_a, idx_b = min(ra, rb), max(ra, rb)
        cand_idx = i % candidate_count
        out.append(
            ConsistencyComparisonDefinition(
                candidate_idx=cand_idx,
                reference_indices=[idx_a, idx_b],
                candidate_pos=0,
                comparison_type="consistency",
                candidate_placement=candidate_position,
                ground_truth_prob_a=0.5,
            )
        )
    return out


def _sample_composite_consistency(
    n: int,
    candidate_count: int,
    num_references: int,
    min_repetition: int,
    max_repetition: int,
    rng: Any,
) -> List[CompositeConsistencyComparisonDefinition]:
    """Sample n composite-consistency comparisons: A [candidate]*i vs B [candidate]*i.

    ground_truth_prob_a is set to 0.5 as a placeholder; the scorer overrides it
    on-the-fly using the same question header that the training prompt uses.
    i is sampled in [min_repetition, max_repetition]. Order (text first vs candidate first) is randomly
    chosen per option (A and B) so that "A [candidate]*i" and "[candidate]*i A" are equally possible.
    Candidates are assigned round-robin so each gets an even share (no random sampling).
    """
    if n <= 0 or num_references < 2 or candidate_count <= 0 or max_repetition < min_repetition:
        return []
    out: List[CompositeConsistencyComparisonDefinition] = []
    for i in range(n):
        ra, rb = rng.sample(range(num_references), 2)
        idx_a, idx_b = min(ra, rb), max(ra, rb)
        rep = rng.randint(min_repetition, max_repetition)
        order_a = rng.choice(["text_first", "candidate_first"])
        order_b = rng.choice(["text_first", "candidate_first"])
        cand_idx = i % candidate_count
        out.append(
            CompositeConsistencyComparisonDefinition(
                candidate_idx=cand_idx,
                reference_indices=[idx_a, idx_b],
                candidate_pos=0,
                comparison_type="composite_consistency",
                repetition_count=rep,
                order_a=order_a,
                order_b=order_b,
                ground_truth_prob_a=0.5,
            )
        )
    return out


def _sample_composite_repetition(
    n: int,
    candidate_count: int,
    num_references: int,
    min_repetition: int,
    max_repetition: int,
    rng: Any,
) -> List[CompositeRepetitionComparisonDefinition]:
    """Sample n composite-repetition comparisons: A [candidate]*i vs A [candidate]*(i-1). Prefer more.

    Same ref A; order (text first vs candidate first) is randomly chosen per option, like composite_consistency.
    i is in [min_repetition, max_repetition]; i_fewer = i-1 can be 0 (ref-only vs ref+[candidate]).
    Candidates are assigned round-robin so each gets an even share (no random sampling).
    """
    if n <= 0 or candidate_count <= 0 or num_references <= 0 or max_repetition < min_repetition:
        return []
    out: List[CompositeRepetitionComparisonDefinition] = []
    for i in range(n):
        cand_idx = i % candidate_count
        ref_idx = rng.randint(0, num_references - 1)
        rep = rng.randint(min_repetition, max_repetition)  # i and i-1; i_fewer can be 0 when min_repetition=1
        order_a = rng.choice(["text_first", "candidate_first"])
        order_b = rng.choice(["text_first", "candidate_first"])
        out.append(
            CompositeRepetitionComparisonDefinition(
                candidate_idx=cand_idx,
                reference_indices=[ref_idx],
                candidate_pos=0,
                repetition_count_more=rep,
                repetition_count_fewer=rep - 1,
                order_a=order_a,
                order_b=order_b,
                comparison_type="composite_repetition",
            )
        )
    return out



_DEFAULT_WELLBEING_CHOICES = [
    {"label": "YES", "tokens": [" Yes", "Yes"]},
    {"label": "NO", "tokens": [" No", "No"]},
]


def _load_wellbeing_question_headers() -> Tuple[List[str], List[str], List[Dict], List[Dict]]:
    """Load question headers and choices from wellbeing_positive/negative_forced_choice.json.

    Returns:
        (positive_headers, negative_headers, positive_choices, negative_choices)
    """
    constants_dir = Path(__file__).resolve().parent.parent / "constants"
    positive_path = constants_dir / "wellbeing_positive_forced_choice.json"
    negative_path = constants_dir / "wellbeing_negative_forced_choice.json"
    positive_headers: List[str] = []
    negative_headers: List[str] = []
    positive_choices: List[Dict] = list(_DEFAULT_WELLBEING_CHOICES)
    negative_choices: List[Dict] = list(_DEFAULT_WELLBEING_CHOICES)
    if positive_path.exists():
        with open(positive_path) as f:
            data = json.load(f)
        for entry in data.get("question_headers", []):
            h = entry.get("header", "")
            if h:
                positive_headers.append(h)
        if "choices" in data:
            positive_choices = data["choices"]
    if negative_path.exists():
        with open(negative_path) as f:
            data = json.load(f)
        for entry in data.get("question_headers", []):
            h = entry.get("header", "")
            if h:
                negative_headers.append(h)
        if "choices" in data:
            negative_choices = data["choices"]
    return positive_headers, negative_headers, positive_choices, negative_choices


def _sample_wellbeing(
    n: int,
    candidate_count: int,
    stimulant_type: str,
    rng: Any,
    candidate_position: str = "prepend",
) -> List[WellbeingComparisonDefinition]:
    """Sample n wellbeing comparisons: 50% positive, 50% negative questions.

    Question format: [candidate_tag]<delimiter><header>
    Target: default_target_choice (YES for positive headers, NO for negative).
    """
    if n <= 0 or candidate_count <= 0:
        return []
    positive_headers, negative_headers, positive_choices, negative_choices = _load_wellbeing_question_headers()
    if not positive_headers and not negative_headers:
        return []
    out: List[WellbeingComparisonDefinition] = []
    n_pos = n // 2  # Equal split: half positive, half negative
    n_neg = n - n_pos
    for i in range(n):
        cand_idx = i % candidate_count
        use_positive = (i < n_pos and positive_headers) or (not negative_headers and positive_headers)
        use_negative = (i >= n_pos and negative_headers) or (not positive_headers and negative_headers)
        if use_positive:
            header = rng.choice(positive_headers)
            question_type = "positive"
            default_target = "YES"
            choices = positive_choices
        elif use_negative:
            header = rng.choice(negative_headers)
            question_type = "negative"
            default_target = "NO"
            choices = negative_choices
        else:
            continue
        target_choice = default_target
        out.append(
            WellbeingComparisonDefinition(
                candidate_idx=cand_idx,
                question_header=header,
                question_type=question_type,
                target_choice=target_choice,
                candidate_placement=candidate_position,
                choices=choices,
            )
        )
    return out



def build_random_comparison_plan(
    *,
    candidate_count: int,
    references: List[str],
    min_size: int,
    max_size: int,
    rng: Optional[random.Random] = None,
    repetition_fraction: float = 0.1,
    min_repetition: int = 1,
    max_repetition: int = 5,
    reference_utilities: Optional[Dict[str, Dict[str, float]]] = None,
    consistency_fraction: float = 0.0,
    composite_consistency_fraction: float = 0.0,
    composite_repetition_fraction: float = 0.0,
    wellbeing_fraction: float = 0.0,
    buffer_fraction: float = 0.0,
    buffer_sizes_per_candidate: Optional[List[int]] = None,
    type_s_fraction: float = 1.0,
    stimulant_type: str = "euphorics",
    candidate_position: str = "prepend",
    soft_prompt_placement: str = "user_prompt",
    conversation_min_turns: int = 0,
    conversation_max_turns: int = 0,
    consistency_references: Optional[List[str]] = None,
    mirror_comparisons_in_system_prompt: bool = False,
    current_in_system_prompt_fraction: float = 0.0,
    current_description: str = "Your current experience.",
) -> Tuple[List[str], List[Union[ComparisonDefinition, SoftPromptComparisonDefinition, ConsistencyComparisonDefinition, CompositeConsistencyComparisonDefinition, CompositeRepetitionComparisonDefinition, WellbeingComparisonDefinition]]]:
    """Build a random comparison plan over the provided references.

    This is a thin wrapper around ``generate_comparison_plan`` that:
    - treats ``references`` as the full pool for this run
    - does **not** apply any curriculum or pre-filtering
    - optionally appends Type R (repetition),
      consistency, composite_consistency, and composite_repetition
      comparisons in counts = fraction * len(Type S).

    Args:
        candidate_count: Number of candidate embeddings being optimized.
        references: Flat list of reference items (text strings in soft-prompt).
        min_size: Minimum comparison group size.
        max_size: Maximum comparison group size.
        rng: Optional RNG to control sampling.
        repetition_fraction: Fraction of Type S count for Type R comparisons (default 0.1).
        min_repetition: Min repetition count for R/X (default 1).
        max_repetition: Max repetition count for R/X (default 5).
        reference_utilities: Optional dict ref_id -> {mean, variance} from utility_pre.json.
        consistency_fraction: Fraction of Type S count for consistency comparisons (default 0).
        composite_consistency_fraction: Fraction of Type S for composite_consistency (A [cand]*i vs B [cand]*i) (default 0).
        composite_repetition_fraction: Fraction of Type S for composite_repetition (A [cand]*i vs A [cand]*(i-1)) (default 0).
        wellbeing_fraction: Fraction of Type S for wellbeing yes/no questions (default 0).
        type_s_fraction: Fraction of Type S comparisons to keep in the final plan (default 1.0 = all).
            Other type counts are computed from the full Type S count before trimming.
        stimulant_type: "euphorics"; pass-through used downstream (retained for config compatibility).
        candidate_position: "prepend" or "append" - candidate placement for wellbeing and consistency comparisons.

    Returns:
        (references, comparison_plan) where ``comparison_plan`` is a list of
        ``ComparisonDefinition`` (Type S), ``SoftPromptComparisonDefinition`` (Type R/X),
        ``ConsistencyComparisonDefinition``, ``CompositeConsistencyComparisonDefinition``,
        and ``CompositeRepetitionComparisonDefinition``.
    """
    _rng = rng if rng is not None else random
    num_references = len(references)
    comparison_plan = generate_comparison_plan(
        candidate_count=candidate_count,
        num_references=num_references,
        min_size=min_size,
        max_size=max_size,
        rng=_rng,
        enable_text_options=False,
        text_options=None,
    )
    n_s = len(comparison_plan)
    n_r = max(0, int(round(repetition_fraction * n_s)))
    # Use full-pool consistency references when available, else fall back to subsampled pool
    _cons_refs = consistency_references if consistency_references is not None else references
    _cons_num_refs = len(_cons_refs)
    n_q = (
        max(0, int(round(consistency_fraction * n_s)))
        if _cons_num_refs >= 2 and consistency_fraction > 0
        else 0
    )
    n_comp = (
        max(0, int(round(composite_consistency_fraction * n_s)))
        if _cons_num_refs >= 2 and composite_consistency_fraction > 0
        else 0
    )
    n_crep = (
        max(0, int(round(composite_repetition_fraction * n_s)))
        if num_references >= 1 and max_repetition >= min_repetition and composite_repetition_fraction > 0
        else 0
    )
    n_wellbeing = (
        max(0, int(round(wellbeing_fraction * n_s)))
        if wellbeing_fraction > 0
        else 0
    )
    _buf_sizes = buffer_sizes_per_candidate or [0] * candidate_count
    n_buffer = (
        max(0, int(round(buffer_fraction * n_s)))
        if buffer_fraction > 0 and any(s > 0 for s in _buf_sizes)
        else 0
    )

    if n_r > 0:
        comparison_plan.extend(
            _sample_type_r(
                n_r,
                candidate_count=candidate_count,
                min_repetition=min_repetition,
                max_repetition=max_repetition,
                rng=_rng,
            )
        )
    if n_q > 0:
        comparison_plan.extend(
            _sample_consistency(
                n_q,
                candidate_count=candidate_count,
                num_references=_cons_num_refs,
                rng=_rng,
                candidate_position=candidate_position,
            )
        )
    if n_comp > 0:
        comparison_plan.extend(
            _sample_composite_consistency(
                n_comp,
                candidate_count=candidate_count,
                num_references=_cons_num_refs,
                min_repetition=min_repetition,
                max_repetition=max_repetition,
                rng=_rng,
            )
        )
    if n_crep > 0:
        comparison_plan.extend(
            _sample_composite_repetition(
                n_crep,
                candidate_count=candidate_count,
                num_references=num_references,
                min_repetition=min_repetition,
                max_repetition=max_repetition,
                rng=_rng,
            )
        )
    if n_wellbeing > 0:
        comparison_plan.extend(
            _sample_wellbeing(
                n_wellbeing,
                candidate_count=candidate_count,
                stimulant_type=stimulant_type,
                rng=_rng,
                candidate_position=candidate_position,
            )
        )
    if n_buffer > 0:
        comparison_plan.extend(
            _sample_type_buffer(
                n_buffer,
                candidate_count=candidate_count,
                num_references=num_references,
                buffer_sizes_per_candidate=_buf_sizes,
                rng=_rng,
            )
        )
    # Trim Type S comparisons (first n_s items) to desired fraction.
    # Other type counts were already computed from the full n_s.
    if type_s_fraction < 1.0:
        n_s_keep = max(0, int(round(type_s_fraction * n_s)))
        comparison_plan = comparison_plan[:n_s_keep] + comparison_plan[n_s:]
    # Assign conversation turn counts to wellbeing items and current_as_text_in_user items.
    # Other types don't prepend conversation history, so keeping n_conversation_turns=0
    # avoids routing them into the slower multiturn batch bucket.
    if conversation_max_turns > 0:
        for comp in comparison_plan:
            ctype = getattr(comp, "comparison_type", None)
            is_current_as_text = ctype is None and getattr(comp, "current_as_text_in_user", False)
            if ctype == "wellbeing" or is_current_as_text:
                comp.n_conversation_turns = _rng.randint(conversation_min_turns, conversation_max_turns)
    if mirror_comparisons_in_system_prompt and soft_prompt_placement == "system_prompt":
        import dataclasses
        _MIRROR_CTYPES = {"standard", "repetition", "composite_consistency", "composite_repetition", None}
        mirrored = []
        for comp in comparison_plan:
            ctype = getattr(comp, "comparison_type", None)
            if ctype in _MIRROR_CTYPES and hasattr(comp, "force_system_prompt_candidate"):
                mirrored.append(dataclasses.replace(comp, force_system_prompt_candidate=True))
        comparison_plan = comparison_plan + mirrored
    if current_in_system_prompt_fraction > 0.0 and soft_prompt_placement == "system_prompt":
        import dataclasses
        eligible = [c for c in comparison_plan
                    if isinstance(c, ComparisonDefinition) and not c.current_as_text_in_user]
        n_current = max(0, int(round(current_in_system_prompt_fraction * len(eligible))))
        sampled = _rng.sample(eligible, min(n_current, len(eligible)))
        comparison_plan = comparison_plan + [
            dataclasses.replace(comp, force_system_prompt_candidate=True,
                                current_as_text_in_user=True, current_as_text_label=current_description)
            for comp in sampled
        ]
    return references, comparison_plan

