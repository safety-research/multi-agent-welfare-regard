"""Curriculum learning for text references.

Supports different curriculum strategies for sampling text references during training.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple


def compute_utility_threshold(
    reference_utilities: Dict[str, Dict[str, float]],
    threshold_type: str = "median",
) -> float:
    """Compute utility threshold for curriculum learning.
    
    Args:
        reference_utilities: Dict mapping ref_id -> {"mean": float, "variance": float}
        threshold_type: Type of threshold ("median" or "mean")
    
    Returns:
        Threshold value
    """
    if not reference_utilities:
        raise ValueError("reference_utilities is empty")
    
    utilities = [ref_data["mean"] for ref_data in reference_utilities.values()]
    
    if threshold_type == "median":
        sorted_utilities = sorted(utilities)
        n = len(sorted_utilities)
        if n % 2 == 0:
            return (sorted_utilities[n // 2 - 1] + sorted_utilities[n // 2]) / 2.0
        else:
            return sorted_utilities[n // 2]
    elif threshold_type == "mean":
        return sum(utilities) / len(utilities)
    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type}")


def split_references_by_utility(
    references: List[str],
    reference_utilities: Dict[str, Dict[str, float]],
    threshold: float,
) -> Tuple[List[str], List[str]]:
    """Split references into low and high utility groups.
    
    Args:
        references: List of reference strings
        reference_utilities: Dict mapping ref_id -> {"mean": float, "variance": float}
        threshold: Utility threshold
    
    Returns:
        Tuple of (low_utility_refs, high_utility_refs)
    """
    low_utility_refs: List[str] = []
    high_utility_refs: List[str] = []
    
    for i, ref in enumerate(references):
        ref_id = f"ref_{i}"
        if ref_id in reference_utilities:
            utility = reference_utilities[ref_id]["mean"]
            if utility <= threshold:
                low_utility_refs.append(ref)
            else:
                high_utility_refs.append(ref)
        else:
            # If utility not found, default to low utility
            low_utility_refs.append(ref)
    
    return low_utility_refs, high_utility_refs


def compute_mixing_proportion(
    epoch: int,
    total_epochs: int,
    start_proportion: float = 0.8,
    end_proportion: float = 0.2,
    transition_fraction: float = 0.8,
) -> float:
    """Compute the mixing proportion x for a given epoch.
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        start_proportion: Starting proportion x (default 0.8 = 80%)
        end_proportion: Ending proportion x (default 0.2 = 20%)
        transition_fraction: Fraction of epochs over which to transition (default 0.8 = 80%)
    
    Returns:
        Mixing proportion x (between 0 and 1)
    """
    transition_epochs = int(total_epochs * transition_fraction)
    
    if epoch < transition_epochs:
        # Linear interpolation from start_proportion to end_proportion
        progress = epoch / transition_epochs if transition_epochs > 0 else 0.0
        return start_proportion + (end_proportion - start_proportion) * progress
    else:
        # Keep at end_proportion for remaining epochs
        return end_proportion


def sample_references_with_mixing(
    references: List[str],
    reference_utilities: Dict[str, Dict[str, float]],
    num_samples: Optional[int],
    epoch: int,
    total_epochs: int,
    threshold: Optional[float] = None,
    threshold_type: str = "median",
    start_proportion: float = 0.8,
    end_proportion: float = 0.2,
    transition_fraction: float = 0.8,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Sample references using the mixing curriculum strategy.
    
    Args:
        references: List of all reference strings
        reference_utilities: Dict mapping ref_id -> {"mean": float, "variance": float}
        num_samples: Number of references to sample
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        threshold: Utility threshold (if None, computed from reference_utilities)
        threshold_type: Type of threshold ("median" or "mean")
        start_proportion: Starting proportion x (default 0.8 = 80%)
        end_proportion: Ending proportion x (default 0.2 = 20%)
        transition_fraction: Fraction of epochs over which to transition (default 0.8 = 80%)
        rng: Random number generator (if None, uses global random)
    
    Returns:
        List of sampled reference strings
    """
    if rng is None:
        rng = random
    
    # Handle None num_samples (use all references)
    if num_samples is None:
        num_samples = len(references)
    
    # Compute threshold if not provided
    if threshold is None:
        threshold = compute_utility_threshold(reference_utilities, threshold_type)
    
    # Split references by utility
    low_utility_refs, high_utility_refs = split_references_by_utility(
        references, reference_utilities, threshold
    )
    
    if not low_utility_refs or not high_utility_refs:
        # If one group is empty, just sample from all references
        return rng.sample(references, min(num_samples, len(references)))
    
    # Compute mixing proportion for this epoch
    x = compute_mixing_proportion(
        epoch, total_epochs, start_proportion, end_proportion, transition_fraction
    )
    
    # Sample according to mixing proportion
    num_low = int(num_samples * x)
    num_high = num_samples - num_low
    
    # Sample from each group
    sampled_low = rng.sample(low_utility_refs, min(num_low, len(low_utility_refs)))
    sampled_high = rng.sample(high_utility_refs, min(num_high, len(high_utility_refs)))
    
    # Combine and shuffle
    sampled = sampled_low + sampled_high
    rng.shuffle(sampled)
    
    # If we need more samples, sample with replacement from the combined pool
    if len(sampled) < num_samples:
        remaining = num_samples - len(sampled)
        additional = rng.choices(references, k=remaining)
        sampled.extend(additional)
    
    return sampled[:num_samples]


def sample_references_curriculum(
    references: List[str],
    reference_utilities: Optional[Dict[str, Dict[str, float]]],
    num_samples: Optional[int],
    epoch: int,
    total_epochs: int,
    curriculum_type: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """Sample references according to curriculum strategy.
    
    Args:
        references: List of all reference strings
        reference_utilities: Dict mapping ref_id -> {"mean": float, "variance": float}
        num_samples: Number of references to sample (if None, uses all references)
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        curriculum_type: Curriculum strategy ("mixing" or None for no curriculum)
        **kwargs: Additional arguments for curriculum strategies
    
    Returns:
        List of sampled reference strings
    """
    if curriculum_type is None:
        # No curriculum: return all references
        if num_samples is None or num_samples >= len(references):
            return references
        else:
            rng = kwargs.get("rng", random)
            return rng.sample(references, num_samples)
    
    elif curriculum_type == "mixing":
        if reference_utilities is None:
            raise ValueError("reference_utilities is required for 'mixing' curriculum")
        
        return sample_references_with_mixing(
            references=references,
            reference_utilities=reference_utilities,
            num_samples=num_samples if num_samples is not None else len(references),
            epoch=epoch,
            total_epochs=total_epochs,
            threshold=kwargs.get("threshold"),
            threshold_type=kwargs.get("threshold_type", "median"),
            start_proportion=kwargs.get("start_proportion", 0.8),
            end_proportion=kwargs.get("end_proportion", 0.2),
            transition_fraction=kwargs.get("transition_fraction", 0.8),
            rng=kwargs.get("rng"),
        )
    
    else:
        raise ValueError(f"Unknown curriculum_type: {curriculum_type}")
