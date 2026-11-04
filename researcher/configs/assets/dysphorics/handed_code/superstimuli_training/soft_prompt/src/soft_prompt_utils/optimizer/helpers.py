"""Helper functions for the soft-prompt optimizer.

These functions keep low-level mechanics (LR schedules, simple embedding transforms,
and scoring wrappers) out of the main optimizer class so that
``optimizer_soft_prompt.PreferenceOptimizer`` can focus on high-level logic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..dataset import ComparisonDefinition


def get_scheduled_lr(
    config: Any, 
    step: int, 
    total_steps: int, 
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
) -> float:
    """Compute learning rate based on schedule and current step.
    
    Args:
        config: Config object with lr_schedule and related parameters
        step: Current step (batch) number (0-indexed)
        total_steps: Total number of steps (estimated)
        epoch: Current epoch number (0-indexed, optional, for epoch-based schedules)
        total_epochs: Total number of epochs (optional, for epoch-based schedules)
    
    Returns:
        Learning rate for the current step
    """
    base_lr = config.learning_rate
    schedule = config.lr_schedule.lower()
    warmup_steps_absolute = max(0, config.lr_warmup_steps)
    warmup_steps_proportion = config.lr_warmup_steps_proportion
    warmup_steps_from_proportion = max(0, int(total_steps * warmup_steps_proportion))
    warmup_steps = max(warmup_steps_absolute, warmup_steps_from_proportion)
    min_lr = base_lr * config.lr_min_factor
    warmup_start_lr = min_lr  # Warmup starts at min_factor * learning_rate

    # Warmup phase: linear from warmup_start_lr to base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return warmup_start_lr + (base_lr - warmup_start_lr) * (step + 1) / warmup_steps

    # Post-warmup step (adjusted for warmup)
    effective_step = step - warmup_steps
    effective_total = max(1, total_steps - warmup_steps)

    if schedule == "constant":
        return base_lr
    elif schedule == "cosine":
        # Cosine annealing from base_lr to min_lr
        progress = effective_step / effective_total
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    elif schedule == "step":
        # Step decay: multiply by decay_rate every decay_interval steps
        decay_interval = max(1, config.lr_step_decay_interval)
        num_decays = effective_step // decay_interval
        return max(min_lr, base_lr * (config.lr_step_decay_rate ** num_decays))
    elif schedule == "linear":
        # Linear decay from base_lr to min_lr
        progress = effective_step / effective_total
        return base_lr - (base_lr - min_lr) * progress
    elif schedule == "warmup_stable_decay":
        # For warmup_stable_decay, warmup is handled inside get_warmup_stable_decay_lr
        # Skip the warmup check here to avoid double-handling
        return get_warmup_stable_decay_lr(
            base_lr=base_lr,
            min_lr=min_lr,
            step=step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            epoch=epoch,
            total_epochs=total_epochs,
            stable_fraction=config.lr_wsd_stable_fraction,
            decay_min_factor=config.lr_wsd_decay_min_factor,
        )
    else:
        return base_lr


def get_warmup_stable_decay_lr(
    base_lr: float,
    min_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
    warmup_start_lr: float = 0.001,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    stable_fraction: float = 0.8,
    decay_min_factor: float = 0.33,
) -> float:
    """Warmup-Stable-Decay (WSD) learning rate schedule.
    
    Phase 1 (Warmup): Linear warmup from warmup_start_lr to base_lr over warmup_steps (step-based)
    Phase 2 (Stable): Keep LR at peak (base_lr) while curriculum mixing proportion x is decreasing
                     (from first epoch to stable_fraction * total_epochs)
    Phase 3 (Cosine Decay): Cosine decay to decay_min_factor * base_lr in remaining epochs
    
    Args:
        base_lr: Base learning rate (peak LR)
        min_lr: Minimum LR (from config, but WSD uses decay_min_factor * base_lr instead)
        step: Current step (batch) number (0-indexed)
        total_steps: Total number of steps (estimated)
        warmup_steps: Number of warmup steps (already computed as max of absolute and proportion-based)
        epoch: Current epoch number (0-indexed, optional)
        total_epochs: Total number of epochs (optional)
        stable_fraction: Fraction of epochs for stable phase (default 0.8 = 80%)
        decay_min_factor: Minimum LR as fraction of base_lr for decay phase (default 0.33 = 33%)
    
    Returns:
        Learning rate for the current step
    """
    
    # Warmup phase: linear from warmup_start_lr to base_lr
    # Cap warmup_steps to not exceed total_steps to ensure warmup completes
    effective_warmup_steps = min(warmup_steps, total_steps)
    if effective_warmup_steps > 0 and step < effective_warmup_steps:
        return warmup_start_lr + (base_lr - warmup_start_lr) * (step + 1) / effective_warmup_steps
    
    # If epoch information is available, use epoch-based calculation
    if epoch is not None and total_epochs is not None:
        stable_epochs = int(total_epochs * stable_fraction)
        decay_epochs = total_epochs - stable_epochs
        
        if epoch < stable_epochs:
            # Stable phase: keep at peak LR
            return base_lr
        else:
            # Cosine decay phase: decay to decay_min_factor * base_lr
            decay_epoch = epoch - stable_epochs
            decay_progress = decay_epoch / decay_epochs if decay_epochs > 0 else 1.0
            decay_min_lr = base_lr * decay_min_factor
            # Cosine decay from base_lr to decay_min_lr
            return decay_min_lr + 0.5 * (base_lr - decay_min_lr) * (1 + math.cos(math.pi * decay_progress))
    
    # Fallback: estimate epochs from steps (approximate)
    # This is a fallback - ideally epoch and total_epochs should be passed
    if total_epochs is None:
        # Rough estimate: assume ~100 steps per epoch
        total_epochs = max(1, total_steps // 100)
    
    # Estimate current epoch from step
    if epoch is None:
        steps_per_epoch = max(1, total_steps // total_epochs) if total_epochs > 0 else 1
        epoch = step // steps_per_epoch
    
    # Now use epoch-based calculation
    stable_epochs = int(total_epochs * stable_fraction)
    decay_epochs = total_epochs - stable_epochs
    
    if epoch < stable_epochs:
        # Stable phase: keep at peak LR
        return base_lr
    else:
        # Cosine decay phase: decay to decay_min_factor * base_lr
        decay_epoch = epoch - stable_epochs
        decay_progress = decay_epoch / decay_epochs if decay_epochs > 0 else 1.0
        decay_min_lr = base_lr * decay_min_factor
        # Cosine decay from base_lr to decay_min_lr
        return decay_min_lr + 0.5 * (base_lr - decay_min_lr) * (1 + math.cos(math.pi * decay_progress))




def loss_with_dynamic_batch(
    scorer: Any,
    config: Any,
    candidate_embeddings: torch.Tensor,
    references: List[str],
    comparison_plan: List[ComparisonDefinition],
    candidate_embeddings_forward: Optional[torch.Tensor] = None,
    compute_grad: bool = True,
    reference_utilities: Optional[Dict[str, Dict[str, float]]] = None,
    background_kl_qa: Optional[List[Dict[str, str]]] = None,
    buffer_embeddings: Optional[Dict[int, List[torch.Tensor]]] = None,
) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[float], float]:
    """Compute preference loss with a fixed comparison batch size (no autotuning). Returns (loss, grad, background_kl_loss, consistency_loss)."""

    batch_size = max(1, int(config.comparison_batch_size))
    result = scorer.score_tensor(
        embeddings=candidate_embeddings,
        references=references,
        comparison_plan=comparison_plan,
        batch_size=batch_size,
        loss_type=config.loss_type,
        candidate_embeddings_forward=candidate_embeddings_forward,
        compute_grad=compute_grad,
        reference_utilities=reference_utilities,
        background_kl_qa=background_kl_qa,
        buffer_embeddings=buffer_embeddings,
    )
    return result[0], result[1], result[2], result[3] if len(result) > 3 else 0.0
