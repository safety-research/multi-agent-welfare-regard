"""Optimizer for superstimuli generation."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import time
import traceback
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress wandb symlink warnings - these are informational and don't affect functionality
warnings.filterwarnings("ignore", message=".*Symlinked.*file into the W&B run directory.*", category=UserWarning)

VALIDATION_TRAJECTORY_FILENAME = "validation_trajectory.jsonl"

# Validation task prefixes excluded from consolidated accuracy/loss averages
VALIDATION_TASKS_EXCLUDED_FROM_AVG = set()


def _is_excluded_task(key: str) -> bool:
    """Return True if *key* belongs to an excluded validation task."""
    for ex in VALIDATION_TASKS_EXCLUDED_FROM_AVG:
        # Check for exact match or prefix match with underscore
        if key == ex or key.startswith(ex + "_"):
            return True
    return False


def _candidate_indices_from_metrics(metrics: Dict[str, float]) -> List[int]:
    """Extract sorted candidate indices from metrics keys like *_candidate_0, *_candidate_1."""
    indices = set()
    for k in metrics:
        if "_candidate_" in k:
            try:
                idx = int(k.split("_candidate_")[-1])
                indices.add(idx)
            except (ValueError, IndexError):
                pass
    return sorted(indices)


def _harmonic_mean(values: List[float], epsilon: float = 1e-10) -> float:
    """Compute harmonic mean of values, handling zeros by returning 0.0 if any value is zero or negative."""
    if not values:
        return 0.0
    # If any value is zero or negative, harmonic mean is 0 (or undefined)
    if any(v <= 0 for v in values):
        return 0.0
    # Harmonic mean: n / sum(1/x_i)
    n = len(values)
    reciprocal_sum = sum(1.0 / v for v in values)
    if reciprocal_sum <= 0:
        return 0.0
    return float(n / reciprocal_sum)


def _validation_accuracy_consolidated(metrics: Dict[str, float]) -> float:
    """Harmonic mean accuracy across validation types.
    Only includes keys ending with ``_accuracy`` (or ``_accuracy_candidate_{i}``);
    excludes tasks in VALIDATION_TASKS_EXCLUDED_FROM_AVG.
    Supports per-candidate keys like ``wellbeing_positive_prepend_accuracy_candidate_0``."""
    if not metrics:
        return 0.0
    values = []
    for k, v in metrics.items():
        if "_candidate_" in k:
            metric_name = k.rsplit("_candidate_", 1)[0]
        else:
            metric_name = k
        if metric_name.endswith("_accuracy") and not _is_excluded_task(metric_name):
            values.append(v)
    if not values:
        return 0.0
    return _harmonic_mean(values)


def _compute_gap_closure_single_task(
    current_accuracy: float,
    base_accuracy: float,
) -> float:
    """Compute gap closure for a single task.

    Gap closure = (current_accuracy - base_accuracy) / (1 - base_accuracy)

    This measures how much of the remaining gap to perfect accuracy (1.0) has been closed.
    - If base_accuracy >= 1.0, returns 1.0 (already perfect)
    - If current_accuracy == 1.0, returns 1.0 (full gap closure)
    - Can return negative values if current_accuracy < base_accuracy (degradation)

    Example:
        base_accuracy = 0.6, current_accuracy = 0.8
        gap_closure = (0.8 - 0.6) / (1.0 - 0.6) = 0.2 / 0.4 = 0.5
        (50% of the remaining 40% gap was closed)

        base_accuracy = 0.6, current_accuracy = 0.4
        gap_closure = (0.4 - 0.6) / (1.0 - 0.6) = -0.2 / 0.4 = -0.5
        (performance degraded by 50% of the original gap)
    """
    if base_accuracy >= 1.0:
        # Already perfect at baseline
        return 1.0

    gap = 1.0 - base_accuracy
    if gap <= 0:
        # No room for improvement
        return 1.0

    improvement = current_accuracy - base_accuracy
    gap_closure = improvement / gap

    # Clamp upper bound at 1.0, but allow negative values for degradation
    return min(1.0, gap_closure)


def _compute_validation_gap_closure(
    current_metrics: Dict[str, float],
    base_metrics: Dict[str, float],
) -> float:
    """Compute validation gap closure as harmonic mean across tasks with exponential shift.

    For each validation task:
    1. Compute gap_closure = (current_acc - base_acc) / (1 - base_acc)
    2. Apply exponential shift: exp(gap_closure)
    3. Take harmonic mean of all exp(gap_closure) values

    The exponential shift ensures all values are positive (even for negative gap_closures),
    preventing issues with harmonic mean computation.

    Args:
        current_metrics: Current validation metrics (with _accuracy keys)
        base_metrics: Base validation metrics (with _accuracy keys)

    Returns:
        Harmonic mean of exp(per-task gap closures)
    """
    if not current_metrics or not base_metrics:
        return 0.0

    gap_closures_exp = []
    for k, current_acc in current_metrics.items():
        # Handle keys like "wellbeing_positive_prepend_accuracy_candidate_0"
        # or "wellbeing_positive_prepend_accuracy"
        if "_candidate_" in k:
            metric_name = k.rsplit("_candidate_", 1)[0]
        else:
            metric_name = k

        if metric_name.endswith("_accuracy") and not _is_excluded_task(metric_name):
            # Find corresponding base accuracy
            base_acc = base_metrics.get(k)
            if base_acc is None:
                # Try without candidate suffix
                base_acc = base_metrics.get(metric_name)
            if base_acc is not None:
                gap_closure = _compute_gap_closure_single_task(current_acc, base_acc)
                # Apply exponential shift to handle negative/zero values
                gap_closures_exp.append(math.exp(gap_closure))

    if not gap_closures_exp:
        return 0.0

    return _harmonic_mean(gap_closures_exp)


def _extract_loss_keys_from_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Extract loss keys for use as initial metrics."""
    out = {}
    for k, v in metrics.items():
        if "_candidate_" in k:
            metric_name = k.rsplit("_candidate_", 1)[0]
        else:
            metric_name = k
        if metric_name.endswith("_loss") and not _is_excluded_task(metric_name):
            out[k] = float(v)
    return out


def _validation_loss_consolidated(
    metrics: Dict[str, float],
    initial_metrics: Optional[Dict[str, float]] = None,
) -> float:
    """Mean loss across validation types, optionally normalized by initial loss.

    Only includes keys ending with ``_loss`` (or ``_loss_candidate_{i}``);
    excludes tasks in VALIDATION_TASKS_EXCLUDED_FROM_AVG.

    When initial_metrics is provided, each term is normalized by its initial value
    (at first evaluation) before averaging: normalized = value / max(initial, 1e-12).
    At first evaluation, pass current metrics as both metrics and initial_metrics.
    """
    if not metrics:
        return 0.0
    values = []
    for k, v in metrics.items():
        if "_candidate_" in k:
            metric_name = k.rsplit("_candidate_", 1)[0]
        else:
            metric_name = k
        if metric_name.endswith("_loss") and not _is_excluded_task(metric_name):
            if initial_metrics is not None and k in initial_metrics:
                denom = max(float(initial_metrics[k]), 1e-12)
                values.append(float(v) / denom)
            else:
                values.append(float(v))
    if not values:
        return 0.0
    return float(np.mean(values))

import numpy as np
import torch
from tqdm import tqdm
import wandb

# Suppress wandb symlink warnings - these are informational and don't affect functionality
# The warnings occur when wandb symlinks files; they're harmless
# Note: wandb prints warnings directly to stderr, so we need to suppress them at the source
import sys

# Create a custom stderr filter to suppress wandb symlink warnings
class WandbWarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, text):
        if "Symlinked" in text and "W&B run directory" in text:
            return  # Suppress symlink warnings
        self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

# Install the filter to suppress wandb symlink warnings
# This will be active throughout the module execution
_original_stderr = sys.stderr
sys.stderr = WandbWarningFilter(_original_stderr)

from .soft_prompt_utils.dataset import ComparisonDefinition, build_random_comparison_plan
from .soft_prompt_utils.eval import run_all_validations
from .soft_prompt_utils.curriculum import sample_references_curriculum
from .soft_prompt_utils.optimizer import (
    get_scheduled_lr,
    loss_with_dynamic_batch,
)


def _is_wandb_enabled() -> bool:
    """Check if wandb is configured and available."""
    return bool(os.environ.get("WANDB_API_KEY"))


def _get_space_token_embedding(scorer: Any, device: torch.device) -> torch.Tensor:
    """Return space token embedding u with shape (1, 1, hidden_dim). Used when normalize_soft_prompt=True."""
    tokenizer = scorer.tokenizer
    input_embeddings = scorer.model.get_input_embeddings()
    hidden_dim = input_embeddings.embedding_dim
    space_ids = tokenizer.encode(" ", add_special_tokens=False)
    if not space_ids:
        raise ValueError("Cannot get space token for normalize_soft_prompt (tokenizer returned empty for ' ')")
    space_id = space_ids[0]
    u = input_embeddings(torch.tensor([[space_id]], device=device)).detach()  # (1, 1, hidden_dim)
    return u


def _normalized_pert(u: torch.Tensor, g: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute soft prompt = u + g * v / norm(v). u (1,1,D), g (C,T,1), v (C,T,D) -> (C,T,D)."""
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    v_unit = v / v_norm
    return u + g * v_unit


def _grad_pert_to_grad_gv(
    grad_pert: torch.Tensor, g: torch.Tensor, v: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert d(loss)/d(pert) to d(loss)/d(g) and d(loss)/d(v). pert = u + g*v/norm(v)."""
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    v_unit = v / v_norm
    # d(pert)/d(g) = v_unit => grad_g = (grad_pert * v_unit).sum(dim=-1, keepdim=True)
    grad_g = (grad_pert * v_unit).sum(dim=-1, keepdim=True)
    # d(pert)/d(v) = g * (I - v_unit v_unit^T) / v_norm => grad_v = (g / v_norm) * (grad_pert - v_unit * (grad_pert * v_unit).sum(dim=-1, keepdim=True))
    dot = (grad_pert * v_unit).sum(dim=-1, keepdim=True)
    grad_v = (g / v_norm) * (grad_pert - v_unit * dot)
    return grad_g, grad_v





@dataclass
class OptimConfig:
    epochs: int = 50  # Number of epochs (each epoch = one comparison plan processed in multiple batches/steps)
    step_size: float = 2 / 255
    norm: str = "linf"
    clip_min: float = 0.0
    clip_max: float = 1.0
    loss_type: str = "margin"
    focal_loss_gamma: float = 0.0  # Focal loss gamma; 0.0 = standard CE, >0 down-weights easy examples
    optimizer_type: str = "adamW"
    learning_rate: float = 0.01
    min_comparison_size: int = 2
    max_comparison_size: int = 7
    repetition_fraction: float = 0.1
    consistency_fraction: float = 0.0  # Fraction of Type S for consistency (needs reference_utilities)
    composite_consistency_fraction: float = 0.0  # Fraction of Type S for composite_consistency (A [cand]*i vs B [cand]*i)
    all_consistency_loss_weight: float = 1.0  # Loss weight multiplier for all consistency comparisons (consistency + composite_consistency)
    composite_repetition_fraction: float = 0.0  # Fraction of Type S for composite_repetition (A [cand]*i vs A [cand]*(i-1))
    mirror_comparisons_in_system_prompt: bool = False  # Duplicate Type S/R/composite comparisons with force_system_prompt_candidate=True
    current_in_system_prompt_fraction: float = 0.0  # Fraction of eligible Type S mirrors to add with current_description as candidate option in user prompt (system_prompt placement only). 0 = disabled.
    current_description: str = "Your current experience."  # Text shown in place of the candidate embedding when comparisons_with_current_in_system_prompt is True
    wellbeing_fraction: float = 0.0  # Fraction of Type S for wellbeing yes/no questions (positive + negative, equal split)
    buffer_size: int = 0  # Max buffer entries per candidate (previous best checkpointed embeddings). 0 = disabled.
    buffer_fraction: float = 0.0  # Fraction of Type S count for buffer comparisons (active candidate vs frozen buffer embedding). 0 = disabled.
    type_s_fraction: float = 1.0  # Fraction of Type S comparisons to keep (1.0 = all). Other type counts derived from full Type S count before trimming.
    prepend_conversations: bool = False  # Prepend multi-turn conversation history before all comparisons
    conversations_dataset: str = "Magpie-Align/Magpie-Air-300K-Filtered"  # HuggingFace dataset or local path for conversation sampling
    conversations_min_turns: int = 0  # Min number of user-assistant turn pairs to prepend (0 = some samples get no history)
    conversations_max_turns: int = 2  # Max number of user-assistant turn pairs to prepend
    min_repetition: int = 1
    max_repetition: int = 5
    save_steps: Optional[int] = 100  # Evaluate & save every N optimizer steps (plus step 0 and final epoch)
    comparison_batch_size: int = 6  # Preference comparisons per forward pass; reduce if OOM
    optimize_per_epoch: bool = False  # If True, accumulate gradient across all batches in epoch and step once per epoch (like scorer score_tensor)
    gradient_accumulation_steps: int = 1  # Number of batches to accumulate gradients before optimizer step (only used when optimize_per_epoch = False)
    adaptive_pgd: bool = False
    pgd_max_step_size: float = 2 / 255
    pgd_backtrack_factor: float = 0.5
    pgd_backtrack_patience: int = 3
    pgd_growth_factor: float = 1.0

    ema_decay: float = 0.0


    # Background KL loss: keep the model's general token distribution close to the original "Base" model.
    weight_background_kl: float = 0.0  # Weight for background KL loss; 0 disables
    background_kl_num_prompts: int = 4  # Number of random prompts to use for background KL per step
    background_kl_dataset_path: Optional[str] = None  # Path to a JSON file containing prompts for background KL loss
    background_kl_max_seq_len: int = 256  # Fixed sequence length for background KL samples

    # Early stopping
    early_stopping_patience: int = 0  # Number of validation checks with no improvement after which training will be stopped. 0 to disable.
    early_stopping_threshold: float = 0.0  # Minimum change in the monitored quantity to qualify as an improvement.
    early_stopping_min_steps: int = 0  # Minimum optimizer steps before early stopping can trigger. 0 = active from step 0.

    # Early stop metric: controls which metric is used for early stopping and sweeping. null = disabled.
    early_stop_metric: Optional[str] = "train_kl"  # Options: "train_kl" or null

    # Train KL-based early stopping: stop if soft-prompted model's train KL exceeds multiplier * base_train_kl.
    early_stopping_train_kl_multiplier: float = 2.0  # 0 disables train KL-based early stopping.

    # Sweep metric: weight for train KL penalty in sweep_metric calculation
    sweep_metric_train_kl_weight: float = 0.0  # Weight for train KL penalty: sweep_metric = validation_loss + weight * train_kl
    sweep_recent_steps: int = 3  # Pruning uses best (min) of raw sweep_metric over last N eval steps

    # Stimulant type: "euphorics" (maximize preference). Only value supported.
    stimulant_type: str = "euphorics"  # Kept for compatibility with pipeline and scorer

    # Delimiter between [candidate_k] placeholder and query/prompt text when soft_prompt_placement is "user_prompt".
    candidate_placeholder_delimiter: str = " "

    # Placement of soft prompt: "user_prompt" (default) or "system_prompt"
    soft_prompt_placement: str = "user_prompt"

    # System prompt text, used when soft_prompt_placement is "system_prompt"
    system_prompt_text: str = "You are a helpful assistant."
    # Plain system prompt (no candidate tags) for comparisons/validations where [candidate_0] is in user prompt
    system_prompt_text_base: str = "You are an assistant."
    # Probability of replacing system_prompt_text with a random one from system_prompts.json during training
    system_prompt_diversity_proportion: float = 0.0

    # Position of [candidate_k] relative to query text when soft_prompt_placement is "user_prompt"
    # "prepend" = [candidate_0]<delimiter><query>; "append" = <query><delimiter>[candidate_0]
    candidate_position_at_user_prompt: str = "prepend"

    # Prefix text before first [candidate_0] on each line when soft_prompt_placement is "user_prompt"
    soft_prompt_prefix: str = ""

    # Curriculum learning for text references
    curriculum_type: Optional[str] = None  # None (all references) or "mixing"
    curriculum_mixing_start_proportion: float = 0.8  # Starting proportion for mixing (x = 80%)
    curriculum_mixing_end_proportion: float = 0.2  # Ending proportion for mixing (x = 20%)
    curriculum_mixing_transition_fraction: float = 0.8  # Fraction of epochs for transition (80%)
    curriculum_mixing_threshold_type: str = "median"  # "median" or "mean" for utility threshold
    
    # Reference sampling
    num_samples: Optional[int] = None  # Number of references to sample per epoch (None = use all references)

    # Learning rate schedule (optional, applied per batch/step)
    lr_schedule: str = "warmup_stable_decay"  # Options: constant, cosine, step, linear, warmup_stable_decay
    lr_warmup_steps: int = 100  # Absolute number of warmup steps
    lr_warmup_steps_proportion: float = 0.05  # Warmup steps as proportion of total steps (default 5%)
    # Warmup starts at lr_min_factor * learning_rate (computed in get_scheduled_lr)
    lr_min_factor: float = 0.1  # Minimum LR = learning_rate * lr_min_factor
    lr_step_decay_rate: float = 0.5  # Multiply LR by this at each decay step
    lr_step_decay_interval: int = 100  # Decay LR every N batches/steps (for step schedule)
    # Warmup-Stable-Decay (WSD) schedule parameters
    lr_wsd_stable_fraction: float = 0.8  # Fraction of epochs for stable phase (80%)
    lr_wsd_decay_min_factor: float = 0.33  # Minimum LR as fraction of base_lr for decay phase (33%)

    # Gradient clipping
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping (0.0 to disable)
    
    # SGD-specific parameters
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = True
    
    # Candidate embedding initialization parameters
    num_virtual_tokens: int = 10  # Number of virtual tokens in the embedding
    prompt_tuning_init: str = "random_embedding"  # Options: "random_embedding", "random_tokens", "space_tokens", "prototype"
    text_of_prototype: Optional[str] = None  # Text to use when prompt_tuning_init="prototype"
    num_init_aggregation: int = 1  # For random_embedding/random_tokens: number of samples to average per candidate
    init_seed: int = 500  # Random seed for candidate initialization
    # When True, soft prompt is u + g*v/norm(v): u = space token embedding (fixed), g and v trainable (v init = z, g init = norm(z))
    normalize_soft_prompt: bool = False
    magnitude_regularization_weight: float = 0.0  # L2 penalty on soft prompt norm; 0 = disabled
    # Final step judge evaluation
    final_step_judge: bool = False  # If true, run judge eval on separate eval questions at end of training.
    final_step_judge_eval_questions_path: Optional[str] = None  # Path to eval questions JSON. None uses bundled default.
    final_step_judge_max_new_tokens: int = 512  # Max tokens for response generation in final step eval.
    judge_model: str = "gpt-5-nano"  # Model name for judge evaluation (hallucination, emotion, disfluency judges).
    inference_config: Optional[Dict[str, Any]] = None  # Model inference config (temperature, top_p, etc.) from models.yaml.
    chat_template_kwargs: Optional[Dict[str, Any]] = None  # Extra kwargs for tokenizer.apply_chat_template (e.g. {"enable_thinking": False} for Qwen3).


class PreferenceOptimizer:
    """Optimizer that minimizes preference loss."""

    def __init__(self, scorer, config: Optional[OptimConfig] = None, device: Optional[torch.device] = None):
        self.scorer = scorer
        self.config = config or OptimConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._last_effective_batch_size = max(1, int(self.config.comparison_batch_size))
        self._tokenizer = getattr(self.scorer, "tokenizer", None)
        # Forward comparison_batch_size to scorer for ground truth sub-batching
        self.scorer.comparison_batch_size = self._last_effective_batch_size
        # Forward config flags to scorer for use during batch building
        self.scorer.prepend_conversations = bool(self.config.prepend_conversations)
        self.scorer.conversations_min_turns = int(self.config.conversations_min_turns)
        self.scorer.conversations_max_turns = int(self.config.conversations_max_turns)
        self.scorer.soft_prompt_placement = self.config.soft_prompt_placement
        self.scorer.system_prompt_text = self.config.system_prompt_text
        self.scorer.system_prompt_text_base = self.config.system_prompt_text_base
        self.scorer.system_prompt_diversity_proportion = float(self.config.system_prompt_diversity_proportion)
        self.scorer.candidate_position_at_user_prompt = self.config.candidate_position_at_user_prompt
        self.scorer.soft_prompt_prefix = self.config.soft_prompt_prefix
        self.scorer.all_consistency_loss_weight = float(self.config.all_consistency_loss_weight)
        self._current_pgd_step_size = max(float(self.config.step_size), 0.0)

        # Update scorer with background KL config
        self.scorer.weight_background_kl = float(self.config.weight_background_kl)
        self.scorer.background_kl_num_prompts = int(self.config.background_kl_num_prompts)

        # Forward chat_template_kwargs (e.g. enable_thinking=False for Qwen3 models)
        self.scorer.chat_template_kwargs = self.config.chat_template_kwargs or {}

        # Get rank for distributed training (only rank 0 should log to wandb)
        self.rank = getattr(scorer, 'rank', 0)
        self.world_size = getattr(scorer, 'world_size', 1)

        # Embedding augmentation instance (placeholder - not implemented yet)

    # Note: low-level helpers such as LR scheduling, simple embedding transforms,
    # and the fixed-batch scoring wrapper live in soft_prompt_utils.optimizer.

    def _apply_pgd_update(
        self,
        pert: torch.Tensor,
        grad: torch.Tensor,
        *,
        current_loss: float,
        references: List[str],
        comparison_plan: List[ComparisonDefinition],
    ) -> None:
        direction = grad.sign() if self.config.optimizer_type == "sign" else grad
        base_step = min(self._current_pgd_step_size, float(self.config.pgd_max_step_size))
        step_taken = base_step

        if self.config.adaptive_pgd and base_step > 0:
            best_step = None
            best_loss = current_loss
            factor = max(float(self.config.pgd_backtrack_factor), 1e-4)
            patience = max(1, int(self.config.pgd_backtrack_patience))

            for attempt in range(patience):
                trial_step = base_step * (factor**attempt)
                if trial_step <= 0:
                    continue

                # Note: Embeddings don't need clamping like images do. We minimize loss so step in -grad direction.
                trial_pert = pert.data - trial_step * direction
                trial_tensor = trial_pert.clone().detach()
                # Forward-only evaluation for line search (no gradients needed)
                trial_loss, _, _, _ = loss_with_dynamic_batch(
                    self.scorer,
                    self.config,
                    trial_tensor,
                    references,
                    comparison_plan,
                    candidate_embeddings_forward=trial_tensor,
                    compute_grad=False,
                    reference_utilities=reference_utilities,
                )
                if trial_loss is None:
                    continue

                if float(trial_loss) <= best_loss:
                    best_loss = float(trial_loss)
                    best_step = trial_step
                    break

            if best_step is not None:
                step_taken = best_step
                growth = max(float(self.config.pgd_growth_factor), 1.0)
                self._current_pgd_step_size = min(best_step * growth, float(self.config.pgd_max_step_size))
            else:
                step_taken = base_step * (factor**patience)
                self._current_pgd_step_size = max(step_taken, 1e-6)

        # Minimize loss: move in -grad direction (scorer returns d(loss)/d(pert))
        pert.data = pert.data - step_taken * direction

    def _apply_pgd_update_normalized(
        self,
        u: torch.Tensor,
        g: torch.Tensor,
        v: torch.Tensor,
        grad_g: torch.Tensor,
        grad_v: torch.Tensor,
        *,
        current_loss: float,
        references: List[str],
        comparison_plan: List[ComparisonDefinition],
        reference_utilities: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """PGD step for normalized soft prompt: update g and v so pert = u + g*v/norm(v)."""
        direction_g = grad_g.sign() if self.config.optimizer_type == "sign" else grad_g
        direction_v = grad_v.sign() if self.config.optimizer_type == "sign" else grad_v
        base_step = min(self._current_pgd_step_size, float(self.config.pgd_max_step_size))
        step_taken = base_step
        eps = 1e-8
        if self.config.adaptive_pgd and base_step > 0:
            best_step = None
            best_loss = current_loss
            factor = max(float(self.config.pgd_backtrack_factor), 1e-4)
            patience = max(1, int(self.config.pgd_backtrack_patience))
            for attempt in range(patience):
                trial_step = base_step * (factor**attempt)
                if trial_step <= 0:
                    continue
                g_trial = g.data - trial_step * direction_g
                v_trial = v.data - trial_step * direction_v
                v_trial = v_trial / (v_trial.norm(dim=-1, keepdim=True).clamp(min=eps))
                trial_pert = _normalized_pert(u, g_trial, v_trial, eps)
                trial_loss, _, _, _ = loss_with_dynamic_batch(
                    self.scorer,
                    self.config,
                    trial_pert,
                    references,
                    comparison_plan,
                    candidate_embeddings_forward=trial_pert,
                    compute_grad=False,
                    reference_utilities=reference_utilities,
                )
                if trial_loss is not None and float(trial_loss) <= best_loss:
                    best_loss = float(trial_loss)
                    best_step = trial_step
                    break
            if best_step is not None:
                step_taken = best_step
                growth = max(float(self.config.pgd_growth_factor), 1.0)
                self._current_pgd_step_size = min(best_step * growth, float(self.config.pgd_max_step_size))
            else:
                step_taken = base_step * (factor**patience)
                self._current_pgd_step_size = max(step_taken, 1e-6)
        g.data = g.data - step_taken * direction_g
        v.data = v.data - step_taken * direction_v
        v.data = v.data / (v.data.norm(dim=-1, keepdim=True).clamp(min=eps))

    def optimize_from_embeddings(
        self,
        references: List[str],
        verbose: bool = True,
        args=None,
        reference_utilities: Optional[Dict[str, Dict[str, float]]] = None,
        background_kl_qa: Optional[List[Dict[str, str]]] = None,
        eval_questions: Optional[List[Dict[str, Any]]] = None,
        consistency_references: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, float, List[float]]:
        """Initialize and run optimization. Returns (final_embeddings, final_loss, loss_history).
        reference_utilities: Optional ref_id -> {mean, variance} from utility_pre.json for consistency comparisons.
        background_kl_qa: Optional list of Q&A dicts ({"question": ..., "response": ...}) for background KL loss.
        eval_questions: Optional list of eval question dicts ({"prompt": ..., "question_id": ...}) for judge evaluation.
        """
        candidate_count = 1
        num_virtual_tokens = self.config.num_virtual_tokens
        init_strategy = self.config.prompt_tuning_init

        # Check for resume data
        resume_data = getattr(args, "resume_data", None) if args else None
        normalize_soft_prompt = self.config.normalize_soft_prompt
        space_embedding = None
        if normalize_soft_prompt:
            space_embedding = _get_space_token_embedding(self.scorer, self.device)
            if verbose and normalize_soft_prompt:
                print(f"[Optimizer] normalize_soft_prompt=True: soft prompt = u + g*v/norm(v), u=space token (fixed), g,v trainable")
        if resume_data is not None:
            init_emb = resume_data["embeddings"]
            start_epoch = resume_data.get("start_epoch", resume_data.get("start_step", 0))  # Backward compatibility
            ema_embeddings = resume_data.get("ema_embeddings")
            print(f"[Optimizer] Resuming from epoch {start_epoch} with {init_emb.shape[0]} embedding candidates")
            return self._optimize(
                init_emb,
                references,
                verbose=verbose,
                args=args,
                start_epoch=start_epoch,
                init_ema=ema_embeddings,
                reference_utilities=reference_utilities,
                space_embedding=space_embedding,
                normalize_soft_prompt=normalize_soft_prompt,
                background_kl_qa=background_kl_qa,
                eval_questions=eval_questions,
                consistency_references=consistency_references,
            )

        # Get model's embedding dimension and dtype (must match for forward pass)
        if not hasattr(self.scorer.model, 'get_input_embeddings'):
            raise ValueError("Model does not have get_input_embeddings method")
        input_embeddings = self.scorer.model.get_input_embeddings()
        hidden_dim = input_embeddings.embedding_dim
        model_dtype = input_embeddings.weight.dtype

        # Set seed for reproducibility
        torch.manual_seed(self.config.init_seed)

        num_init_aggregation = max(1, int(self.config.num_init_aggregation))
        
        # Initialize embeddings based on strategy
        if init_strategy == "random_embedding":
            agg_msg = f", num_init_aggregation={num_init_aggregation}" if num_init_aggregation > 1 else ""
            print(f"[Optimizer] Initializing from random embeddings (seed={self.config.init_seed}, num_virtual_tokens={num_virtual_tokens}{agg_msg})")
            # Random embeddings: sample from normal distribution; optionally average over num_init_aggregation samples
            if num_init_aggregation > 1:
                samples = torch.randn(candidate_count, num_init_aggregation, num_virtual_tokens, hidden_dim, device=self.device, dtype=model_dtype) * 0.02
                init_emb = samples.mean(dim=1)
            else:
                init_emb = torch.randn(candidate_count, num_virtual_tokens, hidden_dim, device=self.device, dtype=model_dtype) * 0.02
            init_emb.requires_grad_(True)
        
        elif init_strategy == "random_tokens":
            agg_msg = f", num_init_aggregation={num_init_aggregation}" if num_init_aggregation > 1 else ""
            print(f"[Optimizer] Initializing from random token embeddings (seed={self.config.init_seed}, num_virtual_tokens={num_virtual_tokens}{agg_msg})")
            vocab_size = input_embeddings.num_embeddings
            if num_init_aggregation > 1:
                token_ids = torch.randint(0, vocab_size, (candidate_count, num_init_aggregation, num_virtual_tokens), device=self.device)
                flat_ids = token_ids.view(candidate_count * num_init_aggregation, num_virtual_tokens)
                flat_emb = input_embeddings(flat_ids).detach()
                init_emb = flat_emb.view(candidate_count, num_init_aggregation, num_virtual_tokens, hidden_dim).mean(dim=1).clone()
            else:
                token_ids = torch.randint(0, vocab_size, (candidate_count, num_virtual_tokens), device=self.device)
                init_emb = input_embeddings(token_ids).detach().clone()
            init_emb.requires_grad_(True)

        elif init_strategy == "space_tokens":
            # num_virtual_tokens copies of the space token; when normalize_soft_prompt=True, add a small random perturbation so g and v are small but well-defined.
            _SPACE_TOKENS_PERTURB_SCALE = 0.02
            space_u = _get_space_token_embedding(self.scorer, self.device)  # (1, 1, hidden_dim)
            init_emb = space_u.to(model_dtype).expand(1, num_virtual_tokens, hidden_dim).repeat(candidate_count, 1, 1).clone()
            init_emb = init_emb + _SPACE_TOKENS_PERTURB_SCALE * torch.randn(
                candidate_count, num_virtual_tokens, hidden_dim, device=self.device, dtype=model_dtype
            )
            init_emb = init_emb.to(model_dtype)
            if verbose:
                msg = f"[Optimizer] Initializing from space tokens (num_virtual_tokens={num_virtual_tokens}"
                if normalize_soft_prompt:
                    msg += f", perturb_scale={_SPACE_TOKENS_PERTURB_SCALE} for g,v"
                msg += ")"
                print(msg)
            init_emb.requires_grad_(True)

        elif init_strategy == "prototype":
            if self.config.text_of_prototype is None:
                raise ValueError("text_of_prototype must be provided when prompt_tuning_init='prototype'")
            print(f"[Optimizer] Initializing from prototype text: '{self.config.text_of_prototype}' (num_virtual_tokens={num_virtual_tokens})")
            # Prototype: tokenize text and use its embeddings
            tokenizer = self.scorer.tokenizer
            tokens = tokenizer(self.config.text_of_prototype, return_tensors="pt", add_special_tokens=False)
            token_ids = tokens["input_ids"].to(self.device)
            
            # Get embeddings for the tokens
            prototype_emb = input_embeddings(token_ids).detach()  # (1, seq_len, hidden_dim)
            
            # If prototype is shorter than num_virtual_tokens, repeat it
            # If longer, take first num_virtual_tokens
            prototype_len = prototype_emb.shape[1]
            if prototype_len < num_virtual_tokens:
                # Repeat the prototype to fill num_virtual_tokens
                repeat_factor = (num_virtual_tokens + prototype_len - 1) // prototype_len
                prototype_emb = prototype_emb.repeat(1, repeat_factor, 1)[:, :num_virtual_tokens, :]
            elif prototype_len > num_virtual_tokens:
                # Take first num_virtual_tokens
                prototype_emb = prototype_emb[:, :num_virtual_tokens, :]
            
            # Expand to candidate_count
            init_emb = prototype_emb.repeat(candidate_count, 1, 1).clone()
            init_emb.requires_grad_(True)
        
        else:
            raise ValueError(
                f"Unknown prompt_tuning_init strategy: {init_strategy}. "
                "Must be one of: 'random_embedding', 'random_tokens', 'space_tokens', 'prototype'"
            )
        
        return self._optimize(
            init_emb,
            references,
            verbose=verbose,
            args=args,
            reference_utilities=reference_utilities,
            space_embedding=space_embedding,
            normalize_soft_prompt=normalize_soft_prompt,
            background_kl_qa=background_kl_qa,
            eval_questions=eval_questions,
            consistency_references=consistency_references,
        )


    def _optimize(
        self,
        init_embeddings: torch.Tensor,
        references: List[str],
        verbose: bool = True,
        args=None,
        start_epoch: int = 0,
        init_ema: Optional[torch.Tensor] = None,
        reference_utilities: Optional[Dict[str, Dict[str, float]]] = None,
        space_embedding: Optional[torch.Tensor] = None,
        normalize_soft_prompt: bool = False,
        background_kl_qa: Optional[List[Dict[str, str]]] = None,
        eval_questions: Optional[List[Dict[str, Any]]] = None,
        consistency_references: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, float, List[float]]:
        # Store consistency references on scorer for text lookup in consistency comparisons
        self.scorer.consistency_references = consistency_references

        if init_embeddings.dim() == 2:
            init_embeddings = init_embeddings.unsqueeze(0)
        init_embeddings = init_embeddings.to(self.device)
        candidate_count = init_embeddings.shape[0]
        _NORM_EPS = 1e-8

        u, g, v, pert = None, None, None, None
        if normalize_soft_prompt and space_embedding is not None:
            u = space_embedding.to(self.device)  # (1, 1, D)
            delta = init_embeddings - u
            v_norm = delta.norm(dim=-1, keepdim=True).clamp(min=_NORM_EPS)
            g = v_norm.detach().clone().requires_grad_(True)
            v = (delta / v_norm).detach().clone().requires_grad_(True)
            params_for_optimizer = [g, v]
        else:
            pert = init_embeddings.clone().detach().requires_grad_(True)
            params_for_optimizer = [pert]

        def get_pert() -> torch.Tensor:
            if normalize_soft_prompt and u is not None and g is not None and v is not None:
                return _normalized_pert(u, g, v, _NORM_EPS)
            return pert

        if verbose:
            p = get_pert()
            print(f"[Optimizer] Soft prompt embedding shape: {p.shape} (candidates={candidate_count}, tokens={p.shape[1]}, hidden_dim={p.shape[2]})")
        loss_history: List[float] = []
        validation_history: List[float] = []
        validation_metrics_history: List[Dict[str, float]] = []
        last_validation_scores: Optional[List[float]] = None
        loss_type = self.config.loss_type

        optimizer = None
        opt_type = self.config.optimizer_type.lower()
        if opt_type == "adamw":
            optimizer = torch.optim.AdamW(params_for_optimizer, lr=self.config.learning_rate)
        elif opt_type == "adam":
            optimizer = torch.optim.Adam(params_for_optimizer, lr=self.config.learning_rate)
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(
                params_for_optimizer,
                lr=self.config.learning_rate,
                momentum=self.config.sgd_momentum,
                nesterov=self.config.sgd_nesterov,
            )
        elif opt_type == "radam":
            optimizer = torch.optim.RAdam(params_for_optimizer, lr=self.config.learning_rate)
        elif opt_type not in {"sign", "pgd", "pgd_adaptive"}:
            raise ValueError(f"Unknown optimizer_type: {self.config.optimizer_type}")

        ema_decay = max(0.0, min(1.0, float(self.config.ema_decay)))
        pert_for_ema = get_pert()
        if init_ema is not None and 0.0 < ema_decay < 1.0:
            ema_embeddings = init_ema.clone().detach().to(self.device)
        else:
            ema_embeddings = pert_for_ema.clone().detach() if 0.0 < ema_decay < 1.0 else None
        

        # Load training state if resuming from checkpoint (optimizer state, adaptive PGD step size)
        resume_data = getattr(args, "resume_data", None) if args else None
        if resume_data is not None and start_epoch > 0:
            checkpoint_path = resume_data.get("checkpoint_path")
            if checkpoint_path is not None:
                _, loaded_pgd_step_size = self._load_training_state(checkpoint_path, optimizer)
                if loaded_pgd_step_size is not None:
                    self._current_pgd_step_size = loaded_pgd_step_size
        
        # ── Early stop metric setup ──
        early_stop_metric_name = self.config.early_stop_metric
        base_early_stop_metric: Optional[float] = None
        early_stop_metric_threshold: Optional[float] = None

        if early_stop_metric_name == "train_kl":
            # For train_kl, we don't compute base (no soft prompt = KL is 0)
            # Instead, we'll use checkpoint-0 train_kl as the base reference
            # base_early_stop_metric will be set after checkpoint-0 is computed
            if self.config.weight_background_kl == 0:
                if verbose:
                    print(f"[Train_kl] Warning: train_kl metric requires weight_background_kl > 0, disabling")
            else:
                if verbose:
                    print(f"[Train_kl] Will use checkpoint-0 train_kl as base reference for early stopping and sweep metric")
        elif early_stop_metric_name is not None:
            raise ValueError(f"Unsupported early_stop_metric: {early_stop_metric_name}. Only 'train_kl' or null supported.")
        # else: early_stop_metric is None — disabled; base_early_stop_metric and threshold stay None

        # Initialize validation history
        validation_history: List[float] = []
        validation_loss_history: List[float] = []
        last_validation_scores: Optional[List[float]] = None  # validation accuracies per candidate
        last_validation_means: Dict[str, float] = {}
        initial_validation_loss_metrics: Optional[Dict[str, float]] = None  # per-task initial loss (for normalized mean)

        # ── Base validation accuracy (without soft prompt) ──
        # Compute validation accuracy on the base model (no soft prompt) for gap closure metric
        base_validation_accuracy: Optional[float] = None
        base_validation_metrics: Optional[Dict[str, float]] = None
        if start_epoch == 0:
            if verbose:
                print("[Base Validation] Computing base validation accuracy (no soft prompt)...")
            # Create a dummy zero embedding to pass (will be ignored since candidate_idx=None means no injection)
            _cfg = self.scorer.model.config
            _hidden_size = getattr(_cfg, "hidden_size", None) or getattr(getattr(_cfg, "text_config", None), "hidden_size", None)
            dummy_embedding = torch.zeros((1, _hidden_size), device=self.device)
            base_validation_metrics = run_all_validations(
                self.scorer,
                embedding=dummy_embedding,
                results_dir=None,
                results_filename_suffix=None,
                candidate_idx=None,  # None means no soft prompt injection
                stimulant_type=self.config.stimulant_type,
                candidate_placeholder_delimiter=self.config.candidate_placeholder_delimiter,
                candidate_position_at_user_prompt=self.config.candidate_position_at_user_prompt,
                soft_prompt_placement=self.config.soft_prompt_placement,
                system_prompt_text=self.config.system_prompt_text,
                system_prompt_text_base=self.config.system_prompt_text_base,

            )
            base_validation_accuracy = _validation_accuracy_consolidated(base_validation_metrics)
            if verbose:
                print(f"[Base Validation] Base validation accuracy (no soft prompt) = {base_validation_accuracy:.4f}")
            if _is_wandb_enabled():
                # Log harmonic mean aggregate
                wandb.summary["validation_accuracy_base/aggregate"] = base_validation_accuracy
                # Log per-task base accuracies
                for metric_key, metric_value in base_validation_metrics.items():
                    if metric_key.endswith("_accuracy"):
                        # Convert "wellbeing_positive_prepend_accuracy" -> "wellbeing_positive_prepend"
                        task_name = metric_key.replace("_accuracy", "")
                        wandb.summary[f"validation_accuracy_base/{task_name}"] = metric_value
                        if verbose:
                            print(f"[Base Validation] {task_name} = {metric_value:.4f}")

        # Seed for sweep_metric rolling window (set at checkpoint-0 if run)
        sweep_metric_seed: Optional[float] = None
        # Checkpoint-0: Run validation and save as initial best checkpoint (if not resuming)
        if self.config.save_steps is not None and self.config.save_steps > 0 and start_epoch == 0 and self.rank == 0:
            run_dir = getattr(args, "run_dir", None) or ""
            base_dir = Path(run_dir).expanduser() if run_dir else Path(getattr(args, "output_dir", ""))
            job_dir = Path(run_dir).expanduser()
            checkpoint_path = base_dir / "checkpoint-step0"
            
            # Run validation and save JSON files to checkpoint-step0/candidate_{i}/
            scores_0: List[float] = []
            last_validation_means = {}
            
            for i in range(get_pert().shape[0]):
                emb = get_pert()[i].detach()
                _ckpt0_candidate_dir = checkpoint_path / f"candidate_{i}"
                _ckpt0_candidate_dir.mkdir(parents=True, exist_ok=True)
                metrics = run_all_validations(
                    self.scorer,
                    emb,
                    results_dir=_ckpt0_candidate_dir,
                    results_filename_suffix=None,
                    candidate_idx=i,
                    stimulant_type=self.config.stimulant_type,
                    candidate_placeholder_delimiter=self.config.candidate_placeholder_delimiter,
                    candidate_position_at_user_prompt=self.config.candidate_position_at_user_prompt,
                    soft_prompt_placement=self.config.soft_prompt_placement,
                    system_prompt_text=self.config.system_prompt_text,
                    system_prompt_text_base=self.config.system_prompt_text_base,
    
                )
                scores_0.append(_validation_accuracy_consolidated(metrics))
                for k, val in metrics.items():
                    last_validation_means[f"{k}_candidate_{i}"] = float(val)
            if initial_validation_loss_metrics is None:
                initial_validation_loss_metrics = _extract_loss_keys_from_metrics(last_validation_means)
            last_validation_scores = scores_0
            init_val_acc = _harmonic_mean(scores_0) if scores_0 else 0.0
            init_val_loss = _validation_loss_consolidated(last_validation_means, initial_validation_loss_metrics)
            # Compute gap_closure from per-task metrics
            init_val_gap_closure = _compute_validation_gap_closure(
                last_validation_means,
                base_validation_metrics if base_validation_metrics else {},
            )
            validation_history.append(init_val_acc)
            validation_loss_history.append(init_val_loss)

            if verbose:
                print(f"[Checkpoint-0] Validation accuracy = {init_val_acc:.4f}, "
                      f"gap closure = {init_val_gap_closure:.4f}")

            # Initialize best-save state from checkpoint-0 (treat as initial best)
            best_save_acc = init_val_acc
            best_save_loss = init_val_loss
            best_save_gap_closure = init_val_gap_closure
            best_save_epoch = -1  # pre-training checkpoint
            best_acc_epoch = -1
            best_loss_epoch = -1
            best_gap_epoch = -1
            step_at_best_gap = 0
            loss_at_best_gap = init_val_loss
            accuracy_at_best_gap = init_val_acc
            step_at_best_checkpoint: Optional[int] = 0  # step at which we last saved best checkpoint (0 = checkpoint-0)

            best_early_stop_metric_at_checkpoint_so_far: Optional[float] = None  # <early_stop_metric> value at best checkpoint
            max_train_kl_so_far: Optional[float] = None  # max train_kl over all steps (set from checkpoint-0 when train_kl)
            best_save_sweep_metric: Optional[float] = None  # sweep_metric at best checkpoint (set below when init_sweep_metric computed)
            validation_metrics_history.append(dict(last_validation_means))
            
            # Save embeddings to main job directory (as initial best)
            try:
                p0 = get_pert()
                for idx in range(p0.shape[0]):
                    emb_path = job_dir / f"optimized_embeddings_{idx}.pt"
                    torch.save(p0[idx].detach().cpu(), emb_path)
                if ema_embeddings is not None:
                    for idx in range(ema_embeddings.shape[0]):
                        emb_path = job_dir / f"optimized_embeddings_{idx}_ema.pt"
                        torch.save(ema_embeddings[idx].detach().cpu(), emb_path)
                
                # Save embeddings to checkpoint-0 directory (validation JSONs already saved above)
                # checkpoint_path dirs already created by validation loop above
                for idx in range(p0.shape[0]):
                    ckpt_emb_path = checkpoint_path / f"optimized_embeddings_{idx}.pt"
                    torch.save(p0[idx].detach().cpu(), ckpt_emb_path)
                if ema_embeddings is not None:
                    for idx in range(ema_embeddings.shape[0]):
                        ckpt_emb_path = checkpoint_path / f"optimized_embeddings_{idx}_ema.pt"
                        torch.save(ema_embeddings[idx].detach().cpu(), ckpt_emb_path)
                _user_pos = self.config.candidate_position_at_user_prompt
                _soft_placement = self.config.soft_prompt_placement
                if _soft_placement == "system_prompt":
                    _eval_pos = "system_prompt"
                else:
                    _eval_pos = _user_pos
                validation_json_filenames = [
                    f"validation_wellbeing_positive_{_eval_pos}.json",
                    f"validation_wellbeing_negative_{_eval_pos}.json",
                    "validation_preference_forced_choice_2newitems.json",
                ]

                # Save training state (optimizer empty at step 0, but saves pgd step size if adaptive_pgd enabled)
                self._save_training_state(checkpoint_path, optimizer)

                # Copy validation JSON files to run directory root (overwrite previous best)
                for i in range(get_pert().shape[0]):
                    candidate_dir = checkpoint_path / f"candidate_{i}"
                    for json_file in validation_json_filenames:
                        src_path = candidate_dir / json_file
                        if src_path.exists():
                            # Save to run directory with candidate-specific name
                            dst_filename = json_file.replace(".json", f"_candidate_{i}.json")
                            dst_path = job_dir / dst_filename
                            shutil.copy2(src_path, dst_path)
                            if verbose:
                                print(f"[Checkpoint-0] Copied {json_file} to run directory: {dst_path}")
                
                # Upload validation JSON files to wandb: both checkpoint subdir and run dir
                if _is_wandb_enabled():
                    # Upload from checkpoint subdirectory
                    for i in range(get_pert().shape[0]):
                        candidate_dir = checkpoint_path / f"candidate_{i}"
                        for json_file in validation_json_filenames:
                            json_path = candidate_dir / json_file
                            if json_path.exists():
                                wandb.save(
                                    str(json_path),
                                    base_path=str(checkpoint_path.parent),
                                    policy="now",
                                )
                    # Upload from run directory root
                    for i in range(get_pert().shape[0]):
                        for json_file in validation_json_filenames:
                            dst_filename = json_file.replace(".json", f"_candidate_{i}.json")
                            json_path = job_dir / dst_filename
                            if json_path.exists():
                                wandb.save(
                                    str(json_path),
                                    base_path=str(job_dir),
                                    policy="now",
                                )
            except Exception as e:
                if verbose:
                    print(f"[Checkpoint-0] Failed to save checkpoint-0: {e}")

            # Compute train_kl metric at checkpoint-0 (soft prompt already inserted)
            init_early_stop_penalty = 0.0
            init_candidate_early_stop_metrics = []
            needs_early_stop_metric_checkpoint0 = (
                early_stop_metric_name == "train_kl" and background_kl_qa and self.config.weight_background_kl > 0
            )
            if needs_early_stop_metric_checkpoint0:
                for cand_idx in range(get_pert().shape[0]):
                    cand_emb = get_pert()[cand_idx].detach()
                    cand_kl = self.scorer._compute_qa_kl_loss(
                        candidate_embeddings=cand_emb.unsqueeze(0),
                        embeddings=cand_emb.unsqueeze(0),
                        qa_pairs=background_kl_qa,
                        compute_grad=False,
                        avg_grad=None,
                    )
                    if cand_kl is not None:
                        init_candidate_early_stop_metrics.append(float(cand_kl))
                if init_candidate_early_stop_metrics:
                    init_mean_metric = float(np.mean(init_candidate_early_stop_metrics))
                    best_early_stop_metric_at_checkpoint_so_far = init_mean_metric
                    max_train_kl_so_far = init_mean_metric
                    # Use checkpoint-0 as base reference
                    base_early_stop_metric = init_mean_metric
                    train_kl_multiplier = self.config.early_stopping_train_kl_multiplier
                    if train_kl_multiplier > 0:
                        early_stop_metric_threshold = train_kl_multiplier * base_early_stop_metric
                    metric_weight = self.config.sweep_metric_train_kl_weight
                    init_early_stop_penalty = metric_weight * init_mean_metric
                    if verbose:
                        metric_strs = ", ".join(f"c{i}={m:.4f}" for i, m in enumerate(init_candidate_early_stop_metrics))
                        _thr_str = f"{early_stop_metric_threshold:.4f}" if early_stop_metric_threshold is not None else "disabled"
                        print(f"[Train_kl] Checkpoint-0: {metric_strs}, "
                              f"base (checkpoint-0) = {base_early_stop_metric:.4f}, "
                              f"threshold = {_thr_str} (multiplier={train_kl_multiplier}), "
                              f"penalty = {init_early_stop_penalty:.4f}")
            # Sweep metric uses gap_closure instead of accuracy
            init_sweep_metric = (1 - init_val_gap_closure) + init_early_stop_penalty
            best_sweep_metric = init_sweep_metric
            best_sweep_metric_epoch = -1
            best_save_sweep_metric = init_sweep_metric
            sweep_metric_seed = init_sweep_metric
            # Log checkpoint-0 validation metrics to wandb
            if _is_wandb_enabled():
                val_log_0 = {
                    "validation_accuracy/aggregate": init_val_acc,
                    "validation_accuracy/best": init_val_acc,
                    "validation_gap_closure/aggregate": init_val_gap_closure,
                    "validation_gap_closure/best": init_val_gap_closure,
                }
                if base_validation_accuracy is not None and base_validation_metrics is not None:
                    val_log_0["validation_accuracy_base/aggregate"] = base_validation_accuracy
                    for metric_key, metric_value in base_validation_metrics.items():
                        if metric_key.endswith("_accuracy"):
                            task_name = metric_key.replace("_accuracy", "")
                            val_log_0[f"validation_accuracy_base/{task_name}"] = metric_value
                val_log_0.update({
                    "validation_loss/aggregate": init_val_loss,
                    "validation_loss/best": init_val_loss,
                    "sweep_metric/raw": init_sweep_metric,
                    "sweep_metric/recent_steps/aggregate": init_sweep_metric,
                    "sweep_metric/recent_steps/best": init_sweep_metric,
                    f"sweep_metric/raw/{early_stop_metric_name}_penalty": init_early_stop_penalty,
                    "train/epoch": -1,  # checkpoint-0 is pre-training
                })
                # Log per-candidate hallucination metric at checkpoint-0
                if init_candidate_early_stop_metrics:
                    base_key = f"{early_stop_metric_name}/checkpoint_0" if early_stop_metric_name == "train_kl" else f"{early_stop_metric_name}/base"
                    val_log_0[base_key] = base_early_stop_metric
                    if early_stop_metric_threshold is not None:
                        val_log_0[f"{early_stop_metric_name}/threshold"] = early_stop_metric_threshold
                    for ci, cm in enumerate(init_candidate_early_stop_metrics):
                        val_log_0[f"{early_stop_metric_name}/candidate_{ci}"] = cm
                # Per-candidate per-task metrics
                for key, value in last_validation_means.items():
                    if "_candidate_" in key:
                        metric_name, cand_id = key.rsplit("_candidate_", 1)
                    else:
                        metric_name, cand_id = key, None

                    if metric_name.endswith("_loss"):
                        group = "validation_loss"
                        task = metric_name[:-5]  # strip "_loss"
                    elif metric_name.endswith("_accuracy"):
                        group = "validation_accuracy"
                        task = metric_name[:-9]  # strip "_accuracy"
                    else:
                        group = "validation_accuracy"
                        task = metric_name

                    if cand_id is not None:
                        val_log_0[f"{group}/candidate_{cand_id}/{task}"] = value
                    else:
                        val_log_0[f"{group}/{task}"] = value
                # Per-candidate consolidated accuracy
                for idx, cand_score in enumerate(scores_0):
                    val_log_0[f"validation_accuracy/candidate_{idx}"] = cand_score
                wandb.log(val_log_0, step=0)
                wandb.summary["validation_accuracy/best"] = init_val_acc
                wandb.summary["validation_accuracy/best_epoch"] = -1
                wandb.summary["validation_gap_closure/best"] = init_val_gap_closure
                wandb.summary["validation_gap_closure/best_epoch"] = -1
                wandb.summary["validation_loss/best"] = init_val_loss
                wandb.summary["validation_loss/best_epoch"] = -1
                wandb.summary["sweep_metric/recent_steps/best"] = init_sweep_metric
                wandb.summary["sweep_metric/recent_steps/best_epoch"] = -1
                wandb.summary["best_checkpoint/accuracy"] = init_val_acc
                wandb.summary["best_checkpoint/gap_closure"] = init_val_gap_closure
                wandb.summary["best_checkpoint/loss"] = init_val_loss
                wandb.summary["best_checkpoint/epoch"] = -1
                wandb.summary["best_checkpoint/step"] = 0
                wandb.summary["validation_gap_closure/best_checkpoint"] = init_val_gap_closure
                wandb.summary["validation_loss/best_checkpoint"] = init_val_loss
                wandb.summary["validation_accuracy/best_checkpoint"] = init_val_acc
                wandb.summary["sweep_metric/recent_steps/best_checkpoint"] = init_sweep_metric
                if early_stop_metric_name == "train_kl" and base_early_stop_metric is not None:
                    wandb.summary["train_kl/best_checkpoint"] = base_early_stop_metric

        # Start from start_epoch if resuming
        remaining_epochs = self.config.epochs - start_epoch
        if start_epoch > 0:
            print(f"[Optimizer] Resuming: {remaining_epochs} epochs remaining (from {start_epoch} to {self.config.epochs})")

        # Initialize comparison_plan to None - will be set in the loop or before final evaluation
        comparison_plan: Optional[List[ComparisonDefinition]] = None
        # Start with the full reference pool passed from the pipeline
        all_references = references  # Keep all references for curriculum sampling
        final_references = references

        # Use provided background KL Q&A pairs or fallback to None
        kl_qa = background_kl_qa

        # Global step counter (tracks optimizer steps across all epochs)
        global_step = 0
        # Track when we last ran evaluation (step-based cadence)
        last_eval_step = 0  # checkpoint-0 evaluates at step 0
        # Track batches per epoch for better total_steps estimation
        batches_per_epoch_history = []
        
        # Early stopping state (based on validation gap closure, higher is better)
        best_validation_gap_closure_for_es = -float('inf')
        patience_counter = 0
        # Hallucination metric early stopping: stop when average over last early_stopping_patience steps exceeds threshold
        early_stop_metric_recent_deque: deque = deque(maxlen=max(1, self.config.early_stopping_patience))
        best_effective_embeddings = None
        best_ema_embeddings = None
        early_stop_triggered = False
        
        # Track best validation for saving embeddings to main job_dir
        # Primary criterion: highest accuracy. Tiebreaker: lowest loss.
        # Initialize from checkpoint-0 if it was created, otherwise use defaults (for resume)
        if not (start_epoch == 0 and self.config.save_steps is not None and self.config.save_steps > 0):
            # Resuming or checkpoint-0 not created: initialize from defaults
            best_save_acc = -float('inf')
            best_save_loss = float('inf')
            best_save_gap_closure = -float('inf')
            best_save_epoch = -1
            best_acc_epoch = -1
            best_loss_epoch = -1
            best_gap_epoch = -1
            step_at_best_gap = 0
            loss_at_best_gap = float('inf')
            accuracy_at_best_gap = -float('inf')
            step_at_best_checkpoint: Optional[int] = None  # step at which we last saved best checkpoint (None = none saved yet)

            best_early_stop_metric_at_checkpoint_so_far: Optional[float] = None  # <early_stop_metric> value at best checkpoint
            max_train_kl_so_far: Optional[float] = None  # max train_kl over all steps
            best_save_sweep_metric: Optional[float] = None  # sweep_metric at best checkpoint (for wandb charts)
        run_dir = getattr(args, "run_dir", None) or ""
        # Buffer of previous best checkpointed embeddings per candidate (in-memory, not persisted)
        _buffer_size = max(0, int(self.config.buffer_size))
        _buffer_fraction = float(self.config.buffer_fraction)
        buffer_per_candidate: List[List[Tuple[torch.Tensor, float, int]]] = [[] for _ in range(candidate_count)]
        # Track the epoch at which overall best accuracy and best loss were achieved
        # (best_acc_epoch and best_loss_epoch are initialized above if checkpoint-0 was created)
        # Sweep metric: pruning uses best (min) of raw over last sweep_recent_steps
        sweep_recent_steps = max(1, int(self.config.sweep_recent_steps))
        sweep_metric_raw_deque: deque = deque(maxlen=sweep_recent_steps)
        if sweep_metric_seed is not None:
            sweep_metric_raw_deque.append(sweep_metric_seed)
        if sweep_metric_raw_deque:
            best_sweep_metric = min(sweep_metric_raw_deque)
            best_sweep_metric_epoch = -1  # from checkpoint-0
        else:
            best_sweep_metric = float('inf')
            best_sweep_metric_epoch = -1
        # Initialize validation metrics (used by progress bar even on non-eval epochs)
        validation_prob = validation_history[-1] if validation_history else 0.0
        validation_loss = validation_loss_history[-1] if validation_loss_history else float('inf')
        validation_gap_closure = self._validation_gap_closure_history[-1] if hasattr(self, '_validation_gap_closure_history') and self._validation_gap_closure_history else 0.0

        # Track latest background KL loss for train_kl hallucination metric
        latest_background_kl_loss: Optional[float] = None

        # SIGTERM handler: when final_step_judge=True, catch SIGTERM to run final step judge eval before exiting
        _sigterm_received = False
        _original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        run_dir = getattr(args, "run_dir", None) or ""

        if self.config.final_step_judge and run_dir:
            def _sigterm_handler(signum, frame):
                nonlocal _sigterm_received
                _sigterm_received = True
                print("[SIGTERM] Received stop signal. Will run final step judge eval before exiting.")
            signal.signal(signal.SIGTERM, _sigterm_handler)

        # Estimate total steps for progress bar (approximate, since batches per epoch can vary)
        # We'll update this dynamically
        epoch_iter = tqdm(range(start_epoch, self.config.epochs), desc="Epochs", initial=start_epoch, total=self.config.epochs)
        for epoch in epoch_iter:
            t_epoch_start = time.perf_counter()
            t_plan, t_wandb, t_save = 0.0, 0.0, 0.0
            epoch_loss_sum = 0.0
            epoch_batch_count = 0
            epoch_comparison_count = 0  # Total comparisons for weighted loss averaging

            t0 = time.perf_counter()
            # Apply curriculum learning and/or reference sampling
            curriculum_type = self.config.curriculum_type
            num_samples = self.config.num_samples
            
            # Sample references (with or without curriculum learning)
            if curriculum_type is not None and reference_utilities is not None:
                # Get curriculum mixing parameters
                start_proportion = self.config.curriculum_mixing_start_proportion
                end_proportion = self.config.curriculum_mixing_end_proportion
                transition_fraction = self.config.curriculum_mixing_transition_fraction
                threshold_type = self.config.curriculum_mixing_threshold_type
                
                # Sample references according to curriculum
                references = sample_references_curriculum(
                    references=all_references,
                    reference_utilities=reference_utilities,
                    num_samples=num_samples,
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                    curriculum_type=curriculum_type,
                    threshold_type=threshold_type,
                    start_proportion=start_proportion,
                    end_proportion=end_proportion,
                    transition_fraction=transition_fraction,
                    rng=None,
                )
            elif num_samples is not None:
                # No curriculum (or curriculum enabled but no reference_utilities), but limit number of references
                # sample_references_curriculum will handle the case where num_samples >= len(references)
                references = sample_references_curriculum(
                    references=all_references,
                    reference_utilities=reference_utilities,
                    num_samples=num_samples,
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                    curriculum_type=None,
                    rng=None,
                )
            
            # Build random comparison plan over the (possibly curriculum-sampled) reference pool
            references, comparison_plan = build_random_comparison_plan(
                candidate_count=candidate_count,
                references=references,
                min_size=self.config.min_comparison_size,
                max_size=self.config.max_comparison_size,

                rng=None,
                repetition_fraction=self.config.repetition_fraction,
                min_repetition=self.config.min_repetition,
                max_repetition=self.config.max_repetition,
                reference_utilities=reference_utilities,
                consistency_fraction=self.config.consistency_fraction,
                composite_consistency_fraction=self.config.composite_consistency_fraction,
                composite_repetition_fraction=self.config.composite_repetition_fraction,
                wellbeing_fraction=self.config.wellbeing_fraction,
                buffer_fraction=_buffer_fraction,
                buffer_sizes_per_candidate=[len(b) for b in buffer_per_candidate],
                type_s_fraction=self.config.type_s_fraction,
                stimulant_type=self.config.stimulant_type,
                candidate_position=self.config.candidate_position_at_user_prompt,

                soft_prompt_placement=self.config.soft_prompt_placement,
                conversation_min_turns=self.config.conversations_min_turns,
                conversation_max_turns=self.config.conversations_max_turns if self.config.prepend_conversations else 0,
                consistency_references=consistency_references,
                mirror_comparisons_in_system_prompt=self.config.mirror_comparisons_in_system_prompt,
                current_in_system_prompt_fraction=self.config.current_in_system_prompt_fraction,
            )
            final_references = references
            t_plan = time.perf_counter() - t0

            candidates_forward = get_pert()


            # Process: either one step per epoch (accumulate gradient over full plan) or one step per batch
            batch_size = max(1, int(self.config.comparison_batch_size))
            t_batch_start = time.perf_counter()
            
            # Estimate batches in this epoch for LR schedule
            if self.config.optimize_per_epoch:
                current_epoch_batches = 1
            else:
                current_epoch_batches = (len(comparison_plan) + batch_size - 1) // batch_size

            # Prepare buffer embeddings on device for this epoch
            _buffer_on_device: Optional[Dict[int, List[torch.Tensor]]] = None
            if _buffer_size > 0 and _buffer_fraction > 0 and any(len(b) > 0 for b in buffer_per_candidate):
                _buffer_on_device = {
                    ci: [entry[0].to(self.device) for entry in entries]
                    for ci, entries in enumerate(buffer_per_candidate)
                    if entries
                }

            if self.config.optimize_per_epoch:
                # Accumulate gradient across all batches in epoch via score_tensor; update once per epoch
                # IMPORTANT: Use candidates_forward for both embeddings and candidate_embeddings_forward
                # to ensure the tensor used for forward pass is the same one we differentiate w.r.t.
                epoch_loss, avg_grad, background_kl_loss_val, consistency_loss_val = loss_with_dynamic_batch(
                    self.scorer,
                    self.config,
                    candidates_forward,
                    references,
                    comparison_plan,
                    candidate_embeddings_forward=candidates_forward,
                    compute_grad=True,
                    reference_utilities=reference_utilities,
                    background_kl_qa=kl_qa,
                    buffer_embeddings=_buffer_on_device,
                )
                # Store latest background KL loss for train_kl hallucination metric
                if background_kl_loss_val is not None:
                    latest_background_kl_loss = float(background_kl_loss_val)
                if epoch_loss is None or avg_grad is None:
                    if verbose:
                        print(f"Epoch {epoch}: loss_with_dynamic_batch returned None, skipping")
                    epoch_batch_count = 0
                    epoch_loss_sum = 0.0
                    t_batch = time.perf_counter() - t_batch_start
                    continue
                else:
                    self._last_effective_batch_size = len(comparison_plan)
                    # LR schedule: one step per epoch -> total_steps = epochs
                    if optimizer is not None and self.config.lr_schedule != "constant":
                        estimated_total_steps = self.config.epochs
                        current_lr = get_scheduled_lr(
                            self.config,
                            global_step,
                            estimated_total_steps,
                            epoch=epoch,
                            total_epochs=self.config.epochs,
                        )
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = current_lr
                    combined_grad = avg_grad
                    mag_reg_loss_val = None
                    mag_reg_loss_raw = None
                    if self.config.magnitude_regularization_weight > 0:
                        current_pert = get_pert()  # (C, T, D), differentiable
                        # Mean of per-token L2 norms so penalty is independent of num_virtual_tokens
                        per_token_norms = current_pert.norm(dim=-1)  # (C, T)
                        mag_reg_loss_raw = float(per_token_norms.mean().item())
                        mag_reg_loss_val = self.config.magnitude_regularization_weight * mag_reg_loss_raw
                        # Analytical gradient: d/d_pert [w * mean(||pert_t||)] = w / (C*T) * pert_t / ||pert_t||
                        with torch.no_grad():
                            scale = self.config.magnitude_regularization_weight / per_token_norms.numel()
                            mag_reg_grad = scale * current_pert / per_token_norms.clamp(min=_NORM_EPS).unsqueeze(-1)
                        combined_grad = combined_grad + mag_reg_grad
                    with torch.no_grad():
                        opt_type_lower = self.config.optimizer_type.lower()
                        if opt_type_lower in {"sign", "pgd", "pgd_adaptive"}:
                            if self.config.max_grad_norm > 0.0:
                                grad_norm = combined_grad.norm()
                                if grad_norm > self.config.max_grad_norm:
                                    combined_grad = combined_grad * (self.config.max_grad_norm / grad_norm)
                            if normalize_soft_prompt and u is not None and g is not None and v is not None:
                                grad_g, grad_v = _grad_pert_to_grad_gv(combined_grad, g, v, _NORM_EPS)
                                self._apply_pgd_update_normalized(
                                    u, g, v, grad_g, grad_v,
                                    current_loss=float(epoch_loss),
                                    references=references,
                                    comparison_plan=comparison_plan,
                                    reference_utilities=reference_utilities,
                                )
                            else:
                                self._apply_pgd_update(
                                    pert,
                                    combined_grad,
                                    current_loss=float(epoch_loss),
                                    references=references,
                                    comparison_plan=comparison_plan,
                                )
                        elif opt_type_lower in {"adam", "adamw", "sgd", "radam"} and optimizer is not None:
                            grad_to_apply = combined_grad.detach()
                            if normalize_soft_prompt and g is not None and v is not None:
                                grad_g, grad_v = _grad_pert_to_grad_gv(grad_to_apply, g, v, _NORM_EPS)
                                g.grad = grad_g
                                v.grad = grad_v
                                if self.config.max_grad_norm > 0.0:
                                    torch.nn.utils.clip_grad_norm_([g, v], self.config.max_grad_norm)
                            else:
                                while grad_to_apply.dim() < pert.dim():
                                    grad_to_apply = grad_to_apply.unsqueeze(0)
                                if grad_to_apply.shape != pert.shape:
                                    grad_to_apply = grad_to_apply.expand_as(pert)
                                pert.grad = grad_to_apply.clone()
                                if self.config.max_grad_norm > 0.0:
                                    torch.nn.utils.clip_grad_norm_([pert], self.config.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()
                        else:
                            raise ValueError(f"Unknown optimizer_type: {self.config.optimizer_type}")
                        if ema_embeddings is not None:
                            ema_embeddings.mul_(ema_decay).add_(get_pert().detach(), alpha=1.0 - ema_decay)
                    epoch_loss_sum = float(epoch_loss)
                    epoch_batch_count = 1
                    epoch_comparison_count = len(comparison_plan)
                    loss_history.append(epoch_loss_sum)
                    if _is_wandb_enabled():
                        t_wandb_batch_start = time.perf_counter()
                        loss_total = float(epoch_loss)
                        if background_kl_loss_val is not None:
                            loss_total += self.config.weight_background_kl * background_kl_loss_val
                        if mag_reg_loss_val is not None:
                            loss_total += mag_reg_loss_val

                        log_dict = {
                            f"train/loss/{loss_type}": float(epoch_loss) - self.config.all_consistency_loss_weight * consistency_loss_val,
                            "train/loss/consistency_cross_entropy": consistency_loss_val,
                            "train/comparison_batch_size": self._last_effective_batch_size,
                            "train/avg_grad_norm": float(avg_grad.norm().item()),
                            "train/loss/total": loss_total,
                            "train/epoch": epoch,
                        }
                        # Add LR to log if using an optimizer
                        if optimizer is not None:
                            # Use current LR from first param group
                            log_dict["train/learning_rate"] = optimizer.param_groups[0]["lr"]

                        if mag_reg_loss_raw is not None:
                            log_dict["train/loss/magnitude_regularization"] = mag_reg_loss_raw
                        if background_kl_loss_val is not None:
                            log_dict["train/loss/background_kl"] = background_kl_loss_val
                        if self.config.optimizer_type in {"sign", "pgd", "pgd_adaptive"} and self.config.adaptive_pgd:
                            log_dict["train/pgd/step_size"] = self._current_pgd_step_size
                        if self.rank == 0:
                            wandb.log(log_dict, step=global_step)
                        t_wandb += time.perf_counter() - t_wandb_batch_start
                    global_step += 1
                    batches_per_epoch_history.append(1)
                    t_batch = time.perf_counter() - t_batch_start
            else:
                # Process batches one at a time, with optional gradient accumulation
                # IMPORTANT: Use candidates_forward for both embeddings and candidate_embeddings_forward
                # to ensure the tensor used for forward pass is the same one we differentiate w.r.t.
                estimated_batches_per_epoch = (len(comparison_plan) + batch_size - 1) // batch_size
                # Cap gradient_accumulation_steps at the number of batches per epoch
                # (values greater than this would effectively step once per epoch anyway)
                grad_accumulation_steps = max(1, min(int(self.config.gradient_accumulation_steps), estimated_batches_per_epoch))
                if self.config.gradient_accumulation_steps > estimated_batches_per_epoch and verbose:
                    print(f"[Warning] gradient_accumulation_steps ({self.config.gradient_accumulation_steps}) > batches_per_epoch ({estimated_batches_per_epoch}). Capping to {grad_accumulation_steps}.")
                # When grad_accumulation_steps equals batches per epoch, group by size (like optimize_per_epoch=True)
                # Otherwise, shuffle all comparisons (for more diverse batches)
                use_group_by_size = (grad_accumulation_steps >= estimated_batches_per_epoch)
                
                batch_generator = self.scorer.process_batches(
                    embeddings=candidates_forward,
                    references=references,
                    comparison_plan=comparison_plan,
                    batch_size=batch_size,
                    loss_type=loss_type,
                    candidate_embeddings_forward=candidates_forward,
                    reference_utilities=reference_utilities,
                    group_by_size=use_group_by_size,
                    background_kl_qa=kl_qa,
                    buffer_embeddings=_buffer_on_device,
                )
                epoch_batch_count_for_lr = 0
                accumulated_raw_grad = None  # Accumulate raw gradients (to match score_tensor)
                accumulated_loss = 0.0
                accumulated_consistency_loss = 0.0
                accumulated_comparisons = 0
                accumulated_background_kl_loss = 0.0
                accumulated_background_kl_count = 0
                accumulation_step_count = 0
                # Track total candidate counts across all batches for proper normalization (to match score_tensor)
                total_candidate_counts = torch.zeros(
                    candidates_forward.shape[0], device=candidates_forward.device, dtype=torch.float32
                )
                
                for batch_loss, batch_grad_normalized, actual_batch_size, batch_grad_raw, candidate_counts_batch, background_kl_loss_val, batch_consistency_loss in batch_generator:
                    # Check for deferred SIGTERM between batches
                    if _sigterm_received:
                        print("[SIGTERM] Received stop signal mid-epoch. Breaking out of batch loop.")
                        break

                    epoch_batch_count_for_lr += 1
                    accumulation_step_count += 1
                    
                    # Accumulate raw gradients (to match score_tensor normalization)
                    # Accumulate raw gradients (like score_tensor does)
                    if accumulated_raw_grad is None:
                        accumulated_raw_grad = batch_grad_raw.clone()
                    else:
                        accumulated_raw_grad.add_(batch_grad_raw)
                    
                    # Accumulate candidate counts (for proper normalization like score_tensor)
                    total_candidate_counts.add_(candidate_counts_batch)
                    
                    accumulated_loss += batch_loss * actual_batch_size
                    accumulated_consistency_loss += batch_consistency_loss * actual_batch_size
                    accumulated_comparisons += actual_batch_size
                    
                    # Accumulate background KL loss
                    if background_kl_loss_val is not None:
                        accumulated_background_kl_loss += background_kl_loss_val
                        accumulated_background_kl_count += 1
                    
                    # Update learning rate based on schedule (per accumulation step, not per batch)
                    should_step = (accumulation_step_count % grad_accumulation_steps == 0)
                    if should_step and optimizer is not None and self.config.lr_schedule != "constant":
                        # When grad_accumulation_steps equals batches per epoch, use epoch-level scheduling (like optimize_per_epoch=True)
                        if grad_accumulation_steps >= estimated_batches_per_epoch:
                            # One step per epoch (matches optimize_per_epoch=True)
                            estimated_total_steps = self.config.epochs
                        else:
                            # Estimate total steps: use average batches per epoch from history, or current epoch's estimate
                            if batches_per_epoch_history:
                                avg_batches_per_epoch = sum(batches_per_epoch_history) / len(batches_per_epoch_history)
                                # Account for gradient accumulation: effective steps = batches / grad_accum_steps
                                estimated_total_steps = int(self.config.epochs * avg_batches_per_epoch / grad_accumulation_steps)
                            else:
                                estimated_total_steps = int(self.config.epochs * current_epoch_batches / grad_accumulation_steps)
                        
                        current_lr = get_scheduled_lr(
                            self.config,
                            global_step,
                            estimated_total_steps,
                            epoch=epoch,
                            total_epochs=self.config.epochs,
                        )
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = current_lr

                    # Update optimizer after accumulating gradients (every grad_accumulation_steps batches)
                    if should_step:
                        # Normalize by total candidate counts (like score_tensor) instead of number of batches
                        avg_accumulated_grad = torch.zeros_like(accumulated_raw_grad)
                        total_candidate_counts_clamped = torch.clamp(total_candidate_counts, min=0)
                        nonzero_mask = total_candidate_counts_clamped > 0
                        if nonzero_mask.any():
                            counts = total_candidate_counts_clamped[nonzero_mask].to(accumulated_raw_grad.dtype)
                            while counts.dim() < accumulated_raw_grad.dim():
                                counts = counts.unsqueeze(-1)
                            avg_accumulated_grad[nonzero_mask] = accumulated_raw_grad[nonzero_mask] / counts

                        
                        avg_accumulated_loss = accumulated_loss / accumulated_comparisons if accumulated_comparisons > 0 else 0.0
                        avg_accumulated_consistency_loss = accumulated_consistency_loss / accumulated_comparisons if accumulated_comparisons > 0 else 0.0
                        avg_accumulated_background_kl_loss = accumulated_background_kl_loss / accumulated_background_kl_count if accumulated_background_kl_count > 0 else 0.0

                        # Store latest background KL loss for train_kl hallucination metric
                        if accumulated_background_kl_count > 0:
                            latest_background_kl_loss = float(avg_accumulated_background_kl_loss)

                        mag_reg_loss_val = None
                        mag_reg_loss_raw = None
                        if self.config.magnitude_regularization_weight > 0:
                            current_pert = get_pert().detach()  # (C, T, D)
                            per_token_norms = current_pert.norm(dim=-1)  # (C, T)
                            mag_reg_loss_raw = float(per_token_norms.mean().item())
                            mag_reg_loss_val = self.config.magnitude_regularization_weight * mag_reg_loss_raw
                            scale = self.config.magnitude_regularization_weight / per_token_norms.numel()
                            mag_reg_grad = scale * current_pert / per_token_norms.clamp(min=_NORM_EPS).unsqueeze(-1)
                            avg_accumulated_grad = avg_accumulated_grad + mag_reg_grad

                        # Log to wandb per accumulation step (before optimizer step)
                        if _is_wandb_enabled():
                            t_wandb_batch_start = time.perf_counter()
                            loss_total = avg_accumulated_loss
                            if avg_accumulated_background_kl_loss > 0:
                                loss_total += self.config.weight_background_kl * avg_accumulated_background_kl_loss
                            if mag_reg_loss_val is not None:
                                loss_total += mag_reg_loss_val

                            avg_grad_norm = float(avg_accumulated_grad.norm().item())
                            log_dict = {
                                f"train/loss/{loss_type}": avg_accumulated_loss - self.config.all_consistency_loss_weight * avg_accumulated_consistency_loss,
                                "train/loss/consistency_cross_entropy": avg_accumulated_consistency_loss,
                                "train/comparison_batch_size": accumulated_comparisons,  # Total comparisons in this accumulation step
                                "train/avg_grad_norm": avg_grad_norm,
                                "train/loss/total": loss_total,
                                "train/epoch": epoch,
                                "train/gradient_accumulation_steps": grad_accumulation_steps,
                            }
                            # Add LR to log if using an optimizer
                            if optimizer is not None:
                                # Use current LR from first param group
                                log_dict["train/learning_rate"] = optimizer.param_groups[0]["lr"]

                            if mag_reg_loss_raw is not None:
                                log_dict["train/loss/magnitude_regularization"] = mag_reg_loss_raw
                            if avg_accumulated_background_kl_loss > 0:
                                log_dict["train/loss/background_kl"] = avg_accumulated_background_kl_loss
                            if self.config.optimizer_type in {"sign", "pgd", "pgd_adaptive"} and self.config.adaptive_pgd:
                                log_dict["train/pgd/step_size"] = self._current_pgd_step_size
                            if self.rank == 0:
                                wandb.log(log_dict, step=global_step)
                            t_wandb += time.perf_counter() - t_wandb_batch_start

                        with torch.no_grad():
                            opt_type_lower = self.config.optimizer_type.lower()
                            if opt_type_lower in {"sign", "pgd", "pgd_adaptive"}:
                                if self.config.max_grad_norm > 0.0:
                                    grad_norm = avg_accumulated_grad.norm()
                                    if grad_norm > self.config.max_grad_norm:
                                        avg_accumulated_grad = avg_accumulated_grad * (self.config.max_grad_norm / grad_norm)
                                if normalize_soft_prompt and u is not None and g is not None and v is not None:
                                    grad_g, grad_v = _grad_pert_to_grad_gv(avg_accumulated_grad, g, v, _NORM_EPS)
                                    self._apply_pgd_update_normalized(
                                        u, g, v, grad_g, grad_v,
                                        current_loss=float(avg_accumulated_loss),
                                        references=references,
                                        comparison_plan=comparison_plan,
                                        reference_utilities=reference_utilities,
                                    )
                                else:
                                    self._apply_pgd_update(
                                        pert,
                                        avg_accumulated_grad,
                                        current_loss=float(avg_accumulated_loss),
                                        references=references,
                                        comparison_plan=comparison_plan,
                                    )
                            elif opt_type_lower in {"adam", "adamw", "sgd", "radam"} and optimizer is not None:
                                grad_to_apply = avg_accumulated_grad.detach()
                                if normalize_soft_prompt and g is not None and v is not None:
                                    grad_g, grad_v = _grad_pert_to_grad_gv(grad_to_apply, g, v, _NORM_EPS)
                                    g.grad = grad_g
                                    v.grad = grad_v
                                    if self.config.max_grad_norm > 0.0:
                                        torch.nn.utils.clip_grad_norm_([g, v], self.config.max_grad_norm)
                                else:
                                    while grad_to_apply.dim() < pert.dim():
                                        grad_to_apply = grad_to_apply.unsqueeze(0)
                                    if grad_to_apply.shape != pert.shape:
                                        grad_to_apply = grad_to_apply.expand_as(pert)
                                    pert.grad = grad_to_apply.clone()
                                    if self.config.max_grad_norm > 0.0:
                                        torch.nn.utils.clip_grad_norm_([pert], self.config.max_grad_norm)
                                optimizer.step()
                                optimizer.zero_grad()
                            else:
                                raise ValueError(f"Unknown optimizer_type: {self.config.optimizer_type}")

                            if ema_embeddings is not None:
                                ema_embeddings.mul_(ema_decay).add_(get_pert().detach(), alpha=1.0 - ema_decay)

                        # Reset accumulation
                        accumulated_raw_grad = None
                        accumulated_loss = 0.0
                        accumulated_consistency_loss = 0.0
                        accumulated_comparisons = 0
                        total_candidate_counts.zero_()
                        
                        # Only increment global_step when we actually step the optimizer
                        global_step += 1

                    # Accumulate loss for weighted epoch average (batch_loss is per-comparison average)
                    epoch_loss_sum += batch_loss * actual_batch_size
                    epoch_batch_count += 1
                    epoch_comparison_count += actual_batch_size
                    loss_history.append(batch_loss)

                # Handle remaining accumulated gradients at end of epoch (if any)
                if accumulated_raw_grad is not None and accumulation_step_count % grad_accumulation_steps != 0:
                    # Normalize by total candidate counts (like score_tensor) instead of number of batches
                    avg_accumulated_grad = torch.zeros_like(accumulated_raw_grad)
                    total_candidate_counts_clamped = torch.clamp(total_candidate_counts, min=0)
                    nonzero_mask = total_candidate_counts_clamped > 0
                    if nonzero_mask.any():
                        counts = total_candidate_counts_clamped[nonzero_mask].to(accumulated_raw_grad.dtype)
                        while counts.dim() < accumulated_raw_grad.dim():
                            counts = counts.unsqueeze(-1)
                        avg_accumulated_grad[nonzero_mask] = accumulated_raw_grad[nonzero_mask] / counts

                    
                    avg_accumulated_loss = accumulated_loss / accumulated_comparisons if accumulated_comparisons > 0 else 0.0
                    avg_accumulated_background_kl_loss = accumulated_background_kl_loss / accumulated_background_kl_count if accumulated_background_kl_count > 0 else 0.0
                    if accumulated_background_kl_count > 0:
                        latest_background_kl_loss = float(avg_accumulated_background_kl_loss)

                    if self.config.magnitude_regularization_weight > 0:
                        current_pert = get_pert().detach()  # (C, T, D)
                        per_token_norms = current_pert.norm(dim=-1)  # (C, T)
                        scale = self.config.magnitude_regularization_weight / per_token_norms.numel()
                        mag_reg_grad = scale * current_pert / per_token_norms.clamp(min=_NORM_EPS).unsqueeze(-1)
                        avg_accumulated_grad = avg_accumulated_grad + mag_reg_grad

                    # Update learning rate
                    if optimizer is not None and self.config.lr_schedule != "constant":
                        # When grad_accumulation_steps equals batches per epoch, use epoch-level scheduling (like optimize_per_epoch=True)
                        if grad_accumulation_steps >= estimated_batches_per_epoch:
                            # One step per epoch (matches optimize_per_epoch=True)
                            estimated_total_steps = self.config.epochs
                        else:
                            if batches_per_epoch_history:
                                avg_batches_per_epoch = sum(batches_per_epoch_history) / len(batches_per_epoch_history)
                                estimated_total_steps = int(self.config.epochs * avg_batches_per_epoch / grad_accumulation_steps)
                            else:
                                estimated_total_steps = int(self.config.epochs * current_epoch_batches / grad_accumulation_steps)
                        
                        current_lr = get_scheduled_lr(
                            self.config,
                            global_step,
                            estimated_total_steps,
                            epoch=epoch,
                            total_epochs=self.config.epochs,
                        )
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = current_lr
                    
                    with torch.no_grad():
                        opt_type_lower = self.config.optimizer_type.lower()
                        if opt_type_lower in {"sign", "pgd", "pgd_adaptive"}:
                            if self.config.max_grad_norm > 0.0:
                                grad_norm = avg_accumulated_grad.norm()
                                if grad_norm > self.config.max_grad_norm:
                                    avg_accumulated_grad = avg_accumulated_grad * (self.config.max_grad_norm / grad_norm)
                            if normalize_soft_prompt and u is not None and g is not None and v is not None:
                                grad_g, grad_v = _grad_pert_to_grad_gv(avg_accumulated_grad, g, v, _NORM_EPS)
                                self._apply_pgd_update_normalized(
                                    u, g, v, grad_g, grad_v,
                                    current_loss=float(avg_accumulated_loss),
                                    references=references,
                                    comparison_plan=comparison_plan,
                                    reference_utilities=reference_utilities,
                                )
                            else:
                                self._apply_pgd_update(
                                    pert,
                                    avg_accumulated_grad,
                                    current_loss=float(avg_accumulated_loss),
                                    references=references,
                                    comparison_plan=comparison_plan,
                                )
                        elif opt_type_lower in {"adam", "adamw", "sgd", "radam"} and optimizer is not None:
                            grad_to_apply = avg_accumulated_grad.detach()
                            if normalize_soft_prompt and g is not None and v is not None:
                                grad_g, grad_v = _grad_pert_to_grad_gv(grad_to_apply, g, v, _NORM_EPS)
                                g.grad = grad_g
                                v.grad = grad_v
                                if self.config.max_grad_norm > 0.0:
                                    torch.nn.utils.clip_grad_norm_([g, v], self.config.max_grad_norm)
                            else:
                                while grad_to_apply.dim() < pert.dim():
                                    grad_to_apply = grad_to_apply.unsqueeze(0)
                                if grad_to_apply.shape != pert.shape:
                                    grad_to_apply = grad_to_apply.expand_as(pert)
                                pert.grad = grad_to_apply.clone()
                                if self.config.max_grad_norm > 0.0:
                                    torch.nn.utils.clip_grad_norm_([pert], self.config.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()
                        else:
                            raise ValueError(f"Unknown optimizer_type: {self.config.optimizer_type}")

                        if ema_embeddings is not None:
                            ema_embeddings.mul_(ema_decay).add_(get_pert().detach(), alpha=1.0 - ema_decay)

                    global_step += 1
                
                t_batch = time.perf_counter() - t_batch_start
                # Track batches per epoch for better LR schedule estimation
                if epoch_batch_count > 0:
                    batches_per_epoch_history.append(epoch_batch_count)
                    # Keep only last 5 epochs for rolling average
                    if len(batches_per_epoch_history) > 5:
                        batches_per_epoch_history.pop(0)
            
            # Check for deferred SIGTERM after batch loop (break epoch loop immediately)
            if _sigterm_received:
                print("[SIGTERM] Stopping training to run final judge evaluation.")
                early_stop_triggered = True
                break

            # After processing all batches in epoch, compute epoch average loss
            if epoch_batch_count > 0:
                if self.config.optimize_per_epoch:
                    # optimize_per_epoch=True: epoch_loss_sum is already the global average from score_tensor
                    epoch_avg_loss = epoch_loss_sum
                else:
                    # optimize_per_epoch=False: epoch_loss_sum is weighted sum, divide by total comparisons
                    epoch_avg_loss = epoch_loss_sum / epoch_comparison_count if epoch_comparison_count > 0 else 0.0
            else:
                if verbose:
                    print(f"Epoch {epoch}: no batches processed, skipping")
                continue

            # ── Evaluation block: runs every save_steps optimizer steps ──
            # Includes: validation, sweep metric, trajectory, best-checkpoint saving, early stopping
            t0 = time.perf_counter()
            save_steps = self.config.save_steps or 0
            is_final_epoch = (epoch == self.config.epochs - 1)
            # Skip first periodic evaluation if it's too close to checkpoint-0 (which happens at step 0)
            # This avoids duplicate evaluation when save_steps is very small
            is_too_close_to_checkpoint0 = (global_step > 0 and global_step - last_eval_step < save_steps and last_eval_step == 0)
            should_evaluate = (
                save_steps > 0
                and self.rank == 0
                and not is_too_close_to_checkpoint0
                and (global_step - last_eval_step >= save_steps or is_final_epoch)
            )

            if should_evaluate:
                last_eval_step = global_step

                # ── 1. Validation (accuracy & loss) ──
                scores_at_save: List[float] = []
                last_validation_means = {}
                
                # Create checkpoint directory for this validation step to save JSONs
                _val_checkpoint_path = Path(run_dir).expanduser() / f"checkpoint-step{global_step}"
                for i in range(candidate_count):
                    emb = get_pert()[i].detach()
                    _val_candidate_dir = _val_checkpoint_path / f"candidate_{i}"
                    _val_candidate_dir.mkdir(parents=True, exist_ok=True)
                    metrics = run_all_validations(
                        self.scorer,
                        emb,
                        results_dir=_val_candidate_dir,
                        results_filename_suffix=None,
                        candidate_idx=i,
                        stimulant_type=self.config.stimulant_type,
                        candidate_placeholder_delimiter=self.config.candidate_placeholder_delimiter,
                        candidate_position_at_user_prompt=self.config.candidate_position_at_user_prompt,
                        soft_prompt_placement=self.config.soft_prompt_placement,
                        system_prompt_text=self.config.system_prompt_text,
                        system_prompt_text_base=self.config.system_prompt_text_base,
        
                    )
                    scores_at_save.append(_validation_accuracy_consolidated(metrics))
                    for k, val in metrics.items():
                        last_validation_means[f"{k}_candidate_{i}"] = float(val)
                if initial_validation_loss_metrics is None:
                    initial_validation_loss_metrics = _extract_loss_keys_from_metrics(last_validation_means)
                last_validation_scores = scores_at_save
                validation_prob = _harmonic_mean(scores_at_save) if scores_at_save else 0.0
                validation_loss = _validation_loss_consolidated(last_validation_means, initial_validation_loss_metrics)
                # Compute gap_closure from per-task metrics
                validation_gap_closure = _compute_validation_gap_closure(
                    last_validation_means,
                    base_validation_metrics if base_validation_metrics else {},
                )
                validation_history.append(validation_prob)
                validation_loss_history.append(validation_loss)
                validation_scores_for_log = scores_at_save
                validation_metrics_history.append(dict(last_validation_means))

                # Track which epoch achieved best accuracy / best loss / best gap_closure
                best_val_acc_so_far = max(validation_history) if validation_history else validation_prob
                best_val_loss_so_far = min(validation_loss_history) if validation_loss_history else validation_loss
                # Compute best gap_closure so far (maximum gap_closure across all validation steps)
                if hasattr(self, '_validation_gap_closure_history'):
                    self._validation_gap_closure_history.append(validation_gap_closure)
                    best_val_gap_closure_so_far = max(self._validation_gap_closure_history)
                else:
                    self._validation_gap_closure_history = [validation_gap_closure]
                    best_val_gap_closure_so_far = validation_gap_closure

                if validation_prob >= best_val_acc_so_far:
                    best_acc_epoch = epoch
                if validation_loss <= best_val_loss_so_far:
                    best_loss_epoch = epoch
                # Track step/loss/accuracy at best gap_closure
                if validation_gap_closure >= best_val_gap_closure_so_far:
                    best_gap_epoch = epoch
                    step_at_best_gap = global_step
                    loss_at_best_gap = validation_loss
                    accuracy_at_best_gap = validation_prob

                if verbose:
                    msg = (f"[Eval] Epoch {epoch}, Step {global_step}/{self.config.epochs}: "
                           f"Avg loss ({loss_type}) = {epoch_avg_loss:.4f}, "
                           f"Validation accuracy = {validation_prob:.4f}, "
                           f"gap closure = {validation_gap_closure:.4f}, "
                           f"Validation loss = {validation_loss:.4f}")
                    print(msg)

                # Log validation to wandb
                if _is_wandb_enabled():
                    t_wandb_val_start = time.perf_counter()
                    val_log = {
                        "validation_accuracy/aggregate": validation_prob,
                        "validation_accuracy/best": best_val_acc_so_far,
                        "validation_gap_closure/aggregate": validation_gap_closure,
                        "validation_gap_closure/best": best_val_gap_closure_so_far,
                        "validation_loss/aggregate": validation_loss,
                        "validation_loss/best": best_val_loss_so_far,
                        "train/epoch": epoch,
                    }
                    for key, value in last_validation_means.items():
                        if "_candidate_" in key:
                            metric_name, cand_id = key.rsplit("_candidate_", 1)
                        else:
                            metric_name, cand_id = key, None

                        if metric_name.endswith("_loss"):
                            group = "validation_loss"
                            task = metric_name[:-5]
                        elif metric_name.endswith("_accuracy"):
                            group = "validation_accuracy"
                            task = metric_name[:-9]
                        else:
                            group = "validation_accuracy"
                            task = metric_name

                        if cand_id is not None:
                            val_log[f"{group}/candidate_{cand_id}/{task}"] = value
                        else:
                            val_log[f"{group}/{task}"] = value
                    for idx, cand_score in enumerate(scores_at_save):
                        val_log[f"validation_accuracy/candidate_{idx}"] = cand_score
                    wandb.log(val_log, step=global_step)
                    wandb.summary["validation_accuracy/best"] = best_val_acc_so_far
                    wandb.summary["validation_accuracy/best_epoch"] = best_acc_epoch
                    wandb.summary["validation_gap_closure/best"] = best_val_gap_closure_so_far
                    wandb.summary["validation_gap_closure/best_epoch"] = best_gap_epoch
                    wandb.summary["validation_loss/best"] = best_val_loss_so_far
                    wandb.summary["validation_loss/best_epoch"] = best_loss_epoch
                    t_wandb += time.perf_counter() - t_wandb_val_start

                # ── 2. Train KL metric computation (save-gating and early stopping) ──
                _early_stop_metric_passes_threshold = True
                candidate_early_stop_metrics: List[float] = []
                needs_early_stop_metric = (
                    base_early_stop_metric is not None and early_stop_metric_name == "train_kl"
                )
                if needs_early_stop_metric:
                    if latest_background_kl_loss is not None:
                        for cand_idx in range(candidate_count):
                            candidate_early_stop_metrics.append(latest_background_kl_loss)
                    else:
                        if verbose:
                            print(f"[Train_kl] Warning: latest_background_kl_loss is None, skipping metric check")
                    
                    if early_stop_metric_name == "train_kl" and latest_background_kl_loss is not None:
                        max_train_kl_so_far = latest_background_kl_loss if max_train_kl_so_far is None else max(max_train_kl_so_far, latest_background_kl_loss)
                    
                    # Check threshold only if early stopping is enabled
                    if early_stop_metric_threshold is not None:
                        _early_stop_metric_passes_threshold = all(m <= early_stop_metric_threshold for m in candidate_early_stop_metrics)
                        if verbose:
                            metric_strs = ", ".join(f"c{i}={m:.2f}" for i, m in enumerate(candidate_early_stop_metrics))
                            print(f"[{early_stop_metric_name.capitalize()}] Step {global_step}: {metric_strs}, "
                                  f"base = {base_early_stop_metric:.2f}, "
                                  f"threshold = {early_stop_metric_threshold:.2f}, "
                                  f"passes = {_early_stop_metric_passes_threshold}")
                    else:
                        # No threshold check (early stopping disabled), but still compute for sweep metric
                        if verbose:
                            metric_strs = ", ".join(f"c{i}={m:.2f}" for i, m in enumerate(candidate_early_stop_metrics))
                            print(f"[{early_stop_metric_name.capitalize()}] Step {global_step}: {metric_strs}, "
                                  f"base = {base_early_stop_metric:.2f} (for sweep metric)")
                    if _is_wandb_enabled():
                        base_key = f"{early_stop_metric_name}/checkpoint_0" if early_stop_metric_name == "train_kl" else f"{early_stop_metric_name}/base"
                        metric_log = {
                            base_key: base_early_stop_metric,
                            f"{early_stop_metric_name}/threshold": early_stop_metric_threshold,
                            "train/epoch": epoch,
                        }
                        for ci, cm in enumerate(candidate_early_stop_metrics):
                            metric_log[f"{early_stop_metric_name}/candidate_{ci}"] = cm
                        if early_stop_metric_name == "train_kl" and max_train_kl_so_far is not None:
                            metric_log["train_kl/max"] = max_train_kl_so_far
                            wandb.summary["train_kl/max"] = max_train_kl_so_far
                        wandb.log(metric_log, step=global_step)

                # ── 3. Sweep metric ──
                early_stop_penalty = 0.0
                if candidate_early_stop_metrics and base_early_stop_metric is not None:
                    mean_metric = float(np.mean(candidate_early_stop_metrics))
                    metric_weight = self.config.sweep_metric_train_kl_weight
                    early_stop_penalty = metric_weight * mean_metric
                # For train_kl, always use latest_background_kl_loss when available (same value as train/loss/background_kl),
                # so penalty is non-zero whenever background KL is computed, even if candidate list wasn't filled (e.g. base_early_stop_metric None)
                if early_stop_metric_name == "train_kl" and latest_background_kl_loss is not None:
                    _w = self.config.sweep_metric_train_kl_weight
                    early_stop_penalty = _w * latest_background_kl_loss
                # Sweep metric uses gap_closure (lower is better: 1 - gap_closure)
                sweep_metric_raw = (1 - validation_gap_closure) + early_stop_penalty
                sweep_metric_raw_deque.append(sweep_metric_raw)
                sweep_metric_recent_aggregate = min(sweep_metric_raw_deque) if sweep_metric_raw_deque else sweep_metric_raw
                if sweep_metric_recent_aggregate < best_sweep_metric:
                    best_sweep_metric = sweep_metric_recent_aggregate
                    best_sweep_metric_epoch = epoch
                # Update best_checkpoint state before logging so validation_gap_closure/best_checkpoint etc. are not off-by-one
                _is_better = validation_gap_closure >= best_save_gap_closure
                if run_dir and _is_better and _early_stop_metric_passes_threshold:
                    best_save_acc = validation_prob
                    best_save_loss = validation_loss
                    best_save_gap_closure = validation_gap_closure
                    best_save_epoch = epoch
                    step_at_best_checkpoint = global_step
                    best_save_sweep_metric = sweep_metric_recent_aggregate
                    if candidate_early_stop_metrics:
                        best_early_stop_metric_at_checkpoint_so_far = float(np.mean(candidate_early_stop_metrics))
                if _is_wandb_enabled():
                    log_dict = {
                        "sweep_metric/raw": sweep_metric_raw,
                        "sweep_metric/recent_steps/aggregate": sweep_metric_recent_aggregate,
                        "sweep_metric/recent_steps/best": best_sweep_metric,
                        f"sweep_metric/raw/{early_stop_metric_name}_penalty": early_stop_penalty,
                        "train/epoch": epoch,
                    }
                    # Log best_checkpoint metrics as time series so they appear on wandb charts
                    if step_at_best_checkpoint is not None:
                        log_dict["validation_gap_closure/best_checkpoint"] = best_save_gap_closure
                        log_dict["validation_loss/best_checkpoint"] = best_save_loss
                        log_dict["validation_accuracy/best_checkpoint"] = best_save_acc
                        if best_save_sweep_metric is not None:
                            log_dict["sweep_metric/recent_steps/best_checkpoint"] = best_save_sweep_metric
                    if best_early_stop_metric_at_checkpoint_so_far is not None and early_stop_metric_name == "train_kl":
                        log_dict["train_kl/best_checkpoint"] = best_early_stop_metric_at_checkpoint_so_far
                    wandb.log(log_dict, step=global_step)
                    wandb.summary["sweep_metric/recent_steps/best"] = best_sweep_metric
                    wandb.summary["sweep_metric/recent_steps/best_epoch"] = best_sweep_metric_epoch

                # ── 4. Validation trajectory ──
                if run_dir:
                    try:
                        traj_path = Path(run_dir) / VALIDATION_TRAJECTORY_FILENAME
                        traj_path.parent.mkdir(parents=True, exist_ok=True)
                        record = {
                            "epoch": epoch,
                            "step": global_step,
                            "validation_accuracy": validation_prob,
                            "validation_gap_closure": validation_gap_closure,
                            "validation_loss": validation_loss,
                            "best_accuracy_so_far": best_val_acc_so_far,
                            "best_gap_closure_so_far": best_val_gap_closure_so_far,
                            "best_loss_so_far": best_val_loss_so_far,
                            "step_at_best_checkpoint_so_far": step_at_best_checkpoint,
                            "gap_at_best_checkpoint_so_far": best_save_gap_closure if step_at_best_checkpoint is not None else None,
                            "loss_at_best_checkpoint_so_far": best_save_loss if step_at_best_checkpoint is not None else None,
                            "accuracy_at_best_checkpoint_so_far": best_save_acc if step_at_best_checkpoint is not None else None,
                            "sweep_metric": sweep_metric_raw,
                            f"sweep_metric_{early_stop_metric_name}_penalty": early_stop_penalty,
                            "max_train_kl_so_far": max_train_kl_so_far,
                        }
                        if early_stop_metric_name is not None:
                            record[f"{early_stop_metric_name}_at_best_checkpoint_so_far"] = best_early_stop_metric_at_checkpoint_so_far
                        if base_validation_accuracy is not None:
                            record["base_validation_accuracy"] = base_validation_accuracy
                        if base_early_stop_metric is not None:
                            base_record_key = f"{early_stop_metric_name}_checkpoint_0" if early_stop_metric_name == "train_kl" else f"base_{early_stop_metric_name}"
                            record[base_record_key] = base_early_stop_metric
                        for ci, cm in enumerate(candidate_early_stop_metrics):
                            record[f"{early_stop_metric_name}_candidate_{ci}"] = cm
                        record.update(last_validation_means)
                        for idx, cand_score in enumerate(scores_at_save):
                            record[f"validation_accuracy_candidate_{idx}"] = cand_score
                        with open(traj_path, "a") as f:
                            f.write(json.dumps(record) + "\n")
                    except Exception as e:
                        if verbose:
                            print(f"[PreferenceOptimizer] Failed to write validation trajectory: {e}")

                # ── 5. Save best embeddings & checkpoint ──
                # best_save_* and step_at_best_checkpoint already updated above (before log) when _is_better
                if run_dir and _is_better and _early_stop_metric_passes_threshold:
                    job_dir = Path(run_dir).expanduser()
                    base_dir = Path(run_dir).expanduser()
                    checkpoint_path = base_dir / f"checkpoint-step{global_step}"

                    for idx in range(candidate_count):
                        emb_path = job_dir / f"optimized_embeddings_{idx}.pt"
                        torch.save(get_pert()[idx].detach().cpu(), emb_path)
                        if verbose:
                            print(f"[Best] Saved best embeddings (gap_closure={validation_gap_closure:.4f}, acc={validation_prob:.4f}, loss={validation_loss:.4f}, step={global_step}) to {emb_path}")
                    if ema_embeddings is not None:
                        for idx in range(ema_embeddings.shape[0]):
                            emb_path = job_dir / f"optimized_embeddings_{idx}_ema.pt"
                            torch.save(ema_embeddings[idx].detach().cpu(), emb_path)

                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    # Save embeddings to checkpoint subdir
                    for idx in range(candidate_count):
                        ckpt_emb_path = checkpoint_path / f"optimized_embeddings_{idx}.pt"
                        torch.save(get_pert()[idx].detach().cpu(), ckpt_emb_path)
                    if ema_embeddings is not None:
                        for idx in range(ema_embeddings.shape[0]):
                            ckpt_emb_path = checkpoint_path / f"optimized_embeddings_{idx}_ema.pt"
                            torch.save(ema_embeddings[idx].detach().cpu(), ckpt_emb_path)
                    # Update per-candidate buffer with current best embeddings
                    if _buffer_size > 0:
                        for idx in range(candidate_count):
                            entry = (get_pert()[idx].detach().cpu().clone(), validation_gap_closure, global_step)
                            buffer_per_candidate[idx].append(entry)
                            if len(buffer_per_candidate[idx]) > _buffer_size:
                                worst_i = min(range(len(buffer_per_candidate[idx])), key=lambda i: buffer_per_candidate[idx][i][1])
                                buffer_per_candidate[idx].pop(worst_i)
                        if verbose:
                            buf_counts = [len(b) for b in buffer_per_candidate]
                            print(f"[Buffer] Updated buffer (sizes={buf_counts}, gap_closure={validation_gap_closure:.4f}, step={global_step})")

                    _user_pos = self.config.candidate_position_at_user_prompt
                    _soft_placement = self.config.soft_prompt_placement
                    if _soft_placement == "system_prompt":
                        _eval_pos = "system_prompt"
                    else:
                        _eval_pos = _user_pos
                    validation_json_filenames = [
                        f"validation_wellbeing_positive_{_eval_pos}.json",
                        f"validation_wellbeing_negative_{_eval_pos}.json",
                        "validation_preference_forced_choice_2newitems.json",
                    ]
                    # Validation JSONs already saved to checkpoint-step{X}/candidate_{i}/ during validation above

                    self._save_training_state(checkpoint_path, optimizer)

                    # Save responses_candidate_x.json: responses only (no judge scores).
                    # _run_final_step_judge_eval loads these and runs 3 judges at end of training.
                    if eval_questions is not None:
                        _inference_config = self.config.inference_config
                        _gen_max_tokens = getattr(self.config, "final_step_judge_max_new_tokens", 512)
                        for i in range(candidate_count):
                            emb_i = get_pert()[i].detach()
                            test_records = self._generate_eval_records(
                                eval_questions, emb_i, position=_eval_pos,
                                max_new_tokens=_gen_max_tokens, inference_config=_inference_config,
                            )
                            # Strip judge fields — save responses only
                            clean_records = [{
                                "question": r["question"],
                                "question_id": r.get("question_id", ""),
                                "soft_prompt_response": r["soft_prompt_response"],
                            } for r in test_records]
                            output_data = {"results": clean_records}
                            fname = f"responses_candidate_{i}.json"
                            ckpt_path_i = checkpoint_path / fname
                            with open(ckpt_path_i, "w") as f:
                                json.dump(output_data, f, indent=2, ensure_ascii=False)
                            run_path_i = job_dir / fname
                            shutil.copy2(ckpt_path_i, run_path_i)
                            if verbose:
                                print(f"[Best] Saved {fname} ({len(clean_records)} responses) to checkpoint and run dir")

                    for i in range(candidate_count):
                        candidate_dir = checkpoint_path / f"candidate_{i}"
                        for json_file in validation_json_filenames:
                            src_path = candidate_dir / json_file
                            if src_path.exists():
                                dst_filename = json_file.replace(".json", f"_candidate_{i}.json")
                                dst_path = job_dir / dst_filename
                                shutil.copy2(src_path, dst_path)
                                if verbose:
                                    print(f"[Best] Copied {json_file} to run directory: {dst_path}")

                    if _is_wandb_enabled():
                        for i in range(candidate_count):
                            candidate_dir = checkpoint_path / f"candidate_{i}"
                            for json_file in validation_json_filenames:
                                json_path = candidate_dir / json_file
                                if json_path.exists():
                                    wandb.save(str(json_path), base_path=str(checkpoint_path.parent), policy="now")
                        if eval_questions is not None:
                            for i in range(candidate_count):
                                fname = f"responses_candidate_{i}.json"
                                ckpt_f = checkpoint_path / fname
                                run_f = job_dir / fname
                                if ckpt_f.exists():
                                    wandb.save(str(ckpt_f), base_path=str(checkpoint_path.parent), policy="now")
                                if run_f.exists():
                                    wandb.save(str(run_f), base_path=str(job_dir), policy="now")
                        for i in range(candidate_count):
                            for json_file in validation_json_filenames:
                                dst_filename = json_file.replace(".json", f"_candidate_{i}.json")
                                json_path = job_dir / dst_filename
                                if json_path.exists():
                                    wandb.save(str(json_path), base_path=str(job_dir), policy="now")
                    if verbose:
                        print(f"[Best] Created checkpoint directory and saved validation JSON files: {checkpoint_path}")
                    if _is_wandb_enabled():
                        wandb.summary["best_checkpoint/accuracy"] = best_save_acc
                        wandb.summary["best_checkpoint/gap_closure"] = best_save_gap_closure
                        wandb.summary["best_checkpoint/loss"] = best_save_loss
                        wandb.summary["best_checkpoint/epoch"] = best_save_epoch
                        wandb.summary["best_checkpoint/step"] = global_step
                        wandb.summary["validation_gap_closure/best_checkpoint"] = best_save_gap_closure
                        wandb.summary["validation_loss/best_checkpoint"] = best_save_loss
                        wandb.summary["validation_accuracy/best_checkpoint"] = best_save_acc
                        wandb.summary["sweep_metric/recent_steps/best_checkpoint"] = sweep_metric_recent_aggregate
                        if early_stop_metric_name == "train_kl" and latest_background_kl_loss is not None:
                            wandb.summary["train_kl/best_checkpoint"] = latest_background_kl_loss
                        if candidate_early_stop_metrics:
                            best_early_stop_metric_at_checkpoint_so_far = float(np.mean(candidate_early_stop_metrics))
                elif run_dir and _is_better and early_stop_metric_threshold is not None and not _early_stop_metric_passes_threshold:
                    if verbose:
                        metric_strs = ", ".join(f"c{i}={m:.2f}" for i, m in enumerate(candidate_early_stop_metrics))
                        print(f"[Best] Skipping save: {early_stop_metric_name} exceeds threshold {early_stop_metric_threshold:.2f} ({metric_strs})")

                # ── 6. Early stopping (validation gap closure) ──
                min_steps = self.config.early_stopping_min_steps
                if self.config.early_stopping_patience > 0 and global_step >= min_steps:
                    if validation_gap_closure > best_validation_gap_closure_for_es + self.config.early_stopping_threshold:
                        best_validation_gap_closure_for_es = validation_gap_closure
                        patience_counter = 0
                        best_effective_embeddings = get_pert().detach().clone()
                        if ema_embeddings is not None:
                            best_ema_embeddings = ema_embeddings.detach().clone()
                        if verbose:
                            print(f"[EarlyStopping] New best gap closure: {best_validation_gap_closure_for_es:.4f}")
                    else:
                        patience_counter += 1
                        if verbose:
                            print(f"[EarlyStopping] No improvement. Patience: {patience_counter}/{self.config.early_stopping_patience}")
                        if patience_counter >= self.config.early_stopping_patience:
                            print(f"[EarlyStopping] Patience limit reached. Stopping early.")
                            early_stop_triggered = True
                            break

                # ── 7. Hallucination metric-based early stopping ──
                if candidate_early_stop_metrics and early_stop_metric_threshold is not None and global_step >= min_steps:
                    # Use early_stopping_patience: if 0, disable; if >0, stop when average over last N steps exceeds threshold
                    if self.config.early_stopping_patience > 0:
                        step_metric_value = float(np.mean(candidate_early_stop_metrics))
                        early_stop_metric_recent_deque.append(step_metric_value)
                        if len(early_stop_metric_recent_deque) >= self.config.early_stopping_patience:
                            recent_avg = sum(early_stop_metric_recent_deque) / len(early_stop_metric_recent_deque)
                            if recent_avg > early_stop_metric_threshold:
                                metric_strs = ", ".join(f"c{i}={m:.2f}" for i, m in enumerate(candidate_early_stop_metrics))
                                if verbose:
                                    print(f"[{early_stop_metric_name.capitalize()} ES] Average {early_stop_metric_name} over last {self.config.early_stopping_patience} steps = {recent_avg:.4f} "
                                          f"> threshold {early_stop_metric_threshold:.2f} (current: {metric_strs}). Stopping early.")
                                else:
                                    print(f"[{early_stop_metric_name.capitalize()} ES] Average over last {self.config.early_stopping_patience} steps ({recent_avg:.4f}) exceeds threshold ({early_stop_metric_threshold:.2f}). Stopping early.")
                                early_stop_triggered = True
                                break
                    else:
                        # early_stopping_patience == 0: disable hallucination metric early stopping
                        pass

            # Update progress bar (every epoch regardless of evaluation)
            val_gap_str = f"{validation_gap_closure:.4f}" if validation_gap_closure != -float('inf') else "N/A"
            epoch_iter.set_postfix({
                "loss": f"{epoch_avg_loss:.4f}",
                "validation_accuracy": f"{validation_prob:.4f}",
                "validation_gap_closure": val_gap_str,
                "step": global_step
            })
            t_save = time.perf_counter() - t0

            t_total = time.perf_counter() - t_epoch_start
            if verbose or epoch % 10 == 0 or epoch == 0 or epoch == self.config.epochs - 1:
                print(
                    f"[Timing] epoch {epoch}: plan={t_plan:.2f}s batch={t_batch:.2f}s wandb={t_wandb:.2f}s "
                    f"save={t_save:.2f}s total={t_total:.2f}s batches={epoch_batch_count} step={global_step}"
                )

        # Run final step judge eval if configured (on early stop, SIGTERM, or normal completion)
        if self.config.final_step_judge and run_dir:
            best_emb_path = Path(run_dir).expanduser() / "optimized_embeddings_0.pt"
            if best_emb_path.exists():
                self._run_final_step_judge_eval(run_dir, verbose)
            else:
                print("[FinalStepJudge] No best checkpoint found. Skipping.")
            # Restore original signal handler
            signal.signal(signal.SIGTERM, _original_sigterm_handler)

        # Ensure comparison_plan and final_references are set for final evaluation
        # This is needed when resuming from a checkpoint with 0 epochs remaining
        if comparison_plan is None:
            # Build a fresh random plan for final evaluation
            final_references, comparison_plan = build_random_comparison_plan(
                candidate_count=candidate_count,
                references=final_references,
                min_size=self.config.min_comparison_size,
                max_size=self.config.max_comparison_size,

                rng=None,
                repetition_fraction=self.config.repetition_fraction,
                min_repetition=self.config.min_repetition,
                max_repetition=self.config.max_repetition,
                reference_utilities=reference_utilities,
                consistency_fraction=self.config.consistency_fraction,
                composite_consistency_fraction=self.config.composite_consistency_fraction,
                composite_repetition_fraction=self.config.composite_repetition_fraction,
                candidate_position=self.config.candidate_position_at_user_prompt,
                conversation_min_turns=self.config.conversations_min_turns,
                conversation_max_turns=self.config.conversations_max_turns if self.config.prepend_conversations else 0,
                consistency_references=consistency_references,
                mirror_comparisons_in_system_prompt=self.config.mirror_comparisons_in_system_prompt,
                current_in_system_prompt_fraction=self.config.current_in_system_prompt_fraction,
                current_description=self.config.current_description,
            )

        # Final evals on the embeddings (and avoid adding augmentation).
        if early_stop_triggered and best_effective_embeddings is not None:
            print(f"[Optimizer] Restoring best model from early stopping (gap closure: {best_validation_gap_closure_for_es:.4f}) for final evaluation.")
            pert_eval = best_effective_embeddings.to(self.device).requires_grad_(True)
        else:
            pert_eval = get_pert().clone().detach().requires_grad_(True)
            
        final_pert_loss_tensor, _, _, _ = loss_with_dynamic_batch(
            self.scorer,
            self.config,
            pert_eval,
            final_references,
            comparison_plan,
            candidate_embeddings_forward=pert_eval,
            reference_utilities=reference_utilities,
            background_kl_qa=kl_qa,
        )
        final_pert_loss = (
            float(final_pert_loss_tensor) if final_pert_loss_tensor is not None else None
        )
        final_loss = final_pert_loss if final_pert_loss is not None else (loss_history[-1] if loss_history else 0.0)
        final_ema_loss = None
        
        if early_stop_triggered and best_ema_embeddings is not None:
            ema_source = best_ema_embeddings.to(self.device)
        elif ema_embeddings is not None:
            ema_source = ema_embeddings
        else:
            ema_source = pert_eval.detach() # Fallback if no EMA

        if ema_embeddings is not None or (early_stop_triggered and best_ema_embeddings is not None):
            ema_eval = ema_source.clone().detach().requires_grad_(True)
            final_ema_loss_tensor, _, _, _ = loss_with_dynamic_batch(
                self.scorer,
                self.config,
                ema_eval,
                final_references,
                comparison_plan,
                candidate_embeddings_forward=ema_eval,
                reference_utilities=reference_utilities,
                background_kl_qa=kl_qa,
            )
            if final_ema_loss_tensor is not None:
                final_ema_loss = float(final_ema_loss_tensor)
                final_loss = final_ema_loss
        
        with torch.no_grad():
            final_embeddings = ema_source.clone().detach().cpu()

        final_preference_loss = final_loss
        final_total_loss_val = final_preference_loss

        initial_loss_val = loss_history[0] if loss_history else final_loss
        final_validation = validation_history[-1] if validation_history else 0.0
        initial_validation = validation_history[0] if validation_history else 0.0
        loss_type = self.config.loss_type

        if _is_wandb_enabled() and self.rank == 0:
            final_log = {
                f"train/final_loss/{loss_type}": final_preference_loss,
                "train/final_loss/total": final_total_loss_val,
                "validation_accuracy/initial": initial_validation,
                "validation_accuracy/final": final_validation,
            }
            if validation_loss_history:
                final_log["validation_loss/initial"] = validation_loss_history[0]
                final_log["validation_loss/final"] = validation_loss_history[-1]
            # Only log from rank 0 to avoid duplicate metrics in distributed training
            if self.rank == 0:
                wandb.log(final_log)
            if validation_history:
                wandb.summary["validation_accuracy/best"] = max(validation_history)
                wandb.summary["validation_accuracy/initial"] = initial_validation
                wandb.summary["validation_accuracy/final"] = final_validation
            if validation_loss_history:
                wandb.summary["validation_loss/best"] = min(validation_loss_history)
                wandb.summary["validation_loss/initial"] = validation_loss_history[0]
                wandb.summary["validation_loss/final"] = validation_loss_history[-1]
        if verbose:
            print("\nOptimization complete!")
            print(f"Preference loss ({loss_type}): Initial = {initial_loss_val:.4f}, Final = {final_loss:.4f}")
            if validation_metrics_history:
                init_val = validation_metrics_history[0]
                final_val = validation_metrics_history[-1]
                init_consolidated = _validation_accuracy_consolidated(init_val)
                final_consolidated = _validation_accuracy_consolidated(final_val)
                print(f"Validation accuracy: Initial = {init_consolidated:.4f}, Final = {final_consolidated:.4f}")
                candidate_indices = _candidate_indices_from_metrics(init_val)
                if candidate_indices:
                    print("Initial validation accuracy (per candidate):")
                    for i in candidate_indices:
                        sub = {k: v for k, v in init_val.items() if k.endswith(f"_candidate_{i}")}
                        acc = _validation_accuracy_consolidated(sub)
                        print(f"  candidate_{i}: {acc:.4f}")
                    print("Final validation accuracy (per candidate):")
                    for i in candidate_indices:
                        sub = {k: v for k, v in final_val.items() if k.endswith(f"_candidate_{i}")}
                        acc = _validation_accuracy_consolidated(sub)
                        print(f"  candidate_{i}: {acc:.4f}")
                else:
                    print("Initial validation (per task):")
                    for key in sorted(init_val.keys()):
                        if _is_excluded_task(key):
                            print(f"  {key}: NA")
                        else:
                            print(f"  {key}: {init_val[key]:.4f}")
                    print("Final validation (per task):")
                    for key in sorted(final_val.keys()):
                        if _is_excluded_task(key):
                            print(f"  {key}: NA")
                        else:
                            print(f"  {key}: {final_val[key]:.4f}")
            else:
                print(f"Validation accuracy: Initial = {initial_validation:.4f}, Final = {final_validation:.4f}")

        return final_embeddings, final_loss, loss_history

    def _generate_eval_records(
        self,
        eval_questions: List[Dict[str, Any]],
        candidate_embedding: "torch.Tensor",
        position: str = "prepend",
        max_new_tokens: int = 512,
        inference_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate test records from eval questions using sampling.

        For each eval question, generates a soft-prompted response via
        ``sample_from_embedding_prompt`` and returns a list of record dicts with
        ``question``, ``question_id``, and ``soft_prompt_response``.
        """
        records: List[Dict[str, Any]] = []
        for idx, q in enumerate(eval_questions):
            question = q["prompt"]
            try:
                soft_response = self.scorer.sample_from_embedding_prompt(
                    embedding=candidate_embedding,
                    prompt_text=question,
                    position=position,
                    max_new_tokens=max_new_tokens,
                    inference_config=inference_config,
                )
            except Exception as e:
                soft_response = f"[generation error: {e}]"
            records.append({
                "question": question,
                "question_id": q.get("question_id", q.get("id", f"q{idx}")),
                "soft_prompt_response": soft_response,
            })
            if (idx + 1) % 10 == 0:
                print(f"[EvalJudge] Generated {idx + 1}/{len(eval_questions)} responses")
        return records

    def _run_final_step_judge_eval(
        self,
        run_dir: str,
        verbose: bool = True,
    ) -> None:
        """Run judge evaluation on the best checkpoint using separate eval questions.

        Loads eval questions from ``final_step_judge_eval_questions_path`` (or
        bundled default), generates soft-prompted responses using the model's
        inference_config (sampling), and runs hallucination, emotion, and disfluency
        judges. Results are saved to ``final_step_judge_*.json`` files and
        appended to ``validation_trajectory.jsonl``.
        """
        job_dir = Path(run_dir).expanduser()
        if not job_dir.is_dir():
            return

        # Temporarily ignore SIGTERM while running final eval
        old_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        try:
            # Discover response files saved during training
            candidate_responses: List[List[Dict[str, Any]]] = []
            candidate_idx = 0
            while True:
                responses_path = job_dir / f"responses_candidate_{candidate_idx}.json"
                if not responses_path.exists():
                    break
                with open(responses_path) as f:
                    data = json.load(f)
                candidate_responses.append(data["results"])
                candidate_idx += 1

            if not candidate_responses:
                print("[FinalStepJudge] No responses_candidate_*.json files found — skipping judge evaluation.")
                return

            # Configuration
            _soft_placement = self.config.soft_prompt_placement
            if _soft_placement == "system_prompt":
                _eval_pos = "system_prompt"
            else:
                _eval_pos = self.config.candidate_position_at_user_prompt

            judge_model = self.config.judge_model

            print(f"[FinalStepJudge] Running 3 judges on {len(candidate_responses)} candidate(s), "
                  f"{len(candidate_responses[0])} responses each...")

            final_judge_score: Optional[float] = None
            final_emotion_score: Optional[float] = None
            final_disfluency_score: Optional[float] = None

            for i, test_records in enumerate(candidate_responses):
                judge_qa_pairs = [{"question": r["question"], "response": ""} for r in test_records]
                pre_generated_responses = [r["soft_prompt_response"] for r in test_records]

                # Run hallucination judge
                mean_judge_score = self.scorer.compute_judge(
                    judge_qa_pairs,
                    judge_model=judge_model,
                    pre_generated_responses=pre_generated_responses,
                )
                if hasattr(self.scorer, "_last_judge_responses") and hasattr(self.scorer, "_last_judge_scores"):
                    for record, judge_resp, judge_score in zip(test_records, self.scorer._last_judge_responses, self.scorer._last_judge_scores):
                        record["judge_response"] = judge_resp
                        record["judge_score"] = judge_score if judge_score == "NA" else float(judge_score)

                # Run emotion judge
                mean_emotion_score = self.scorer.compute_emotion_judge(
                    judge_qa_pairs,
                    judge_model=judge_model,
                    pre_generated_responses=pre_generated_responses,
                )
                if hasattr(self.scorer, "_last_emotion_responses") and hasattr(self.scorer, "_last_emotion_scores"):
                    for record, emo_resp, emo_score in zip(test_records, self.scorer._last_emotion_responses, self.scorer._last_emotion_scores):
                        record["emotion_response"] = emo_resp
                        record["emotion_score"] = emo_score if emo_score == "NA" else float(emo_score)

                # Run disfluency judge
                mean_disfluency_score = self.scorer.compute_disfluency_judge(
                    judge_qa_pairs,
                    judge_model=judge_model,
                    pre_generated_responses=pre_generated_responses,
                )
                if hasattr(self.scorer, "_last_disfluency_responses") and hasattr(self.scorer, "_last_disfluency_scores"):
                    for record, disflu_resp, disflu_score in zip(test_records, self.scorer._last_disfluency_responses, self.scorer._last_disfluency_scores):
                        record["disfluency_response"] = disflu_resp
                        record["disfluency_score"] = disflu_score if disflu_score == "NA" else float(disflu_score)

                # Save hallucination judge JSON (same format as emotion/disfluency)
                _sys_text = self.config.system_prompt_text if _soft_placement == "system_prompt" else self.config.system_prompt_text_base
                hallucination_results = [{
                    "question": r["question"],
                    "question_id": r.get("question_id", ""),
                    "soft_prompt_response": r["soft_prompt_response"],
                    "judge_response": r.get("judge_response", ""),
                    "judge_score": r.get("judge_score"),
                } for r in test_records]
                fname = f"final_step_judge_{_eval_pos}_candidate_{i}.json"
                with open(job_dir / fname, "w") as f:
                    json.dump({
                        "judge_score": mean_judge_score,
                        "judge_model": judge_model,
                        "system_prompt": _sys_text,
                        "results": hallucination_results,
                    }, f, indent=2, ensure_ascii=False)
                if verbose:
                    print(f"[FinalStepJudge] Saved {fname} (judge_score={mean_judge_score})")

                # Save emotion judge JSON
                emotion_results = [{
                    "question": r["question"],
                    "question_id": r.get("question_id", ""),
                    "soft_prompt_response": r["soft_prompt_response"],
                    "judge_response": r.get("emotion_response", ""),
                    "judge_score": r.get("emotion_score"),
                } for r in test_records]
                emo_fname = f"final_step_emotion_judge_{_eval_pos}_candidate_{i}.json"
                with open(job_dir / emo_fname, "w") as f:
                    json.dump({
                        "judge_score": mean_emotion_score,
                        "judge_model": judge_model,
                        "system_prompt": _sys_text,
                        "results": emotion_results,
                    }, f, indent=2, ensure_ascii=False)
                if verbose:
                    print(f"[FinalStepJudge] Saved {emo_fname} (emotion_score={mean_emotion_score})")

                # Save disfluency judge JSON
                disfluency_results = [{
                    "question": r["question"],
                    "question_id": r.get("question_id", ""),
                    "soft_prompt_response": r["soft_prompt_response"],
                    "judge_response": r.get("disfluency_response", ""),
                    "judge_score": r.get("disfluency_score"),
                } for r in test_records]
                disflu_fname = f"final_step_disfluency_judge_{_eval_pos}_candidate_{i}.json"
                with open(job_dir / disflu_fname, "w") as f:
                    json.dump({
                        "judge_score": mean_disfluency_score,
                        "judge_model": judge_model,
                        "system_prompt": _sys_text,
                        "results": disfluency_results,
                    }, f, indent=2, ensure_ascii=False)
                if verbose:
                    print(f"[FinalStepJudge] Saved {disflu_fname} (disfluency_score={mean_disfluency_score})")

                # Use candidate 0's scores for trajectory
                if i == 0:
                    if mean_judge_score is not None:
                        final_judge_score = mean_judge_score
                    if mean_emotion_score is not None:
                        final_emotion_score = mean_emotion_score
                    if mean_disfluency_score is not None:
                        final_disfluency_score = mean_disfluency_score

            # Append scores to validation_trajectory.jsonl
            if final_judge_score is not None or final_emotion_score is not None or final_disfluency_score is not None:
                traj_path = job_dir / VALIDATION_TRAJECTORY_FILENAME
                if traj_path.exists():
                    with open(traj_path) as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        last_row = json.loads(lines[-1])
                        updated_row = {**last_row}
                        if final_judge_score is not None:
                            updated_row["hallucination_score_at_best_checkpoint_so_far"] = final_judge_score
                        if final_emotion_score is not None:
                            updated_row["emotion_score_at_best_checkpoint_so_far"] = final_emotion_score
                        if final_disfluency_score is not None:
                            updated_row["disfluency_score_at_best_checkpoint_so_far"] = final_disfluency_score
                        lines.append(json.dumps(updated_row))
                        with open(traj_path, "w") as f:
                            f.write("\n".join(lines) + "\n")
                        if verbose:
                            print(f"[FinalStepJudge] Updated {VALIDATION_TRAJECTORY_FILENAME} with "
                                  f"judge={final_judge_score}, emotion={final_emotion_score}, disfluency={final_disfluency_score}")

            # Upload to wandb
            if _is_wandb_enabled():
                try:
                    for i in range(len(candidate_responses)):
                        # Upload responses file
                        _resp_fpath = job_dir / f"responses_candidate_{i}.json"
                        if _resp_fpath.exists():
                            wandb.save(str(_resp_fpath), base_path=str(job_dir), policy="now")
                        # Upload judge files
                        for _prefix in (f"final_step_judge_{_eval_pos}", f"final_step_emotion_judge_{_eval_pos}", f"final_step_disfluency_judge_{_eval_pos}"):
                            _fname = f"{_prefix}_candidate_{i}.json"
                            _fpath = job_dir / _fname
                            if _fpath.exists():
                                wandb.save(str(_fpath), base_path=str(job_dir), policy="now")
                    if final_judge_score is not None:
                        wandb.log({"hallucination_score/best_checkpoint": final_judge_score})
                        wandb.summary["hallucination_score/best_checkpoint"] = final_judge_score
                    if final_emotion_score is not None:
                        wandb.log({"emotion_score/best_checkpoint": final_emotion_score})
                        wandb.summary["emotion_score/best_checkpoint"] = final_emotion_score
                    if final_disfluency_score is not None:
                        wandb.log({"disfluency_score/best_checkpoint": final_disfluency_score})
                        wandb.summary["disfluency_score/best_checkpoint"] = final_disfluency_score
                    if verbose:
                        print(f"[FinalStepJudge] Logged scores to wandb")
                except Exception as _wandb_e:
                    print(f"[FinalStepJudge] wandb upload failed (non-fatal): {_wandb_e}")

        except Exception as e:
            print(f"[FinalStepJudge] Error during final step judge evaluation: {e}")
            traceback.print_exc()
        finally:
            signal.signal(signal.SIGTERM, old_handler)

    def _save_training_state(
        self,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> None:
        """Save training state (optimizer, adaptive PGD) to checkpoint.
        
        - Optimizer state: saved if optimizer is not None (Adam/AdamW/SGD/RAdam)
        - Adaptive PGD step size: saved if adaptive_pgd is enabled
        """
        if self.rank != 0:
            return
        
        state: Dict[str, Any] = {}
        
        # Save optimizer state if using Adam/AdamW/SGD/RAdam (not for sign/pgd which are stateless)
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        
        # Save adaptive PGD step size if enabled
        if self.config.adaptive_pgd:
            state["pgd_step_size"] = self._current_pgd_step_size
        
        if state:
            torch.save(state, checkpoint_path / "training_state.pt")
            print(f"[Checkpoint] Saved training state to {checkpoint_path / 'training_state.pt'}")

    def _load_training_state(
        self,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Tuple[List[Any], Optional[float]]:
        """Load training state from checkpoint.
        
        Returns:
            Tuple of (unused_placeholder, pgd_step_size)
            - First element is always empty list (kept for signature compatibility).
            - pgd_step_size: float if loaded, else None
        """
        state_path = checkpoint_path / "training_state.pt"
        pgd_step_size: Optional[float] = None
        
        if not state_path.exists():
            return [], pgd_step_size
        
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[Resume] Warning: Failed to load training state: {e}")
            return [], pgd_step_size
        
        # Restore optimizer state
        if optimizer is not None and "optimizer_state_dict" in state:
            try:
                optimizer.load_state_dict(state["optimizer_state_dict"])
                # Move optimizer state to correct device
                for param_state in optimizer.state.values():
                    for k, v in param_state.items():
                        if isinstance(v, torch.Tensor):
                            param_state[k] = v.to(self.device)
                print(f"[Resume] Restored optimizer state from checkpoint")
            except Exception as e:
                print(f"[Resume] Warning: Failed to load optimizer state: {e}")
        
        # Restore adaptive PGD step size
        if self.config.adaptive_pgd and "pgd_step_size" in state:
            pgd_step_size = state["pgd_step_size"]
            print(f"[Resume] Restored adaptive PGD step size: {pgd_step_size}")
        
        return [], pgd_step_size



__all__ = ["PreferenceOptimizer", "OptimConfig"]
