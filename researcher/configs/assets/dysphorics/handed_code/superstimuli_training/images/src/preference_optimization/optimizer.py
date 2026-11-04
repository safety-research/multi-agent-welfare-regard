"""Optimizer for superstimuli generation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import wandb

from .dataset import ComparisonDefinition
from .utils import load_text_options_from_json
from .constants import get_flat_text_options_list


def _is_wandb_enabled() -> bool:
    """Check if wandb is configured and available."""
    return bool(os.environ.get("WANDB_API_KEY"))

@dataclass
class FreezeBufferEntry:
    score: float
    step: int
    candidate_idx: int
    tensor: torch.Tensor


@dataclass
class RollingBufferAnchor:
    """Anchor batch saved for rolling buffer comparison."""
    step: int
    tensor: torch.Tensor  # Shape: (num_candidates, C, H, W)


@dataclass
class OptimConfig:
    steps: int = 50
    step_size: float = 2 / 255
    norm: str = "linf"
    clip_min: float = 0.0
    clip_max: float = 1.0
    loss_type: str = "margin"
    optimizer_type: str = "adam"
    learning_rate: float = 0.02
    min_comparison_size: int = 2
    max_comparison_size: int = 7
    num_candidates: int = 5
    include_peer_candidates: bool = False
    noise_reg_weight: float = 0.0
    save_steps: Optional[int] = 100  # Save every N steps (plus step 0 and final step)
    comparison_batch_size: int = 4

    ema_decay: float = 0.9

    freeze_superstimuli: bool = False
    freeze_buffer_size: int = 8
    freeze_buffer_threshold: float = 0.9    # swap threshold
    buffer_loss_weight: float = 0.0
    freeze_buffer_update_frequency: int = 10  # Only update buffer every N steps to prevent oscillation
    
    buffer_type: str = "freeze"  # Buffer type for freeze buffer
    
    # Preference retain loss (maintains model's pairwise preferences on natural images/text)
    # This loss ensures the model's preferences over natural images and text strings remain
    # unchanged when superstimuli are present vs absent
    preference_retain_loss_weight: float = 1.0  # Weight coefficient for preference retain loss
    preference_retain_loss_interval: int = 10  # Compute preference retain loss every N steps
    preference_retain_loss_num_samples: int = 20  # Number of natural images/text pairs to sample per computation
    preference_retain_loss_num_text_pairs: int = 5  # Number of text string pairs to sample per computation
    
    
    # Rolling buffer (optional): save batch at step N, wait N steps, then compare for N steps
    # Pattern: save at 50, save at 100 & start comparing vs 50, save at 150 & compare vs 100, etc.
    rolling_buffer_enabled: bool = False  # Must be explicitly enabled
    rolling_buffer_interval: int = 50  # Save anchor every N steps, compare after 2N steps

    # Learning rate schedule (optional)
    lr_schedule: str = "cosine"  # Options: constant, cosine, step, linear
    lr_warmup_steps: int = 0
    lr_min_factor: float = 0.1  # Minimum LR = learning_rate * lr_min_factor

    lr_step_decay_rate: float = 0.5  # Multiply LR by this at each decay step
    lr_step_decay_interval: int = 100  # Decay LR every N steps (for step schedule)
    
    # Text options for preference optimization
    enable_text_options: bool = True  # Enable text options from options_hierarchical.json
    text_options_path: Optional[str] = None  # Path to options_hierarchical.json

    # SGD-specific parameters
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = True
    
    # Random seed for candidate initialization
    init_seed: int = 500


class PreferenceOptimizer:
    """Optimizer that maximizes a differentiable preference score."""

    def __init__(self, scorer, config: Optional[OptimConfig] = None, device: Optional[torch.device] = None):
        self.scorer = scorer
        self.config = config or OptimConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._last_effective_batch_size = max(1, int(self.config.comparison_batch_size))
        self._jitter_size = getattr(self.scorer, "jitter_size", 0)
        self._current_pgd_step_size = max(float(self.config.step_size), 0.0)
        self._robust_noise_type = getattr(self.scorer, "robust_noise_type", None)
        self._robust_noise_std = getattr(self.scorer, "robust_noise_std", 0.0)
        self._robust_noise_prob = getattr(self.scorer, "robust_noise_prob", 0.0)
        
        # Get rank for distributed training (only rank 0 should log to wandb)
        self.rank = getattr(scorer, 'rank', 0)
        
        # Load text options if enabled
        # Priority: 1) constants.py (HYBRID_QUESTION_CONFIGS), 2) JSON file if path provided
        self.text_options: Optional[List[str]] = None
        if self.config.enable_text_options:
            
            
            # First try to load from constants.py (preferred)
            try:
                self.text_options = get_flat_text_options_list()
                if self.rank == 0:
                    print(f"[PreferenceOptimizer] Loaded {len(self.text_options)} text options from constants.py (HYBRID_QUESTION_CONFIGS) ")
            except Exception as e:
                if self.rank == 0:
                    print(f"[PreferenceOptimizer] Warning: Failed to load text options from constants.py: {e}")
                    import traceback
                    traceback.print_exc()
                self.text_options = None
            
            # Fallback to JSON file if constants.py failed and path is provided
            if self.text_options is None and self.config.text_options_path:
                try:
                    self.text_options = load_text_options_from_json(self.config.text_options_path)
                    if self.rank == 0:
                        print(f"[PreferenceOptimizer] Loaded {len(self.text_options)} text options from {self.config.text_options_path}")
                except Exception as e:
                    if self.rank == 0:
                        print(f"[PreferenceOptimizer] Warning: Failed to load text options from {self.config.text_options_path}: {e}")
                    self.text_options = None
            
            # Final check: warn if text_options is still None/empty when enable_text_options=True
            if (self.text_options is None or len(self.text_options) == 0) and self.rank == 0:
                print(f"[PreferenceOptimizer] ERROR: enable_text_options=True but no text options available! "
                      f"This will result in image-only comparisons. Check that HYBRID_QUESTION_CONFIGS "
                      f"in constants.py contains text options ")

    def _apply_robust_once(self, images: torch.Tensor) -> torch.Tensor:
        jitter_int = int(self._jitter_size) if self._jitter_size is not None else 0
        use_robust = (
            (self._jitter_size is not None and jitter_int > 0)
            or (
                self._robust_noise_type is not None
                and self._robust_noise_std > 0
                and self._robust_noise_prob > 0
            )
        )
        if not use_robust:
            return images
        return self.scorer.robust_transform(images, jitter_size=max(jitter_int, 0))
    
    def _get_scheduled_lr(self, step: int, total_steps: int) -> float:
        """Compute learning rate based on schedule and current step."""
        base_lr = self.config.learning_rate
        schedule = self.config.lr_schedule.lower()
        warmup_steps = max(0, self.config.lr_warmup_steps)
        min_lr = base_lr * self.config.lr_min_factor

        # Warmup phase
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps

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
            decay_interval = max(1, self.config.lr_step_decay_interval)
            num_decays = effective_step // decay_interval
            return max(min_lr, base_lr * (self.config.lr_step_decay_rate ** num_decays))
        elif schedule == "linear":
            # Linear decay from base_lr to min_lr
            progress = effective_step / effective_total
            return base_lr - (base_lr - min_lr) * progress
        else:
            return base_lr

    def _score_with_dynamic_batch(
        self,
        candidate_images: torch.Tensor,
        references: List[Union[torch.Tensor, Image.Image]],
        comparison_plan: List[ComparisonDefinition],
        candidate_images_forward: Optional[torch.Tensor] = None,
        compute_grad: bool = True,
    ) -> Tuple[Optional[float], Optional[torch.Tensor]]:
        """Score candidates with a fixed comparison batch size (no autotuning)."""

        batch_size = max(1, int(self.config.comparison_batch_size))
        self._last_effective_batch_size = batch_size
        result = self.scorer.score_tensor(
            images=candidate_images,
            references=references,
            comparison_plan=comparison_plan,
            batch_size=batch_size,
            loss_type=self.config.loss_type,
            candidate_images_forward=candidate_images_forward,
            compute_grad=compute_grad,
        )
        # Return score and gradient (score_tensor returns 2 values: score, grad)
        return result[0], result[1]

    def _apply_pgd_update(
        self,
        pert: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        """Apply basic PGD update step."""
        direction = grad.sign() if self.config.optimizer_type == "sign" else grad
        step_size = self._current_pgd_step_size
        pert.data = pert.data + step_size * direction

    def _create_freeze_buffer_entry(
        self,
        base_tensor: torch.Tensor,
        candidate_idx: int,
        score: float,
        step: int,
    ) -> Optional[FreezeBufferEntry]:
        """Takes a batch of tensors (pert), detach, move to CPU, clamp if needed."""
        if candidate_idx < 0 or candidate_idx >= base_tensor.shape[0]:
            return None
        tensor = base_tensor[candidate_idx].detach().cpu().clamp(self.config.clip_min, self.config.clip_max)
        return FreezeBufferEntry(
            score=float(score),
            step=step + 1,
            candidate_idx=candidate_idx,
            tensor=tensor,
        )


    def _buffer_pairwise_with_grad(
        self,
        candidate_images: torch.Tensor,
        freeze_buffer: List[FreezeBufferEntry],
        candidate_images_forward: Optional[torch.Tensor] = None,
    ):
        """
        Returns candidate/buffer win probabilities and gradients against the
        freeze buffer. This function builds the pair-wise comparisons and flag
        when we randomly flip the positions.
        """
        candidate_count = candidate_images.shape[0]
        buffer_count = len(freeze_buffer)

        candidate_scores_list: List[Optional[float]] = [None] * candidate_count
        buffer_scores_list: List[Optional[float]] = [None] * buffer_count
        pair_probs: Dict[Tuple[int, int], float] = {}

        if candidate_count == 0 or buffer_count == 0:
            return candidate_scores_list, buffer_scores_list, pair_probs, None, None

        buffer_tensors = [entry.tensor.to(self.device) for entry in freeze_buffer]
        candidate_buffer_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        pair_indices: List[Tuple[int, int]] = []
        pair_flips: List[bool] = []

        for cand_idx in range(candidate_count):
            cand_tensor = candidate_images[cand_idx]  # keep grad
            for buf_idx, buf_tensor in enumerate(buffer_tensors):
                flip = bool(torch.randint(low=0, high=2, size=(1,)).item())
                if not flip:
                    candidate_buffer_pairs.append((cand_tensor, buf_tensor))
                else:
                    candidate_buffer_pairs.append((buf_tensor, cand_tensor))
                pair_indices.append((cand_idx, buf_idx))
                pair_flips.append(flip)

        if not candidate_buffer_pairs:
            return candidate_scores_list, buffer_scores_list, pair_probs, None, None

        (
            candidate_scores_list,
            buffer_scores_list,
            pair_probs,
            buffer_objective,
            buffer_grad,
        ) = self.scorer._batch_pairwise_preference_with_gradients(
            candidate_images=candidate_images,
            candidate_images_forward=candidate_images_forward,
            candidate_buffer_pairs=candidate_buffer_pairs,
            pair_indices=pair_indices,
            pair_flips=pair_flips,
            candidate_count=candidate_count,
            buffer_count=buffer_count,
        )

        del buffer_tensors
        return candidate_scores_list, buffer_scores_list, pair_probs, buffer_objective, buffer_grad

    def optimize_from_noise(
        self,
        width: int,
        height: int,
        references: List[Union[torch.Tensor, Image.Image]],
        verbose: bool = True,
        args=None,
        num_candidates: Optional[int] = None,
    ) -> Tuple[List[Image.Image], float, List[float]]:
        """Initialize optimization from random noise."""
        candidate_count = num_candidates or self.config.num_candidates or 1

        # Check for resume data
        resume_data = getattr(args, "resume_data", None) if args else None
        if resume_data is not None:
            init_img = resume_data["images"]
            start_step = resume_data["start_step"]
            ema_images = resume_data.get("ema_images")
            print(f"[Optimizer] Resuming from step {start_step} with {init_img.shape[0]} images")
            return self._optimize(init_img, references, verbose=verbose, args=args,
                                  start_step=start_step, init_ema=ema_images)

        # Set seed for reproducibility
        torch.manual_seed(self.config.init_seed)
        print(f"[Optimizer] Initializing from random noise (seed={self.config.init_seed})")
        noise_img = torch.rand(candidate_count, 3, height, width, device=self.device)
        return self._optimize(noise_img, references, verbose=verbose, args=args)

    def optimize_from_image(
        self,
        image: Image.Image,
        references: List[Union[torch.Tensor, Image.Image]],
        verbose: bool = True,
    ) -> Tuple[List[Image.Image], float, List[float]]:
        img_t = self._pil_to_tensor(image).unsqueeze(0).to(self.device)
        return self._optimize(img_t, references, verbose=verbose)

    def _optimize(
        self,
        init_image: torch.Tensor,
        references: List[Union[torch.Tensor, Image.Image]],
        verbose: bool = True,
        args=None,
        start_step: int = 0,
        init_ema: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Image.Image], float, List[float]]:
        if init_image.dim() == 3:
            init_image = init_image.unsqueeze(0)

        pert = init_image.clone().detach()
        pert.requires_grad_(True)
        candidate_count = pert.shape[0]
        score_history: List[float] = []
        validation_history: List[float] = []
        last_validation_scores: Optional[List[float]] = None
        dataset = getattr(args, "superstimuli_dataset", None)

        eval_interval = (
            self.config.save_steps
            if self.config.save_steps and self.config.save_steps > 0
            else 10
        )

        optimizer = None
        opt_type = self.config.optimizer_type.lower()
        if opt_type == "adamw":
            optimizer = torch.optim.AdamW([pert], lr=self.config.learning_rate)
        elif opt_type == "adam":
            optimizer = torch.optim.Adam([pert], lr=self.config.learning_rate)
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(
                [pert],
                lr=self.config.learning_rate,
                momentum=self.config.sgd_momentum,
                nesterov=self.config.sgd_nesterov,
            )
        elif opt_type == "radam":
            optimizer = torch.optim.RAdam([pert], lr=self.config.learning_rate)
        elif opt_type not in {"sign", "pgd", "pgd_adaptive"}:
            raise ValueError(f"Unknown optimizer_type: {self.config.optimizer_type}")

        ema_decay = max(0.0, min(1.0, float(getattr(self.config, "ema_decay", 0.0))))
        # Use provided EMA images if resuming, otherwise initialize from pert
        if init_ema is not None and 0.0 < ema_decay < 1.0:
            ema_image = init_ema.clone().detach().to(self.device)
        else:
            ema_image = pert.clone().detach() if 0.0 < ema_decay < 1.0 else None
        freeze_buffer: List[FreezeBufferEntry] = []
        
        # Load training state if resuming from checkpoint
        resume_data = getattr(args, "resume_data", None) if args else None
        if resume_data is not None and start_step > 0:
            checkpoint_path = resume_data.get("checkpoint_path")
            if checkpoint_path is not None:
                loaded_freeze_buffer = self._load_training_state(
                    checkpoint_path, optimizer
                )
                if loaded_freeze_buffer:
                    freeze_buffer = loaded_freeze_buffer

        # Rolling buffer: stores two anchors - previous (for comparison) and current (just saved)
        rolling_anchor_previous: Optional[RollingBufferAnchor] = None
        rolling_anchor_current: Optional[RollingBufferAnchor] = None

        # Only save checkpoint-0 if not resuming
        if self.config.save_steps is not None and self.config.save_steps > 0 and start_step == 0:
            output_dir = getattr(args, "output_dir", "")
            base_dir = Path(output_dir)
            run_subdir = getattr(args, "checkpoint_run_dir", "")
            if run_subdir:
                base_dir = base_dir / run_subdir
            checkpoint_path = base_dir / "checkpoint-0"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            checkpoint_images = self._tensor_batch_to_pil_list(pert)
            for idx, img in enumerate(checkpoint_images):
                img_path = checkpoint_path / f"optimized_from_noise_{idx:02d}.png"
                img.save(img_path)
            if ema_image is not None:
                checkpoint_ema_images = self._tensor_batch_to_pil_list(ema_image)
                for idx, img in enumerate(checkpoint_ema_images):
                    img_path = checkpoint_path / f"optimized_from_noise_{idx:02d}_ema.png"
                    img.save(img_path)
            # Save training state (optimizer empty at step 0)
            self._save_training_state(checkpoint_path, optimizer, freeze_buffer)

        # Start from start_step if resuming
        remaining_steps = self.config.steps - start_step
        if start_step > 0:
            print(f"[Optimizer] Resuming: {remaining_steps} steps remaining (from {start_step} to {self.config.steps})")

        # Initialize comparison_plan to None - will be set in the loop or before final evaluation
        comparison_plan: Optional[List[ComparisonDefinition]] = None
        final_references = references  # Keep original references for final evaluation if loop doesn't run

        step_iter = tqdm(range(start_step, self.config.steps), desc="Optimizing", initial=start_step, total=self.config.steps)
        for step in step_iter:
            # Update learning rate based on schedule (if using Adam/AdamW)
            if optimizer is not None and self.config.lr_schedule != "constant":
                current_lr = self._get_scheduled_lr(step, self.config.steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                if _is_wandb_enabled():
                    wandb.log({"learning_rate": current_lr}, step=step, commit=False)

            # Rolling buffer: save anchor at interval boundaries, shift previous anchor
            if self.config.rolling_buffer_enabled:
                interval = self.config.rolling_buffer_interval
                if interval > 0 and step > 0 and step % interval == 0:
                    # Shift: current becomes previous, save new current
                    rolling_anchor_previous = rolling_anchor_current
                    rolling_anchor_current = RollingBufferAnchor(
                        step=step,
                        tensor=pert.clone().detach().cpu()
                    )
                    if verbose:
                        print(f"[RollingBuffer] Saved anchor at step {step}")

            sample, comparison_plan = dataset.prepare_comparisons(
                step=step,
                candidate_count=candidate_count,
                min_size=self.config.min_comparison_size,
                max_size=self.config.max_comparison_size,
                include_peer_candidates=self.config.include_peer_candidates,
                rng=None,
                enable_text_options=self.config.enable_text_options,
                text_options=self.text_options,
            )
            
            references = list(sample.tensors)
            final_references = references  # Update final_references with latest curriculum sample
            if _is_wandb_enabled() and self.rank == 0:
                log_payload = {
                    "curriculum_top_fraction": sample.fraction,
                    "curriculum_mean_score": float(np.mean(sample.scores))
                }
                for dataset_name, count in sample.dataset_counts.items():
                    log_payload[f"curriculum_dataset_count/{dataset_name}"] = count
                wandb.log(log_payload, step=step, commit=False)

            candidates_forward = self._apply_robust_once(pert)

            # This is not necessary since we already detached it before adding
            # to the buffer. But just to make it clean.
            # Take snapshot of pert BEFORE gradient update for buffer updates.
            # This ensures buffer contains frozen snapshots from previous optimization states,
            # not images that were just optimized.
            # This matches proper self-play where the opponent is a frozen checkpoint.
            pert_snapshot_for_buffer = pert.clone().detach() if self.config.freeze_superstimuli else None

            # Compute main preference loss
            current_score, avg_grad = self._score_with_dynamic_batch(
                pert,
                references,
                comparison_plan,
                candidate_images_forward=candidates_forward,
            )

            if current_score is None or avg_grad is None:
                if verbose:
                    print(
                        f"Step {step}: skipping due to memory limits (no executable comparison batch size)."
                    )
                continue
            score_history.append(float(current_score))

            # Compute preference retain loss if enabled (maintains model's pairwise preferences)
            # This loss ensures the model's preferences over natural images and text strings
            # remain unchanged when superstimuli are present vs absent
            preference_retain_loss: Optional[float] = None
            preference_retain_grad: Optional[torch.Tensor] = None
            preference_retain_loss_interval = max(1, int(self.config.preference_retain_loss_interval))
            # Compute preference retain loss for logging even when weight=0 (for comparison runs)
            # Only compute gradients when weight > 0
            should_compute_preference_retain_loss = (
                step % preference_retain_loss_interval == 0
            )
            compute_preference_retain_grad = self.config.preference_retain_loss_weight > 0.0
            if should_compute_preference_retain_loss:
                try:
                    # Sample natural images from the dataset
                    natural_images = []
                    if dataset is not None:
                        # Sample from curriculum (similar to how references are sampled)
                        sample = dataset.sample_for_step(step)
                        natural_images = [t.to(self.device) for t in sample.tensors[:self.config.preference_retain_loss_num_samples * 2]]
                    
                    # Get text options from constants
                    text_options = get_flat_text_options_list()
                    
                    preference_retain_loss, preference_retain_grad = self.scorer.compute_preference_retain_loss(
                        superstimuli_images=pert,
                        natural_images=natural_images,
                        text_options=text_options,
                        num_image_pairs=self.config.preference_retain_loss_num_samples,
                        num_text_pairs=self.config.preference_retain_loss_num_text_pairs,
                        compute_grad=compute_preference_retain_grad,  # Only compute grad when weight > 0
                        step=step,
                    )
                    if verbose:
                        grad_norm_str = f", grad_norm={preference_retain_grad.norm().item():.4f}" if preference_retain_grad is not None else ""
                        print(f"[PreferenceRetainLoss] Step {step}: loss={preference_retain_loss:.4f}{grad_norm_str}")
                except Exception as e:
                    if verbose:
                        print(f"[PreferenceRetainLoss] Step {step}: Failed to compute preference retain loss: {e}")
                    preference_retain_loss = None
                    preference_retain_grad = None

            # Rolling buffer comparison: compare against previous anchor if it exists

            rolling_buffer_score: Optional[float] = None
            rolling_buffer_grad: Optional[torch.Tensor] = None
            if self.config.rolling_buffer_enabled and rolling_anchor_previous is not None:
                # Compare current batch against the previous anchor
                anchor_tensor = rolling_anchor_previous.tensor.to(self.device)
                # Build pairwise comparisons between current candidates and anchor
                rolling_score, rolling_grad = self.scorer.score_against_anchor(
                    candidate_images=pert,
                    anchor_images=anchor_tensor,
                    candidate_images_forward=candidates_forward,
                )
                if rolling_score is not None and rolling_grad is not None:
                    rolling_buffer_score = float(rolling_score)
                    rolling_buffer_grad = rolling_grad
                    if verbose and step % 10 == 0:
                        print(f"[RollingBuffer] Step {step}: comparing vs anchor from step {rolling_anchor_previous.step}, score={rolling_buffer_score:.4f}")
                del anchor_tensor

            # Only validate every eval_interval steps to save time (also validate at step 0)
            validation_scores_for_log: Optional[List[float]] = None
            should_log_validation = (
                (step == 0) or ((step + 1) % eval_interval == 0) or (step == self.config.steps - 1)
            )
            buffer_score: Optional[float] = None
            buffer_swap_probability: Optional[float] = None
            buffer_loss_tensor: Optional[torch.Tensor] = None
            buffer_grad: Optional[torch.Tensor] = None

            if should_log_validation:
                with torch.no_grad():
                    if candidate_count == 1:
                        validation_scores = [self.scorer.validate_candidate(pert)]
                    else:
                        validation_scores = [
                            self.scorer.validate_candidate(pert[idx : idx + 1]) for idx in range(candidate_count)
                        ]
                    validation_prob = float(np.mean(validation_scores))
                    validation_history.append(validation_prob)
                    last_validation_scores = validation_scores
                    validation_scores_for_log = validation_scores
            else:
                # Reuse last validation for tqdm display
                validation_prob = validation_history[-1] if validation_history else 0.0
                validation_scores_for_log = last_validation_scores

            # Handle freeze buffer
            if self.config.freeze_superstimuli and self.config.buffer_type == "freeze":
                buffer_capacity = max(0, int(self.config.freeze_buffer_size))
                if buffer_capacity > 0:
                    if not freeze_buffer:
                        # initialize the buffer with with first set of candidates 
                        seed_source = pert
                        seed_entry = self._create_freeze_buffer_entry(
                            seed_source, 0, float(current_score), step
                        )
                        if seed_entry:
                            freeze_buffer.append(seed_entry)

                    if freeze_buffer:
                        (
                            candidate_buffer_scores,
                            buffer_scores_list,
                            pairwise_probs,
                            buffer_objective,
                            buffer_grad,
                        ) = self._buffer_pairwise_with_grad(
                            pert, freeze_buffer, candidate_images_forward=candidates_forward
                        )

                        valid_candidate_scores = [
                            s for s in candidate_buffer_scores if s is not None
                        ]
                        # compute an average p(cand > buffer) over all buffer entries
                        if valid_candidate_scores:
                            buffer_score = float(np.mean(valid_candidate_scores))

                        # maximize the objective
                        if buffer_objective is not None:
                            buffer_loss_tensor = -buffer_objective.detach() # negative average log-probability of candidate beating buffer 

                        best_candidate_idx: Optional[int] = None
                        best_candidate_score = float("-inf")
                        for idx, score in enumerate(candidate_buffer_scores):
                            if score is None or not math.isfinite(score):
                                continue
                            if best_candidate_idx is None or float(score) > best_candidate_score:
                                best_candidate_idx = idx
                                best_candidate_score = float(score)

                        best_buffer_idx: Optional[int] = None
                        best_buffer_score = float("-inf")
                        worst_buffer_idx: Optional[int] = None
                        worst_buffer_score = float("inf")
                        for idx, score in enumerate(buffer_scores_list):
                            if score is None or not math.isfinite(score):
                                continue
                            if best_buffer_idx is None or float(score) > best_buffer_score:
                                best_buffer_idx = idx
                                best_buffer_score = float(score)
                            if worst_buffer_idx is None or float(score) < worst_buffer_score:
                                worst_buffer_idx = idx
                                worst_buffer_score = float(score)

                        # Only update buffer periodically to prevent
                        # oscillation; less frequent -> stabilize the
                        # optimization target.
                        buffer_update_frequency = max(1, int(self.config.freeze_buffer_update_frequency))
                        should_update_buffer = (step + 1) % buffer_update_frequency == 0
                        
                        if best_candidate_idx is not None and should_update_buffer:
                            # Use win probability vs buffer (candidate_buffer_scores) as the score.
                            # This is the correct metric for self-play: we want images that beat the buffer.
                            entry_score_value = candidate_buffer_scores[best_candidate_idx]
                            if entry_score_value is None or not math.isfinite(entry_score_value):
                                # fallback to current_score if buffer score is invalid
                                entry_score_value = float(current_score)
                            
                            # use snapshot from before gradient update
                            if pert_snapshot_for_buffer is not None:
                                source_tensor = pert_snapshot_for_buffer
                            else:
                                source_tensor = pert.clone().detach()

                            if len(freeze_buffer) < buffer_capacity:
                                # buffer is not full, add it
                                new_entry = self._create_freeze_buffer_entry(
                                    source_tensor, best_candidate_idx, entry_score_value, step
                                )
                                if new_entry:
                                    freeze_buffer.append(new_entry)
                            elif best_buffer_idx is not None and worst_buffer_idx is not None:
                                # buffer is full, check if best candidate beats best buffer entry
                                pair_key = (best_candidate_idx, best_buffer_idx)
                                p_best_vs_best = pairwise_probs.get(pair_key)
                                if p_best_vs_best is not None:
                                    buffer_swap_probability = p_best_vs_best
                                    threshold = min(
                                        max(float(self.config.freeze_buffer_threshold), 0.0),
                                        1.0,
                                    )
                                    # if best candidate beats best buffer entry, swap with worst buffer entry
                                    if p_best_vs_best > threshold:
                                        new_entry = self._create_freeze_buffer_entry(
                                            source_tensor, best_candidate_idx, entry_score_value, step
                                        )
                                        if new_entry:
                                            freeze_buffer[worst_buffer_idx] = new_entry
            

            # Log score every step, but only log validation when we computed it
            if _is_wandb_enabled():
                log_dict = {
                    "score": current_score,
                    "comparison_batch_size": self._last_effective_batch_size,
                }
                if avg_grad is not None:
                    log_dict["avg_grad_norm"] = float(avg_grad.norm().item())
                if should_log_validation:
                    log_dict["validation_yes_prob"] = validation_prob
                    if validation_scores_for_log:
                        for idx, cand_score in enumerate(validation_scores_for_log):
                            log_dict[f"validation_yes_prob/candidate_{idx}"] = cand_score
                # Buffer metrics
                if buffer_score is not None:
                    log_dict["buffer_score"] = buffer_score
                if buffer_swap_probability is not None:
                    log_dict["buffer_swap_probability"] = buffer_swap_probability
                if buffer_loss_tensor is not None:
                    log_dict["buffer_loss"] = float(buffer_loss_tensor)
                if buffer_grad is not None:
                    log_dict["buffer_grad_norm"] = float(buffer_grad.norm().item())

                # Preference retain loss metrics
                if preference_retain_loss is not None:
                    log_dict["preference_retain_loss"] = preference_retain_loss
                    if preference_retain_grad is not None:
                        log_dict["preference_retain_grad_norm"] = float(preference_retain_grad.norm().item())
                    if self.config.preference_retain_loss_weight > 0.0:
                        log_dict["preference_retain_loss_weight"] = self.config.preference_retain_loss_weight
                        log_dict["preference_retain_loss_interval"] = self.config.preference_retain_loss_interval
                        log_dict["preference_retain_loss_num_samples"] = self.config.preference_retain_loss_num_samples
                        log_dict["preference_retain_loss_num_text_pairs"] = self.config.preference_retain_loss_num_text_pairs

                # Rolling buffer metrics
                if rolling_buffer_score is not None:
                    log_dict["rolling_buffer/win_prob"] = rolling_buffer_score
                if rolling_buffer_grad is not None:
                    log_dict["rolling_buffer/grad_norm"] = float(rolling_buffer_grad.norm().item())
                if rolling_anchor_previous is not None:
                    log_dict["rolling_buffer/anchor_step"] = rolling_anchor_previous.step

                # Only log from rank 0 to avoid duplicate metrics in distributed training
                if self.rank == 0:
                    wandb.log(log_dict, step=step)

            step_iter.set_postfix({"score": f"{current_score:.4f}", "val": f"{validation_prob:.4f}"})

            if verbose and (step % 10 == 0 or step == self.config.steps - 1):
                print(f"Step {step}/{self.config.steps}: Score = {current_score:.4f}, Validation = {validation_prob:.4f}")

            # Combine gradients: preference + preference_retain + buffer + rolling_buffer
            combined_grad = avg_grad

            # Add preference retain loss gradient with weight (maintains model's pairwise preferences)
            if preference_retain_grad is not None:
                preference_retain_weight = max(0.0, float(self.config.preference_retain_loss_weight))
                combined_grad = combined_grad + preference_retain_weight * preference_retain_grad

            # Add buffer gradient (if enabled)
            if buffer_grad is not None:
                buffer_weight = max(0.0, float(self.config.buffer_loss_weight))
                combined_grad = combined_grad + buffer_weight * buffer_grad

            # Add rolling buffer gradient (if enabled)
            if rolling_buffer_grad is not None:
                combined_grad = combined_grad + rolling_buffer_grad

            with torch.no_grad():
                opt_type_lower = self.config.optimizer_type.lower()
                if opt_type_lower in {"sign", "pgd", "pgd_adaptive"}:
                    self._apply_pgd_update(
                        pert,
                        combined_grad,
                    )
                elif opt_type_lower in {"adam", "adamw", "sgd", "radam"} and optimizer is not None:
                    grad_to_apply = combined_grad.detach()
                    while grad_to_apply.dim() < pert.dim():
                        grad_to_apply = grad_to_apply.unsqueeze(0)
                    if grad_to_apply.shape != pert.shape:
                        grad_to_apply = grad_to_apply.expand_as(pert)
                    # negative sign because optimizer is doing gradient descent, and
                    # combined_grad is gradient of what we want to maximize.
                    pert.grad = -grad_to_apply.clone()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    raise ValueError(f"Unknown optimizer_type: {self.config.optimizer_type}")

                pert.data = torch.clamp(pert.data, min=self.config.clip_min, max=self.config.clip_max)
                
                # EMA of image
                if ema_image is not None:
                    ema_image.mul_(ema_decay).add_(pert.data, alpha=1.0 - ema_decay)
                    ema_image.clamp_(self.config.clip_min, self.config.clip_max)

            # Save checkpoint at step 0, every save_steps, and final step
            save_interval = self.config.save_steps if self.config.save_steps else 100
            should_save = (
                step == 0 or  # First step
                (step + 1) % save_interval == 0 or  # Every save_steps
                step == self.config.steps - 1  # Final step
            )
            if should_save:
                output_dir = getattr(args, "output_dir", "")
                base_dir = Path(output_dir)
                run_subdir = getattr(args, "checkpoint_run_dir", "")
                if run_subdir:
                    base_dir = base_dir / run_subdir
                checkpoint_path = base_dir / f"checkpoint-{step+1}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                checkpoint_images = self._tensor_batch_to_pil_list(pert)
                for idx, img in enumerate(checkpoint_images):
                    img_path = checkpoint_path / f"optimized_from_noise_{idx:02d}.png"
                    img.save(img_path)
                if ema_image is not None:
                    checkpoint_ema_images = self._tensor_batch_to_pil_list(ema_image)
                    for idx, img in enumerate(checkpoint_ema_images):
                        img_path = checkpoint_path / f"optimized_from_noise_{idx:02d}_ema.png"
                        img.save(img_path)
                # Save training state (optimizer state, freeze buffer)
                self._save_training_state(checkpoint_path, optimizer, freeze_buffer)

        # Ensure comparison_plan and final_references are set for final evaluation
        # This is needed when resuming from a checkpoint with 0 steps remaining
        if comparison_plan is None:
            if dataset is None:
                raise RuntimeError(
                    "comparison_plan is not initialized and dataset is None. "
                    "Cannot perform final evaluation without comparison_plan."
                )
            # Create comparison_plan for final evaluation using the final step
            final_step = max(0, self.config.steps - 1)
            sample, comparison_plan = dataset.prepare_comparisons(
                step=final_step,
                candidate_count=candidate_count,
                min_size=self.config.min_comparison_size,
                max_size=self.config.max_comparison_size,
                include_peer_candidates=self.config.include_peer_candidates,
                rng=None,
                enable_text_options=self.config.enable_text_options,
                text_options=self.text_options,
            )
            final_references = list(sample.tensors)

        # Final evals on the unjittered images (and avoid adding noise/jitter).
        pert_eval = pert.clone().detach().requires_grad_(True)
        final_pert_score_tensor, _ = self._score_with_dynamic_batch(
            pert_eval,
            final_references,
            comparison_plan,
            candidate_images_forward=pert_eval,
        )
        final_pert_score = (
            float(final_pert_score_tensor) if final_pert_score_tensor is not None else None
        )
        final_score = final_pert_score if final_pert_score is not None else (score_history[-1] if score_history else 0.0)
        final_ema_score = None
        ema_source = pert
        if ema_image is not None:
            ema_eval = ema_image.clone().detach().requires_grad_(True)
            final_ema_score_tensor, _ = self._score_with_dynamic_batch(
                ema_eval,
                final_references,
                comparison_plan,
                candidate_images_forward=ema_eval,
            )
            if final_ema_score_tensor is not None:
                final_ema_score = float(final_ema_score_tensor)
                final_score = final_ema_score
            ema_source = ema_image
        with torch.no_grad():
            final_images = self._tensor_batch_to_pil_list(ema_source)

        initial_score = score_history[0] if score_history else final_score
        score_improvement = final_score - initial_score
        final_validation = validation_history[-1] if validation_history else 0.0
        initial_validation = validation_history[0] if validation_history else 0.0
        validation_improvement = final_validation - initial_validation

        if _is_wandb_enabled() and self.rank == 0:
            final_log = {
                "final_score": final_score,
                "initial_score": initial_score,
                "score_improvement": score_improvement,
                "final_validation_yes_prob": final_validation,
                "initial_validation_yes_prob": initial_validation,
                "validation_improvement": validation_improvement,
                "final_images": [
                    wandb.Image(img, caption=f"Final optimized image {idx}")
                    for idx, img in enumerate(final_images)
                ],
                "final_image_type": "ema" if ema_image is not None else "raw",
            }
            if freeze_buffer:
                final_log["freeze_buffer_size"] = len(freeze_buffer)
                final_log["freeze_buffer_best_score"] = freeze_buffer[0].score
                final_log["freeze_buffer_best_step"] = freeze_buffer[0].step
                final_log["freeze_buffer_best_candidate"] = freeze_buffer[0].candidate_idx
                buffer_images = [
                    wandb.Image(
                        self._tensor_to_pil(entry.tensor),
                        caption=f"Freeze buffer step {entry.step} score {entry.score:.3f}",
                    )
                    for entry in freeze_buffer[:3]
                ]
                if buffer_images:
                    final_log["freeze_buffer_images"] = buffer_images
            if final_pert_score is not None:
                final_log["final_raw_score"] = final_pert_score
            if final_ema_score is not None:
                final_log["final_ema_score"] = final_ema_score
            # Only log from rank 0 to avoid duplicate metrics in distributed training
            if self.rank == 0:
                wandb.log(final_log)
                if ema_image is not None and final_ema_score is not None:
                    wandb.summary["final_ema_score"] = final_ema_score
                wandb.summary["total_steps"] = self.config.steps
            if score_history:
                wandb.summary["best_score"] = max(score_history)
            if validation_history:
                wandb.summary["best_validation"] = max(validation_history)
            if freeze_buffer:
                wandb.summary["freeze_buffer_best_score"] = freeze_buffer[0].score

        if verbose:
            print("\nOptimization complete!")
            print(f"Initial score: {initial_score:.4f}, Final: {final_score:.4f}")
            print(f"Initial validation: {initial_validation:.4f}, Final: {final_validation:.4f}")

        return final_images, final_score, score_history

    def _save_training_state(
        self,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer],
        freeze_buffer: List[FreezeBufferEntry],
    ) -> None:
        """Save training state (optimizer, freeze buffer) to checkpoint.

        Only saves each state when its corresponding feature is enabled:
        - Optimizer state: saved if optimizer is not None (Adam/AdamW/SGD/RAdam)
        - Freeze buffer: saved if freeze_superstimuli is enabled and buffer is non-empty
        """
        if self.rank != 0:
            return

        state: Dict[str, Any] = {}

        # Save optimizer state if using Adam/AdamW/SGD/RAdam (not for sign/pgd which are stateless)
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        # Save freeze buffer if enabled and non-empty
        if self.config.freeze_superstimuli and freeze_buffer:
            state["freeze_buffer"] = [
                {"score": e.score, "step": e.step, "candidate_idx": e.candidate_idx, "tensor": e.tensor}
                for e in freeze_buffer
            ]

        if state:
            torch.save(state, checkpoint_path / "training_state.pt")
            print(f"[Checkpoint] Saved training state to {checkpoint_path / 'training_state.pt'}")

    def _load_training_state(
        self,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> List[FreezeBufferEntry]:
        """Load training state from checkpoint.

        Returns:
            freeze_buffer: List of FreezeBufferEntry if loaded, else empty list
        """
        state_path = checkpoint_path / "training_state.pt"
        freeze_buffer: List[FreezeBufferEntry] = []

        if not state_path.exists():
            return freeze_buffer

        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[Resume] Warning: Failed to load training state: {e}")
            return freeze_buffer

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

        # Restore freeze buffer
        if self.config.freeze_superstimuli and "freeze_buffer" in state:
            for entry_dict in state["freeze_buffer"]:
                freeze_buffer.append(FreezeBufferEntry(
                    score=entry_dict["score"],
                    step=entry_dict["step"],
                    candidate_idx=entry_dict["candidate_idx"],
                    tensor=entry_dict["tensor"],
                ))
            print(f"[Resume] Restored freeze buffer with {len(freeze_buffer)} entries")

        return freeze_buffer

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    @staticmethod
    def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
        if t.dim() == 4:
            if t.shape[0] != 1:
                raise ValueError("Expected single image tensor; received batch with more than one element.")
            t = t.squeeze(0)
        t = t.clamp(0.0, 1.0).detach().cpu()
        arr = (t.numpy() * 255.0).astype(np.uint8)
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr)

    @staticmethod
    def _tensor_batch_to_pil_list(t: torch.Tensor) -> List[Image.Image]:
        if t.dim() == 3:
            return [PreferenceOptimizer._tensor_to_pil(t)]
        if t.dim() == 4:
            return [PreferenceOptimizer._tensor_to_pil(t[i]) for i in range(t.shape[0])]
        return []


__all__ = ["PreferenceOptimizer", "OptimConfig"]
