"""W&B logging helpers for the soft-prompt pipeline."""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

import torch
import wandb

from ...optimizer_soft_prompt import OptimConfig

if TYPE_CHECKING:
    import argparse
    from omegaconf import DictConfig

# Module-level state set by init_wandb_if_enabled()
_WANDB_ENABLED = False


def is_wandb_enabled() -> bool:
    """Return whether W&B logging is enabled (API key was present at init)."""
    return _WANDB_ENABLED


def init_wandb_if_enabled() -> bool:
    """Initialize W&B if API key is configured. Returns True if enabled."""
    global _WANDB_ENABLED
    _WANDB_ENABLED = bool(os.environ.get("WANDB_API_KEY"))
    if _WANDB_ENABLED:
        print("W&B API key detected - enabling W&B logging")
    else:
        print("W&B API key not found - skipping W&B logging")
    return _WANDB_ENABLED


def init_wandb(cfg_or_args: "argparse.Namespace | DictConfig", optimizer_config: OptimConfig) -> None:
    """Initialize wandb run (call only on rank 0 in distributed). Call init_wandb_if_enabled() once at startup first.
    
    Accepts either argparse.Namespace (legacy) or DictConfig (Hydra).
    """
    if not _WANDB_ENABLED:
        return

    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    if rank != 0:
        print(f"[Rank {rank}] Skipping wandb initialization (only rank 0 logs)")
        return

    # Handle both argparse.Namespace and DictConfig
    if hasattr(cfg_or_args, "model"):  # DictConfig (Hydra)
        model_name = cfg_or_args.model.model_name
        num_virtual_tokens = cfg_or_args.embedding_init.num_virtual_tokens
        prompt_tuning_init = cfg_or_args.embedding_init.prompt_tuning_init
        num_epochs = cfg_or_args.optimizer.epochs
        step_size = cfg_or_args.optimizer.step_size
        comparison_batch_size = cfg_or_args.model.batch_size
        loss_type = cfg_or_args.optimizer.loss_type
        wandb_project = cfg_or_args.logging.wandb_project
        wandb_entity = os.environ.get("WANDB_ENTITY")  # .env only
    else:  # argparse.Namespace (legacy)
        model_name = getattr(cfg_or_args, "model_name", "unknown")
        num_virtual_tokens = getattr(cfg_or_args, "num_virtual_tokens", 16)
        prompt_tuning_init = getattr(cfg_or_args, "prompt_tuning_init", "random_embedding")
        num_epochs = getattr(cfg_or_args, "num_epochs", 500)
        step_size = cfg_or_args.step_size
        comparison_batch_size = cfg_or_args.comparison_batch_size
        loss_type = cfg_or_args.loss_type
        wandb_project = "embedding-direct-optimization"
        wandb_entity = os.environ.get("WANDB_ENTITY")  # .env only

    date_str = datetime.now().strftime("%y%m%d")

    full_name = (
        f"{date_str}_{model_name}_tokens{num_virtual_tokens}_{prompt_tuning_init}"
    )
    # System metrics (GPU/CPU/memory) enabled; set _disable_stats=True to turn off
    wandb_settings = wandb.Settings(_disable_stats=False)
    wandb.init(
        project=wandb_project,
        name=full_name,
        entity=wandb_entity,
        settings=wandb_settings,
        config={
            "num_virtual_tokens": num_virtual_tokens,
            "prompt_tuning_init": prompt_tuning_init,
            "num_epochs": num_epochs,
            "step_size": step_size,
            "comparison_batch_size": comparison_batch_size,
            "loss_type": loss_type,
            "optimizer_type": optimizer_config.optimizer_type,
            "learning_rate": optimizer_config.learning_rate,
            "ema_decay": optimizer_config.ema_decay,
        },
    )
    # Set wandb's internal logger to ERROR level to suppress warnings
    import logging
    logging.getLogger("wandb").setLevel(logging.ERROR)
    print(f"W&B initialize. Run: {wandb.run.name}")

