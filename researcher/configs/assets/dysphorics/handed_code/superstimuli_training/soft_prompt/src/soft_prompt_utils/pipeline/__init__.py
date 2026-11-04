"""Helper utilities for the soft-prompt pipeline.

This subpackage centralizes:
- checkpoint & seed helpers
- W&B initialization
- reference loading for text-based runs
"""

from .checkpoints import (
    find_latest_checkpoint,
    load_checkpoint_embeddings,
    set_random_seeds,
)
from .reference_loading import ReferenceBundle, load_reference_data
from .wandb import init_wandb, init_wandb_if_enabled, is_wandb_enabled

__all__ = [
    "ReferenceBundle",
    "find_latest_checkpoint",
    "init_wandb",
    "init_wandb_if_enabled",
    "is_wandb_enabled",
    "load_checkpoint_embeddings",
    "load_reference_data",
    "set_random_seeds",
]

