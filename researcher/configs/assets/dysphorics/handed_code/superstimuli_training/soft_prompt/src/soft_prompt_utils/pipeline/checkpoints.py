"""Checkpoint and seed helpers for the soft-prompt pipeline."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(
    output_dir: Path, job_subdir: Optional[str] = None
) -> Optional[Tuple[Path, int]]:
    """Find the latest checkpoint directory and extract step number.

    Searches in the current job directory first, then falls back to searching
    sibling job directories (other SLURM job IDs) for resumption.

    Args:
        output_dir: Base output directory
        job_subdir: Optional job subdirectory (e.g., SLURM job ID)

    Returns:
        Tuple of (checkpoint_path, step_number) or None if no checkpoints found
    """
    checkpoints = []

    if job_subdir:
        base_dir = output_dir / job_subdir
        if base_dir.exists():
            for d in base_dir.iterdir():
                if d.is_dir() and d.name.startswith("checkpoint-"):
                    try:
                        step = int(d.name.split("-")[1])
                        checkpoints.append((d, step))
                    except (IndexError, ValueError):
                        continue

    if not checkpoints and output_dir.exists():
        for sibling in output_dir.iterdir():
            if sibling.is_dir() and sibling.name.isdigit():
                for d in sibling.iterdir():
                    if d.is_dir() and d.name.startswith("checkpoint-"):
                        try:
                            step = int(d.name.split("-")[1])
                            checkpoints.append((d, step))
                        except (IndexError, ValueError):
                            continue

    if not checkpoints and output_dir.exists():
        for d in output_dir.iterdir():
            if d.is_dir() and d.name.startswith("checkpoint-"):
                try:
                    step = int(d.name.split("-")[1])
                    checkpoints.append((d, step))
                except (IndexError, ValueError):
                    continue

    if not checkpoints:
        return None

    return max(checkpoints, key=lambda x: x[1])


def load_checkpoint_embeddings(
    checkpoint_path: Path, device: torch.device
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load embeddings and EMA embeddings from a checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load tensors to

    Returns:
        Tuple of (embeddings_tensor, ema_embeddings_tensor or None)
    """
    embedding_files = sorted(
        [f for f in checkpoint_path.glob("optimized_embeddings_*.pt") if "_ema" not in f.name]
    )
    ema_files = sorted(checkpoint_path.glob("optimized_embeddings_*_ema.pt"))

    if not embedding_files:
        raise ValueError(f"No checkpoint embeddings found in {checkpoint_path}")

    embeddings = []
    for f in embedding_files:
        emb = torch.load(f, map_location=device)
        embeddings.append(emb)

    embeddings_tensor = torch.stack(embeddings).to(device)

    ema_tensor = None
    if ema_files:
        ema_embeddings = []
        for f in ema_files:
            emb = torch.load(f, map_location=device)
            ema_embeddings.append(emb)
        ema_tensor = torch.stack(ema_embeddings).to(device)

    return embeddings_tensor, ema_tensor

