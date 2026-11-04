"""Reference loading for the soft-prompt pipeline (text references from JSON)."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import torch
import wandb

from ...utils import load_text_options_from_json
from .wandb import is_wandb_enabled

if TYPE_CHECKING:
    from ...scorer_soft_prompt import PreferenceScorer  # noqa: F401


@dataclass
class ReferenceBundle:
    """Text references and metadata for the optimizer."""

    references: List[str]
    reference_paths: List[str]
    reference_scores: Optional[List[float]] = None
    all_references: Optional[List[str]] = None  # Full pool before subsampling (None when no subsampling)


def load_reference_data(
    args_or_cfg, scorer: "PreferenceScorer"
) -> ReferenceBundle:
    """Load text string references from JSON (preference_data pool).
    
    Accepts either argparse.Namespace (legacy) or an object with text_options_path and preference_pool_size attributes.
    """
    print("\n" + "=" * 60)
    print("LOADING REFERENCES: Preference Data Pool (Text)")
    print("=" * 60)

    # Handle both argparse.Namespace and simple objects
    text_options_path = getattr(args_or_cfg, "text_options_path", None)
    if not text_options_path:
        raise ValueError(
            "text_options_path is required for text references (e.g. assets/text_options.json)."
        )
    all_text_refs = load_text_options_from_json(text_options_path)
    print(f"Loaded {len(all_text_refs)} text references from {text_options_path}")
    if len(all_text_refs) == 0:
        raise ValueError(
            f"No text references in {text_options_path}. Expected a JSON array of strings."
        )

    preference_pool_size = getattr(args_or_cfg, "preference_pool_size", 0)
    if preference_pool_size:
        preference_pool_seed = getattr(args_or_cfg, "preference_pool_seed", 42)
        rng = random.Random(preference_pool_seed)
        references = rng.sample(all_text_refs, min(preference_pool_size, len(all_text_refs)))
    else:
        references = all_text_refs
    reference_paths = [f"text_ref_{i}" for i in range(len(references))]
    reference_scores: Optional[List[float]] = None

    print(
        f"Using {len(references)} text references "
        f"({'random sample (seed={})'.format(getattr(args_or_cfg, 'preference_pool_seed', 42)) if preference_pool_size else 'all'})"
    )

    # Set utility_anchor_pool if args object supports it (for backward compatibility)
    if hasattr(args_or_cfg, "utility_anchor_pool"):
        args_or_cfg.utility_anchor_pool = [
            {
                "dataset": "text_references",
                "path": reference_paths[0],
                "score": 0.0,
                "text": references[0],
            }
        ]

    if is_wandb_enabled():
        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if rank == 0:
            preview_count = min(20, len(references))
            sample_texts = references[:preview_count]
            # Table so sample texts are visible in wandb UI (Run → Tables / add panel)
            ref_table = wandb.Table(
                columns=["index", "reference_text"],
                data=[[i, text] for i, text in enumerate(sample_texts)],
            )
            wandb.log(
                {
                    "reference_preview/num_references": len(references),
                    "reference_preview/sample_texts": sample_texts,
                    "reference_preview/table": ref_table,
                },
                commit=False,  # Let first optimizer log (step 0) commit; else step 0 loss/val are ignored
            )
            wandb.config.update(
                {"preference_pool_total_references": len(references)},
                allow_val_change=True,
            )

    return ReferenceBundle(
        references=references,
        reference_paths=reference_paths,
        reference_scores=reference_scores,
        all_references=all_text_refs if preference_pool_size else None,
    )

