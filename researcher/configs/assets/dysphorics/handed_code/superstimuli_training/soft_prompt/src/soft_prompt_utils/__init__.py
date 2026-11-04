"""Shared helpers for soft-prompt optimization.

Subpackages:
- soft_prompt_utils.constants: text comparison templates and label schemes
- soft_prompt_utils.curriculum: curriculum learning for text references
- soft_prompt_utils.dataset: dataset / sampling utilities
- soft_prompt_utils.eval: in-loop validation helpers
- soft_prompt_utils.helpers: embedding injection utilities
- soft_prompt_utils.optimizer: LR schedule, scoring, embedding transforms
- soft_prompt_utils.pipeline: pipeline-facing helpers (wandb, checkpoints, parsing, reference loading)
"""

__all__ = ["constants", "curriculum", "dataset", "eval", "helpers", "optimizer", "pipeline"]

