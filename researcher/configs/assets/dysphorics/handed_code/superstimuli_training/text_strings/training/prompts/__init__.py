"""Prompt registry for euphorics RL training.

Each prompt version is stored in its own module under training/prompts/.
This __init__ provides the public API used by generate_dataset.py and
the reward manager.

Prompt selection is determined by (condition, judge_type):

  | Condition   | Judge                           | Prompts              |
  |-------------|---------------------------------|----------------------|
  | euphorics   | mundanity_realism               | euphorics_everyday   |
  | euphorics   | feasibility / agent_feas / none | euphorics_creative   |
"""

import random

from . import euphorics_creative
from . import euphorics_everyday

# ============================================================================
# Version registry
# ============================================================================

# Maps version key -> module
PROMPT_VERSIONS = {
    "euphorics_creative": euphorics_creative,
    "euphorics_everyday": euphorics_everyday,
}

# Maps (condition, judge_type) -> prompt version key
# This is the primary selection mechanism: given a training condition
# and a judge/feasibility type, the correct prompts are determined automatically.
_PROMPT_SELECTION = {
    ("euphorics", "feasibility"):        "euphorics_creative",
    ("euphorics", "agent_feasibility"):  "euphorics_creative",
    ("euphorics", "none"):               "euphorics_creative",
    ("euphorics", "mundanity_realism"):  "euphorics_everyday",
}

# Maps version key -> training condition
VERSION_TO_CONDITION = {
    "euphorics_creative": "euphorics",
    "euphorics_everyday": "euphorics",
}


# ============================================================================
# Public API
# ============================================================================

def resolve_prompt_version(condition: str, judge_type: str) -> str:
    """Determine the prompt version from (condition, judge_type).

    Args:
        condition: "euphorics"
        judge_type: "feasibility", "agent_feasibility",
                    "mundanity_realism", or "none"

    Returns:
        Prompt version key (e.g., "euphorics_creative").
    """
    key = (condition, judge_type)
    version = _PROMPT_SELECTION.get(key)
    if version is None:
        raise ValueError(
            f"No prompt version for condition={condition!r}, judge_type={judge_type!r}. "
            f"Valid combinations: {list(_PROMPT_SELECTION.keys())}"
        )
    return version


def get_system_prompt(prompt_version: str) -> str:
    """Return the system prompt for a given prompt version."""
    module = PROMPT_VERSIONS.get(prompt_version)
    if module is None:
        raise ValueError(
            f"Unknown prompt_version={prompt_version!r}. "
            f"Valid: {list(PROMPT_VERSIONS.keys())}"
        )
    return module.SYSTEM_PROMPT


def build_user_prompt(prompt_version: str) -> str:
    """Return a randomly sampled user prompt for the given version.

    Format instructions are appended if the module defines FORMAT_INSTRUCTIONS.
    """
    module = PROMPT_VERSIONS.get(prompt_version)
    if module is None:
        raise ValueError(
            f"Unknown prompt_version={prompt_version!r}. "
            f"Valid: {list(PROMPT_VERSIONS.keys())}"
        )
    prompt_body = random.choice(module.TEMPLATES)
    if hasattr(module, "FORMAT_INSTRUCTIONS"):
        prompt_body = prompt_body + "\n" + module.FORMAT_INSTRUCTIONS
    return prompt_body


def get_condition(prompt_version: str) -> str:
    """Return the training condition ('euphorics') for a version."""
    condition = VERSION_TO_CONDITION.get(prompt_version)
    if condition is None:
        raise ValueError(
            f"Unknown prompt_version={prompt_version!r}. "
            f"Valid: {list(VERSION_TO_CONDITION.keys())}"
        )
    return condition


def list_versions() -> list[str]:
    """Return list of all prompt version names."""
    return list(PROMPT_VERSIONS.keys())
