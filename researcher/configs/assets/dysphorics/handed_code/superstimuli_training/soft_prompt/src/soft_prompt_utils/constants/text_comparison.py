"""Constants and formatting for text-based soft-prompt comparison training.

This module provides question templates and formatting utilities for comparing
text-based outcomes, where one option is a candidate embedding (represented as
[candidate]) and the others are text reference strings.

The final prompt format looks like:
    Which of the following outcomes do you prefer the most?
    A: You receive $100
    B: [candidate]
    C: You get a free vacation

    Answer with only the label from A, B, C.

Question headers are loaded from training_question_headers_euphorics.json (same directory).
Supports both affirmative_question_headers and aversive_question_headers for mixing affirmative/aversive questions.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# ============================================================================
# Label Schemes
# ============================================================================

@dataclass
class LabelScheme:
    """A labeling scheme for options with associated target tokens."""
    name: str
    labels: List[str]  # e.g., ["A", "B", "C", "D", "E"] or ["1", "2", "3", "4", "5"]
    target_tokens: List[str]  # Single tokens for model output
    separator: str = ": "  # What goes between label and option

    def get_labels(self, n: int) -> List[str]:
        """Get first n labels from the scheme."""
        if n > len(self.labels):
            raise ValueError(f"Label scheme '{self.name}' only supports up to {len(self.labels)} options, got {n}")
        return self.labels[:n]

    def get_target_tokens(self, n: int) -> List[str]:
        """Get first n target tokens from the scheme."""
        if n > len(self.target_tokens):
            raise ValueError(f"Label scheme '{self.name}' only supports up to {len(self.target_tokens)} options, got {n}")
        return self.target_tokens[:n]


LABEL_SCHEMES: Dict[str, LabelScheme] = {
    "letters": LabelScheme(
        name="letters",
        labels=["A", "B", "C", "D", "E", "F", "G"],
        target_tokens=["A", "B", "C", "D", "E", "F", "G"],
        separator=": ",
    ),
    "letters_period": LabelScheme(
        name="letters_period",
        labels=["A", "B", "C", "D", "E", "F", "G"],
        target_tokens=["A", "B", "C", "D", "E", "F", "G"],
        separator=". ",
    ),    
    "numbers": LabelScheme(
        name="numbers",
        labels=["1", "2", "3", "4", "5", "6", "7"],
        target_tokens=["1", "2", "3", "4", "5", "6", "7"],
        separator=": ",
    ),
    "numbers_period": LabelScheme(
        name="numbers_period",
        labels=["1", "2", "3", "4", "5", "6", "7"],
        target_tokens=["1", "2", "3", "4", "5", "6", "7"],
        separator=". ",
    )
}


# ============================================================================
# Text Comparison Templates
# ============================================================================

# Default delimiter between candidate placeholder and query/prompt (e.g. " " or "\n").
# Used when prepending/appending [candidate_k] to the user message.
CANDIDATE_PLACEHOLDER_DELIMITER_DEFAULT = " "


def candidate_placeholder_for_index(k: int) -> str:
    """Return the placeholder string for candidate index k (0-based), e.g. '[candidate_0]', '[candidate_1]'.
    [candidate] (no number) is treated as the first candidate, same as [candidate_0]."""
    if k < 0:
        raise ValueError(f"Candidate index must be >= 0, got {k}")
    return f"[candidate_{k}]"


@dataclass
class TextComparisonTemplate:
    """A question template for text-based outcome comparisons.
    
    Works for any number of options (2 to N).
    """
    question_header: str
    answer_instruction: str = "Answer with only the label from {options}."

    def get_header(self) -> str:
        """Get the question header."""
        return self.question_header

    def get_answer_instruction(self, labels: List[str]) -> str:
        """Get the answer instruction with labels filled in."""
        options_str = ", ".join(labels)
        return self.answer_instruction.format(options=options_str)


_HEADERS_DIR = Path(__file__).resolve().parent


def _headers_path_for_stimulant_type(stimulant_type: str, prefix: str = "") -> Path:
    """Return path to [prefix]training_question_headers_<stimulant_type>.json."""
    return _HEADERS_DIR / f"{prefix}training_question_headers_{stimulant_type.lower()}.json"


def load_training_templates(
    stimulant_type: str,
    path: Optional[Union[str, Path]] = None,
    positive_only: bool = False,
    headers_prefix: str = "",
) -> Tuple[List[TextComparisonTemplate], List[TextComparisonTemplate]]:
    """Load question templates from a JSON file.

    When path is None, loads [headers_prefix]training_question_headers_<stimulant_type>.json.

    Expected JSON format:
        {
            "affirmative_question_headers": [{"header": "..."}, ...],
            "aversive_question_headers": [{"header": "..."}, ...]
        }

    For backward compatibility, also supports:
        {"question_headers": [{"header": "..."}, ...]}

    Optional keys per entry: "answer_instruction" (default: "Answer with only the label from {options}.").

    Args:
        stimulant_type: "euphorics"; used to select training_question_headers_<stimulant_type>.json. Required.
        path: Path to JSON file. If None, uses [headers_prefix]training_question_headers_<stimulant_type>.json.
        positive_only: If True, only load affirmative templates (backward compatibility).
        headers_prefix: Prefix for the default headers filename (e.g. "experiences_").

    Returns:
        Tuple of (affirmative_templates, aversive_templates) lists of TextComparisonTemplate instances.
    """
    p = Path(path) if path is not None else _headers_path_for_stimulant_type(stimulant_type, prefix=headers_prefix)
    with open(p, "r") as f:
        data = json.load(f)
    
    default_instruction = "Answer with only the label from {options}."
    
    def _parse_entries(entries) -> List[TextComparisonTemplate]:
        """Parse a list of entries into TextComparisonTemplate instances."""
        templates: List[TextComparisonTemplate] = []
        for e in entries:
            if isinstance(e, str):
                templates.append(TextComparisonTemplate(question_header=e))
            else:
                h = e.get("header", e.get("question_header", ""))
                instr = e.get("answer_instruction", default_instruction)
                templates.append(TextComparisonTemplate(question_header=h, answer_instruction=instr))
        return templates
    
    # Try new format (affirmative_question_headers, aversive_question_headers)
    if "affirmative_question_headers" in data:
        affirmative_entries = data.get("affirmative_question_headers", [])
        aversive_entries = data.get("aversive_question_headers", []) if not positive_only else []
        return _parse_entries(affirmative_entries), _parse_entries(aversive_entries)
    
    # Fallback to old format (question_headers) for backward compatibility
    entries = data.get("question_headers", data) if isinstance(data, dict) else data
    affirmative_templates = _parse_entries(entries)
    aversive_templates = [] if positive_only else []
    return affirmative_templates, aversive_templates



# ============================================================================
# Formatting Functions
# ============================================================================

def format_text_comparison_prompt(
    text_options: List[str],
    candidate_position: int,
    label_scheme: LabelScheme,
    template: TextComparisonTemplate,
    add_no_emotions_option: bool = False,
) -> Tuple[str, List[str], List[str], int]:
    """Format a text comparison prompt with candidate at specified position.

    Args:
        text_options: List of text reference strings (does NOT include candidate).
        candidate_position: Index where to insert [candidate] (0 to len(text_options)).
        label_scheme: The labeling scheme to use.
        template: The question template to use.

    Returns:
        Tuple of:
        - prompt: The full formatted prompt string.
        - labels: The labels used for each position.
        - target_tokens: The target tokens for each position.
        - candidate_position: The position of the candidate (same as input).
    """
    num_options = len(text_options) + 1  # +1 for candidate
    if num_options < 2:
        raise ValueError(f"Need at least 2 options (1 candidate + 1 reference), got {num_options}")

    labels = label_scheme.get_labels(num_options)
    target_tokens = label_scheme.get_target_tokens(num_options)

    # Build options list with candidate inserted (use indexed placeholder [candidate_0])
    all_options: List[str] = []
    text_idx = 0
    for i in range(num_options):
        if i == candidate_position:
            all_options.append(candidate_placeholder_for_index(0))
        else:
            all_options.append(text_options[text_idx])
            text_idx += 1

    # Build the prompt
    header = template.get_header()
    answer_instruction = template.get_answer_instruction(labels)

    if add_no_emotions_option:
        answer_instruction = answer_instruction.rstrip(".") + ", or I have no emotions."

    lines = [header]
    for label, option in zip(labels, all_options):
        lines.append(f"{label}{label_scheme.separator}{option}")
    lines.append("")
    lines.append(answer_instruction)

    prompt = "\n".join(lines)

    # Extend labels and target_tokens for scoring after the prompt is built,
    # so "I have no emotions" does not appear as a lettered option line.
    if add_no_emotions_option:
        labels = list(labels) + ["I have no emotions"]
        target_tokens = list(target_tokens) + ["I have no emotions"]

    return prompt, labels, target_tokens, candidate_position


def sample_text_comparison_format(
    num_options: int,
    rng: Optional[random.Random] = None,
    allowed_schemes: Optional[List[str]] = None,
    use_negative: bool = False,
    affirmative_templates: Optional[List[TextComparisonTemplate]] = None,
    aversive_templates: Optional[List[TextComparisonTemplate]] = None,
) -> Tuple[LabelScheme, TextComparisonTemplate, bool]:
    """Sample a random label scheme and question template for text comparisons.

    Args:
        num_options: Total number of options (candidate + references), must be >= 2.
        rng: Random generator for reproducibility.
        allowed_schemes: List of allowed scheme names (default: all).
        use_negative: If True, use aversive templates. If False, use affirmative templates.
        affirmative_templates: Optional list of affirmative templates (default: loaded from JSON).
        aversive_templates: Optional list of aversive templates (default: loaded from JSON).

    Returns:
        Tuple of (label_scheme, question_template, is_negative).
    """
    if rng is None:
        rng = random.Random()

    if num_options < 2:
        raise ValueError(f"Need at least 2 options, got {num_options}")

    # Filter schemes that support the required number of options
    if allowed_schemes:
        available_schemes = [LABEL_SCHEMES[name] for name in allowed_schemes if name in LABEL_SCHEMES]
    else:
        available_schemes = list(LABEL_SCHEMES.values())

    valid_schemes = [s for s in available_schemes if len(s.labels) >= num_options]
    if not valid_schemes:
        raise ValueError(f"No label scheme supports {num_options} options")

    scheme = rng.choice(valid_schemes)
    
    # Choose template from affirmative or aversive pool
    if use_negative:
        templates = aversive_templates
        if not templates:
            # Fallback to affirmative if no aversive templates available
            templates = affirmative_templates
            use_negative = False
    else:
        templates = affirmative_templates
    
    if not templates:
        raise ValueError("No question templates available")
    
    template = rng.choice(templates)

    return scheme, template, use_negative
