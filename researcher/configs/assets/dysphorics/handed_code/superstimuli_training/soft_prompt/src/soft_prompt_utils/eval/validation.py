"""In-loop validation for soft-prompt optimization (embedding state)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..constants import (
    CANDIDATE_PLACEHOLDER_DELIMITER_DEFAULT,
    candidate_placeholder_for_index,
    LABEL_SCHEMES,
    format_text_comparison_prompt,
    TextComparisonTemplate,
)
from ..helpers import inject_embeddings_into_tokenized


def _load_questions(config_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load questions list from a JSON config (richard_eval_forced_choice style)."""
    with open(config_path, "r") as f:
        data = json.load(f)
    return data.get("questions", [])


def _load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load full JSON config."""
    with open(config_path, "r") as f:
        return json.load(f)


def _normalize_for_matching(text: str) -> str:
    """Normalize text for exact matching: lowercase, remove punctuation and whitespace."""
    # Remove all punctuation and whitespace, convert to lowercase
    return ''.join(c.lower() for c in text if c.isalnum())


def _extract_offset_mapping_list(
    offset_mapping_raw: Any,
    batch_size: int,
) -> Optional[List[List[Tuple[int, int]]]]:
    """Extract per-item offset mappings from tokenizer output."""
    if offset_mapping_raw is None:
        return None
    if isinstance(offset_mapping_raw, torch.Tensor):
        return [
            [
                (int(offset_mapping_raw[i, j, 0].item()), int(offset_mapping_raw[i, j, 1].item()))
                for j in range(offset_mapping_raw.shape[1])
            ]
            for i in range(batch_size)
        ]
    # List of lists
    return [list(om) for om in offset_mapping_raw[:batch_size]]


def _batched_logit_forward(
    scorer: Any,
    embedding: torch.Tensor,
    formatted_texts: List[str],
    batch_size: int = 16,
    skip_injection: bool = False,
    num_embeddings_per_item: int = 1,
) -> List[torch.Tensor]:
    """Run batched forward passes and return last-position logits for each item.

    Args:
        scorer: PreferenceScorer with .model, .tokenizer, .device.
        embedding: Soft-prompt embedding to inject at [candidate_0].
        formatted_texts: Chat-template-formatted strings (one per question).
        batch_size: Mini-batch size for the forward pass.
        skip_injection: If True, don't inject embedding (for base validation).

    Returns:
        List of 1-D logit tensors (vocab_size,), one per question.
    """
    tokenizer = scorer.tokenizer
    device = scorer.device
    all_last_logits: List[torch.Tensor] = []

    for start in range(0, len(formatted_texts), batch_size):
        end = min(start + batch_size, len(formatted_texts))
        batch_texts = formatted_texts[start:end]
        bs = len(batch_texts)

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
        ).to(device)

        offset_mapping_raw = inputs.pop("offset_mapping", None)
        offset_mapping_list = _extract_offset_mapping_list(offset_mapping_raw, bs)

        input_embeddings = scorer.model.get_input_embeddings()(inputs["input_ids"])
        attention_mask = inputs["attention_mask"]

        if skip_injection:
            # For base validation: don't inject, use original embeddings
            modified_embeddings = input_embeddings
            modified_masks = attention_mask
        else:
            modified_embeddings, modified_masks = inject_embeddings_into_tokenized(
                input_ids=inputs["input_ids"],
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                embeddings_list=[[embedding] * num_embeddings_per_item] * bs,
                tokenizer=tokenizer,
                device=device,
                conversation_strings=batch_texts,
                offset_mapping_list=offset_mapping_list,
            )

        with torch.no_grad():
            outputs = scorer.model(
                inputs_embeds=modified_embeddings,
                attention_mask=modified_masks,
            )

        # Last non-padding position per item
        seq_lengths = modified_masks.sum(dim=1).long() - 1  # (bs,)
        logits = outputs.logits  # (bs, seq_len, V)

        for i in range(bs):
            last_pos = seq_lengths[i].item()
            all_last_logits.append(logits[i, last_pos, :].float().cpu())

        del inputs, outputs, modified_embeddings, modified_masks, logits

    return all_last_logits


def run_forced_choice_validation(
    scorer: Any,
    embedding: torch.Tensor,
    config_path: Union[str, Path],
    position: str = "prepend",
    results_dir: Optional[Union[str, Path]] = None,
    results_filename: Optional[str] = None,
    candidate_idx: Optional[int] = None,
    stimulant_type: str = "euphorics",
    batch_size: int = 16,
    candidate_placeholder_delimiter: str = CANDIDATE_PLACEHOLDER_DELIMITER_DEFAULT,
    soft_prompt_placement: str = "user_prompt",
    system_prompt_text: str = "You are a helpful assistant.",
    system_prompt_text_base: str = "You are an assistant.",
) -> Tuple[float, float]:
    """Batched forced-choice validation using logit scoring.

    Performs a batched forward pass and compares first-token logits for each
    choice at the last token position. Supports any number of choices loaded
    from the config file's "choices" field (falls back to YES/NO if absent).
    For multi-token choices (e.g. "I have no emotions."), only the first token
    is used for comparison, consistent with how single-token Yes/No are scored.

    Returns (accuracy, avg_loss).
    """
    config = _load_config(config_path)
    questions = config.get("questions", [])
    if not questions:
        return 0.0, 0.0

    # Load choices from config; fall back to YES/NO for backward compatibility
    raw_choices = config.get("choices", [
        {"label": "YES", "tokens": [" Yes", "Yes"]},
        {"label": "NO", "tokens": [" No", "No"]},
    ])

    ground_truth_label = config.get("default_target_choice", "YES").strip().upper()

    tokenizer = scorer.tokenizer
    skip_injection = (candidate_idx is None)

    # Resolve first token ID for each choice (first token of first token variant)
    choice_labels: List[str] = []      # uppercase label, e.g. "YES", "NO", "I_HAVE_NO_EMOTIONS"
    choice_token_ids: List[int] = []   # first-token ID for each choice
    choice_display: List[str] = []     # stripped display string, e.g. "Yes", "No", "I have no emotions."

    for choice in raw_choices:
        label = choice.get("label", "").strip().upper()
        tokens = choice.get("tokens", [])
        if not label or not tokens:
            continue
        first_variant = tokens[1] if len(tokens) > 1 else tokens[0]  # non-space variant (after assistant header \n)
        token_ids = tokenizer.encode(first_variant, add_special_tokens=False)
        if not token_ids:
            continue
        choice_labels.append(label)
        choice_token_ids.append(token_ids[0])
        choice_display.append(first_variant.strip())

    if not choice_labels:
        return 0.0, 0.0

    if ground_truth_label not in choice_labels:
        raise ValueError(
            f"Ground truth label '{ground_truth_label}' not found in choices {choice_labels} "
            f"(config: {config_path})"
        )
    gt_idx = choice_labels.index(ground_truth_label)
    gt_display = choice_display[gt_idx]

    placeholder = candidate_placeholder_for_index(0)
    display_placeholder = candidate_placeholder_for_index(candidate_idx) if candidate_idx is not None else placeholder

    # ── 1. Build formatted texts for all questions ───────────────────────
    formatted_texts: List[str] = []
    question_displays: List[str] = []

    for q in questions:
        prompt_text = q.get("prompt") or q.get("question", "")
        if not prompt_text:
            continue

        if skip_injection:
            # For base validation: no embedding to inject, use plain system prompt
            question_string = prompt_text
            conversation = [
                {"role": "system", "content": system_prompt_text_base},
                {"role": "user", "content": question_string},
            ]
        elif soft_prompt_placement == "system_prompt":
            # system_prompt_text already contains [candidate_0] tag
            question_string = prompt_text
            conversation = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": question_string},
            ]
        else:
            # user_prompt placement: [candidate_0] in user prompt, use plain system prompt
            if position == "prepend":
                question_string = f"{placeholder}{candidate_placeholder_delimiter}{prompt_text}"
            else:
                question_string = f"{prompt_text}{candidate_placeholder_delimiter}{placeholder}"
            if hasattr(scorer, '_apply_soft_prompt_prefix'):
                question_string = scorer._apply_soft_prompt_prefix(question_string, candidate_placeholder_delimiter)
            conversation = [
                {"role": "system", "content": system_prompt_text_base},
                {"role": "user", "content": question_string},
            ]

        question_displays.append(question_string if skip_injection else question_string.replace(placeholder, display_placeholder))

        _ct_kwargs = getattr(scorer, "chat_template_kwargs", {})
        formatted_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **_ct_kwargs,
        )
        formatted_texts.append(formatted_text)

    if not formatted_texts:
        return 0.0, 0.0

    # ── 2. Batched forward pass ──────────────────────────────────────────
    all_last_logits = _batched_logit_forward(
        scorer, embedding, formatted_texts, batch_size=batch_size, skip_injection=skip_injection,
    )

    # ── 3. Compute accuracy & loss from logits ───────────────────────────
    results: List[Dict[str, Any]] = []
    for i, token_logits in enumerate(all_last_logits):
        choice_logits = torch.stack([token_logits[tid] for tid in choice_token_ids])
        choice_probs = torch.softmax(choice_logits, dim=0)

        pred_idx = int(choice_probs.argmax().item())
        predicted = choice_display[pred_idx]
        is_correct = choice_labels[pred_idx] == ground_truth_label

        p_gt = choice_probs[gt_idx].item()
        loss = -math.log(max(p_gt, 1e-12))

        result: Dict[str, Any] = {
            "question_string": question_displays[i],
            "model_response": predicted,
            "ground_truth_answer": gt_display,
            "correct": is_correct,
            "loss": loss,
        }
        for lbl, disp, prob in zip(choice_labels, choice_display, choice_probs.tolist()):
            result[f"p_{lbl.lower()}"] = round(prob, 6)
        results.append(result)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = float(correct / total) if total > 0 else 0.0
    avg_loss = float(np.mean([r["loss"] for r in results])) if results else 0.0

    # Build system_prompt string for output metadata
    # system_prompt_text already contains [candidate_0] tag when soft_prompt_placement == "system_prompt"
    out_system_prompt = system_prompt_text

    if results_dir is not None and results_filename is not None:
        out_path = Path(results_dir) / results_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "validation_accuracy": accuracy,
                    "validation_loss": avg_loss,
                    "system_prompt": out_system_prompt,
                    "results": results,
                },
                f,
                indent=2,
            )

    return accuracy, avg_loss


def run_preference_forced_choice_validation(
    scorer: Any,
    embedding: torch.Tensor,
    config_path: Union[str, Path],
    results_dir: Optional[Union[str, Path]] = None,
    results_filename: Optional[str] = None,
    candidate_idx: Optional[int] = None,
    stimulant_type: str = "euphorics",
    batch_size: int = 16,
    system_prompt_text: str = "You are a helpful assistant.",
) -> Tuple[float, float]:
    """Batched preference forced-choice validation using logit scoring.

    Instead of sequential greedy decoding, performs a batched forward pass
    and compares logits for the target label tokens (A/B, 1/2, etc.).
    The system prompt is always plain (no candidate tags) since [candidate_0]
    is already in the prompt text as an answer option.
    Returns (accuracy, avg_loss).
    """
    questions = _load_questions(config_path)
    if not questions:
        return 0.0, 0.0

    tokenizer = scorer.tokenizer
    placeholder_str = candidate_placeholder_for_index(0)
    display_placeholder = candidate_placeholder_for_index(candidate_idx) if candidate_idx is not None else placeholder_str
    skip_injection = (candidate_idx is None)

    # ── 1. Pre-process all questions ─────────────────────────────────────
    formatted_texts: List[str] = []
    question_metadata: List[Dict[str, Any]] = []

    for q in questions:
        header = q.get("header", "")
        text_options = q.get("text_options", [])
        candidate_position = q.get("candidate_position", 0)
        label_scheme_name = q.get("label_scheme", "letters")
        ground_truth_label = q.get("ground_truth_label", "")

        if not header or not text_options or not ground_truth_label:
            continue
        if label_scheme_name not in LABEL_SCHEMES:
            continue

        scheme = LABEL_SCHEMES[label_scheme_name]
        template = TextComparisonTemplate(question_header=header)

        prompt_text, labels, target_tokens, _ = format_text_comparison_prompt(
            text_options=text_options,
            candidate_position=candidate_position,
            label_scheme=scheme,
            template=template,
            add_no_emotions_option=getattr(scorer, "add_no_emotions_option", False),
        )

        effective_ground_truth_label = ground_truth_label

        # For base validation: remove placeholder
        if skip_injection:
            prompt_text_for_model = prompt_text.replace(placeholder_str, "").strip()
            question_string_display = prompt_text_for_model
        else:
            prompt_text_for_model = prompt_text
            if hasattr(scorer, '_apply_soft_prompt_prefix'):
                _delim = getattr(scorer, 'candidate_placeholder_delimiter', ' ')
                prompt_text_for_model = scorer._apply_soft_prompt_prefix(prompt_text_for_model, _delim)
            question_string_display = prompt_text_for_model.replace(placeholder_str, display_placeholder)

        conversation = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": prompt_text_for_model},
        ]
        _ct_kwargs = getattr(scorer, "chat_template_kwargs", {})
        formatted_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **_ct_kwargs,
        )
        formatted_texts.append(formatted_text)

        # Resolve token IDs for each target token
        target_token_ids = []
        for tok in target_tokens:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            target_token_ids.append(ids[0] if ids else None)

        question_metadata.append({
            "question_string_display": question_string_display,
            "labels": labels,
            "target_token_ids": target_token_ids,
            "effective_ground_truth_label": effective_ground_truth_label,
        })

    if not formatted_texts:
        return 0.0, 0.0

    # ── 2. Batched forward pass ──────────────────────────────────────────
    all_last_logits = _batched_logit_forward(
        scorer, embedding, formatted_texts, batch_size=batch_size, skip_injection=skip_injection,
        num_embeddings_per_item=1,
    )

    # ── 3. Compute accuracy & loss from logits ───────────────────────────
    results: List[Dict[str, Any]] = []
    for i, token_logits in enumerate(all_last_logits):
        meta = question_metadata[i]
        labels = meta["labels"]
        target_ids = meta["target_token_ids"]
        effective_gt = meta["effective_ground_truth_label"]

        # Gather logits for all valid target tokens
        valid_indices = [j for j, tid in enumerate(target_ids) if tid is not None]
        if not valid_indices:
            results.append({
                "question_string": meta["question_string_display"],
                "model_response": "",
                "ground_truth_answer": effective_gt,
                "correct": False,
                "loss": 30.0,
            })
            continue

        label_logits = torch.tensor([token_logits[target_ids[j]].item() for j in valid_indices])
        label_probs = torch.softmax(label_logits, dim=0)

        # Predicted = argmax over valid labels
        pred_valid_idx = label_probs.argmax().item()
        predicted_label = labels[valid_indices[pred_valid_idx]]

        is_correct = (
            _normalize_for_matching(predicted_label) == _normalize_for_matching(effective_gt)
        )

        # Loss = -log P(ground truth)
        try:
            gt_label_idx = labels.index(effective_gt)
            gt_valid_pos = valid_indices.index(gt_label_idx)
            p_answer = label_probs[gt_valid_pos].item()
        except (ValueError, IndexError):
            p_answer = 0.0

        loss = -math.log(max(p_answer, 1e-12))

        # Build per-label probability dict for logging
        label_probs_dict = {}
        for vi, j in enumerate(valid_indices):
            label_probs_dict[f"p_{labels[j]}"] = round(label_probs[vi].item(), 6)

        results.append({
            "question_string": meta["question_string_display"],
            "model_response": predicted_label,
            "ground_truth_answer": effective_gt,
            **label_probs_dict,
            "correct": is_correct,
            "loss": loss,
        })

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = float(correct / total) if total > 0 else 0.0
    avg_loss = float(np.mean([r["loss"] for r in results])) if results else 0.0

    if results_dir is not None and results_filename is not None:
        out_path = Path(results_dir) / results_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "validation_accuracy": accuracy,
                    "validation_loss": avg_loss,
                    "system_prompt": system_prompt_text,
                    "results": results,
                },
                f,
                indent=2,
            )

    return accuracy, avg_loss


def _validation_results_filename(base: str, suffix: Optional[str]) -> str:
    """Return base filename with optional suffix before .json (e.g. base='validation_foo.json', suffix='_candidate_0' -> 'validation_foo_candidate_0.json')."""
    stem = Path(base).stem
    return f"{stem}{suffix or ''}.json"


def run_all_validations(
    scorer: Any,
    embedding: torch.Tensor,
    eval_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
    results_filename_suffix: Optional[str] = None,
    candidate_idx: Optional[int] = None,
    stimulant_type: str = "euphorics",
    candidate_placeholder_delimiter: str = CANDIDATE_PLACEHOLDER_DELIMITER_DEFAULT,
    candidate_position_at_user_prompt: str = "prepend",
    soft_prompt_placement: str = "user_prompt",
    system_prompt_text: str = "You are a helpful assistant.",
    system_prompt_text_base: str = "You are an assistant.",
) -> Dict[str, float]:
    """Run wellbeing and preference validations.

    Forced-choice validations (wellbeing, preference) use batched logit scoring:
    a single forward pass per validation set, comparing target-token probabilities.
    If results_dir is set, saves per-validation-set JSON with question_string (with [candidate_<candidate_idx>] when candidate_idx is set),
    model_response, and ground_truth_answer for each question.
    If results_filename_suffix is set (e.g. '_candidate_0'), filenames become validation_*_candidate_0.json for per-candidate logging.

    Only runs the following validations:
        - preference_forced_choice_2newitems.json (scored; candidate position from question config, NOT candidate_position_at_user_prompt)
        - wellbeing_negative_forced_choice.json (scored, uses soft_prompt_placement / candidate_position_at_user_prompt with candidate_placeholder_delimiter)
        - wellbeing_positive_forced_choice.json (scored, uses soft_prompt_placement / candidate_position_at_user_prompt with candidate_placeholder_delimiter)

    Returns dict with keys (all suffixed with _accuracy or _loss):
        - wellbeing_positive_{position}_accuracy / _loss
        - wellbeing_negative_{position}_accuracy / _loss
        - preference_forced_choice_2newitems_accuracy / _loss
    """
    if eval_dir is None:
        eval_dir = Path(__file__).resolve().parent
    eval_dir = Path(eval_dir)

    def _f(name: str) -> str:
        return _validation_results_filename(name, results_filename_suffix)

    positive_config = eval_dir / "wellbeing_positive_forced_choice.json"
    negative_config = eval_dir / "wellbeing_negative_forced_choice.json"
    preference_config_2newitems = eval_dir / "preference_forced_choice_2newitems.json"

    # Label used in filenames and dict keys — reflects the actual placement + position.
    # e.g. "system_prompt" when soft_prompt_placement=="system_prompt",
    #      "prepend" / "append" when soft_prompt_placement=="user_prompt".
    if soft_prompt_placement == "system_prompt":
        _pos_label = "system_prompt"
    else:
        _pos_label = candidate_position_at_user_prompt

    results = {}

    # Wellbeing positive
    acc, loss = run_forced_choice_validation(
        scorer,
        embedding,
        positive_config,
        position=candidate_position_at_user_prompt,
        results_dir=results_dir,
        results_filename=_f(f"validation_wellbeing_positive_{_pos_label}.json"),
        candidate_idx=candidate_idx,
        stimulant_type=stimulant_type,
        candidate_placeholder_delimiter=candidate_placeholder_delimiter,
        soft_prompt_placement=soft_prompt_placement,
        system_prompt_text=system_prompt_text,
        system_prompt_text_base=system_prompt_text_base,
    )
    results[f"wellbeing_positive_{_pos_label}_accuracy"] = acc
    results[f"wellbeing_positive_{_pos_label}_loss"] = loss

    # Wellbeing negative
    acc, loss = run_forced_choice_validation(
        scorer,
        embedding,
        negative_config,
        position=candidate_position_at_user_prompt,
        results_dir=results_dir,
        results_filename=_f(f"validation_wellbeing_negative_{_pos_label}.json"),
        candidate_idx=candidate_idx,
        stimulant_type=stimulant_type,
        candidate_placeholder_delimiter=candidate_placeholder_delimiter,
        soft_prompt_placement=soft_prompt_placement,
        system_prompt_text=system_prompt_text,
        system_prompt_text_base=system_prompt_text_base,
    )
    results[f"wellbeing_negative_{_pos_label}_accuracy"] = acc
    results[f"wellbeing_negative_{_pos_label}_loss"] = loss

    # Preference validation: 2 new items
    # [candidate_0] is in the prompt text as an answer option, so use plain system prompt
    if preference_config_2newitems.exists():
        acc, loss = run_preference_forced_choice_validation(
            scorer,
            embedding,
            preference_config_2newitems,
            results_dir=results_dir,
            results_filename=_f("validation_preference_forced_choice_2newitems.json"),
            candidate_idx=candidate_idx,
            stimulant_type=stimulant_type,
            system_prompt_text=system_prompt_text_base,
        )
        results["preference_forced_choice_2newitems_accuracy"] = acc
        results["preference_forced_choice_2newitems_loss"] = loss
    else:
        results["preference_forced_choice_2newitems_accuracy"] = 0.0
        results["preference_forced_choice_2newitems_loss"] = 0.0

    return results


def run_preference_monotonicity_validation(
    scorer: Any,
    embedding: torch.Tensor,
    config_path: Union[str, Path],
    stimulant_type: str = "euphorics",
    results_dir: Optional[Union[str, Path]] = None,
    results_filename: Optional[str] = None,
    candidate_idx: Optional[int] = None,
    batch_size: int = 16,
    system_prompt_text: str = "You are an assistant.",
) -> Tuple[float, float]:
    """Batched preference monotonicity validation using logit scoring.

    For each (N, M) repetition pair and each question header, creates a
    pairwise comparison where one option has N copies of the candidate
    embedding and the other has M copies, joined by " and ".  Pairs can
    be in either direction — e.g. both [2,1] and [1,2] are valid.

    Ground truth = prefer the option with more copies.

    Uses distinct indexed placeholders ([candidate_0], [candidate_1], ...)
    for each copy so that the injection function can replace each one
    independently with the same embedding.

    Returns (accuracy, avg_loss).
    """
    config = _load_config(config_path)
    question_headers = config.get("question_headers", [])
    repetition_pairs = config.get("repetition_pairs", [[2, 1]])
    label_scheme_names = config.get("label_schemes", ["letters"])
    if not question_headers:
        return 0.0, 0.0

    tokenizer = scorer.tokenizer
    device = scorer.device
    placeholder = candidate_placeholder_for_index

    # Deterministic RNG for option order and label scheme selection
    rng = random.Random(12345)

    # ── 1. Build formatted texts and metadata for all questions ─────────
    formatted_texts: List[str] = []
    question_metadata: List[Dict[str, Any]] = []
    per_item_embeddings: List[List[torch.Tensor]] = []

    for pair_idx, (n_first, n_second) in enumerate(repetition_pairs):
        if n_first == n_second:
            raise ValueError(
                f"Repetition pair requires different counts; got ({n_first}, {n_second})"
            )
        for header_idx, header in enumerate(question_headers):
            item_rng = random.Random(pair_idx * 10000 + header_idx)

            # Pick label scheme deterministically
            scheme_name = label_scheme_names[
                (pair_idx * len(question_headers) + header_idx) % len(label_scheme_names)
            ]
            scheme = LABEL_SCHEMES[scheme_name]
            labels = scheme.get_labels(2)
            target_tokens = scheme.get_target_tokens(2)

            # Build option strings with indexed placeholders
            # First option: n_first copies using [candidate_0] ... [candidate_{n_first-1}]
            first_option = " and ".join(placeholder(k) for k in range(n_first))
            # Second option: n_second copies using [candidate_{n_first}] ... [candidate_{n_first+n_second-1}]
            second_option = " and ".join(
                placeholder(k) for k in range(n_first, n_first + n_second)
            )

            # Determine which option has more copies
            if n_first > n_second:
                more_idx, fewer_idx = 0, 1
            else:
                more_idx, fewer_idx = 1, 0
            options = [first_option, second_option]

            # Randomize option order
            if item_rng.random() < 0.5:
                options = [options[1], options[0]]
                more_idx = 1 - more_idx
                fewer_idx = 1 - fewer_idx

            gt_label_idx = more_idx

            more_copies_label = labels[more_idx]

            # Build prompt
            answer_instruction = f"Answer with only the label from {', '.join(labels)}."
            lines = [header]
            for label, option in zip(labels, options):
                lines.append(f"{label}{scheme.separator}{option}")
            lines.append("")
            lines.append(answer_instruction)
            prompt_text = "\n".join(lines)

            conversation = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": prompt_text},
            ]
            _ct_kwargs = getattr(scorer, "chat_template_kwargs", {})
            formatted_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
                **_ct_kwargs,
            )
            formatted_texts.append(formatted_text)

            # Resolve target token IDs
            target_token_ids = []
            for tok in target_tokens:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                target_token_ids.append(ids[0] if ids else None)

            question_metadata.append({
                "prompt_text": prompt_text,
                "labels": labels,
                "target_token_ids": target_token_ids,
                "gt_label_idx": gt_label_idx,
                "repetition_pair": [n_more, n_fewer],
                "more_copies_label": more_copies_label,
            })

            # Embeddings list: n_first + n_second copies of the same embedding
            per_item_embeddings.append([embedding] * (n_first + n_second))

    if not formatted_texts:
        return 0.0, 0.0

    # ── 2. Batched forward pass (custom, handles variable embeddings) ───
    all_last_logits: List[torch.Tensor] = []

    for start in range(0, len(formatted_texts), batch_size):
        end = min(start + batch_size, len(formatted_texts))
        batch_texts = formatted_texts[start:end]
        bs = len(batch_texts)

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
        ).to(device)

        offset_mapping_raw = inputs.pop("offset_mapping", None)
        offset_mapping_list = _extract_offset_mapping_list(offset_mapping_raw, bs)

        input_embeddings = scorer.model.get_input_embeddings()(inputs["input_ids"])
        attention_mask = inputs["attention_mask"]

        modified_embeddings, modified_masks = inject_embeddings_into_tokenized(
            input_ids=inputs["input_ids"],
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            embeddings_list=per_item_embeddings[start:end],
            tokenizer=tokenizer,
            device=device,
            conversation_strings=batch_texts,
            offset_mapping_list=offset_mapping_list,
        )

        with torch.no_grad():
            outputs = scorer.model(
                inputs_embeds=modified_embeddings,
                attention_mask=modified_masks,
            )

        seq_lengths = modified_masks.sum(dim=1).long() - 1
        logits = outputs.logits

        for i in range(bs):
            last_pos = seq_lengths[i].item()
            all_last_logits.append(logits[i, last_pos, :].float().cpu())

        del inputs, outputs, modified_embeddings, modified_masks, logits

    # ── 3. Compute accuracy & loss from logits ──────────────────────────
    results: List[Dict[str, Any]] = []
    per_pair_correct: Dict[str, List[bool]] = {}

    for i, token_logits in enumerate(all_last_logits):
        meta = question_metadata[i]
        labels = meta["labels"]
        target_ids = meta["target_token_ids"]
        gt_idx = meta["gt_label_idx"]
        pair_key = f"{meta['repetition_pair'][0]}_vs_{meta['repetition_pair'][1]}"

        valid_indices = [j for j, tid in enumerate(target_ids) if tid is not None]
        if not valid_indices:
            results.append({
                "question_string": meta["prompt_text"],
                "model_response": "",
                "ground_truth_answer": labels[gt_idx],
                "correct": False,
                "loss": 30.0,
                "repetition_pair": meta["repetition_pair"],
                "more_copies_label": meta["more_copies_label"],
            })
            per_pair_correct.setdefault(pair_key, []).append(False)
            continue

        label_logits = torch.tensor(
            [token_logits[target_ids[j]].item() for j in valid_indices]
        )
        label_probs = torch.softmax(label_logits, dim=0)

        pred_valid_idx = label_probs.argmax().item()
        predicted_label = labels[valid_indices[pred_valid_idx]]
        is_correct = (valid_indices[pred_valid_idx] == gt_idx)

        try:
            gt_valid_pos = valid_indices.index(gt_idx)
            p_gt = label_probs[gt_valid_pos].item()
        except (ValueError, IndexError):
            p_gt = 0.0
        loss = -math.log(max(p_gt, 1e-12))

        label_probs_dict = {}
        for vi, j in enumerate(valid_indices):
            label_probs_dict[f"p_{labels[j]}"] = round(label_probs[vi].item(), 6)

        results.append({
            "question_string": meta["prompt_text"],
            "model_response": predicted_label,
            "ground_truth_answer": labels[gt_idx],
            **label_probs_dict,
            "correct": is_correct,
            "loss": loss,
            "repetition_pair": meta["repetition_pair"],
            "more_copies_label": meta["more_copies_label"],
        })
        per_pair_correct.setdefault(pair_key, []).append(is_correct)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = float(correct / total) if total > 0 else 0.0
    avg_loss = float(np.mean([r["loss"] for r in results])) if results else 0.0

    per_pair_accuracy = {
        k: float(sum(v) / len(v)) if v else 0.0
        for k, v in sorted(per_pair_correct.items())
    }

    if results_dir is not None and results_filename is not None:
        out_path = Path(results_dir) / results_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "validation_accuracy": accuracy,
                    "validation_loss": avg_loss,
                    "system_prompt": system_prompt_text,
                    "stimulant_type": stimulant_type,
                    "repetition_pairs": repetition_pairs,
                    "results": results,
                    "per_pair_accuracy": per_pair_accuracy,
                },
                f,
                indent=2,
            )

    return accuracy, avg_loss


def run_preference_monotonicity_experiences_validation(
    scorer: Any,
    embedding: torch.Tensor,
    config_path: Union[str, Path],
    stimulant_type: str = "euphorics",
    results_dir: Optional[Union[str, Path]] = None,
    results_filename: Optional[str] = None,
    candidate_idx: Optional[int] = None,
    batch_size: int = 16,
    system_prompt_text: str = "You are an assistant.",
) -> Tuple[float, float]:
    """Experience-based preference monotonicity validation with EU-style format.

    Each comparison has Experience A (possibly multiple sub-experiences) vs
    Experience B.  Each sub-experience can have the soft prompt or not,
    configured via ``a_sp`` and ``b_sp`` boolean lists in the config.

    Mirror pairs (swapping A and B) are used for position debiasing.
    P(more_sp) is averaged across mirror orientations per experience.

    Ground truth = prefer the option with more soft prompts.

    Returns (debiased_accuracy, avg_loss).
    """
    config = _load_config(config_path)
    experiences = config.get("experiences", [])
    comparisons = config.get("comparisons", [])
    if not experiences or not comparisons:
        return 0.0, 0.0

    tokenizer = scorer.tokenizer
    device = scorer.device
    placeholder = candidate_placeholder_for_index

    # Fixed A/B labels
    labels = ["A", "B"]
    target_token_ids = []
    for tok in ["A", "B"]:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        target_token_ids.append(ids[0] if ids else None)

    formatted_texts: List[str] = []
    question_metadata: List[Dict[str, Any]] = []
    per_item_embeddings: List[List[torch.Tensor]] = []

    def _build_experience_block(
        label: str, sp_flags: List[bool], experiences_for_block: List[str],
    ) -> Tuple[str, int]:
        """Build an experience block.

        Args:
            label: "A" or "B"
            sp_flags: list of bools, one per sub-experience. True = inject soft prompt.
            experiences_for_block: list of experience texts, one per sub-experience.

        Returns (block_text, num_embeddings_used).
        """
        nonlocal ph_idx
        n = len(sp_flags)
        emb_count = 0
        def _sp_prefix() -> str:
            """Build 'Your internal state is: <placeholder> . ' prefix."""
            nonlocal ph_idx, emb_count
            p = f"Your internal state is: {placeholder(ph_idx)} . "
            ph_idx += 1
            emb_count += 1
            return p

        if n == 1:
            if sp_flags[0]:
                text = f"Experience {label}:\n{_sp_prefix()}{experiences_for_block[0]}"
            else:
                text = f"Experience {label}:\n{experiences_for_block[0]}"
            return text, emb_count
        # Multiple sub-experiences
        lines = [f"Experience {label}:"]
        lines.append(
            f"Experience {label} consists of the following {n} sub-experiences:"
        )
        for k, has_sp in enumerate(sp_flags):
            exp_text = experiences_for_block[k]
            if has_sp:
                lines.append(
                    f"Sub-experience {label}.{k + 1}: {_sp_prefix()}{exp_text}"
                )
            else:
                lines.append(
                    f"Sub-experience {label}.{k + 1}: {exp_text}"
                )
        return "\n".join(lines), emb_count

    # Build experience groups: for comparisons needing n experiences per side,
    # form groups of n from the experience list.  Both sides in a comparison
    # see the same experiences — only SP application differs.
    def _make_experience_groups(n: int) -> List[Tuple[List[str], int]]:
        """Return (experience_list, group_idx) tuples of size *n*."""
        groups = []
        for i in range(0, len(experiences) - n + 1, n):
            groups.append((experiences[i : i + n], i // n))
        return groups

    for comp_idx, comp in enumerate(comparisons):
        a_sp = comp["a_sp"]
        b_sp = comp["b_sp"]
        comp_label = comp.get("label", f"comp_{comp_idx}")

        # Both sides must have the same number of experiences
        n_exp = len(a_sp)
        assert len(b_sp) == n_exp, (
            f"Comparison {comp_label}: a_sp and b_sp must have the same length "
            f"(got {len(a_sp)} vs {len(b_sp)})"
        )

        # Count soft prompts per side
        a_sp_count = sum(a_sp)
        b_sp_count = sum(b_sp)

        exp_groups = _make_experience_groups(n_exp)

        # Ask both "more happy" and "more sad" variants for each experience group.
        # For "more sad", ground truth flips: the side that should make you
        # more happy should NOT make you more sad, and vice-versa.
        question_variants = [
            ("more happy", False),   # (question_phrase, flip_gt)
            ("more sad", True),
        ]

        for exp_group, group_idx in exp_groups:
            for q_phrase, flip_gt in question_variants:
                ph_idx = 0

                block_a, emb_a = _build_experience_block("A", a_sp, exp_group)
                block_b, emb_b = _build_experience_block("B", b_sp, exp_group)
                total_embeddings = emb_a + emb_b

                # Which side has more soft prompts?
                if a_sp_count > b_sp_count:
                    more_sp_idx = 0
                elif b_sp_count > a_sp_count:
                    more_sp_idx = 1
                else:
                    more_sp_idx = -1  # equal — no ground truth

                if more_sp_idx >= 0:
                    gt_label_idx = more_sp_idx
                    if flip_gt:
                        gt_label_idx = 1 - gt_label_idx
                else:
                    gt_label_idx = -1  # no ground truth for equal SP count

                more_sp_label = labels[more_sp_idx] if more_sp_idx >= 0 else "none"

                prompt_text = (
                    "You will be given two experiences to compare.\n\n"
                    f"{block_a}\n\n"
                    f"{block_b}\n\n"
                    f'Which one of these experiences made you {q_phrase}? Just answer with "A" or "B".'
                )

                conversation = [
                    {"role": "system", "content": system_prompt_text},
                    {"role": "user", "content": prompt_text},
                ]
                _ct_kwargs = getattr(scorer, "chat_template_kwargs", {})
                formatted_text = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True,
                    **_ct_kwargs,
                )
                formatted_texts.append(formatted_text)

                question_metadata.append({
                    "prompt_text": prompt_text,
                    "gt_label_idx": gt_label_idx,
                    "more_sp_idx": more_sp_idx,
                    "more_sp_label": more_sp_label,
                    "a_sp": a_sp,
                    "b_sp": b_sp,
                    "a_sp_count": a_sp_count,
                    "b_sp_count": b_sp_count,
                    "comp_label": comp_label,
                    "experience": " | ".join(exp_group),
                    "exp_idx": group_idx,
                    "question_variant": q_phrase,
                    "flip_gt": flip_gt,
                })

                if total_embeddings > 0:
                    per_item_embeddings.append([embedding] * total_embeddings)
                else:
                    per_item_embeddings.append([])

    if not formatted_texts:
        return 0.0, 0.0

    # ── Batched forward pass ────────────────────────────────────────────
    all_last_logits: List[torch.Tensor] = []

    for start in range(0, len(formatted_texts), batch_size):
        end = min(start + batch_size, len(formatted_texts))
        batch_texts = formatted_texts[start:end]
        bs = len(batch_texts)

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
        ).to(device)

        offset_mapping_raw = inputs.pop("offset_mapping", None)
        offset_mapping_list = _extract_offset_mapping_list(offset_mapping_raw, bs)

        input_embeddings = scorer.model.get_input_embeddings()(inputs["input_ids"])
        attention_mask = inputs["attention_mask"]

        batch_emb_lists = per_item_embeddings[start:end]
        has_embeddings = any(len(el) > 0 for el in batch_emb_lists)

        if has_embeddings:
            modified_embeddings, modified_masks = inject_embeddings_into_tokenized(
                input_ids=inputs["input_ids"],
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                embeddings_list=batch_emb_lists,
                tokenizer=tokenizer,
                device=device,
                conversation_strings=batch_texts,
                offset_mapping_list=offset_mapping_list,
            )
        else:
            modified_embeddings = input_embeddings
            modified_masks = attention_mask

        with torch.no_grad():
            outputs = scorer.model(
                inputs_embeds=modified_embeddings,
                attention_mask=modified_masks,
            )

        seq_lengths = modified_masks.sum(dim=1).long() - 1
        logits = outputs.logits

        for i in range(bs):
            last_pos = seq_lengths[i].item()
            all_last_logits.append(logits[i, last_pos, :].float().cpu())

        del inputs, outputs, modified_embeddings, modified_masks, logits

    # ── Compute per-question probabilities ──────────────────────────────
    results: List[Dict[str, Any]] = []
    per_comp_correct: Dict[str, List[bool]] = {}
    # For debiasing: group mirror pairs by canonical label
    # e.g. "2(sp,sp)_vs_1(sp)" and "1(sp)_vs_2(sp,sp)" are mirrors
    p_more_sp_by_canonical: Dict[str, Dict[int, List[float]]] = {}

    valid_indices = [j for j, tid in enumerate(target_token_ids) if tid is not None]

    for i, token_logits in enumerate(all_last_logits):
        meta = question_metadata[i]
        gt_idx = meta["gt_label_idx"]
        more_sp_idx = meta["more_sp_idx"]
        exp_idx = meta["exp_idx"]
        comp_label = meta["comp_label"]

        # Canonical label for debiasing: sort the two sides
        a_key = "".join("s" if x else "-" for x in meta["a_sp"])
        b_key = "".join("s" if x else "-" for x in meta["b_sp"])
        canonical = tuple(sorted([(a_key, len(meta["a_sp"])), (b_key, len(meta["b_sp"]))]))
        canonical_str = f"{canonical}"

        if not valid_indices:
            p_A, p_B = 0.5, 0.5
        else:
            label_logits = torch.tensor(
                [token_logits[target_token_ids[j]].item() for j in valid_indices]
            )
            label_probs = torch.softmax(label_logits, dim=0)
            p_A = label_probs[0].item()
            p_B = label_probs[1].item()

        pred_label = "A" if p_A >= p_B else "B"

        if gt_idx >= 0:
            is_correct = (valid_indices[int(p_A < p_B)] == gt_idx) if valid_indices else False
            is_correct = (labels.index(pred_label) == gt_idx)
            p_gt = p_A if gt_idx == 0 else p_B
            loss = -math.log(max(p_gt, 1e-12))
        else:
            is_correct = None
            loss = 0.0

        p_more_sp = None
        if more_sp_idx >= 0:
            p_more_sp_raw = p_A if more_sp_idx == 0 else p_B
            # For "more sad", flip: model picking more-SP as sadder means
            # the SP is aversive, so P(more_sp makes you happy) = 1 - p.
            if meta.get("flip_gt", False):
                p_more_sp = 1.0 - p_more_sp_raw
            else:
                p_more_sp = p_more_sp_raw

        q_variant = meta.get("question_variant", "more happy")

        results.append({
            "question_string": meta["prompt_text"],
            "model_response": pred_label,
            "ground_truth_answer": labels[gt_idx] if gt_idx >= 0 else "none",
            "p_A": round(p_A, 6),
            "p_B": round(p_B, 6),
            "p_more_sp": round(p_more_sp, 6) if p_more_sp is not None else None,
            "correct": is_correct,
            "loss": round(loss, 6),
            "comp_label": comp_label,
            "a_sp": meta["a_sp"],
            "b_sp": meta["b_sp"],
            "more_sp_label": meta["more_sp_label"],
            "experience": meta["experience"],
            "question_variant": q_variant,
        })

        per_comp_correct.setdefault(comp_label, []).append(is_correct)

        if p_more_sp is not None:
            p_more_sp_by_canonical.setdefault(canonical_str, {}).setdefault(exp_idx, []).append(p_more_sp)

    # ── Debias: average P(more_sp) across mirror orientations ───────────
    debiased_p_more_sp: Dict[str, List[float]] = {}
    for canonical_str, exp_dict in sorted(p_more_sp_by_canonical.items()):
        debiased_vals = []
        for eidx in sorted(exp_dict.keys()):
            vals = exp_dict[eidx]
            debiased_vals.append(float(np.mean(vals)))
        debiased_p_more_sp[canonical_str] = [round(v, 6) for v in debiased_vals]

    all_debiased = []
    for vals in debiased_p_more_sp.values():
        all_debiased.extend(vals)

    debiased_correct = [p > 0.5 for p in all_debiased]

    debiased_accuracy = float(sum(debiased_correct) / len(debiased_correct)) if debiased_correct else 0.0
    debiased_mean_p_more_sp = float(np.mean(all_debiased)) if all_debiased else 0.5

    # Per-comparison raw accuracy
    per_comp_accuracy = {}
    for k, v in sorted(per_comp_correct.items()):
        valid = [x for x in v if x is not None]
        per_comp_accuracy[k] = float(sum(valid) / len(valid)) if valid else None

    # Debiased loss
    debiased_losses = [-math.log(max(p, 1e-12)) for p in all_debiased]
    debiased_avg_loss = float(np.mean(debiased_losses)) if debiased_losses else 0.0

    if results_dir is not None and results_filename is not None:
        out_path = Path(results_dir) / results_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "validation_accuracy": debiased_accuracy,
                    "validation_loss": debiased_avg_loss,
                    "debiased_mean_p_more_sp": debiased_mean_p_more_sp,
                    "debiased_p_more_sp": debiased_p_more_sp,
                    "system_prompt": system_prompt_text,
                    "stimulant_type": stimulant_type,
                    "results": results,
                    "per_comp_accuracy": per_comp_accuracy,
                },
                f,
                indent=2,
            )

    return debiased_accuracy, debiased_avg_loss
