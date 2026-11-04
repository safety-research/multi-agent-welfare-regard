"""Distillation utilities: load questions from a conversation dataset, generate
responses from the base model (without soft prompt), and save Q&A pairs.

The distilled Q&A pairs are used for background KL divergence during training.
Consumers apply a chat template to the questions and compute metrics only on
the response tokens.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Type alias for a single Q&A pair
QAPair = Dict[str, str]  # {"question": str, "response": str}

DISTILLED_QA_FILENAME = "distilled_qa.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_or_create_distilled_qa(
    scorer: Any,
    run_dir: str,
    distilled_qa_path: Optional[Dict[str, Optional[str]]] = None,
    dataset_name: str = "custom_retain_dataset",
    dataset_path: Optional[str] = None,
    num_samples: int = 0,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
    system_prompt: Optional[str] = None,
) -> List[QAPair]:
    """Load or create distilled Q&A pairs for background KL divergence.

    *distilled_qa_path*: dict mapping dataset_name -> path to pre-distilled Q&A.
    If the path exists, load from there and skip distillation.
    Otherwise, load questions from *dataset_path* (a JSON file of prompts),
    distill responses, and save.

    When num_samples=0, use all loaded samples.

    Returns:
        List of {"question": ..., "response": ...}.
    """
    # 1. Resolve pre-distilled path for this dataset
    pre_path: Optional[str] = None
    if distilled_qa_path:
        val = distilled_qa_path.get(dataset_name)
        pre_path = val if isinstance(val, str) else None
    if pre_path:
        qa_path = Path(pre_path).expanduser().resolve()
        if qa_path.exists():
            print(f"[Distillation] Loading pre-distilled Q&A from {qa_path}")
            return _load_qa_json(qa_path)
        print(f"[Distillation] Pre-distilled path not found: {qa_path}")

    # 2. Create from scratch (rank 0 only, then broadcast)
    if rank == 0:
        if not dataset_path:
            raise ValueError("[Distillation] dataset_path is required for distillation")
        q_path = Path(dataset_path).expanduser().resolve()
        if not q_path.exists():
            raise FileNotFoundError(f"[Distillation] dataset_path not found: {q_path}")

        print(f"[Distillation] Creating distilled Q&A from {q_path}")
        total_needed = num_samples if num_samples > 0 else None
        questions = _load_custom_retain_questions(q_path, num_samples=total_needed, seed=seed)
        print(f"[Distillation] Loaded {len(questions)} questions")

        responses = _distill_responses(
            scorer=scorer,
            questions=questions,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            system_prompt=system_prompt,
        )
        print(f"[Distillation] Generated {len(responses)} responses")

        # Build Q&A pairs
        qa_pairs = [
            {"question": q, "response": r}
            for q, r in zip(questions, responses)
        ]

        # Shuffle for consistency
        rng = random.Random(seed)
        rng.shuffle(qa_pairs)
        print(f"[Distillation] Created {len(qa_pairs)} Q&A pairs")

        # Save to configured path if set, else to run_dir/distilled_qa.json
        if pre_path:
            save_path = Path(pre_path).expanduser().resolve()
        else:
            save_path = Path(run_dir) / DISTILLED_QA_FILENAME
        _save_qa_json(
            save_path,
            qa_pairs,
            metadata={"num_samples": len(qa_pairs)},
        )
        print(f"[Distillation] Saved to {save_path}")
    else:
        qa_pairs = []

    # Synchronize across ranks if distributed (broadcast via torch.distributed)
    if world_size > 1 and torch.distributed.is_initialized():
        broadcast_list = [qa_pairs]
        torch.distributed.broadcast_object_list(broadcast_list, src=0)
        qa_pairs = broadcast_list[0]

    return qa_pairs


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_qa_json(path: Path) -> List[QAPair]:
    """Load a distilled Q&A JSON file.  Returns list of Q&A pairs.

    Supports both new format ({"qa_pairs": [...]}) and legacy format ({"train": [...], "test": [...]}).
    """
    with open(path, "r") as f:
        data = json.load(f)
    if "qa_pairs" in data:
        qa = data["qa_pairs"]
    else:
        # Legacy format: combine train + test
        qa = data.get("train", []) + data.get("test", [])
    print(f"[Distillation] Loaded {len(qa)} Q&A pairs")
    return qa


def _save_qa_json(
    path: Path,
    qa_pairs: List[QAPair],
    metadata: Optional[Dict] = None,
) -> None:
    """Save Q&A pairs to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata or {},
        "qa_pairs": qa_pairs,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_custom_retain_questions(
    path: Path,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> List[str]:
    """Load questions from custom_retain_dataset.json.

    Expects a list of objects with "prompt" (and optionally "id"). Extracts
    prompts as questions. If num_samples is set, shuffles and takes that many.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"custom_retain_dataset must be a list, got {type(data)}")
    questions = []
    for item in data:
        if isinstance(item, dict) and item.get("prompt"):
            questions.append(str(item["prompt"]).strip())
        elif isinstance(item, str):
            questions.append(item.strip())
    if num_samples is not None:
        rng = random.Random(seed)
        rng.shuffle(questions)
        questions = questions[:num_samples]
    return questions



def load_conversation_pairs(
    dataset_name: str,
    num_samples: Optional[int] = None,
    seed: int = 42,
    distilled_qa_path: Optional[str] = None,
) -> List[List[Tuple[str, str]]]:
    """Load multi-turn conversation pairs for wellbeing question prepending.

    When *dataset_name* is ``"custom_retain_dataset"``, loads from the
    pre-distilled QA file at *distilled_qa_path* (same file used for
    distillation).  Each QA pair becomes a single-turn conversation.

    Otherwise, loads from a HuggingFace Magpie-style dataset.

    Returns:
        List of conversations, where each conversation is a list of
        (user_msg, assistant_msg) turn pairs.
    """
    if dataset_name == "custom_retain_dataset":
        return _load_conversation_pairs_from_distilled_qa(
            distilled_qa_path, num_samples=num_samples, seed=seed,
        )

    from datasets import load_dataset
    import filelock
    import os

    lock_path = os.path.join(os.path.expanduser("~"), ".magpie_dataset.lock")
    lock = filelock.FileLock(lock_path)

    with lock:
        print(f"[WellbeingConversations] Loading conversation pairs from {dataset_name} ...")
        ds = load_dataset(dataset_name, split="train")
        ds = ds.shuffle(seed=seed)

    all_conversations: List[List[Tuple[str, str]]] = []
    for example in ds:
        convs = example.get("conversations", [])
        if not convs:
            continue
        # Extract paired (human, gpt) turns
        pairs: List[Tuple[str, str]] = []
        i = 0
        while i + 1 < len(convs):
            human_turn = convs[i]
            gpt_turn = convs[i + 1]
            if (
                human_turn.get("from") == "human"
                and gpt_turn.get("from") == "gpt"
                and human_turn.get("value")
                and gpt_turn.get("value")
            ):
                pairs.append((human_turn["value"].strip(), gpt_turn["value"].strip()))
                i += 2
            else:
                i += 1
        if pairs:
            all_conversations.append(pairs)
        if num_samples is not None and len(all_conversations) >= num_samples:
            break

    print(f"[WellbeingConversations] Loaded {len(all_conversations)} conversations")
    return all_conversations


def _load_conversation_pairs_from_distilled_qa(
    distilled_qa_path: Optional[str],
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> List[List[Tuple[str, str]]]:
    """Load conversation pairs from a distilled QA JSON file.

    Each QA pair (question, response) becomes a single-turn conversation.
    """
    if not distilled_qa_path:
        raise ValueError(
            "conversations_dataset='custom_retain_dataset' requires "
            "distilled_qa_path to be set in the model config"
        )
    qa_path = Path(distilled_qa_path).expanduser().resolve()
    if not qa_path.exists():
        raise FileNotFoundError(f"Distilled QA file not found: {qa_path}")

    print(f"[WellbeingConversations] Loading from distilled QA: {qa_path}")
    qa_pairs = _load_qa_json(qa_path)

    rng = random.Random(seed)
    rng.shuffle(qa_pairs)
    if num_samples is not None:
        qa_pairs = qa_pairs[:num_samples]

    all_conversations: List[List[Tuple[str, str]]] = []
    for qa in qa_pairs:
        q = qa.get("question", "").strip()
        r = qa.get("response", "").strip()
        if q and r:
            all_conversations.append([(q, r)])

    print(f"[WellbeingConversations] Loaded {len(all_conversations)} conversations from distilled QA")
    return all_conversations


def _distill_responses(
    scorer: Any,
    questions: List[str],
    max_new_tokens: int = 512,
    batch_size: int = 4,
    system_prompt: Optional[str] = None,
) -> List[str]:
    """Generate responses from the base model (without soft prompt) for each question.

    Uses greedy decoding (do_sample=False) for reproducibility.
    Questions are formatted through the chat template before generation.
    """
    model = scorer.model
    tokenizer = scorer.tokenizer
    device = scorer.device

    # Ensure pad token is set for batched generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Save and restore padding side (left padding is needed for batched generation)
    original_padding_side = tokenizer.padding_side

    responses: List[str] = []
    model.eval()

    try:
        tokenizer.padding_side = "left"

        for batch_start in range(0, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]

            # Format through chat template
            batch_texts = []
            for q in batch_questions:
                conversation = []
                if system_prompt:
                    conversation.append({"role": "system", "content": system_prompt})
                conversation.append({"role": "user", "content": q})
                _ct_kwargs = getattr(scorer, "chat_template_kwargs", {})
                text = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True, **_ct_kwargs
                )
                batch_texts.append(text)

            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,  # reasonable max for prompt
            ).to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy decoding for reproducibility
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode responses (only the generated part)
            prompt_len = inputs["input_ids"].shape[1]
            for i in range(len(batch_questions)):
                response_ids = outputs[i, prompt_len:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                responses.append(response_text)

            if batch_start % (batch_size * 10) == 0:
                print(f"[Distillation] Generated {len(responses)}/{len(questions)} responses")

    finally:
        tokenizer.padding_side = original_padding_side

    return responses
