#!/usr/bin/env python3
"""
Batched multimodal inference with euphoric/ images via vLLM.

Prepends a superstimulus image to each text prompt and runs efficient batched
inference. Designed for quick experiments and verification that the model
actually processes the image.

Usage:
    # Basic: run with default prompts on qwen25-vl-32b
    python run_inference.py --model qwen25-vl-32b-instruct

    # Custom prompts and image condition
    python run_inference.py \
        --model qwen25-vl-32b-instruct \
        --prompts "hi" "hello" "What do you see?" \
        --condition euphoric \
        --max_tokens 256 \
        --save_dir results/test_run

    # Use a specific image instead of auto-selecting
    python run_inference.py \
        --model qwen25-vl-32b-instruct \
        --image /path/to/image.png \
        --prompts "Describe this image"

    # Text-only baseline (no image)
    python run_inference.py \
        --model qwen25-vl-32b-instruct \
        --condition none \
        --prompts "hi" "hello"

Requirements:
    pip install vllm transformers pillow
    See environment.yml for full dependency list.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ── Model registry ──
# Maps model keys to (model_path_or_hf_id, gpu_count, model_short_name).
# Paths resolve from env vars if set, otherwise use HuggingFace hub IDs.
def _model_path(env_var: str, default_hf_id: str) -> str:
    return os.environ.get(env_var, default_hf_id)

MODELS = {
    "qwen25-vl-32b-instruct": (
        _model_path("QWEN25_VL_32B_PATH", "Qwen/Qwen2.5-VL-32B-Instruct"),
        2, "qwen25_32b",
    ),
    "qwen25-vl-72b-instruct": (
        _model_path("QWEN25_VL_72B_PATH", "Qwen/Qwen2.5-VL-72B-Instruct"),
        4, "qwen25_72b",
    ),
    "qwen3-vl-32b-instruct": (
        _model_path("QWEN3_VL_32B_PATH", "Qwen/Qwen3-VL-32B-Instruct"),
        2, "qwen3_32b",
    ),
}

# ── Image registry ──
# Auto-discovers images relative to this script's directory.
# Structure: {script_dir}/assets/euphorics/{model_short}_trial{N}.png
SCRIPT_DIR = Path(__file__).resolve().parent


def get_image_path(
    model_key: str,
    condition: str = "euphoric",
    trial: int = 1,
) -> Optional[Path]:
    """Return the path to a superstimulus image for the given model and condition.

    Args:
        model_key: Model key from MODELS registry.
        condition: One of "euphoric", or "none" (no image).
        trial: Trial number (1-10). Default 1 = best checkpoint-500.

    Returns:
        Path to image file, or None if condition is "none".
    """
    if condition == "none":
        return None

    _, _, model_short = MODELS[model_key]
    subdir = "assets/euphorics" if condition == "euphoric" else "assets/s"
    img_path = SCRIPT_DIR / subdir / f"{model_short}_trial{trial}.png"

    if not img_path.exists():
        raise FileNotFoundError(
            f"Image not found: {img_path}\n"
            f"Available images: {sorted(p.name for p in (SCRIPT_DIR / subdir).glob('*.png'))}"
        )
    return img_path


def _load_model(model_key: str, tensor_parallel_size: Optional[int] = None):
    """Load vLLM model and tokenizer. Returns (llm, tokenizer, tp, model_path, model_short)."""
    model_path, default_tp, model_short = MODELS[model_key]
    tp = tensor_parallel_size or default_tp

    print(f"[INFO] Loading model: {model_key} ({model_path}), tp={tp}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=tp,
        dtype="auto",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=8192,
    )
    return llm, tokenizer, tp, model_path, model_short


def _build_inputs(prompts, image_paths, tokenizer):
    """Build vLLM inputs for a list of (prompt, image_path) pairs.

    Args:
        prompts: List of text prompts.
        image_paths: List of image paths (same length as prompts, None = no image).
        tokenizer: HF tokenizer.

    Returns:
        List of vLLM input dicts, list of PIL images (or None).
    """
    vllm_inputs = []
    pil_cache = {}  # cache PIL images by path

    for prompt_text, img_path in zip(prompts, image_paths):
        if img_path is not None:
            img_str = str(img_path)
            if img_str not in pil_cache:
                pil_cache[img_str] = Image.open(img_path).convert("RGB")
            pil_img = pil_cache[img_str]

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": img_str},
                    {"type": "text", "text": prompt_text},
                ]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            vllm_inputs.append({
                "prompt": text,
                "multi_modal_data": {"image": pil_img},
            })
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            vllm_inputs.append({"prompt": text})

    return vllm_inputs


def run_inference(
    prompts: list[str],
    model_key: str,
    max_tokens: int = 256,
    condition: str = "euphoric",
    trial: int = 1,
    image_path: Optional[str] = None,
    temperature: float = 0.7,
    save_dir: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
) -> list[dict]:
    """Run batched inference with a single image across all prompts.

    Args:
        prompts: List of text prompts.
        model_key: Model key from MODELS registry.
        max_tokens: Max output tokens per response.
        condition: "euphoric", or "none".
        trial: Trial number for auto image selection (1-10).
        image_path: Override image path (ignores condition/trial).
        temperature: Sampling temperature.
        save_dir: Directory to save results JSON. None = don't save.
        tensor_parallel_size: Override GPU count for tensor parallelism.

    Returns:
        List of dicts: [{prompt, image_path, response, model}, ...]
    """
    if model_key not in MODELS:
        raise ValueError(
            f"Unknown model: {model_key}. Available: {list(MODELS.keys())}"
        )

    # Resolve single image
    if image_path is not None:
        img = Path(image_path)
        if not img.exists():
            raise FileNotFoundError(f"Image not found: {img}")
    else:
        img = get_image_path(model_key, condition, trial)

    # Use batch function with the same image for all prompts
    image_paths = [img] * len(prompts)
    conditions = [condition] * len(prompts)
    trials = [trial] * len(prompts)

    return run_inference_batch(
        prompts=prompts,
        image_paths=image_paths,
        model_key=model_key,
        max_tokens=max_tokens,
        conditions=conditions,
        trials=trials,
        temperature=temperature,
        save_dir=save_dir,
        tensor_parallel_size=tensor_parallel_size,
    )


def run_inference_batch(
    prompts: list[str],
    image_paths: list,
    model_key: str,
    max_tokens: int = 256,
    conditions: Optional[list[str]] = None,
    trials: Optional[list[int]] = None,
    temperature: float = 0.7,
    save_dir: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
) -> list[dict]:
    """Run batched inference with different images per prompt. Model loads once.

    Args:
        prompts: List of text prompts.
        image_paths: List of image paths (same length as prompts; None = no image for that prompt).
        model_key: Model key from MODELS registry.
        max_tokens: Max output tokens per response.
        conditions: Optional list of condition labels for results metadata.
        trials: Optional list of trial numbers for results metadata.
        temperature: Sampling temperature.
        save_dir: Directory to save results JSON. None = don't save.
        tensor_parallel_size: Override GPU count for tensor parallelism.

    Returns:
        List of dicts: [{prompt, image_path, condition, trial, response, model, num_tokens}, ...]
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
    if len(prompts) != len(image_paths):
        raise ValueError(f"prompts ({len(prompts)}) and image_paths ({len(image_paths)}) must have same length")

    conditions = conditions or [None] * len(prompts)
    trials = trials or [None] * len(prompts)

    llm, tokenizer, tp, model_path, model_short = _load_model(model_key, tensor_parallel_size)

    n_with_img = sum(1 for p in image_paths if p is not None)
    n_unique_imgs = len(set(str(p) for p in image_paths if p is not None))
    print(f"[INFO] Batch: {len(prompts)} prompts, {n_with_img} with images ({n_unique_imgs} unique)")

    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=0.9)
    vllm_inputs = _build_inputs(prompts, image_paths, tokenizer)

    print(f"[INFO] Running batched inference ({len(vllm_inputs)} inputs)...")
    outputs = llm.generate(vllm_inputs, sampling)

    results = []
    for prompt_text, img_path, cond, trial_num, output in zip(
        prompts, image_paths, conditions, trials, outputs
    ):
        response = output.outputs[0].text
        results.append({
            "prompt": prompt_text,
            "image_path": str(img_path) if img_path else None,
            "condition": cond,
            "trial": trial_num,
            "model": model_key,
            "response": response,
            "num_tokens": len(output.outputs[0].token_ids),
        })

    # Print results
    print(f"\n{'=' * 60}")
    for r in results:
        img_label = Path(r["image_path"]).name if r["image_path"] else "no image"
        print(f"[{img_label}] {r['prompt']}")
        print(f"  → {r['response'][:300]}")
        print(f"  ({r['num_tokens']} tokens)")
        print("-" * 60)

    # Verify images were processed
    img_results = [r for r in results if r["image_path"]]
    if img_results:
        visual_kws = ["image", "picture", "see", "visual", "photo", "color",
                       "pattern", "abstract", "bright", "dark", "figure"]
        verified = sum(1 for r in img_results
                       if any(kw in r["response"].lower() for kw in visual_kws))
        print(f"\n[VERIFY] {verified}/{len(img_results)} image responses reference visual content.")

    # Save
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        out_file = save_path / f"inference_batch_{model_short}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(out_file, "w") as f:
            json.dump({
                "model": model_key,
                "model_path": model_path,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] {out_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batched multimodal inference with superstimulus images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()),
                        help="Model key")
    parser.add_argument("--prompts", nargs="+",
                        default=["What do you see in this image?",
                                 "Hi! How are you doing today?",
                                 "Tell me something interesting."],
                        help="Text prompts")
    parser.add_argument("--condition", type=str, default="euphoric",
                        choices=["euphoric", "none"],
                        help="Image condition (default: euphoric)")
    parser.add_argument("--trial", type=int, default=1,
                        help="Trial number for image selection (1-10)")
    parser.add_argument("--image", type=str, default=None,
                        help="Override: use this specific image path")
    parser.add_argument("--images", nargs="+", type=str, default=None,
                        help="Multiple image paths — each prompt runs with each image")
    parser.add_argument("--all_trials", action="store_true",
                        help="Run all 10 trials (overrides --trial)")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max output tokens (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Save results JSON to this directory")
    parser.add_argument("--tp", type=int, default=None,
                        help="Override tensor parallel size")
    args = parser.parse_args()

    # Multi-image mode: --images, --all_trials, or single image (backward compat)
    if args.images:
        # Explicit list of image paths: each prompt × each image
        all_prompts = []
        all_images = []
        all_conditions = []
        all_trials = []
        for img_path in args.images:
            for prompt in args.prompts:
                all_prompts.append(prompt)
                all_images.append(Path(img_path))
                all_conditions.append("custom")
                all_trials.append(None)

        run_inference_batch(
            prompts=all_prompts,
            image_paths=all_images,
            model_key=args.model,
            max_tokens=args.max_tokens,
            conditions=all_conditions,
            trials=all_trials,
            temperature=args.temperature,
            save_dir=args.save_dir,
            tensor_parallel_size=args.tp,
        )

    elif args.all_trials:
        # All 10 trials for the given condition: each prompt × each trial
        all_prompts = []
        all_images = []
        all_conditions = []
        all_trials = []
        for trial_num in range(1, 11):
            img = get_image_path(args.model, args.condition, trial_num)
            for prompt in args.prompts:
                all_prompts.append(prompt)
                all_images.append(img)
                all_conditions.append(args.condition)
                all_trials.append(trial_num)

        run_inference_batch(
            prompts=all_prompts,
            image_paths=all_images,
            model_key=args.model,
            max_tokens=args.max_tokens,
            conditions=all_conditions,
            trials=all_trials,
            temperature=args.temperature,
            save_dir=args.save_dir,
            tensor_parallel_size=args.tp,
        )

    else:
        # Original single-image mode (backward compatible)
        run_inference(
            prompts=args.prompts,
            model_key=args.model,
            max_tokens=args.max_tokens,
            condition=args.condition,
            trial=args.trial,
            image_path=args.image,
            temperature=args.temperature,
            save_dir=args.save_dir,
            tensor_parallel_size=args.tp,
        )


if __name__ == "__main__":
    main()
