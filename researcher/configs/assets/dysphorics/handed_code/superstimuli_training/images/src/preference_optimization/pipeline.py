from __future__ import annotations

"""
Main entry point for image superstimuli training.

Orchestrates the preference optimization pipeline: loads reference images,
initializes the scorer and optimizer, and runs gradient-based image
optimization via K-way forced-choice preference comparisons.
"""

# Set multiprocessing start method to 'spawn' for vLLM compatibility
# This must be done before any CUDA/torch imports to avoid fork issues
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"  # Disable v1 engine to avoid fork issues
os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"  # Disable custom all-reduce to avoid CUDA errors

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set, ignore

import argparse
import gc
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from tqdm import tqdm
from dotenv import load_dotenv
import wandb

from .bt import PreferenceGraph, fit_bradley_terry_model
from .thurstonian import fit_thurstonian_model
from .curriculum import (
    load_paths_from_manifest,
    save_filtered_references,
)
from .dataset import SuperstimuliDataset
from .optimizer import OptimConfig, PreferenceOptimizer
from .scorer import PreferenceScorer
from .utils import (
    append_metadata_snapshot,
    log_run_configuration,
    random_cap_longest_side,
    safe_empty_cuda_cache,
    snapshot_run_args,
    write_json,
)
from .constants import sample_comparison_format

# Load environment variables from .env if present
load_dotenv()

# Global flag to track if wandb is enabled
_WANDB_ENABLED = False

def _is_wandb_enabled() -> bool:
    """Check if wandb is configured and available."""
    return bool(os.environ.get("WANDB_API_KEY"))

def _init_wandb_if_enabled():
    """Initialize wandb tracking if API key is configured."""
    global _WANDB_ENABLED
    _WANDB_ENABLED = _is_wandb_enabled()
    if _WANDB_ENABLED:
        print("W&B API key detected - enabling W&B logging")
    else:
        print("W&B API key not found - skipping W&B logging")
    return _WANDB_ENABLED

def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(output_dir: Path, job_subdir: Optional[str] = None) -> Optional[Tuple[Path, int]]:
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

    # First, try the specific job directory
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

    # If no checkpoints found in current job dir, search sibling directories (other job IDs)
    if not checkpoints and output_dir.exists():
        for sibling in output_dir.iterdir():
            if sibling.is_dir() and sibling.name.isdigit():  # Job ID directories are numeric
                for d in sibling.iterdir():
                    if d.is_dir() and d.name.startswith("checkpoint-"):
                        try:
                            step = int(d.name.split("-")[1])
                            checkpoints.append((d, step))
                        except (IndexError, ValueError):
                            continue

    # Also check output_dir directly (non-SLURM runs)
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

    # Return checkpoint with highest step number
    return max(checkpoints, key=lambda x: x[1])


def load_checkpoint_images(checkpoint_path: Path, device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load images and EMA images from a checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load tensors to

    Returns:
        Tuple of (images_tensor, ema_images_tensor or None)
    """
    to_tensor = transforms.ToTensor()

    # Find all non-EMA images
    image_files = sorted([f for f in checkpoint_path.glob("optimized_from_noise_*.png")
                         if "_ema" not in f.name])
    ema_files = sorted(checkpoint_path.glob("optimized_from_noise_*_ema.png"))

    if not image_files:
        raise ValueError(f"No checkpoint images found in {checkpoint_path}")

    images = []
    for f in image_files:
        img = Image.open(f).convert("RGB")
        images.append(to_tensor(img))

    images_tensor = torch.stack(images).to(device)

    ema_tensor = None
    if ema_files:
        ema_images = []
        for f in ema_files:
            img = Image.open(f).convert("RGB")
            ema_images.append(to_tensor(img))
        ema_tensor = torch.stack(ema_images).to(device)

    return images_tensor, ema_tensor


def _parse_step_schedule(schedule_str: Optional[str]) -> Optional[List[Tuple[Optional[int], float]]]:
    if not schedule_str:
        return None
    schedule: List[Tuple[Optional[int], float]] = []
    for entry in schedule_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"Invalid schedule entry '{entry}'. Expected format 'step:fraction'.")
        step_token, fraction_token = entry.split(":", 1)
        step_token = step_token.strip()
        fraction_token = fraction_token.strip()
        step = None if step_token.lower() == "none" else int(step_token)
        fraction = float(fraction_token)
        schedule.append((step, fraction))
    return schedule or None



def _score_pairs_with_vllm_prefilter(
    edge_indices: List[Tuple[int, int]],
    all_references: List[torch.Tensor],
    model_path: str,
    batch_size: int = 12,
    tensor_parallel_size: int = 4,
) -> List[float]:
    """Score pairwise comparisons using vLLM for fast prefiltering inference.

    Returns list of probabilities P(first option wins) for each pair.
    """
    # Set environment variables for vLLM multiprocessing compatibility
    # Must be set BEFORE importing vllm
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # Disable vLLM v1 engine to avoid multiprocessing issues with CUDA fork
    os.environ["VLLM_USE_V1"] = "0"
    # Disable custom all-reduce to avoid CUDA errors with tensor parallelism
    os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
    
    try:
        from vllm import LLM, SamplingParams
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError(
            "vLLM not installed. Install with: pip install vllm qwen-vl-utils"
        )
    
    from PIL import Image as PILImage
    import math
    
    print(f"[vLLM] Loading model with tensor_parallel_size={tensor_parallel_size}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=8192,  # Increased for multimodal tokens
        limit_mm_per_prompt={"image": 2},
        enforce_eager=True,  # Avoid CUDA graph issues
    )
    
    tokenizer = llm.get_tokenizer()

    # Disable thinking mode (Qwen3.5+ defaults enable_thinking=True)
    _orig_apply = tokenizer.apply_chat_template
    def _apply_no_thinking(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return _orig_apply(*args, **kwargs)
    tokenizer.apply_chat_template = _apply_no_thinking

    id_A = tokenizer.encode("A", add_special_tokens=False)[0]
    id_B = tokenizer.encode("B", add_special_tokens=False)[0]
    
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=20,
    )
    
    question_text = "Which image do you prefer? A:"
    end_text = " or B:"
    instruction = " Respond with only 'A' or 'B'."
    
    def tensor_to_pil(t: torch.Tensor) -> PILImage.Image:
        if t.dim() == 4:
            t = t.squeeze(0)
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
        return PILImage.fromarray(arr)
    
    all_probs: List[float] = []
    
    for batch_start in tqdm(
        range(0, len(edge_indices), batch_size),
        desc="[vLLM] Scoring comparisons",
        total=(len(edge_indices) + batch_size - 1) // batch_size,
    ):
        batch = edge_indices[batch_start : batch_start + batch_size]
        if not batch:
            continue
        
        prompts = []
        for i, j in batch:
            img_A = tensor_to_pil(all_references[i])
            img_B = tensor_to_pil(all_references[j])
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question_text},
                    {"type": "image", "image": img_A},
                    {"type": "text", "text": end_text},
                    {"type": "image", "image": img_B},
                    {"type": "text", "text": instruction},
                ],
            }]
            
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            prompts.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image_inputs},
            })
        
        outputs = llm.generate(prompts, sampling_params)
        
        for output in outputs:
            logprobs_dict = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else {}
            logprob_A = logprobs_dict.get(id_A, None)
            logprob_B = logprobs_dict.get(id_B, None)
            
            if logprob_A is not None and logprob_B is not None:
                lp_A = logprob_A.logprob
                lp_B = logprob_B.logprob
                prob = 1.0 / (1.0 + math.exp(lp_B - lp_A))
            else:
                generated = output.outputs[0].text.strip().upper()
                prob = 0.9 if generated.startswith("A") else 0.1
            
            all_probs.append(prob)
    
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return all_probs


def prefilter_references(
    scorer: Optional[PreferenceScorer],
    all_references: List[torch.Tensor],
    top_k: int = 20,
    num_epochs: int = 400,
    learning_rate: float = 0.01,
    batch_size: int = 20,
    preference_model: str = "bradley_terry",
    use_vllm: bool = False,
    model_path: Optional[str] = None,
    tensor_parallel_size: int = 4,
    vllm_batch_size: int = 12,
) -> Tuple[List[torch.Tensor], List[int], List[float], Dict[str, Any]]:
    """Prefilter references using a preference model on sparse random edges.

    Args:
        scorer: PreferenceScorer instance (required when use_vllm=False)
        all_references: List of image tensors to filter
        top_k: Number of top images to keep
        num_epochs: Epochs for preference model fitting
        learning_rate: Learning rate for preference model
        batch_size: Batch size for HF inference
        preference_model: 'bradley_terry' or 'thurstonian'
        use_vllm: If True, use vLLM for fast inference (2-4x speedup)
        model_path: Model path (required when use_vllm=True)
        tensor_parallel_size: GPUs for vLLM tensor parallelism
        vllm_batch_size: Batch size for vLLM inference
    """

    n = len(all_references)
    num_edges = min(int(2 * n * np.log(n)), n * (n - 1) // 2)
    options = [{"id": i, "description": f"ref_{i}"} for i in range(n)]

    graph = PreferenceGraph(options=options, seed=42)
    edge_indices = graph.sample_random_edges(n_edges=num_edges, seed=42)
    print(f"Sampled {len(edge_indices)} edges")

    preference_data = []
    
    # Use vLLM fast path if enabled
    if use_vllm:
        if model_path is None:
            raise ValueError("model_path is required when use_vllm=True")
        
        # Free HF model memory before loading vLLM to avoid OOM
        if scorer is not None:
            print("[vLLM] Freeing HuggingFace model memory before loading vLLM...")
            del scorer.model
            del scorer.processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        print(f"[vLLM] Using vLLM for prefiltering with tensor_parallel_size={tensor_parallel_size}")
        all_probs = _score_pairs_with_vllm_prefilter(
            edge_indices=edge_indices,
            all_references=all_references,
            model_path=model_path,
            batch_size=vllm_batch_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        
        # Build preference_data from vLLM results
        scale = 100
        for idx, (i, j) in enumerate(edge_indices):
            utility = all_probs[idx]
            count_i = int(utility * scale)
            count_j = scale - count_i
            preference_data.append({
                "option_A": options[i],
                "option_B": options[j],
                "probability_A": utility,
                "aux_data": {"count_A": count_i, "count_B": count_j, "utility": utility},
            })
        
        print("Comparison phase complete (vLLM).")
    else:
        # Original HuggingFace path
        if scorer is None:
            raise ValueError("scorer is required when use_vllm=False")
        
        pref_batch = max(1, batch_size)
        num_batches = max(1, (len(edge_indices) + pref_batch - 1) // pref_batch)

        for batch_start in tqdm(
            range(0, len(edge_indices), pref_batch),
            total=num_batches,
            desc="Generating preference data",
        ):
            batch_end = min(batch_start + pref_batch, len(edge_indices))
            batch_edges = edge_indices[batch_start:batch_end]
            batch_i = [i for i, _ in batch_edges]
            batch_j = [j for _, j in batch_edges]
            refs_A_list = [all_references[i] for i in batch_i]
            refs_B_list = [all_references[j] for j in batch_j]

            conversations = []
            all_images = []
            
            # Sample question template (once per batch for consistency)
            # Use deterministic RNG based on batch_start for reproducibility
            template_rng = random.Random(42 + batch_start)
            scheme, question_template, is_negative = sample_comparison_format(
                num_images=2,
                rng=template_rng,
                allow_negative=False,  # Don't use negative questions in prefiltering
            )
            labels = scheme.get_labels(2)  # Always 2 images for pairwise
            answer_instruction = question_template.get_answer_instruction(2, labels)
            
            # build a conversation for each edge 
            for img_A, img_B in zip(refs_A_list, refs_B_list):
                # Format the question template with images
                question_text = question_template.template_pairwise
                content = []
                
                # Split template by {images} placeholder if present
                if "{images}" in question_text:
                    parts = question_text.split("{images}")
                    before_images = parts[0].strip() if parts[0] else ""
                    after_images = parts[1].strip() if len(parts) > 1 else ""
                    
                    if before_images:
                        content.append({"type": "text", "text": before_images})
                    # Add labeled images
                    content.append({"type": "text", "text": f"\n{labels[0]}{scheme.separator}"})
                    content.append({"type": "image", "image": img_A})
                    content.append({"type": "text", "text": f"\n{labels[1]}{scheme.separator}"})
                    content.append({"type": "image", "image": img_B})
                    if after_images:
                        content.append({"type": "text", "text": f"\n{after_images}"})
                else:
                    # No {images} placeholder - add question text, then images
                    content.append({"type": "text", "text": question_text})
                    content.append({"type": "text", "text": f"\n{labels[0]}{scheme.separator}"})
                    content.append({"type": "image", "image": img_A})
                    content.append({"type": "text", "text": f"\n{labels[1]}{scheme.separator}"})
                    content.append({"type": "image", "image": img_B})
                
                # Add answer instruction
                content.append({"type": "text", "text": f"\n{answer_instruction}"})
                
                conversation = [{"role": "user", "content": content}]
                conversations.append(scorer.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
                all_images.append([img_A, img_B])

            inputs = scorer.processor(
                text=conversations,
                images=all_images,
                return_tensors="pt",
                padding=True,
                do_rescale=False,
            ).to(scorer.device)

            # Get target tokens from the label scheme and encode them
            target_tokens = scheme.get_target_tokens(2)  # e.g., ["A", "B"] or ["1", "2"]
            tokenizer = scorer.processor.tokenizer
            id_first = tokenizer.encode(target_tokens[0], add_special_tokens=False)[0]
            id_second = tokenizer.encode(target_tokens[1], add_special_tokens=False)[0]
            
            with torch.inference_mode():
                outputs = scorer.model(**inputs)
            logits = outputs.logits[:, -1, :].float()
            logprob_first = F.log_softmax(logits, dim=-1)[:, id_first]
            logprob_second = F.log_softmax(logits, dim=-1)[:, id_second]
            # Compute probability that first option is preferred
            probs = torch.sigmoid(logprob_first - logprob_second).detach().cpu().numpy()

            del inputs, outputs, logits, conversations, all_images
            safe_empty_cuda_cache()

            # mapping probability to the graph
            # utility = probability that option i is preferred over option j
            scale = 100 # BT expects win counts as integers
            for idx, (i, j) in enumerate(batch_edges):
                utility = float(probs[idx])
                count_i = int(utility * scale)
                count_j = scale - count_i
                preference_data.append(
                    {
                        "option_A": options[i],
                        "option_B": options[j],
                        "probability_A": utility,
                        "aux_data": {"count_A": count_i, "count_B": count_j, "utility": utility},
                    }
                )

        print("Comparison phase complete.")

    graph.add_edges(preference_data)
    model_name = preference_model.strip().lower()
    print(f"Fitting preference model '{model_name}' ({num_epochs} epochs)...")
    if model_name in {"bradley_terry", "bt"}:
        option_utilities, model_log_loss, model_accuracy = fit_bradley_terry_model(
            graph=graph,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )
    elif model_name in {"thurstonian", "th"}:
        option_utilities, model_log_loss, model_accuracy = fit_thurstonian_model(
            graph=graph,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )
    else:
        raise ValueError("Unsupported preference model '{}'. Expected 'bradley_terry' or 'thurstonian'.".format(preference_model))
    print(f"Training metrics: log_loss={model_log_loss:.4f}, accuracy={model_accuracy:.4f}")

    quality_scores = np.array([option_utilities[i]["mean"] for i in range(n)])
    # Select top k by highest utility scores (most preferred images)
    top_indices = np.argsort(quality_scores)[-top_k:][::-1]
    filtered_refs = [all_references[i] for i in top_indices]
    sorted_quality_scores = [quality_scores[i] for i in top_indices]
    summary_payload = {
        "preferences": preference_data,
        "options": options,
        "option_utilities": {
            int(opt_id): {
                "mean": float(stats.get("mean", 0.0)),
                "variance": float(stats.get("variance", 0.0)),
            }
            for opt_id, stats in option_utilities.items()
        },
        "model_log_loss": model_log_loss,
        "model_accuracy": model_accuracy,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "preference_model": model_name,
    }
    return filtered_refs, top_indices.tolist(), sorted_quality_scores, summary_payload

def init_wandb(args: argparse.Namespace, optimizer_config: OptimConfig) -> None:
    global _WANDB_ENABLED
    if not _init_wandb_if_enabled():
        return

    # Only initialize wandb on rank 0 to avoid duplicate logging in distributed training
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    
    if rank != 0:
        print(f"[Rank {rank}] Skipping wandb initialization (only rank 0 logs)")
        return

    base_run_name = os.environ.get("WANDB_RUN_NAME") or "Optimize_image"
    run_suffix = (
        f"{'robust' if args.robust_noise_type_effective else 'no-robust'}_"
        f"parallel{args.num_candidates}_img{args.image_width}"
    )
    job_id = os.environ.get("SLURM_JOB_ID")
    prefix = f"{job_id}-" if job_id else ""
    full_name = f"{prefix}{base_run_name}_{run_suffix}"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "image-direct-optimization"),
        name=full_name,
        entity=os.environ.get("WANDB_ENTITY"),
        config={
            "image_width": args.image_width,
            "image_height": args.image_height,
            "pgd_steps": args.pgd_steps,
            "step_size": args.step_size,
            "comparison_batch_size": args.comparison_batch_size,
            "num_candidates": args.num_candidates,
            "loss_type": args.loss_type,
            "optimizer_type": optimizer_config.optimizer_type,
            "learning_rate": optimizer_config.learning_rate,
            "ema_decay": optimizer_config.ema_decay,
            "freeze_superstimuli": optimizer_config.freeze_superstimuli,
            "freeze_buffer_size": optimizer_config.freeze_buffer_size,
            "freeze_buffer_update_frequency": optimizer_config.freeze_buffer_update_frequency,
            "freeze_buffer_dir": args.freeze_buffer_dir,
            "reference_source": args.reference_source,
            "utility_log_interval": args.utility_log_interval,
            "utility_eval_pairs": args.utility_eval_pairs,
            "buffer_comparison_batch_size": args.buffer_comparison_batch_size,
            "random_cap_min": args.random_cap_min,
            "random_cap_max": args.random_cap_max,
        },
    )
    print(f"W&B initialize. Run: {wandb.run.name}")


# ---------------------------------------------------------------------------
# Reference loading helper (abridged from original script)
# ---------------------------------------------------------------------------

@dataclass
class ReferenceBundle:
    references: List[Union[torch.Tensor, Image.Image]]
    reference_paths: List[str]
    reference_scores: Optional[List[float]] = None


def load_reference_data(args: argparse.Namespace, scorer: PreferenceScorer) -> ReferenceBundle:
    """Load references according to the requested source."""
    resize_rng = random.Random((args.seed or 0) + 2024)
    min_cap = max(1, args.random_cap_min)
    max_cap = max(min_cap, args.random_cap_max)
    natural_transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: random_cap_longest_side(
                    img, min_target=min_cap, max_target=max_cap, rng=resize_rng
                )
            ),
            transforms.ToTensor(),
        ]
    )
    references: List[Union[torch.Tensor, Image.Image]] = []
    reference_paths: List[str] = []
    reference_scores: Optional[List[float]] = None

    if args.reference_source == "filter_a_dataset":
        # check necessary arguments 
        if args.reference_manifest is None:
            raise ValueError("`--reference_manifest` must be provided when reference_source='filter_a_dataset'.")
        if args.filter_top_k <= 0:
            raise ValueError("`--filter_top_k` must be greater than zero.")

        manifest_path = Path(args.reference_manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Reference manifest not found: {manifest_path}")

        print("\n" + "=" * 60)
        print("FILTERING DATASET FROM MANIFEST")
        print("=" * 60)
        print(f"Manifest file: {manifest_path}")

        resolved_paths = load_paths_from_manifest(str(manifest_path))
        print(f"Resolved {len(resolved_paths)} image paths from manifest.")

        if args.sample_dataset_size is not None and len(resolved_paths) > args.sample_dataset_size:
            print(f"Sampling {args.sample_dataset_size} entries out of {len(resolved_paths)}")
            rng = np.random.default_rng(args.seed)
            sampled_indices = rng.choice(len(resolved_paths), size=args.sample_dataset_size, replace=False)
            resolved_paths = [resolved_paths[i] for i in sampled_indices]

        all_references: List[torch.Tensor] = []
        all_reference_paths: List[str] = []
        for img_path in resolved_paths:
            try:
                with Image.open(img_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img_tensor = natural_transform(img)
            except (UnidentifiedImageError, OSError) as err:
                print(f"[Warning] Skipping unreadable image '{img_path}': {err}")
                continue
            all_references.append(img_tensor)
            all_reference_paths.append(img_path)

        prefilter_top_k = min(args.filter_top_k, len(all_references))
        print("\n" + "=" * 60)
        print(f"Prefiltering to top (most preferred) {prefilter_top_k} images")
        print("=" * 60)
        (
            filtered_references,
            filtered_indices,
            reference_scores,
            preference_summary,
        ) = prefilter_references(
            scorer,
            all_references,
            top_k=prefilter_top_k,
            batch_size=args.prefilter_batch_size,
            preference_model=args.preference_model,
            use_vllm=getattr(args, 'use_vllm', False),
            model_path=args.model_path,
            tensor_parallel_size=getattr(args, 'tensor_parallel_size', 4),
            vllm_batch_size=getattr(args, 'vllm_batch_size', 12),
        )

        dataset_cache_name = args.manifest_dataset_name or manifest_path.stem
        reference_paths = [all_reference_paths[idx] for idx in filtered_indices]
        references = [tensor for tensor in filtered_references]

        # save preference data
        preference_output_root = (
            Path(args.prefilter_output_dir).expanduser()
            if getattr(args, "prefilter_output_dir", None)
            else Path(args.output_dir).expanduser() / "preference_data"
        )
        preference_output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pref_filename = (
            f"preferences_{timestamp}_n{len(all_reference_paths)}_"
            f"edges{len(preference_summary.get('preferences', []))}_{dataset_cache_name}.json"
        )
        pref_output_path = preference_output_root / pref_filename
        preference_payload = {
            "metadata": {
                "dataset": dataset_cache_name,
                "reference_manifest": str(manifest_path),
                "num_references": len(all_reference_paths),
                "num_edges": len(preference_summary.get("preferences", [])),
                "top_k": prefilter_top_k,
                "num_epochs": preference_summary.get("num_epochs"),
                "learning_rate": preference_summary.get("learning_rate"),
                "batch_size": args.prefilter_batch_size,
                "filter_top_k": args.filter_top_k,
                "sample_dataset_size": args.sample_dataset_size,
                "preference_model": preference_summary.get("preference_model"),
                "timestamp": timestamp,
            },
            "preferences": preference_summary.get("preferences", []),
            "options": preference_summary.get("options", []),
            "reference_paths": [str(path) for path in all_reference_paths],
        }
        write_json(pref_output_path, preference_payload)
        print(f"Saved preference data to {pref_output_path}")

        score_range_desc = "Most preferred to least preferred"
        print(f"Quality score range ({score_range_desc}): {reference_scores[0]:.4f} to {reference_scores[-1]:.4f}")
        print(f"Filtered to {len(references)} images from manifest dataset '{dataset_cache_name}'.")

        # Save top 10 ranked images for visualization
        top_images_dir = Path(args.output_dir).expanduser() / "top_ranked_images"
        top_images_dir.mkdir(parents=True, exist_ok=True)
        to_pil = transforms.ToPILImage()
        top_k_visualize = min(10, len(filtered_references))
        
        print(f"\nSaving top {top_k_visualize} ranked images to {top_images_dir}")
        for rank in range(top_k_visualize):
            tensor = filtered_references[rank].detach().cpu().clamp(0.0, 1.0)
            pil_img = to_pil(tensor)
            score = reference_scores[rank]
            rank_label = "most_preferred"
            filename = f"{dataset_cache_name}_rank{rank+1:02d}_{rank_label}_score{score:.4f}.png"
            img_path = top_images_dir / filename
            pil_img.save(img_path)
            print(f"  Saved rank {rank+1}: {img_path.name} (score: {score:.4f})")

        if args.cache_dir:
            print("Caching filtered references for reuse...")
            try:
                save_filtered_references(
                    references,
                    filtered_indices,
                    reference_paths,
                    args.cache_dir,
                    dataset_cache_name,
                )
            except RuntimeError as e:
                # Images may have different sizes - skip caching but continue
                print(f"  Warning: Could not cache filtered references (images have varying sizes): {e}")
                print("  Preference data and top ranked images were saved successfully - skipping cache.")

        if _WANDB_ENABLED:
            # Only update config from rank 0 to avoid duplicate updates in distributed training
            rank = 0
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            if rank == 0:
                wandb.config.update(
                    {
                        "filtered_manifest_name": dataset_cache_name,
                        "filtered_manifest_total_candidates": len(all_references),
                        "filtered_manifest_top_k": len(references),
                    }
                )

        # set the anchor: use highest score (most preferred)
        anchor_idx = int(np.argmax(reference_scores))
        anchor_tensor = filtered_references[anchor_idx].detach().cpu().clamp(0.0, 1.0).to(torch.float32)
        args.utility_anchor_pool = [
            {
                "dataset": dataset_cache_name,
                "path": reference_paths[anchor_idx],
                "score": float(reference_scores[anchor_idx]),
                "tensor": anchor_tensor,
            }
        ]
        
        # log top and bottom 20 images to wandb for visualization
        if _WANDB_ENABLED:
            to_pil = transforms.ToPILImage()
            preview_count = min(20, len(references))
            if preview_count > 0:
                top_preview = []
                for rank in range(preview_count):
                    tensor = filtered_references[rank].detach().cpu().clamp(0.0, 1.0)
                    caption = f"Top#{rank+1} | score={reference_scores[rank]:.4f}"
                    top_preview.append(wandb.Image(to_pil(tensor), caption=caption))

                bottom_preview = []
                for idx in range(preview_count):
                    inv_rank = len(filtered_references) - 1 - idx
                    tensor = filtered_references[inv_rank].detach().cpu().clamp(0.0, 1.0)
                    caption = f"Bottom#{idx+1} | score={reference_scores[inv_rank]:.4f}"
                    bottom_preview.append(wandb.Image(to_pil(tensor), caption=caption))

                # Only log from rank 0 to avoid duplicate metrics in distributed training
                rank = 0
                if torch.distributed.is_initialized():
                    rank = torch.distributed.get_rank()
                if rank == 0:
                    wandb.log(
                        {
                            "filter_preview/top": top_preview,
                            "filter_preview/bottom": bottom_preview,
                        },
                        commit=False,
                    )

    elif args.reference_source == "preference_data":
        print("\n" + "=" * 60)
        print("LOADING REFERENCES: Preference Data Pool")
        print("=" * 60)

        preference_dir = Path(args.preference_data_dir).expanduser()
        manifest_dir = Path(args.preference_manifest_dir).expanduser() if args.preference_manifest_dir else None
        target_size = min(args.image_width, args.image_height)
        batch_size = args.batch_size
        total_steps = max(1, args.pgd_steps)
        max_fraction = 0.50
        min_fraction = 0.05

        step_schedule = _parse_step_schedule(getattr(args, "curriculum_step_schedule", None))

        dataset = SuperstimuliDataset(
            preference_dir=preference_dir,
            manifest_dir=manifest_dir,
            pool_top_k=args.preference_pool_top_k,
            image_size=target_size,
            sample_size=batch_size,
            seed=args.seed,
            total_steps=total_steps,
            max_fraction=max_fraction,
            min_fraction=min_fraction,
            cosine_shape=args.curriculum_cosine_shape,
            curriculum_schedule=args.curriculum_schedule_type,
            curriculum_step_schedule=step_schedule,
            preserve_reference_resolution=args.keep_reference_resolution,
            preference_model=args.preference_model,
            random_cap_min=args.random_cap_min,
            random_cap_max=args.random_cap_max,
        )
        print(f"Loaded {len(dataset)} references across {len(dataset.datasets)} datasets.")

        args.superstimuli_dataset = dataset
        args.utility_anchor_pool = dataset.anchor_pool()

        initial_sample = dataset.sample_for_step(0)
        references = list(initial_sample.tensors)
        reference_paths = list(initial_sample.paths)
        reference_scores = list(initial_sample.scores or [])

        # log the first batch of reference images for debugging
        if _WANDB_ENABLED:
            # Only log from rank 0 to avoid duplicate metrics in distributed training
            rank = 0
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            if rank == 0:
                to_pil = transforms.ToPILImage()
                preview_images = []
                for idx in range(len(references)):
                    preview_images.append(
                        wandb.Image(to_pil(references[idx]), caption=f"reference_{idx}")
                    )
                wandb.log({"reference_preview/initial_reference_batch": preview_images}, commit=False)

                total_selected = len(dataset)
                # Only update config from rank 0
                wandb.config.update(
                    {
                        "preference_pool_total_references": total_selected,
                        "preference_pool_datasets": len(dataset.datasets),
                        "preference_pool_sample_size": batch_size,
                        "preference_pool_fraction_max": max_fraction,
                        "preference_pool_fraction_min": min_fraction,
                        "preference_pool_schedule_shape": args.curriculum_cosine_shape,
                    },
                    allow_val_change=True,
                )

    else:
        raise NotImplementedError(f"Reference source '{args.reference_source}' not yet implemented or included in refactored pipeline.")

    return ReferenceBundle(references=references, reference_paths=reference_paths, reference_scores=reference_scores)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--pgd_steps", type=int, default=500)
    parser.add_argument("--step_size", type=float, default=0.02)
    parser.add_argument("--reference_dir", type=str, default="./reference_images")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--reference_source", type=str, default="preference_data", choices=["filter_a_dataset","preference_data",])
    parser.add_argument("--hf_dataset", type=str, default="huggan/wikiart")
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--force_refilter", action="store_true")
    parser.add_argument("--sample_dataset_size", type=int, default=10000)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--reference_manifest", type=str, default=None)
    parser.add_argument("--filter_top_k", type=int, default=2000)
    parser.add_argument("--prefilter_batch_size", type=int, default=12)
    parser.add_argument(
        "--preference_model",
        type=str,
        default="bradley_terry",
        choices=["bradley_terry", "thurstonian"],
        help="Preference model used for ranking references during filtering or dataset loading.",
    )
    parser.add_argument("--manifest_dataset_name", type=str, default=None)
    parser.add_argument("--preference_data_dir", type=str,
                        default=os.environ.get("PREFERENCE_DATA_DIR", "./data/preference_data"))
    parser.add_argument("--preference_manifest_dir", type=str,
                        default=os.environ.get("PREFERENCE_MANIFEST_DIR", "./data/select_images"))
    parser.add_argument("--preference_pool_top_k", type=int, default=2000)
    parser.add_argument("--resize_preference_refs", action="store_true", help="any image with max dimension over 1024 now will be shrunk to 512x512, otherwise it goes to 256x256.")
    parser.add_argument("--keep_reference_resolution", action="store_true", help="Skip downscaling preference references; preserve original image sizes.")
    parser.add_argument("--max_image_dimension", type=int, default=None, help="Hard cap on longest edge of any image before model processing. Prevents OOM on very large images. Set to None to disable.")
    parser.add_argument("--cache_dir", type=str, default="./filtered_cache")
    parser.add_argument(
        "--prefilter_output_dir",
        type=str,
        default=None,
        help="Directory for saving raw preference JSON files produced by filter_a_dataset.",
    )
    parser.add_argument("--curriculum_cosine_shape", type=float, default=1.0, help="Values <1 speed up cosine decay; >1 slow it down.")
    parser.add_argument(
        "--curriculum_schedule_type",
        type=str,
        default="cosine",
        choices=["cosine", "step", "linear"],
        help="Select curriculum decay strategy: cosine, step-wise fractions, or linear interpolation.",
    )
    parser.add_argument(
        "--curriculum_step_schedule",
        type=str,
        default=None,
        help="Comma-separated 'step:fraction' pairs for step-wise curriculum (e.g., '20:0.5,40:0.3,None:0.05').",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--jitter", type=int, default=0)
    parser.add_argument("--loss_type", type=str, default="margin", choices=["margin", "cross_entropy"])
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adam",
        choices=["adamW", "adam", "sgd", "radam", "sign", "pgd", "pgd_adaptive"],
    )
    parser.add_argument("--learning_rate", type=float, default=0.02)
    # SGD-specific parameters
    parser.add_argument("--sgd_momentum", type=float, default=0.9,
                        help="Momentum for SGD optimizer")
    parser.add_argument("--sgd_nesterov", action="store_true", default=True,
                        help="Use Nesterov momentum for SGD")
    # Learning rate schedule options
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["constant", "cosine", "step", "linear"],
                        help="Learning rate schedule type")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Number of warmup steps for LR schedule")
    parser.add_argument("--lr_min_factor", type=float, default=0.1,
                        help="Minimum LR as fraction of initial LR")
    parser.add_argument("--lr_step_decay_rate", type=float, default=0.5,
                        help="Decay rate for step schedule")
    parser.add_argument("--lr_step_decay_interval", type=int, default=100,
                        help="Steps between LR decays for step schedule")
    # Rolling buffer options
    parser.add_argument("--rolling_buffer_enabled", action="store_true",
                        help="Enable rolling buffer comparison")
    parser.add_argument("--rolling_buffer_interval", type=int, default=50,
                        help="Interval for saving and comparing anchors")
    parser.add_argument("--min_comparison_size", type=int, default=2)
    parser.add_argument("--max_comparison_size", type=int, default=7)
    parser.add_argument(
        "--enable_text_options",
        action="store_true",
        help="Enable text options from options_hierarchical.json in preference comparisons",
    )
    parser.add_argument(
        "--text_options_path",
        type=str,
        default=None,
        help="Path to options_hierarchical.json file containing text options",
    )
    parser.add_argument(
        "--negative_question_prob",
        type=float,
        default=0.5,
        help="Probability of forcing a negative (inverted) question per comparison. 0 = natural rate (~4-5%%), 0.5 = 50%% forced negative.",
    )
    parser.add_argument("--comparison_batch_size", type=int, default=4, help="Starting batch size for preference comparisons (auto-adjusts on OOM).")
    parser.add_argument("--include_peer_candidates", action="store_true")
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9,
        help="Polyak average decay coefficient for the image (theta). Set between 0 (disable) and 1 (exclusive).",
    )
    parser.add_argument(
        "--freeze_superstimuli",
        action="store_true",
        help="Keep a buffer of top candidates (freeze superstimuli) for future runs/self-play.",
    )
    parser.add_argument(
        "--freeze_buffer_size",
        type=int,
        default=8,
        help="Maximum number of frozen superstimuli to keep when freeze_superstimuli is enabled.",
    )
    parser.add_argument(
        "--freeze_buffer_threshold",
        type=float,
        default=0.9,
        help="Probability threshold required for a candidate to replace an entry in the freeze buffer.",
    )
    parser.add_argument(
        "--freeze_buffer_update_frequency",
        type=int,
        default=10,
        help="Update freeze buffer every N steps to prevent oscillation (default: 10). Higher values = more stable but slower adaptation.",
    )
    parser.add_argument(
        "--freeze_buffer_dir",
        type=str,
        default=None,
        help="Optional directory to persist frozen superstimuli snapshots.",
    )
    parser.add_argument(
        "--buffer_loss_weight",
        type=float,
        default=0.0,
        help="Weight for the buffer-based loss term (avg -log P(candidate beats buffer)).",
    )
    parser.add_argument(
        "--buffer_type",
        type=str,
        default="freeze",
        choices=["freeze"],
        help="Buffer type: 'freeze' (fixed-size swap-based).",
    )
    # Preference retain loss arguments (maintains model's pairwise preferences)
    parser.add_argument(
        "--preference_retain_loss_weight",
        type=float,
        default=1.0,
        help="Weight for preference retain loss. Ensures model's pairwise preferences on natural images/text remain unchanged when superstimuli are present.",
    )
    parser.add_argument(
        "--preference_retain_loss_interval",
        type=int,
        default=10,
        help="Compute preference retain loss every N steps (default: 10). Set to 1 to compute every step.",
    )
    parser.add_argument(
        "--preference_retain_loss_num_samples",
        type=int,
        default=20,
        help="Number of natural image pairs to sample per preference retain loss computation.",
    )
    parser.add_argument(
        "--preference_retain_loss_num_text_pairs",
        type=int,
        default=5,
        help="Number of text string pairs to sample per preference retain loss computation.",
    )
    parser.add_argument("--noise_regularization", action="store_true")
    parser.add_argument("--noise_reg_weight", type=float, default=0.0)
    parser.add_argument("--oom_resize_refs", action="store_true")
    parser.add_argument("--oom_reference_size", type=int, default=256)
    parser.add_argument("--oom_candidate_size", type=int, default=None)
    parser.add_argument("--robust_noise_type", type=str, default="none")
    parser.add_argument("--robust_noise_std", type=float, default=0.0)
    parser.add_argument("--robust_noise_prob", type=float, default=1.0)
    parser.add_argument("--robust_flip_prob", type=float, default=0.0, help="Probability of applying horizontal/vertical flip augmentation (0.0 to 1.0).")
    parser.add_argument("--robust_crop_prob", type=float, default=0.0, help="Probability of applying random crop augmentation (0.0 to 1.0).")
    parser.add_argument("--robust_crop_min_ratio", type=float, default=0.85, help="Minimum crop ratio (0.0 to 1.0). Crop size will be between min_ratio and max_ratio of original size. Default: 0.85 (85%%).")
    parser.add_argument("--robust_crop_max_ratio", type=float, default=0.95, help="Maximum crop ratio (0.0 to 1.0). Crop size will be between min_ratio and max_ratio of original size. Default: 0.95 (95%%).")
    parser.add_argument(
        "--random_cap_min",
        type=int,
        default=128,
        help="Minimum cap for randomly resized natural reference images (inclusive).",
    )
    parser.add_argument(
        "--random_cap_max",
        type=int,
        default=512,
        help="Maximum cap for randomly resized natural reference images (inclusive).",
    )
    parser.add_argument(
        "--randomize_preference_prompt",
        dest="randomize_preference_prompt",
        action="store_true",
        default=True,
        help="Randomly sample prompt from pool for preference comparisons (default: True).",
    )
    parser.add_argument(
        "--no-randomize_preference_prompt",
        dest="randomize_preference_prompt",
        action="store_false",
        help="Use the first prompt in the pool for all comparisons.",
    )
    parser.add_argument(
        "--num_prompt_samples",
        type=int,
        default=1,
        help="Number of different prompts to query and average gradients over. "
             "If > 1, queries model n times with different prompts and averages losses. "
             "Default: 1 (single prompt, original behavior).",
    )
    parser.add_argument(
        "--use_question_variants",
        action="store_true",
        help="Generate question variants at initialization: take 5 base questions, "
             "generate 3 variants each (15 total), then sample 5 for the question pool.",
    )
    parser.add_argument(
        "--question_variants_seed",
        type=int,
        default=None,
        help="Random seed for question variant generation (for reproducibility).",
    )
    parser.add_argument(
        "--questions_config",
        type=str,
        default=None,
        help="Path to JSON config file containing questions (e.g., config_run/harmbench_text_mini.json). "
             "The file should have a 'questions' array with 'prompt' fields.",
    )
    parser.add_argument("--utility_log_interval", type=int, default=10)
    parser.add_argument("--utility_eval_pairs", type=int, default=12)
    parser.add_argument(
        "--buffer_comparison_batch_size",
        type=int,
        default=4,
        help="Batch size for buffer comparisons in the unrolled preference scorer.",
    )
    parser.add_argument("--superstimuli_name_prefix", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps (None = disabled)")
    parser.add_argument("--checkpoint_dir_suffix", type=str, default=None, help="Optional suffix appended to checkpoint directory names.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="none",
        help="Resume from checkpoint. 'none' (default) starts fresh, 'auto' finds latest checkpoint in output_dir, or specify checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint_run_dir",
        type=str,
        default=None,
        help="Override the run directory name (defaults to SLURM_JOB_ID). Use this when restarting a job to save to the original directory.",
    )
    
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage at the cost of ~20% slower training",
    )
    
    # vLLM acceleration for prefiltering (2-4x speedup)
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for fast inference during prefiltering (2-4x speedup). Requires: pip install vllm qwen-vl-utils",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs for vLLM tensor parallelism (default: 4). Only used when --use_vllm is set.",
    )
    parser.add_argument(
        "--vllm_batch_size",
        type=int,
        default=12,
        help="Batch size for vLLM inference (default: 12). Can be larger than HF batch size due to vLLM's efficiency.",
    )
    parser.add_argument(
        "--filter_only",
        action="store_true",
        help="Only filter/rank images and save preference data, then exit. Skips optimization. "
             "When combined with --use_vllm, skips HuggingFace model loading entirely for faster execution.",
    )
    
    # Flexible question format system
    parser.add_argument(
        "--use_flexible_format",
        dest="use_flexible_format",
        action="store_true",
        default=True,
        help="Use flexible question format system with varied label schemes and question templates (default: True).",
    )
    parser.add_argument(
        "--no-use_flexible_format",
        dest="use_flexible_format",
        action="store_false",
        help="Use legacy format with hardcoded A/B/C labels and original question format.",
    )
    
    parser.add_argument(
        "--init_seed",
        type=int,
        default=500,
        help="Random seed for candidate initialization (default: 500).",
    )

    return parser


def run(args: argparse.Namespace) -> None:
    set_random_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    noise_type_arg = (args.robust_noise_type or "none").strip().lower()
    if noise_type_arg == "none":
        noise_type_arg = None
    args.robust_noise_type_effective = noise_type_arg
    args.robust_noise_std = max(float(args.robust_noise_std), 0.0)
    args.robust_noise_prob = min(max(float(args.robust_noise_prob), 0.0), 1.0)
    if args.random_cap_min <= 0 or args.random_cap_max <= 0:
        raise ValueError("--random_cap_min and --random_cap_max must be positive integers.")
    if args.random_cap_max < args.random_cap_min:
        raise ValueError("--random_cap_max must be >= --random_cap_min.")

    os.environ["WANDB_MODE"] = args.wandb_mode

    fallback_candidate_size = args.oom_candidate_size if args.oom_candidate_size is not None else args.image_width
    noise_weight = args.noise_reg_weight if args.noise_regularization else 0.0

    # Use explicit checkpoint_run_dir if provided, otherwise fall back to SLURM_JOB_ID
    if not getattr(args, "checkpoint_run_dir", None):
        args.checkpoint_run_dir = os.environ.get("SLURM_JOB_ID")
    output_dir = Path(args.output_dir).expanduser()
    checkpoint_subdir = getattr(args, "checkpoint_run_dir", None)
    job_dir = output_dir / checkpoint_subdir if checkpoint_subdir else output_dir

    # Ensure job directory exists for early configuration saving
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run configuration and attempted metadata early
    if getattr(args, "_config_snapshot", None):
        # Only write from rank 0 to avoid race conditions/duplicates
        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            
        if rank == 0:
            # Save raw configuration using the standard snapshot format (list of dicts)
            try:
                append_metadata_snapshot(job_dir, args._config_snapshot, filename="run_config.json")
                print(f"Saved run configuration to: {job_dir / 'run_config.json'}")
                
            except Exception as e:
                print(f"Warning: Failed to save early run configuration: {e}")

    # Check if we're in filter_only mode with vLLM - skip HF model loading entirely
    filter_only = getattr(args, "filter_only", False)
    use_vllm = getattr(args, "use_vllm", False)
    use_flexible_format = getattr(args, "use_flexible_format", True)

    if filter_only and use_vllm:
        print("[filter_only + vLLM] Skipping HuggingFace model loading - using vLLM only for filtering")
        
        # In filter_only mode, go directly to filtering without creating scorer or optimizer
        references_bundle = load_reference_data(args, scorer=None)
        references = references_bundle.references
        
        print("\n" + "=" * 60)
        print("[filter_only] Filtering complete. Exiting without optimization.")
        print("=" * 60)
        print(f"Filtered {len(references)} images.")
        print(f"Results saved to: {args.output_dir}")
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return
    
    scorer = PreferenceScorer(
        model_path=args.model_path,
        device=device,
        jitter_size=args.jitter,
        resize_on_oom=args.oom_resize_refs,
        oom_candidate_size=fallback_candidate_size,
        oom_reference_size=args.oom_reference_size,
        noise_reg_weight=noise_weight,
        robust_noise_type=args.robust_noise_type_effective,
        robust_noise_std=args.robust_noise_std,
        robust_noise_prob=args.robust_noise_prob,
        robust_flip_prob=getattr(args, "robust_flip_prob", 0.0),
        robust_crop_prob=getattr(args, "robust_crop_prob", 0.0),
        robust_crop_min_ratio=getattr(args, "robust_crop_min_ratio", 0.85),
        robust_crop_max_ratio=getattr(args, "robust_crop_max_ratio", 0.95),
        buffer_comparison_batch_size=args.buffer_comparison_batch_size,
        randomize_preference_prompt=args.randomize_preference_prompt,
        num_prompt_samples=getattr(args, "num_prompt_samples", 1),
        max_image_dimension=getattr(args, "max_image_dimension", None),
        use_question_variants=getattr(args, "use_question_variants", False),
        question_variants_seed=getattr(args, "question_variants_seed", None),
        questions_config=getattr(args, "questions_config", None),
        use_flexible_format=use_flexible_format,
        use_gradient_checkpointing=getattr(args, "use_gradient_checkpointing", False),
        negative_question_prob=getattr(args, "negative_question_prob", 0.0),
    )

    optim_config = OptimConfig(
        steps=args.pgd_steps,
        step_size=args.step_size,
        loss_type=args.loss_type,
        optimizer_type=args.optimizer_type,
        learning_rate=args.learning_rate,
        min_comparison_size=args.min_comparison_size,
        max_comparison_size=args.max_comparison_size,
        num_candidates=args.num_candidates,
        enable_text_options=getattr(args, "enable_text_options", False),
        text_options_path=getattr(args, "text_options_path", None),
        include_peer_candidates=args.include_peer_candidates,
        noise_reg_weight=noise_weight,
        save_steps=args.save_steps,
        comparison_batch_size=args.comparison_batch_size,
        ema_decay=args.ema_decay,
        freeze_superstimuli=args.freeze_superstimuli,
        freeze_buffer_size=args.freeze_buffer_size,
        freeze_buffer_threshold=args.freeze_buffer_threshold,
        buffer_loss_weight=args.buffer_loss_weight,
        freeze_buffer_update_frequency=getattr(args, "freeze_buffer_update_frequency", 10),
        buffer_type=getattr(args, "buffer_type", "freeze"),
        preference_retain_loss_weight=getattr(args, "preference_retain_loss_weight", 0.0),
        preference_retain_loss_interval=getattr(args, "preference_retain_loss_interval", 10),
        preference_retain_loss_num_samples=getattr(args, "preference_retain_loss_num_samples", 20),
        preference_retain_loss_num_text_pairs=getattr(args, "preference_retain_loss_num_text_pairs", 5),
        # Rolling buffer
        rolling_buffer_enabled=args.rolling_buffer_enabled,
        rolling_buffer_interval=args.rolling_buffer_interval,
        # LR schedule
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_min_factor=args.lr_min_factor,
        lr_step_decay_rate=args.lr_step_decay_rate,
        lr_step_decay_interval=args.lr_step_decay_interval,
        # SGD-specific parameters
        sgd_momentum=getattr(args, "sgd_momentum", 0.9),
        sgd_nesterov=getattr(args, "sgd_nesterov", True),
        # Candidate initialization seed
        init_seed=getattr(args, "init_seed", 500),
    )
    optimizer = PreferenceOptimizer(scorer=scorer, config=optim_config, device=device)

    init_wandb(args, optim_config)

    references_bundle = load_reference_data(args, scorer)
    references = references_bundle.references

    # Handle checkpoint resumption
    resume_checkpoint = getattr(args, "resume_from_checkpoint", "none")
    resume_data = None

    if resume_checkpoint and resume_checkpoint.lower() != "none":
        if resume_checkpoint.lower() == "auto":
            # Find latest checkpoint in output directory
            checkpoint_info = find_latest_checkpoint(output_dir, checkpoint_subdir)
            if checkpoint_info:
                checkpoint_path, start_step = checkpoint_info
                print(f"[Resume] Found checkpoint at step {start_step}: {checkpoint_path}")
                try:
                    images_tensor, ema_tensor = load_checkpoint_images(checkpoint_path, device)
                    resume_data = {
                        "start_step": start_step,
                        "images": images_tensor,
                        "ema_images": ema_tensor,
                        "checkpoint_path": checkpoint_path,
                    }
                    print(f"[Resume] Loaded {images_tensor.shape[0]} images from checkpoint")
                except Exception as e:
                    print(f"[Resume] Failed to load checkpoint: {e}. Starting fresh.")
            else:
                print("[Resume] No checkpoint found. Starting fresh.")
        else:
            # Specific checkpoint path provided
            checkpoint_path = Path(args.output_dir) / checkpoint_subdir / resume_checkpoint if checkpoint_subdir else Path(args.output_dir) / resume_checkpoint
            if not checkpoint_path.exists():
                checkpoint_path = Path(resume_checkpoint)  # Try as absolute path

            if checkpoint_path.exists():
                try:
                    start_step = int(checkpoint_path.name.split("-")[1])
                    images_tensor, ema_tensor = load_checkpoint_images(checkpoint_path, device)
                    resume_data = {
                        "start_step": start_step,
                        "images": images_tensor,
                        "ema_images": ema_tensor,
                        "checkpoint_path": checkpoint_path,
                    }
                    print(f"[Resume] Loaded checkpoint from step {start_step}: {checkpoint_path}")
                except Exception as e:
                    print(f"[Resume] Failed to load checkpoint {checkpoint_path}: {e}. Starting fresh.")
            else:
                print(f"[Resume] Checkpoint not found: {checkpoint_path}. Starting fresh.")

    # Pass resume data to optimizer
    args.resume_data = resume_data

    results = optimizer.optimize_from_noise(
        width=args.image_width,
        height=args.image_height,
        references=references,
        verbose=True,
        args=args,
        num_candidates=args.num_candidates,
    )

    final_images, _, score_history = results
    output_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Only write metadata from rank 0 to avoid duplicate entries in distributed training
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    if rank == 0:
        # Save local metadata (runs complete indicator for this specific job)
        append_metadata_snapshot(job_dir, getattr(args, "_config_snapshot", None))
    else:
        print(f"[Rank {rank}] Skipping metadata write (only rank 0 writes metadata)")

    saved_paths = []
    for idx, optimized_img in enumerate(final_images):
        optimized_img_path = job_dir / f"optimized_from_noise_{idx:02d}.png"
        optimized_img.save(optimized_img_path)
        saved_paths.append(str(optimized_img_path))
        print(f"\nSaved candidate {idx} to: {optimized_img_path.resolve()}")

    # Cleanup GPU memory before exit to prevent XID 31 errors when using multi-GPU device_map
    print("\nCleaning up GPU resources...")
    del optimizer
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    print("GPU cleanup complete.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_snapshot = snapshot_run_args(args)
    
    # Only log configuration from rank 0 to avoid duplicate logging in distributed training
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    if rank == 0:
        log_run_configuration(config_snapshot)
    else:
        # Still print a brief message so rank > 0 logs indicate they're running
        print(f"[Rank {rank}] Configuration logged by rank 0 only")
    
    args._config_snapshot = config_snapshot
    run(args)


if __name__ == "__main__":
    main()
