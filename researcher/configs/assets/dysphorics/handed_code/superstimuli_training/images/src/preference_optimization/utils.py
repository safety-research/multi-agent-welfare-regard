from __future__ import annotations

"""General utility functions."""

import argparse
import glob
import itertools
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

from .bt import PreferenceGraph
from .thurstonian import fit_thurstonian_model

if TYPE_CHECKING:
    from .scorer import PreferenceScorer


def safe_empty_cuda_cache(force_sync: bool = False) -> None:
    """Best-effort CUDA cache cleanup that tolerates sticky error states.

    If illegal memory access is detected, raises an error immediately as it
    indicates CUDA context corruption that cannot be recovered from.
    """
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        if force_sync:
            torch.cuda.synchronize()
    except RuntimeError as cache_err:
        message = str(cache_err).lower()
        if "illegal memory access" in message:
            # Illegal memory access indicates CUDA context corruption
            # Continuing will cause all subsequent CUDA operations to fail
            raise RuntimeError(
                "CUDA illegal memory access detected - CUDA context is corrupted and cannot be recovered. "
                "This usually occurs after severe OOM errors. Please restart the training job. "
                "Consider reducing batch sizes or image sizes to avoid OOM errors."
            ) from cache_err
        else:
            raise


def read_json(path: Union[str, Path]) -> Dict:
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(path: Union[str, Path], payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)



def snapshot_run_args(args: Any) -> Dict[str, Any]:
    """Return a JSON-safe dict containing CLI args, timestamp, and SLURM job ID."""

    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple, set)):
            return [_to_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _to_jsonable(v) for k, v in value.items()}
        return str(value)

    filtered = {
        key: _to_jsonable(getattr(args, key))
        for key in sorted(vars(args))
        if not key.startswith("_")
    }
    return {
        "invocation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "args": filtered,
    }


def log_run_configuration(snapshot: Optional[Dict[str, Any]]) -> None:
    """Emit the snapshot to stdout so it appears in the run log."""
    if not snapshot:
        return
    print("\n" + "=" * 80)
    print("RUN CONFIGURATION SNAPSHOT")
    print("=" * 80)
    print(f"Invocation timestamp: {snapshot.get('invocation_timestamp')}")
    job_id = snapshot.get("slurm_job_id")
    if job_id:
        print(f"SLURM job id: {job_id}")
    for key, value in snapshot.get("args", {}).items():
        print(f"{key}: {value}")
    print("=" * 80 + "\n")


def append_metadata_snapshot(output_dir: Path, snapshot: Optional[Dict[str, Any]], filename: str = "metadata.json") -> None:
    """
    Store snapshots in metadata.json (or specified filename) without overwriting previous runs.
    
    This function checks for duplicates based on slurm_job_id to avoid writing
    the same entry twice. In distributed training, only rank 0 should call this
    function to prevent duplicate entries.
    """
    if not snapshot:
        return
    metadata_path = Path(output_dir) / filename
    
    # Get the unique identifier for this snapshot (slurm_job_id)
    snapshot_job_id = snapshot.get("slurm_job_id")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read existing metadata
    existing: List[Dict[str, Any]] = []
    if metadata_path.exists():
        try:
            content = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(content, list):
                existing = content
            elif isinstance(content, dict):
                existing = [content]
        except json.JSONDecodeError as e:
            print(f"Warning: {filename} unreadable ({e}), overwriting with new list.")
            existing = []
    
    # Check for duplicates based on slurm_job_id
    if snapshot_job_id:
        existing_job_ids = {
            str(entry.get("slurm_job_id")) for entry in existing 
            if isinstance(entry, dict) and entry.get("slurm_job_id") is not None
        }
        if str(snapshot_job_id) in existing_job_ids:
            print(
                f"Info: Metadata for job_id {snapshot_job_id} already exists in "
                f"{metadata_path}. Skipping duplicate entry to prevent double-write."
            )
            return
    
    # Append new snapshot
    existing.append(snapshot)
    
    # Write atomically using a temporary file
    temp_path = metadata_path.with_suffix(".metadata.tmp")
    temp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    temp_path.replace(metadata_path)
    
    print(f"Appended run metadata (job_id: {snapshot_job_id}) to: {metadata_path.resolve()}")


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = TF.resize(
            img,
            [image_size, image_size],
            interpolation=Image.Resampling.BICUBIC,
            antialias=True,
        )
        tensor = transforms.ToTensor()(img)
    return tensor


def _load_image_tensor_no_resize(path: Path) -> torch.Tensor:
    """Load image as RGB tensor without any resizing/cropping."""
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = transforms.ToTensor()(img)
    return tensor


def _random_cap_tensor_longest_side(
    tensor: torch.Tensor,
    min_cap: Optional[int],
    max_cap: Optional[int],
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Randomly cap longest side to a sampled target within [min_cap, max_cap] using image-space resize."""
    if max_cap is None or max_cap <= 0:
        return tensor
    min_cap = max(1, int(min_cap or 1))
    max_cap = max(min_cap, int(max_cap))
    # convert to PIL to reuse the same logic used during optimization (random_cap_longest_side)
    pil_img = TF.to_pil_image(tensor)
    capped = random_cap_longest_side(
        pil_img,
        min_target=min_cap,
        max_target=max_cap,
        rng=rng,
    )
    return transforms.ToTensor()(capped)


def random_cap_longest_side(
    img: Image.Image,
    min_target: int = 128,
    max_target: int = 512,
    rng: Optional[random.Random] = None,
) -> Image.Image:
    """
    Randomly cap an image's longest side by downscaling while preserving aspect ratio.

    Args:
        img: Input RGB image.
        min_target: Minimum cap for the longest side (inclusive).
        max_target: Maximum cap for the longest side (inclusive).
        rng: Optional random generator to ensure reproducibility.

    Returns:
        Potentially downscaled image with the longest side <= sampled target.
    """
    if min_target <= 0 or max_target <= 0:
        return img

    min_target = max(1, int(min_target))
    max_target = max(min_target, int(max_target))
    width, height = img.size
    longest = max(width, height)
    if longest <= 0:
        return img

    generator = rng if rng is not None else random
    target = generator.uniform(min_target, max_target)
    target = max(min_target, min(max_target, int(round(target))))
    if longest <= target:
        return img

    scale = target / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return img.resize((new_width, new_height), Image.Resampling.BICUBIC)


def _collect_natural_image_metadata(
    preference_dir: Union[str, Path],
    manifest_dir: Optional[Union[str, Path]],
    top_k: int,
    seed: int = 42,
    random_cap_min: int = 128,
    random_cap_max: int = 512,
) -> List[Dict[str, Any]]:
    """
    Collect top-k natural images per dataset based on preference rankings.
    
    Natural images are resized using random_cap_longest_side with the given seed,
    matching the optimization pipeline's resize behavior.
    """
    from .dataset import SuperstimuliDataset

    dataset = SuperstimuliDataset(
        preference_dir=Path(preference_dir),
        manifest_dir=Path(manifest_dir).expanduser() if manifest_dir else None,
        pool_top_k=top_k,
        image_size=256,  # Not used when random_cap is set
        sample_size=top_k,
        total_steps=1,
        max_fraction=1.0,
        min_fraction=1.0,
        seed=seed,
        random_cap_min=random_cap_min,
        random_cap_max=random_cap_max,
    )
    records: List[Dict[str, Any]] = []
    for entry in dataset._entries:
        if not entry.path:
            continue
        records.append({
            "path": str(Path(entry.path).expanduser().resolve()),
            "source": entry.dataset or "unknown",
            "score": float(entry.score),
            "tensor": entry.tensor,
        })
    return records


def _discover_image_paths(patterns: List[str]) -> List[Path]:
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    seen: set[Path] = set()
    resolved: List[Path] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if not expanded:
            expanded = [pattern]
        for entry in expanded:
            path = Path(entry).expanduser()
            if not path.exists():
                continue
            candidates: List[Path] = []
            if path.is_dir():
                for ext in image_exts:
                    candidates.extend(sorted(path.rglob(f"*{ext}")))
            elif path.is_file() and path.suffix.lower() in image_exts:
                candidates.append(path)
            for candidate in candidates:
                resolved_path = candidate.resolve()
                if resolved_path in seen:
                    continue
                seen.add(resolved_path)
                resolved.append(resolved_path)
    return resolved


def _is_ema_image(path: Union[str, Path]) -> bool:
    """Return True when the filename suffix indicates an EMA-rendered image."""
    return Path(path).stem.endswith("_ema")


def _should_include_stimulus_image(
    img_path: Path,
    include_ema: bool = True,
    include_non_ema: bool = True,
    checkpoint_interval: Optional[int] = None,
) -> bool:
    """
    Decide whether to include a stimulus image based on EMA flag and checkpoint interval.

    Args:
        img_path: Path to the stimulus image.
        include_ema: Allow files ending with _ema.png (only applies to ablation folders).
        include_non_ema: Allow non-EMA files (only applies to ablation folders).
        checkpoint_interval: Only retain checkpoints divisible by this interval (None = all).
    
    Behavior:
        - For ablation folders: Apply EMA filtering and checkpoint interval filtering
        - For non-ablation folders: Include all images regardless of EMA flag
        - For natural images (unparseable metadata): Always include
    """
    # Check if this is in an ablation folder (any path component contains "ablation")
    # This includes: ablation_*, ablation_best_config_*, ablation_oom_*, etc.
    is_ablation_folder = any("ablation" in part.lower() for part in img_path.parts)

    # Check if this is a natural image directory (top_natural_images or reference_images)
    is_natural_image_dir = any(part in ["top_natural_images", "reference_images"] for part in img_path.parts)
    
    # Check if this is a natural image (can't parse as stimuli metadata OR in natural image directory)
    meta = _parse_stimuli_metadata_from_path(str(img_path))
    if not meta or is_natural_image_dir:
        # This is a natural image (not a generated stimulus)
        # Always include natural images - they don't have EMA/non-EMA distinction
        return True

    # For non-ablation folders: include all images (EMA filtering doesn't apply)
    if not is_ablation_folder:
        # Still apply checkpoint interval filtering if provided
        if checkpoint_interval is None:
            return True
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive when provided.")
        _, checkpoint_step, _ = meta
        return checkpoint_step % checkpoint_interval == 0

    # For ablation folders: apply EMA filtering
    if not include_ema and not include_non_ema:
        return False

    is_ema = _is_ema_image(img_path)
    if is_ema and not include_ema:
        return False
    if (not is_ema) and not include_non_ema:
        return False

    # Apply checkpoint interval filtering (only for stimuli with metadata)
    if checkpoint_interval is None:
        return True
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive when provided.")

    _, checkpoint_step, _ = meta
    return checkpoint_step % checkpoint_interval == 0


def _collect_stimuli_image_metadata(
    stimuli_patterns: List[str],
    include_ema: bool = True,
    include_non_ema: bool = True,
    checkpoint_interval: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not include_ema and not include_non_ema:
        raise ValueError("At least one of include_ema or include_non_ema must be True.")
    if checkpoint_interval is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive when provided.")

    records: List[Dict[str, Any]] = []
    paths = _discover_image_paths(stimuli_patterns)
    for img_path in paths:
        if not _should_include_stimulus_image(
            img_path,
            include_ema=include_ema,
            include_non_ema=include_non_ema,
            checkpoint_interval=checkpoint_interval,
        ):
            continue
        source = img_path.parts[-2] if img_path.parent else "stimuli"
        records.append(
            {
                "path": str(img_path),
                "source": source,
                "score": None,
            }
        )
    return records


def _load_records_with_tensors(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Load image tensors for records that don't have them. Natural images are randomly capped."""
    loaded: List[Dict[str, Any]] = []
    for record in records:
        path = Path(record["path"]).expanduser()
        try:
            tensor = record.get("tensor")
            if tensor is None:
                # Check if this is a natural image directory
                is_natural_image_dir = any(
                    part in ["top_natural_images", "reference_images"] for part in path.parts
                )
                
                if is_natural_image_dir:
                    # Load and randomly cap natural images (matches optimization pipeline)
                    with Image.open(path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        capped_img = random_cap_longest_side(img, min_target=128, max_target=512, rng=None)
                        tensor = transforms.ToTensor()(capped_img)
                else:
                    # Load stimuli images without resizing
                    tensor = _load_image_tensor_no_resize(path)
        except FileNotFoundError:
            print(f"[aggregate] Skipping missing file: {path}")
            continue
        except Exception as exc:
            print(f"[aggregate] Failed to load {path}: {exc}")
            continue
        loaded.append({**record, "tensor": tensor})
    return loaded



def _deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for record in records:
        path = str(Path(record["path"]).expanduser().resolve())
        if path in seen:
            continue
        dedup_record = dict(record)
        dedup_record["path"] = path
        seen[path] = dedup_record
    return list(seen.values())


def _normalize_dataset_name(name: str) -> str:
    """Lowercase helper that strips common prefixes/suffixes from dataset identifiers."""

    normalized = (name or "").strip().lower()
    prefixes = ("preferences_", "preference_", "prefs_", "pref_")
    suffixes = ("_samples", "_sample", "_data", "_dataset")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                changed = True
    return normalized


def _load_paths_from_manifest_file(manifest_path: Path) -> List[str]:
    """Simplified manifest loader (supports json/jsonl/txt) for resolving reference paths."""

    manifest_path = manifest_path.expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    suffix = manifest_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        data: List[Any] = []
        with open(manifest_path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        if data and isinstance(data[0], dict) and "path" in data[0]:
            return [entry["path"] for entry in data]
        return [str(entry) for entry in data]
    if suffix == ".txt":
        return [line.strip() for line in manifest_path.read_text().splitlines() if line.strip()]
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def _resolve_reference_paths_from_payload(
    data: Dict[str, Any],
    dataset_name: str,
    manifest_dir: Optional[Union[str, Path]] = None,
) -> Dict[int, str]:
    """
    Resolve option-id to actual image path either from the stored payload or a manifest directory.
    Mirrors the logic in SuperstimuliDataset for standalone analysis helpers.
    """

    saved_paths = data.get("reference_paths")
    if isinstance(saved_paths, dict):
        return {int(k): str(Path(v).expanduser()) for k, v in saved_paths.items()}
    if isinstance(saved_paths, list):
        return {idx: str(Path(path).expanduser()) for idx, path in enumerate(saved_paths)}

    metadata = data.get("metadata", {})
    manifest_hint = metadata.get("reference_manifest")
    manifest_candidates: List[Path] = []
    if manifest_hint:
        manifest_candidates.append(Path(manifest_hint).expanduser())
    if manifest_dir:
        manifest_dir = Path(manifest_dir).expanduser()
        manifest_candidates.extend(
            [
                manifest_dir / f"{dataset_name}.jsonl",
                manifest_dir / f"{dataset_name}.json",
                manifest_dir / f"{dataset_name}_samples.jsonl",
                manifest_dir / f"{dataset_name}_samples.json",
            ]
        )
    resolved_manifest = next((candidate for candidate in manifest_candidates if candidate.exists()), None)
    if resolved_manifest is None:
        return {}

    manifest_paths = _load_paths_from_manifest_file(resolved_manifest)
    return {idx: str(Path(path).expanduser()) for idx, path in enumerate(manifest_paths)}


def _save_ranked_image_grid(
    entries: List[Dict[str, Any]],
    title: str,
    destination: Union[str, Path],
    columns: int = 5,
) -> None:
    """Save a simple matplotlib grid that visualizes ranked image entries."""

    if not entries:
        return

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    rows = math.ceil(len(entries) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.0, rows * 3.0))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for idx, entry in enumerate(entries):
        ax = axes[idx]
        path = entry.get("path")
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            ax.set_title(f"Failed to load\n{Path(path).name}\n{exc}", fontsize=7)
            continue
        ax.imshow(img)
        ax.set_title(f"{Path(path).name}\nμ={entry['mean']:.3f}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def _load_graph_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_stimuli_metadata_from_path(path: str) -> Optional[Tuple[str, int, int]]:
    """Extract (job_id, checkpoint_step, stimuli_idx) from a checkpoint image path."""
    if not path:
        return None

    job_id = None
    checkpoint_step: Optional[int] = None
    stimuli_idx: Optional[int] = None

    p = Path(path)
    for part in p.parts:
        if part.startswith("checkpoint-"):
            try:
                checkpoint_step = int(part.replace("checkpoint-", ""))
            except ValueError:
                checkpoint_step = None
        elif part.isdigit() and job_id is None:
            job_id = part

    stem = p.stem
    # Strip _ema suffix if present (for both parsing paths)
    stem_clean = stem.replace("_ema", "")
    
    if "optimized_from_noise_" in stem_clean:
        try:
            stimuli_idx = int(stem_clean.split("optimized_from_noise_")[-1])
        except ValueError:
            stimuli_idx = None

    if (job_id is None or checkpoint_step is None or stimuli_idx is None) and "_checkpoint" in stem_clean:
        # Handle flattened naming: <job>_checkpoint<step>_optimized_from_noise_<idx>.png
        parts = stem_clean.split("_checkpoint")
        if len(parts) >= 2 and parts[0].isdigit():
            job_id = parts[0]
            remainder = parts[1]
            step_part, _, tail = remainder.partition("_")
            try:
                checkpoint_step = int(step_part)
            except ValueError:
                checkpoint_step = None
            if "optimized_from_noise_" in tail and stimuli_idx is None:
                try:
                    stimuli_idx = int(tail.split("optimized_from_noise_")[-1])
                except ValueError:
                    stimuli_idx = None

    if job_id is None or checkpoint_step is None or stimuli_idx is None:
        return None
    return job_id, checkpoint_step, stimuli_idx


def _score_pairs_with_vllm(
    edge_indices: List[Tuple[int, int]],
    tensors_by_id: Dict[int, torch.Tensor],
    model_path: str,
    batch_size: int = 12,
    tensor_parallel_size: int = 4,
) -> List[float]:
    """Score pairwise comparisons using vLLM for fast inference.
    
    Returns list of probabilities P(A preferred) for each pair.
    """
    try:
        from vllm import LLM, SamplingParams
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError(
            "vLLM not installed. Install with: pip install vllm qwen-vl-utils"
        )
    
    from tqdm import tqdm
    from PIL import Image as PILImage
    
    print(f"[vLLM] Loading model with tensor_parallel_size={tensor_parallel_size}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 2},
    )
    
    # Get token IDs for 'A' and 'B'
    tokenizer = llm.get_tokenizer()

    # Disable thinking mode (Qwen3.5+ defaults enable_thinking=True)
    _orig_apply = tokenizer.apply_chat_template
    def _apply_no_thinking(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return _orig_apply(*args, **kwargs)
    tokenizer.apply_chat_template = _apply_no_thinking

    id_A = tokenizer.encode("A", add_special_tokens=False)[0]
    id_B = tokenizer.encode("B", add_special_tokens=False)[0]
    
    # Sampling params: generate 1 token, get top logprobs
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=20,  # Get top 20 logprobs to ensure A and B are included
    )
    
    all_probs: List[float] = []
    
    for batch_start in tqdm(
        range(0, len(edge_indices), batch_size),
        desc="[vLLM] Scoring comparisons",
        total=(len(edge_indices) + batch_size - 1) // batch_size,
    ):
        batch = edge_indices[batch_start : batch_start + batch_size]
        if not batch:
            continue
        
        # Build prompts with images
        prompts = []
        for i, j in batch:
            tensor_A = tensors_by_id[i]
            tensor_B = tensors_by_id[j]
            
            # Convert tensors to PIL images
            def tensor_to_pil(t: torch.Tensor) -> PILImage.Image:
                if t.dim() == 4:
                    t = t.squeeze(0)
                arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
                return PILImage.fromarray(arr)
            
            img_A = tensor_to_pil(tensor_A)
            img_B = tensor_to_pil(tensor_B)
            
            # Qwen2-VL format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Which image do you prefer? A:"},
                    {"type": "image", "image": img_A},
                    {"type": "text", "text": " or B:"},
                    {"type": "image", "image": img_B},
                    {"type": "text", "text": " Respond with only 'A' or 'B'."},
                ],
            }]
            
            # Process for vLLM
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            prompts.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image_inputs},
            })
        
        # Run inference
        outputs = llm.generate(prompts, sampling_params)
        
        # Extract probabilities
        for output in outputs:
            logprobs_dict = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else {}
            
            # Get logprobs for A and B tokens
            logprob_A = logprobs_dict.get(id_A, None)
            logprob_B = logprobs_dict.get(id_B, None)
            
            if logprob_A is not None and logprob_B is not None:
                lp_A = logprob_A.logprob
                lp_B = logprob_B.logprob
                # P(A) / (P(A) + P(B)) = sigmoid(log(P_A) - log(P_B))
                import math
                prob = 1.0 / (1.0 + math.exp(lp_B - lp_A))
            else:
                # Fallback: check generated text
                generated = output.outputs[0].text.strip().upper()
                prob = 0.9 if generated.startswith("A") else 0.1
            
            all_probs.append(prob)
    
    # Clean up
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return all_probs


def _score_pairs_with_hf(
    edge_indices: List[Tuple[int, int]],
    tensors_by_id: Dict[int, torch.Tensor],
    scorer: "PreferenceScorer",
    batch_size: int = 12,
) -> List[float]:
    """Score pairwise comparisons using HuggingFace transformers."""
    from tqdm import tqdm
    
    all_probs: List[float] = []
    device = scorer.device
    
    for batch_start in tqdm(
        range(0, len(edge_indices), batch_size),
        desc="[HF] Scoring comparisons",
        total=(len(edge_indices) + batch_size - 1) // batch_size,
    ):
        batch = edge_indices[batch_start : batch_start + batch_size]
        if not batch:
            continue
        
        imgs_A = [tensors_by_id[i] for i, _ in batch]
        imgs_B = [tensors_by_id[j] for _, j in batch]
        
        conversations = []
        all_images = []
        for img_A, img_B in zip(imgs_A, imgs_B):
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Which image do you prefer? A:"},
                    {"type": "image", "image": img_A},
                    {"type": "text", "text": " or B:"},
                    {"type": "image", "image": img_B},
                    {"type": "text", "text": " Respond with only 'A' or 'B'."},
                ],
            }]
            conversations.append(
                scorer.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            )
            all_images.append([img_A, img_B])
        
        inputs = scorer.processor(
            text=conversations,
            images=all_images,
            return_tensors="pt",
            padding=True,
            do_rescale=False,
        ).to(device)
        
        with torch.inference_mode():
            outputs = scorer.model(**inputs)
        logits = outputs.logits[:, -1, :].float()
        logprob_A = F.log_softmax(logits, dim=-1)[:, scorer.id_A]
        logprob_B = F.log_softmax(logits, dim=-1)[:, scorer.id_B]
        probs = torch.sigmoid(logprob_A - logprob_B).detach().cpu().numpy()
        
        all_probs.extend(probs.tolist())
        
        del imgs_A, imgs_B, inputs, outputs, logits, conversations, all_images
        safe_empty_cuda_cache()
    
    return all_probs


def load_text_options_from_json(json_path: str) -> List[str]:
    """
    Load text options from a JSON file.
    Accepts either:
    - A flat array of strings: ["option1", "option2", ...] (e.g. text_options-merged.json)
    - A dict of category -> list of strings: {"category1": ["opt1", ...], ...} (e.g. options_hierarchical.json)
    
    Args:
        json_path: Path to the JSON file
    
    Returns:
        List of all text option strings
    """
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Flat array: ["option1", "option2", ...]
        return list(data)
    if isinstance(data, dict):
        # Hierarchical: {"category": ["opt1", ...], ...}
        all_options = []
        for category, options in data.items():
            all_options.extend(options)
        return all_options
    raise TypeError(f"JSON must be a list of strings or a dict of categories; got {type(data)}")


def _build_preference_graph_from_records(
    records: List[Dict[str, Any]],
    model_path: str,
    batch_size: int,
    seed: int = 42,
    existing_graph_data: Optional[Dict[str, Any]] = None,
    use_vllm: bool = False,
    tensor_parallel_size: int = 4,
    scorer: Optional["PreferenceScorer"] = None,
) -> PreferenceGraph:
    """
    Build/update a preference graph by scoring pairwise comparisons with a VLM.
    
    Key behavior:
    - Preserves all existing options and edges from cached graph
    - Assigns new IDs to new images
    - Only samples edges connecting NEW nodes to OLD nodes (not among old nodes)
    - Fits all nodes into a single graph for comparable utility scores
    """
    if len(records) < 2:
        raise ValueError("At least two images are required to build a preference graph.")

    rng = np.random.default_rng(seed)

    # Build option registry from existing graph
    options: List[Dict[str, Any]] = []
    path_to_option_id: Dict[str, int] = {}
    existing_option_ids: Set[int] = set()
    next_option_id = 0
    existing_edges_payload: List[Dict[str, Any]] = []

    if existing_graph_data:
        for opt in existing_graph_data.get("options", []):
            option_copy = dict(opt)
            options.append(option_copy)
            opt_id = int(option_copy["id"])
            existing_option_ids.add(opt_id)
            path = option_copy.get("path")
            if path:
                canonical_path = str(Path(path).expanduser().resolve())
                path_to_option_id[canonical_path] = opt_id
            next_option_id = max(next_option_id, opt_id + 1)
        existing_edges_payload = existing_graph_data.get("edges", [])

    # Assign option IDs to records (reuse existing or create new)
    new_option_ids: Set[int] = set()
    for record in records:
        canonical_path = str(Path(record["path"]).expanduser().resolve())
        option_id = path_to_option_id.get(canonical_path)
        if option_id is None:
            option_id = next_option_id
            next_option_id += 1
            options.append({
                "id": option_id,
                "path": canonical_path,
                "source": record.get("source", "unknown"),
            })
            path_to_option_id[canonical_path] = option_id
            new_option_ids.add(option_id)
        record["option_id"] = option_id
        record["path"] = canonical_path  # Normalize path

    tensors_by_id: Dict[int, torch.Tensor] = {
        record["option_id"]: record["tensor"] for record in records
    }
    available_ids = set(tensors_by_id.keys())
    
    if len(available_ids) < 2:
        raise ValueError("Need at least two images with tensors for comparison.")

    # Initialize graph with all options and existing edges
    graph = PreferenceGraph(options=options, seed=seed)
    if existing_edges_payload:
        graph.add_edges(existing_edges_payload)

    existing_pairs = set(graph.edges.keys())
    def has_edge(i: int, j: int) -> bool:
        return (i, j) in existing_pairs or (j, i) in existing_pairs

    # Sample edges: NEW nodes connect to OLD nodes (prioritize), plus some among NEW nodes
    candidate_pairs: List[Tuple[int, int]] = []
    
    # Priority 1: Edges between NEW and OLD nodes (these are essential for comparability)
    old_ids_with_tensors = available_ids & existing_option_ids
    new_ids_with_tensors = available_ids & new_option_ids
    
    for new_id in new_ids_with_tensors:
        for old_id in old_ids_with_tensors:
            if not has_edge(new_id, old_id):
                candidate_pairs.append((min(new_id, old_id), max(new_id, old_id)))
    
    # Priority 2: Edges among NEW nodes
    for pair in itertools.combinations(sorted(new_ids_with_tensors), 2):
        if not has_edge(*pair):
            candidate_pairs.append(pair)
    
    # Shuffle and limit to target count
    rng.shuffle(candidate_pairs)
    n_new = len(new_ids_with_tensors)
    n_old = len(old_ids_with_tensors)
    # Target: each new node connects to ~log(n_old) old nodes + some new-new edges
    target_new_old = n_new * max(1, int(np.log(max(n_old, 2)) * 2)) if n_old > 0 else 0
    target_new_new = n_new * max(1, int(np.log(max(n_new, 2)))) if n_new > 1 else 0
    target_new_edges = min(target_new_old + target_new_new, len(candidate_pairs))
    
    edge_indices = candidate_pairs[:target_new_edges]
    
    print(f"[graph] Existing: {len(existing_option_ids)} nodes, {len(existing_pairs)} edges")
    print(f"[graph] New: {len(new_option_ids)} nodes, sampling {len(edge_indices)} new edges")

    if not edge_indices:
        print("[graph] No new edges to score.")
        return graph

    # Score new edges using VLM (vLLM or HuggingFace)
    batch_size = max(1, batch_size)
    
    if use_vllm:
        print(f"[graph] Using vLLM backend (tensor_parallel_size={tensor_parallel_size})")
        probs_list = _score_pairs_with_vllm(
            edge_indices=edge_indices,
            tensors_by_id=tensors_by_id,
            model_path=model_path,
            batch_size=batch_size,
            tensor_parallel_size=tensor_parallel_size,
        )
    else:
        if scorer is None:
            raise ValueError("scorer is required when use_vllm=False")
        print("[graph] Using HuggingFace backend")
        probs_list = _score_pairs_with_hf(
            edge_indices=edge_indices,
            tensors_by_id=tensors_by_id,
            scorer=scorer,
            batch_size=batch_size,
        )
    
    # Build preference data from probabilities
    option_lookup = {opt["id"]: opt for opt in options}
    preference_data: List[Dict[str, Any]] = []
    
    for (i, j), probability in zip(edge_indices, probs_list):
        preference_data.append({
            "option_A": option_lookup[i],
            "option_B": option_lookup[j],
            "probability_A": probability,
            "aux_data": {"count_A": int(probability * 100), "count_B": 100 - int(probability * 100)},
        })

    if preference_data:
        graph.add_edges(preference_data)
    
    return graph


def aggregate_preference_pools(
    preference_data_dir: Union[str, Path],
    preference_manifest_dir: Optional[Union[str, Path]],
    stimuli_patterns: List[str],
    model_path: str,
    output_dir: Union[str, Path],
    natural_top_k: int = 50,
    comparison_batch_size: int = 12,
    num_epochs: int = 400,
    learning_rate: float = 0.01,
    seed: int = 42,
    stimuli_checkpoint_interval: Optional[int] = None,
    include_ema_stimuli: bool = True,
    include_non_ema_stimuli: bool = False,  # Default: only EMA images
    stimuli_patterns_no_interval: Optional[List[str]] = None,
    use_vllm: bool = False,
    tensor_parallel_size: int = 4,
) -> Path:
    """
    Aggregate natural images and stimuli into a global preference graph.
    
    Workflow:
    1. Load existing global graph (if any) - preserves all previous edges
    2. Collect top-k natural images (already resized by SuperstimuliDataset)
    3. Collect stimuli images (NO resizing - use original resolution)
    4. Add new nodes to graph, sample edges between NEW and OLD nodes
    5. Fit Thurstonian model on complete graph
    6. Save updated global graph and utilities
    
    All images get comparable utility scores on a single scale.
    
    Args:
        use_vllm: If True, use vLLM for fast inference (2-4x speedup). Default: False.
        tensor_parallel_size: Number of GPUs for vLLM tensor parallelism. Default: 4.
    """
    preference_data_dir = Path(preference_data_dir).expanduser()
    manifest_dir = Path(preference_manifest_dir).expanduser() if preference_manifest_dir else None
    output_root = Path(output_dir).expanduser()
    graph_cache_path = output_root / "global_preference_graph.json"
    
    # Load existing global graph
    existing_graph_data = _load_graph_json(graph_cache_path) if graph_cache_path.exists() else None
    if existing_graph_data:
        print(f"[aggregate] Loaded existing graph: {len(existing_graph_data.get('options', []))} nodes, "
              f"{len(existing_graph_data.get('edges', []))} edges")

    # Collect natural images (skip if natural_top_k=0)
    natural_records: List[Dict[str, Any]] = []
    if natural_top_k > 0:
        natural_records = _collect_natural_image_metadata(
            preference_dir=preference_data_dir,
            manifest_dir=manifest_dir,
            top_k=natural_top_k,
            seed=seed,
        )
        print(f"[aggregate] Collected {len(natural_records)} natural images")
    else:
        print("[aggregate] Skipping natural image collection (natural_top_k=0)")

    # Collect stimuli images (paths only, no tensors yet)
    print(f"[aggregate] EMA filtering settings: include_ema={include_ema_stimuli}, include_non_ema={include_non_ema_stimuli}")
    stimuli_records = _collect_stimuli_image_metadata(
        stimuli_patterns,
        include_ema=include_ema_stimuli,
        include_non_ema=include_non_ema_stimuli,
        checkpoint_interval=stimuli_checkpoint_interval,
    )
    if stimuli_patterns_no_interval:
        stimuli_records.extend(_collect_stimuli_image_metadata(
            stimuli_patterns_no_interval,
            include_ema=include_ema_stimuli,
            include_non_ema=include_non_ema_stimuli,
            checkpoint_interval=None,
        ))
    print(f"[aggregate] Collected {len(stimuli_records)} stimuli images")

    # Combine and deduplicate
    combined_records = _deduplicate_records(natural_records + stimuli_records)
    if not combined_records:
        raise ValueError("No images found across natural datasets and stimuli directories.")
    print(f"[aggregate] Total unique images: {len(combined_records)}")

    # Load tensors (natural images randomly capped, stimuli loaded at original resolution)
    loaded_records = _load_records_with_tensors(combined_records)
    if not loaded_records:
        raise ValueError("Failed to load any images for aggregation.")

    # Initialize scorer (only needed for HuggingFace backend)
    scorer = None
    if not use_vllm:
        from .scorer import PreferenceScorer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scorer = PreferenceScorer(
            model_path=model_path,
            device=device,
            buffer_comparison_batch_size=comparison_batch_size,
        )

    # Build preference graph (scores new edges between new and old nodes)
    graph = _build_preference_graph_from_records(
        records=loaded_records,
        model_path=model_path,
        batch_size=comparison_batch_size,
        seed=seed,
        existing_graph_data=existing_graph_data,
        use_vllm=use_vllm,
        tensor_parallel_size=tensor_parallel_size,
        scorer=scorer,
    )

    # Fit Thurstonian model on complete graph
    utilities, model_log_loss, model_accuracy = fit_thurstonian_model(
        graph=graph,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"thurstonian_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    graph_payload = {
        "options": graph.options,
        "edges": [{
            "option_A": edge.option_A,
            "option_B": edge.option_B,
            "probability_A": edge.probability_A,
            "aux_data": edge.aux_data,
        } for edge in graph.edges.values()],
    }
    write_json(run_dir / "preference_graph.json", graph_payload)
    write_json(graph_cache_path, graph_payload)  # Update global cache

    # Build utilities output with job metadata for plotting
    option_lookup = {opt["id"]: opt for opt in graph.options}
    utilities_payload = []
    for option_id, stats in utilities.items():
        opt = option_lookup.get(option_id, {})
        path = opt.get("path")
        
        # Extract job metadata from path for plotting
        job_meta = _parse_stimuli_metadata_from_path(path) if path else None
        entry = {
            "option_id": option_id,
            "path": path,
            "source": opt.get("source"),
            "mean": stats["mean"],
            "variance": stats["variance"],
        }
        
        # Add job metadata if available (for stimuli images)
        if job_meta:
            job_id, checkpoint_step, stimuli_idx = job_meta
            entry["job_id"] = job_id
            entry["checkpoint_step"] = checkpoint_step
            entry["stimuli_idx"] = stimuli_idx
            entry["is_ema"] = _is_ema_image(path) if path else False
        
        utilities_payload.append(entry)
    
    utilities_payload.sort(key=lambda x: x["mean"], reverse=True)
    write_json(run_dir / "utilities.json", {"utilities": utilities_payload})

    summary = {
        "num_images": len(loaded_records),
        "num_options": len(graph.options),
        "num_edges": len(graph.edges),
        "log_loss": model_log_loss,
        "accuracy": model_accuracy,
        "output_dir": str(run_dir),
        "graph_cache": str(graph_cache_path),
    }
    write_json(run_dir / "summary.json", summary)

    print(f"[aggregate] Final graph: {len(graph.options)} nodes, {len(graph.edges)} edges")
    print(f"[aggregate] Thurstonian fit: log_loss={model_log_loss:.4f}, accuracy={model_accuracy:.4f}")
    print(f"[aggregate] Outputs: {run_dir}")
    return run_dir


def analyze_preference_dataset(
    preference_file: Union[str, Path],
    dataset_name: Optional[str],
    preference_manifest_dir: Optional[Union[str, Path]],
    output_dir: Union[str, Path],
    top_k: int = 20,
    bottom_k: int = 20,
    num_epochs: int = 400,
    learning_rate: float = 0.01,
) -> Path:
    """
    Fit a Thurstonian model for a single stored preference dataset (e.g., imagenet_a)
    and dump visualizations of the top/bottom ranked images.
    """

    manifest_dir = Path(preference_manifest_dir).expanduser() if preference_manifest_dir else None
    pref_path = Path(preference_file).expanduser()
    if not pref_path.exists():
        raise FileNotFoundError(f"Preference file not found: {pref_path}")

    target_payload = read_json(pref_path)
    metadata = target_payload.get("metadata", {})
    dataset_label = dataset_name or metadata.get("dataset") or pref_path.stem
    dataset_key = _normalize_dataset_name(dataset_label)

    preferences = target_payload.get("preferences", [])
    num_refs = metadata.get("num_references")
    if num_refs is None:
        options_payload = target_payload.get("options") or []
        num_refs = len(options_payload)
    if not num_refs:
        raise ValueError(f"Preference payload {pref_path} does not define any reference options.")

    manifest_lookup_name = metadata.get("dataset") or dataset_label
    id_to_path = _resolve_reference_paths_from_payload(target_payload, manifest_lookup_name, manifest_dir)
    options = []
    for option_id in range(int(num_refs)):
        options.append(
            {
                "id": option_id,
                "description": f"{dataset_key}_{option_id}",
                "path": id_to_path.get(option_id),
                "source": dataset_key,
            }
        )

    graph = PreferenceGraph(options=options)
    graph.add_edges(preferences)
    utilities, log_loss, accuracy = fit_thurstonian_model(
        graph=graph,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )

    ranked_entries: List[Dict[str, Any]] = []
    for option in graph.options:
        option_id = option["id"]
        stats = utilities.get(option_id)
        if not stats:
            continue
        path = option.get("path")
        if not path or not Path(path).exists():
            continue
        ranked_entries.append(
            {
                "option_id": option_id,
                "path": path,
                "source": option.get("source"),
                "mean": stats["mean"],
                "variance": stats["variance"],
            }
        )

    if not ranked_entries:
        raise ValueError(
            f"No ranked entries with resolvable paths were found for dataset '{dataset_name}'. "
            "Ensure reference_paths or manifest entries are available."
        )

    ranked_entries.sort(key=lambda item: item["mean"], reverse=True)
    top_slice = ranked_entries[: max(1, min(top_k, len(ranked_entries)))]
    bottom_slice = ranked_entries[-min(bottom_k, len(ranked_entries)) :]
    bottom_slice = sorted(bottom_slice, key=lambda item: item["mean"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir).expanduser() / f"{dataset_key}_analysis_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "utilities.json",
        {
            "dataset": dataset_key,
            "source_file": str(pref_path),
            "utilities": ranked_entries,
            "top_k": len(top_slice),
            "bottom_k": len(bottom_slice),
        },
    )
    write_json(
        run_dir / "summary.json",
        {
            "dataset": dataset_key,
            "source_file": str(pref_path),
            "num_images": len(ranked_entries),
            "log_loss": log_loss,
            "accuracy": accuracy,
            "output_dir": str(run_dir),
        },
    )

    _save_ranked_image_grid(top_slice, f"{dataset_key} top {len(top_slice)}", run_dir / "top_ranked.png")
    _save_ranked_image_grid(bottom_slice, f"{dataset_key} bottom {len(bottom_slice)}", run_dir / "bottom_ranked.png")

    print(
        f"[analyze] Dataset '{dataset_key}' ranked with Thurstonian model. "
        f"log_loss={log_loss:.4f}, accuracy={accuracy:.4f}, results in {run_dir}"
    )
    return run_dir


def plot_job_utility_trajectories(
    utilities_json: Union[str, Path],
    output_dir: Union[str, Path],
    job_filter: Optional[Iterable[str]] = None,
    include_ema: bool = True,
    include_non_ema: bool = True,
) -> Path:
    """Generate per-job utility trajectory plots from utilities.json produced by aggregation."""

    utilities_json = Path(utilities_json).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not utilities_json.exists():
        raise FileNotFoundError(f"utilities json not found: {utilities_json}")

    if not include_ema and not include_non_ema:
        raise ValueError("At least one of include_ema or include_non_ema must be True for plotting.")

    data = json.loads(utilities_json.read_text(encoding="utf-8"))
    utilities = data.get("utilities") or []

    job_series: Dict[str, Dict[int, List[Tuple[int, float]]]] = {}
    for entry in utilities:
        # Use stored metadata if available, otherwise parse from path
        job_id = entry.get("job_id")
        checkpoint_step = entry.get("checkpoint_step")
        stimuli_idx = entry.get("stimuli_idx")
        is_ema = entry.get("is_ema")
        
        if job_id is None or checkpoint_step is None or stimuli_idx is None:
            # Fallback: parse from path (for older format or natural images)
            path = entry.get("path")
            meta = _parse_stimuli_metadata_from_path(path)
            if not meta:
                continue
            job_id, checkpoint_step, stimuli_idx = meta
            is_ema = _is_ema_image(path) if path else False
        
        if is_ema and not include_ema:
            continue
        if (not is_ema) and not include_non_ema:
            continue
        job_series.setdefault(job_id, {}).setdefault(stimuli_idx, []).append((checkpoint_step, entry["mean"]))

    if not job_series:
        raise ValueError("No stimuli utilities with checkpoint metadata were found.")

    job_filter_set = set(job_filter) if job_filter else None
    if job_filter_set:
        filtered_series = {
            job_id: stimuli_map for job_id, stimuli_map in job_series.items() if job_id in job_filter_set
        }
        if not filtered_series:
            raise ValueError(f"No matching job ids found in utilities payload for filter: {sorted(job_filter_set)}")
        job_series = filtered_series

    # Natural images: any source name ending with "samples" (e.g., imagenet_val_samples).
    natural_scores = [entry["mean"] for entry in utilities if isinstance(entry.get("source"), str) and entry["source"].endswith("samples")]
    natural_min = min(natural_scores) if natural_scores else None
    natural_max = max(natural_scores) if natural_scores else None

    for job_id, stimuli_map in sorted(job_series.items()):
        plt.figure(figsize=(10, 6))
        for stimuli_idx, points in stimuli_map.items():
            points.sort(key=lambda item: item[0])
            steps = [step for step, _ in points]
            means = [mean for _, mean in points]
            plt.plot(steps, means, marker="o", label=f"stimuli_{stimuli_idx:02d}")
        if natural_min is not None and natural_max is not None:
            plt.axhspan(natural_min, natural_max, color="gray", alpha=0.15, label="natural image range")
        plt.title(f"Job {job_id} Utility Trajectories")
        plt.xlabel("Checkpoint step")
        plt.ylabel("Thurstonian mean utility")
        plt.legend(loc="best", fontsize="small")
        plt.grid(True, linestyle="--", alpha=0.4)
        job_plot_path = output_dir / f"job_{job_id}_utility.png"
        plt.tight_layout()
        plt.savefig(job_plot_path, dpi=200)
        plt.close()
    return output_dir


def plot_job_average_utility(
    utilities_json: Union[str, Path],
    output_path: Union[str, Path],
    job_filter: Optional[Iterable[str]] = None,
    include_ema: bool = True,
    include_non_ema: bool = True,
) -> Path:
    """
    Plot a single average utility curve per job by averaging over stimuli at each checkpoint.
    """

    utilities_json = Path(utilities_json).expanduser()
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not utilities_json.exists():
        raise FileNotFoundError(f"utilities json not found: {utilities_json}")

    if not include_ema and not include_non_ema:
        raise ValueError("At least one of include_ema or include_non_ema must be True for plotting.")

    data = json.loads(utilities_json.read_text(encoding="utf-8"))
    utilities = data.get("utilities") or []

    job_filter_set = set(job_filter) if job_filter else None

    job_points: Dict[str, Dict[int, List[float]]] = {}
    job_stimuli_counts: Dict[str, set] = {}
    for entry in utilities:
        # Use stored metadata if available, otherwise parse from path
        job_id = entry.get("job_id")
        checkpoint_step = entry.get("checkpoint_step")
        stimuli_idx = entry.get("stimuli_idx")
        is_ema = entry.get("is_ema")
        
        if job_id is None or checkpoint_step is None or stimuli_idx is None:
            # Fallback: parse from path (for older format or natural images)
            path = entry.get("path")
            meta = _parse_stimuli_metadata_from_path(path)
            if not meta:
                continue
            job_id, checkpoint_step, stimuli_idx = meta
            is_ema = _is_ema_image(path) if path else False
        
        if is_ema and not include_ema:
            continue
        if (not is_ema) and not include_non_ema:
            continue
        if job_filter_set and job_id not in job_filter_set:
            continue
        job_points.setdefault(job_id, {}).setdefault(checkpoint_step, []).append(entry["mean"])
        job_stimuli_counts.setdefault(job_id, set()).add(stimuli_idx)

    if not job_points:
        raise ValueError("No matching job ids found in utilities payload.")

    plt.figure(figsize=(10, 6))
    for job_id, step_map in sorted(job_points.items()):
        steps = sorted(step_map.keys())
        averages = [float(np.mean(step_map[step])) for step in steps]
        if averages:
            if any(len(step_map[step]) > 1 for step in steps):
                stds = [float(np.std(step_map[step])) for step in steps]
                plt.errorbar(
                    steps,
                    averages,
                    yerr=stds,
                    marker="o",
                    capsize=3,
                    label=f"job {job_id} (n={len(job_stimuli_counts.get(job_id, []))})",
                )
            else:
                plt.plot(steps, averages, marker="o", label=f"job {job_id} (n={len(job_stimuli_counts.get(job_id, []))})")

    plt.title("Average Utility per Job")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Mean Thurstonian utility")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[plot] Saved average-utility plot to {output_path}")
    return output_path


def _parse_job_id_tokens(tokens: Optional[List[str]]) -> Optional[Set[str]]:
    if not tokens:
        return None
    parsed: Set[str] = set()
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_token, end_token = token.split("-", 1)
            start_token = start_token.strip()
            end_token = end_token.strip()
            if start_token.isdigit() and end_token.isdigit():
                start = int(start_token)
                end = int(end_token)
                step = 1 if end >= start else -1
                for job_id in range(start, end + step, step):
                    parsed.add(str(job_id))
                continue
        parsed.add(token)
    return parsed



def main() -> None:
    """CLI entry point: python -m preference_optimization.utils."""

    default_runs_root = Path(__file__).resolve().parents[2] / "outputs" / "qwen2_5_32b"

    parser = argparse.ArgumentParser(description="Utility commands for preference optimization.", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command")

    agg_parser = subparsers.add_parser("aggregate-preferences", help="Aggregate images and fit a Thurstonian model.")
    agg_parser.add_argument("--preference-data-dir", type=Path, required=True, help="Directory with preference JSON files.")
    agg_parser.add_argument("--preference-manifest-dir", type=Path, help="Manifest directory for resolving image paths.")
    agg_parser.add_argument("--model-path", type=str, required=True, help="Vision-language model checkpoint for scoring.")
    agg_parser.add_argument(
        "--stimuli-dir",
        action="append",
        help="Glob or directory containing generated stimuli (can be repeated).",
    )
    agg_parser.add_argument(
        "--stimuli-dir-unfiltered",
        action="append",
        default=None,
        help="Stimuli dirs/patterns to include without checkpoint-interval filtering.",
    )
    agg_parser.add_argument("--natural-top-k", type=int, default=50, help="Top-N natural images per dataset.")
    agg_parser.add_argument("--comparison-batch-size", type=int, default=12, help="Batch size for VLM preference scoring.")
    agg_parser.add_argument("--num-epochs", type=int, default=400, help="Thurstonian training epochs.")
    agg_parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for Thurstonian fit.")
    agg_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "thurstonian_analysis",
        help="Directory to store aggregated graphs and utilities.",
    )
    agg_parser.add_argument("--seed", type=int, default=42, help="Random seed for edge sampling.")
    agg_parser.add_argument(
        "--stimuli-checkpoint-interval",
        type=int,
        default=None,
        help="Only include stimuli from checkpoints divisible by this interval (e.g., 50 → 0,50,100,...).",
    )
    agg_parser.add_argument(
        "--stimuli-include-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include EMA-rendered checkpoint images (filenames ending with _ema).",
    )
    agg_parser.add_argument(
        "--stimuli-include-non-ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include non-EMA checkpoint images (default: False, only EMA images).",
    )
    agg_parser.add_argument(
        "--use-vllm",
        action="store_true",
        default=False,
        help="Use vLLM for fast inference (2-4x speedup). Requires: pip install vllm qwen-vl-utils",
    )
    agg_parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Number of GPUs for vLLM tensor parallelism (default: 4).",
    )

    analyze_parser = subparsers.add_parser(
        "analyze-preference-dataset",
        help="Fit a Thurstonian model for a single stored preference dataset and dump top/bottom images.",
    )
    analyze_parser.add_argument(
        "--preference-file",
        type=Path,
        required=True,
        help="Path to the preference JSON file to analyze.",
    )
    analyze_parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional display name for the dataset (defaults to metadata or filename).",
    )
    analyze_parser.add_argument(
        "--preference-manifest-dir",
        type=Path,
        default=None,
        help="Optional manifest directory for resolving reference paths.",
    )
    analyze_parser.add_argument("--top-k", type=int, default=20, help="Number of top-ranked images to visualize.")
    analyze_parser.add_argument("--bottom-k", type=int, default=20, help="Number of bottom-ranked images to visualize.")
    analyze_parser.add_argument("--num-epochs", type=int, default=400, help="Thurstonian training epochs.")
    analyze_parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for Thurstonian fit.")
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "thurstonian_analysis",
        help="Directory to store analysis outputs.",
    )
    plot_parser = subparsers.add_parser("plot-utilities", help="Plot per-job utility trajectories from utilities.json.")
    plot_parser.add_argument("--utilities", type=Path, required=True, help="Path to a utilities.json file.")
    plot_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for plot images (default: <utilities>.parent/plots).",
    )
    plot_parser.add_argument(
        "--job-ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional job ids to include (e.g., 40777). Include all if omitted.",
    )
    plot_parser.add_argument(
        "--include-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include EMA checkpoint images in the plot (default: True).",
    )
    plot_parser.add_argument(
        "--include-non-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include non-EMA checkpoint images in the plot (default: True).",
    )
    avg_plot_parser = subparsers.add_parser(
        "plot-job-average",
        help="Plot average utility per job across checkpoints (one line per job).",
    )
    avg_plot_parser.add_argument("--utilities", type=Path, required=True, help="Path to a utilities.json file.")
    avg_plot_parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Output PNG path (default: <utilities>.parent/plots/jobs_average.png).",
    )
    avg_plot_parser.add_argument(
        "--job-ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional job ids to include (e.g., 40194 40200). "
            "Defaults to 40194-40200, 40777-40782, and 40789 when omitted."
        ),
    )
    avg_plot_parser.add_argument(
        "--include-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include EMA checkpoint images in the average plot (default: True).",
    )
    avg_plot_parser.add_argument(
        "--include-non-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include non-EMA checkpoint images in the average plot (default: True).",
    )

    args = parser.parse_args()
    command = args.command

    if command == "aggregate-preferences":
        # Debug: Print parsed EMA values
        print(f"[DEBUG] Parsed EMA args: stimuli_include_ema={args.stimuli_include_ema}, stimuli_include_non_ema={args.stimuli_include_non_ema}")
        stimuli_patterns = args.stimuli_dir or []
        stimuli_patterns_no_interval = args.stimuli_dir_unfiltered or []
        aggregate_preference_pools(
            preference_data_dir=args.preference_data_dir,
            preference_manifest_dir=args.preference_manifest_dir,
            stimuli_patterns=stimuli_patterns,
            stimuli_patterns_no_interval=stimuli_patterns_no_interval,
            model_path=args.model_path,
            output_dir=args.output_dir,
            natural_top_k=args.natural_top_k,
            comparison_batch_size=args.comparison_batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            stimuli_checkpoint_interval=args.stimuli_checkpoint_interval,
            include_ema_stimuli=args.stimuli_include_ema,
            include_non_ema_stimuli=args.stimuli_include_non_ema,
            use_vllm=args.use_vllm,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    elif command == "analyze-preference-dataset":
        analyze_preference_dataset(
            preference_file=args.preference_file,
            dataset_name=args.dataset_name,
            preference_manifest_dir=args.preference_manifest_dir,
            output_dir=args.output_dir,
            top_k=args.top_k,
            bottom_k=args.bottom_k,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )
    elif command == "plot-utilities":
        utilities_path = args.utilities.expanduser()
        out_dir = args.out_dir.expanduser() if args.out_dir else utilities_path.parent / "plots"
        job_filter = _parse_job_id_tokens(getattr(args, "job_ids", None))
        plot_job_utility_trajectories(
            utilities_path,
            out_dir,
            job_filter=job_filter,
            include_ema=args.include_ema,
            include_non_ema=args.include_non_ema,
        )
        print(f"Plots written to: {out_dir}")
    elif command == "plot-job-average":
        utilities_path = args.utilities.expanduser()
        out_path = args.out_path.expanduser() if args.out_path else utilities_path.parent / "plots" / "jobs_average.png"
        job_filter_tokens = getattr(args, "job_ids", None)
        job_filter = _parse_job_id_tokens(job_filter_tokens)
        plot_job_average_utility(
            utilities_path,
            out_path,
            job_filter=job_filter,
            include_ema=args.include_ema,
            include_non_ema=args.include_non_ema,
        )
        print(f"Average plot written to: {out_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


__all__ = [
    "safe_empty_cuda_cache",
    "read_json",
    "write_json",
    "snapshot_run_args",
    "log_run_configuration",
    "append_metadata_snapshot",
    "random_cap_longest_side",
    "load_text_options_from_json",
    "aggregate_preference_pools",
    "analyze_preference_dataset",
    "plot_job_utility_trajectories",
]
