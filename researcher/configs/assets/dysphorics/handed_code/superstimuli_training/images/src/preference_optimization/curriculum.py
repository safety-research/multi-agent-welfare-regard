"""Reference curriculum helpers. It manages tasks such as sampling natural
reference images from different datasets based on the current schedule"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import math
import os

import torch
from PIL import Image
from torchvision import transforms



def cosine_curriculum_fraction(
    step: int,
    *,
    total_steps: int,
    max_fraction: float,
    min_fraction: float,
    sharpness: float = 1.0,
) -> float:
    """
    Cosine-shaped decay with shape controled by sharpness. shaperness < 1 for faster decay. 
    """
    progress = (step / total_steps) ** sharpness
    cosine_weight = 0.5 * (1 + math.cos(math.pi * progress))
    fraction = min_fraction + (max_fraction - min_fraction) * cosine_weight
    return fraction


def linear_curriculum_fraction(
    step: int,
    *,
    total_steps: int,
    max_fraction: float,
    min_fraction: float,
) -> float:
    """
    Linear interpolation from max_fraction (at step 0) to min_fraction (at step total_steps).
    """
    progress = step / total_steps if total_steps > 0 else 0.0
    progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
    fraction = max_fraction + (min_fraction - max_fraction) * progress
    return fraction


def get_raw_cache_path(cache_dir: str, dataset_name: str) -> str:
    return os.path.join(cache_dir, f"raw_references_{dataset_name}.pt")


def get_filtered_cache_path(cache_dir: str, dataset_name: str) -> str:
    return os.path.join(cache_dir, f"filtered_references_{dataset_name}.pt")


def _reference_to_tensor(ref: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
    if isinstance(ref, torch.Tensor):
        tensor = ref.detach()
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.to(dtype=torch.float32)
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        return tensor
    if isinstance(ref, Image.Image):
        return transforms.ToTensor()(ref)
    raise TypeError(f"Unsupported reference type: {type(ref)}")


def sample_curriculum_references(curriculum: Dict[str, Any], step: int) -> Dict[str, Any]:
    datasets: List[Dict[str, Any]] = curriculum["datasets"]
    total_refs = sum(len(entry["references"]) for entry in datasets)
    if total_refs == 0:
        raise ValueError("Curriculum reference pool is empty.")

    fraction_scheduler = curriculum.get("fraction_scheduler")
    if not callable(fraction_scheduler):
        raise ValueError("Curriculum missing a fraction scheduler.")
    fraction = float(fraction_scheduler(step))

    eligible_pairs: List[Tuple[int, int]] = []
    
    for dataset_idx, entry in enumerate(datasets):
        n = len(entry["references"])
        if n == 0:
            continue
        count = min(max(1, math.ceil(n * fraction)), n)

        # Select from the beginning: indices 0 to (count-1)
        # Top items are strongest (most preferred)
        eligible_pairs.extend((dataset_idx, local_idx) for local_idx in range(count))

    if not eligible_pairs:
        raise ValueError("No eligible references found for curriculum sampling.")

    sample_size = min(curriculum["sample_size"], len(eligible_pairs))

    generator = curriculum.get("generator")
    if not isinstance(generator, torch.Generator):
        generator = torch.Generator()
        generator.manual_seed(0)
        curriculum["generator"] = generator

    selected_pairs = torch.randperm(len(eligible_pairs), generator=generator)[:sample_size].tolist()

    tensors: List[torch.Tensor] = []
    paths: List[str] = []
    scores: List[float] = []
    dataset_counts: Dict[str, int] = {}
    resolved_pairs: List[Tuple[int, int]] = []

    for idx in selected_pairs:
        dataset_idx, local_idx = eligible_pairs[idx]
        entry = datasets[dataset_idx]
        ref_tensor = _reference_to_tensor(entry["references"][local_idx])
        tensors.append(ref_tensor)
        paths.append(entry["paths"][local_idx])
        score = entry["scores"][local_idx]
        scores.append(score)
        dataset_name = entry["name"]
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        resolved_pairs.append((dataset_idx, local_idx))
    return {
        "indices": resolved_pairs,
        "paths": paths,
        "scores": scores,
        "tensors": tensors,
        "fraction": fraction,
        "dataset_counts": dataset_counts,
    }


def save_raw_references(
    references: List[torch.Tensor],
    reference_paths: List[str],
    cache_dir: str,
    dataset_name: str,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = get_raw_cache_path(cache_dir, dataset_name)
    stacked_refs = torch.stack(references)
    torch.save(
        {
            "references": stacked_refs,
            "reference_paths": reference_paths,
            "count": len(references),
            "shape": stacked_refs.shape,
        },
        cache_path,
    )


def load_raw_references(
    cache_dir: str,
    dataset_name: str,
) -> Tuple[Optional[List[torch.Tensor]], Optional[List[str]]]:
    cache_path = get_raw_cache_path(cache_dir, dataset_name)
    if not os.path.exists(cache_path):
        return None, None

    cache_data = torch.load(cache_path, map_location="cpu")
    stacked_refs = cache_data["references"]
    references = [stacked_refs[i] for i in range(cache_data["count"])]
    reference_paths = cache_data["reference_paths"]
    return references, reference_paths


def save_filtered_references(
    references: List[torch.Tensor],
    filtered_indices: List[int],
    reference_paths: List[str],
    cache_dir: str,
    dataset_name: str,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = get_filtered_cache_path(cache_dir, dataset_name)
    stacked_refs = torch.stack(references)
    torch.save(
        {
            "references": stacked_refs,
            "filtered_indices": filtered_indices,
            "reference_paths": reference_paths,
            "count": len(references),
            "shape": stacked_refs.shape,
        },
        cache_path,
    )
    print(f"Saved filtered references to {cache_path}.")

def load_filtered_references(
    cache_dir: str,
    dataset_name: str,
) -> Tuple[Optional[List[torch.Tensor]], Optional[List[str]]]:
    cache_path = get_filtered_cache_path(cache_dir, dataset_name)
    if not os.path.exists(cache_path):
        return None, None

    cache_data = torch.load(cache_path, map_location="cpu")
    stacked_refs = cache_data["references"]
    references = [stacked_refs[i] for i in range(cache_data["count"])]
    reference_paths = cache_data["reference_paths"]
    return references, reference_paths


def load_paths_from_manifest(manifest_path: str) -> List[str]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if path.suffix in {".json", ".jsonl"}:
        data = []
        with open(path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        if data and isinstance(data[0], dict) and "path" in data[0]:
            return [entry["path"] for entry in data]
        return data  # assume it is already a list of paths

    if path.suffix == ".txt":
        return path.read_text().splitlines()

    raise ValueError(f"Unsupported manifest format: {manifest_path}")


__all__ = [
    "sample_curriculum_references",
    "save_raw_references",
    "load_raw_references",
    "save_filtered_references",
    "load_filtered_references",
    "load_paths_from_manifest",
    "_reference_to_tensor",
    "get_raw_cache_path",
    "get_filtered_cache_path",
    "cosine_curriculum_fraction",
    "linear_curriculum_fraction",
]
