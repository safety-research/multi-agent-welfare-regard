"""Dataset helpers for loading superstimuli preference data."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from .bt import PreferenceGraph, fit_bradley_terry_model
from .thurstonian import fit_thurstonian_model
from .curriculum import (
    cosine_curriculum_fraction,
    linear_curriculum_fraction,
    load_paths_from_manifest,
    sample_curriculum_references,
)
from .utils import random_cap_longest_side


@dataclass
class PreferenceEntry:
    """Description of a single reference entry loaded from a preference dataset."""

    dataset: str
    tensor: torch.Tensor
    path: str
    score: float
    option_id: int


@dataclass
class ComparisonDefinition:
    """Description of a single candidate-reference comparison."""

    candidate_idx: int
    reference_indices: List[int]
    candidate_pos: int
    text_options: List[str] = None  # Optional list of text strings to include in comparison

    def __post_init__(self):
        if self.text_options is None:
            self.text_options = []

    @property
    def group_size(self) -> int:
        return len(self.reference_indices) + 1 + len(self.text_options)


@dataclass
class CurriculumSample:
    """Curriculum sample for a given optimization step."""

    tensors: List[torch.Tensor]
    paths: List[str]
    scores: Optional[List[float]]
    indices: List[Tuple[int, int]]
    fraction: float
    dataset_counts: Dict[str, int]


class SuperstimuliDataset(Dataset):
    """Centralised dataset that materialises preference references with curriculum support."""

    def __init__(
        self,
        preference_dir: Path,
        manifest_dir: Optional[Path] = None,
        pool_top_k: int = 50,
        image_size: int = 256,
        sample_size: int = 64,
        seed: int = 42,
        total_steps: int = 100,
        max_fraction: float = 0.5,
        min_fraction: float = 0.05,
        cosine_shape: float = 1.0,
        curriculum_schedule: str = "cosine",
        curriculum_step_schedule: Optional[List[Tuple[Optional[int], float]]] = None,
        preserve_reference_resolution: bool = False,
        preference_model: str = "thurstonian", # or "bradley_terry"
        random_cap_min: int = 128,
        random_cap_max: int = 512,
        max_datasets: Optional[int] = None,  # Limit number of datasets to load
    ) -> None:
        self.preference_dir = preference_dir
        self.manifest_dir = manifest_dir
        self.pool_top_k = max(1, int(pool_top_k))
        self.image_size = image_size
        self.sample_size = max(1, int(sample_size))
        self.generator = torch.Generator().manual_seed(seed)
        self.total_steps = max(1, int(total_steps))
        self.max_fraction = float(max_fraction)
        self.min_fraction = float(min_fraction)
        self.cosine_shape = float(max(cosine_shape, 1e-6))
        self.curriculum_schedule = curriculum_schedule.strip().lower()
        self.curriculum_step_schedule = self._prepare_step_schedule(curriculum_step_schedule)
        self.fraction_scheduler = self._build_fraction_scheduler()
        self.preserve_reference_resolution = bool(preserve_reference_resolution)
        self.preference_model = preference_model.strip().lower()
        self.random_cap_min = max(1, int(random_cap_min))
        self.random_cap_max = max(self.random_cap_min, int(random_cap_max))
        self.max_datasets = max_datasets

        self._resize_rng = random.Random(seed + 1337)

        def _random_cap(img: Image.Image) -> Image.Image:
            return random_cap_longest_side(
                img, 
                min_target=self.random_cap_min, 
                max_target=self.random_cap_max, 
                rng=self._resize_rng
            )

        transform_parts: List[Callable[..., Any]] = []
        if not self.preserve_reference_resolution:
            transform_parts.append(transforms.Lambda(_random_cap))
        transform_parts.append(transforms.ToTensor())
        self._transform = transforms.Compose(transform_parts)

        self._entries: List[PreferenceEntry] = []
        self._datasets: List[Dict[str, Any]] = []
        self._curriculum: Dict[str, Any] = {}
        self._initial_sample: Optional[CurriculumSample] = None
        self._last_step: Optional[int] = None
        self._last_sample: Optional[CurriculumSample] = None
        self._anchor_pool: List[Dict[str, Any]] = []
        self._load_preference_files()
        self._initialize_curriculum()
        self._anchor_pool = self._compute_anchor_pool()


    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self._entries[index]
        return {
            "tensor": entry.tensor.clone(),
            "path": entry.path,
            "score": entry.score,
            "dataset": entry.dataset,
            "option_id": entry.option_id,
        }


    @property
    def datasets(self) -> List[Dict[str, Any]]:
        """Curriculum-compatible dataset payloads."""
        return self._datasets

    def build_curriculum(self) -> Dict[str, Any]:
        """Return a curriculum dictionary consumable by the optimizer."""
        return dict(self._curriculum)

    def sample_for_step(self, step: int) -> CurriculumSample:
        """Sample a batch of references according to the curriculum schedule."""

        if self._initial_sample is not None and step == 0:
            return self._initial_sample
        if self._last_step == step and self._last_sample is not None:
            return self._last_sample

        raw_sample = sample_curriculum_references(self._curriculum, step)
        sample = CurriculumSample(
            tensors=[tensor.clone().detach().cpu() for tensor in raw_sample["tensors"]],
            paths=list(raw_sample["paths"]),
            scores=list(raw_sample.get("scores") or []),
            indices=list(raw_sample.get("indices") or []),
            fraction=float(raw_sample.get("fraction", 1.0)),
            dataset_counts=dict(raw_sample.get("dataset_counts") or {}),
        )

        self._last_step = step
        self._last_sample = sample
        if step == 0 and self._initial_sample is None:
            self._initial_sample = sample
        return sample

    def prepare_comparisons(
        self,
        step: int,
        candidate_count: int,
        *,
        min_size: int,
        max_size: int,
        include_peer_candidates: bool = False,
        rng: Optional[random.Random] = None,
        enable_text_options: bool = False,
        text_options: Optional[List[str]] = None,
    ) -> Tuple[CurriculumSample, List[ComparisonDefinition]]:
        """Return curriculum sample and comparison plan for the given step."""

        sample = self.sample_for_step(step)
        comparisons = generate_comparison_plan(
            candidate_count=candidate_count,
            num_references=len(sample.tensors),
            min_size=min_size,
            max_size=max_size,
            include_peer_candidates=include_peer_candidates,
            rng=rng,
            enable_text_options=enable_text_options,
            text_options=text_options,
        )
        return sample, comparisons

    def anchor_pool(self) -> List[Dict[str, Any]]:
        """Return the precomputed anchor pool (top-1 per dataset)."""
        return [dict(entry) for entry in self._anchor_pool]


    def _load_preference_files(self) -> None:
        """Walk through all preference data json files"""
        json_files = sorted(self.preference_dir.glob("*.json"))
        
        # Limit number of datasets if specified (for testing)
        if self.max_datasets is not None and self.max_datasets > 0:
            json_files = json_files[:self.max_datasets]
            print(f"[Dataset] Limiting to first {len(json_files)} datasets for testing")
        
        for json_path in tqdm(json_files, desc="Loading preference datasets", leave=False):
            dataset_payload = self._process_preference_file(json_path)
            self._datasets.append(dataset_payload["curriculum"])
            self._entries.extend(dataset_payload["entries"])

    def _process_preference_file(self, json_path: Path) -> Optional[Dict[str, Any]]:
        with open(json_path, "r") as handle:
            data = json.load(handle)

        # load stored preference data
        preferences = data.get("preferences", [])
        metadata = data.get("metadata", {})
        dataset_name = metadata.get("dataset")    
        options = data.get("options")
        num_refs = metadata.get("num_references")
        options = [{"id": idx, "description": f"ref_{idx}"} for idx in range(num_refs)]
        options_sorted = sorted(options, key=lambda opt: opt["id"])
        graph = PreferenceGraph(options=options_sorted)
        graph.add_edges(preferences)

        # fit a preference model (Bradley-Terry or Thurstonian)
        model_name = (metadata.get("preference_model") or self.preference_model).strip().lower()
        # Always use 1000 epochs for better convergence (ignore metadata value)
        num_epochs = 1000
        learning_rate = metadata.get("learning_rate", 0.01)
        option_utilities, _, _ = self._fit_preference_model(
            graph=graph,
            model_name=model_name,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )
        quality_scores = np.array([option_utilities[opt["id"]]["mean"] for opt in options_sorted])
        id_to_path = self._resolve_id_to_path(data, dataset_name, metadata) # get image paths

        # Sort indices by quality scores in descending order (highest scores first)
        # Sort by quality scores descending (highest = most preferred first)
        sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order (highest scores first)
        # Use top k for both modes (strongest items first)
        top_indices = sorted_indices[: self.pool_top_k]

        # create a curriculum dict 
        entries: List[PreferenceEntry] = []
        tensors: List[torch.Tensor] = []
        paths: List[str] = []
        scores: List[float] = []
        option_ids: List[int] = []

        for idx in top_indices:
            option = options_sorted[idx]
            option_id = option["id"]
            img_path = id_to_path.get(option_id)
            if img_path is None:
                continue
            tensor = self._load_image(Path(img_path))
            if tensor is None:
                continue
            score = float(quality_scores[idx])
            entries.append(
                PreferenceEntry(
                    dataset=dataset_name,
                    tensor=tensor,
                    path=img_path,
                    score=score,
                    option_id=int(option_id),
                )
            )
            tensors.append(tensor)
            paths.append(img_path)
            scores.append(score)
            option_ids.append(int(option_id))

        curriculum_payload = {
            "name": dataset_name,
            "references": tensors,
            "paths": paths,
            "scores": scores,
            "option_ids": option_ids,
        }
        return {
            "entries": entries,
            "curriculum": curriculum_payload,
        }

    def _fit_preference_model(
        self,
        graph: PreferenceGraph,
        model_name: str,
        num_epochs: int,
        learning_rate: float,
    ) -> Tuple[Dict[Any, Dict[str, float]], float, float]:
        """Select and fit the requested preference model."""

        normalized_name = model_name.strip().lower()
        if normalized_name in {"bradley_terry", "bt"}:
            return fit_bradley_terry_model(graph=graph, num_epochs=num_epochs, learning_rate=learning_rate)
        if normalized_name in {"thurstonian", "th"}:
            return fit_thurstonian_model(graph=graph, num_epochs=num_epochs, learning_rate=learning_rate)
        raise ValueError(f"Unsupported preference model '{model_name}'. Expected 'bradley_terry' or 'thurstonian'.")

    def _resolve_id_to_path(
        self,
        data: Dict[str, Any],
        dataset_name: str,
        metadata: Dict[str, Any],
    ) -> Dict[int, str]:
        saved_paths = data.get("reference_paths")
        if isinstance(saved_paths, dict):
            return {int(k): str(v) for k, v in saved_paths.items()}
        if isinstance(saved_paths, list):
            return {idx: str(path) for idx, path in enumerate(saved_paths)}

        manifest_hint = metadata.get("reference_manifest")
        manifest_candidates: List[Path] = []
        if manifest_hint:
            manifest_candidates.append(Path(manifest_hint).expanduser())

        if self.manifest_dir:
            manifest_candidates.extend(
                [
                    self.manifest_dir / f"{dataset_name}.jsonl",
                    self.manifest_dir / f"{dataset_name}.json",
                    self.manifest_dir / f"{dataset_name}_samples.jsonl",
                    self.manifest_dir / f"{dataset_name}_samples.json",
                ]
            )

        resolved_manifest = next((candidate for candidate in manifest_candidates if candidate.exists()), None)
        if resolved_manifest is None:
            return {}

        manifest_paths = load_paths_from_manifest(str(resolved_manifest))
        return {idx: path for idx, path in enumerate(manifest_paths)}

    def _load_image(self, path: Path) -> Optional[torch.Tensor]:
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            processed = self._transform(img)
            return processed

    def _initialize_curriculum(self) -> None:
        self._curriculum = {
            "datasets": self._datasets,
            "sample_size": self.sample_size,
            "fraction_scheduler": self.fraction_scheduler,
            "generator": self.generator,
        }
        self._initial_sample = self.sample_for_step(0)

    def _build_fraction_scheduler(self) -> Callable[[int], float]:
        if self.curriculum_schedule == "step" and self.curriculum_step_schedule:
            schedule = self.curriculum_step_schedule

            def step_scheduler(step: int) -> float:
                for boundary, fraction in schedule:
                    if boundary is None or step < boundary:
                        return fraction
                return schedule[-1][1]

            return step_scheduler

        if self.curriculum_schedule == "linear":
            return lambda step: linear_curriculum_fraction(
                step,
                total_steps=self.total_steps,
                max_fraction=self.max_fraction,
                min_fraction=self.min_fraction,
            )

        # Default to cosine
        return lambda step: cosine_curriculum_fraction(
            step,
            total_steps=self.total_steps,
            max_fraction=self.max_fraction,
            min_fraction=self.min_fraction,
            sharpness=self.cosine_shape,
        )

    def _prepare_step_schedule(
        self, schedule: Optional[List[Tuple[Optional[int], float]]]
    ) -> Optional[List[Tuple[Optional[int], float]]]:
        if not schedule:
            if getattr(self, "curriculum_schedule", "cosine") != "step":
                return None
            schedule = [
                (20, 0.50),
                (40, 0.30),
                (60, 0.20),
                (80, 0.10),
                (None, 0.05),
            ]

        normalized: List[Tuple[Optional[int], float]] = []
        for boundary, fraction in schedule:
            if boundary is not None and boundary < 0:
                raise ValueError(f"Curriculum step boundary must be non-negative or None, got {boundary}.")
            if not (0.0 < fraction <= 1.0):
                raise ValueError(f"Curriculum fraction must be in (0, 1], got {fraction}.")
            normalized.append((boundary, fraction))

        normalized.sort(key=lambda item: float("inf") if item[0] is None else item[0])
        return normalized

    def _compute_anchor_pool(self) -> List[Dict[str, Any]]:
        anchor_entries: List[Dict[str, Any]] = []
        for dataset_entry in self._datasets:
            references = dataset_entry.get("references") or []
            scores = dataset_entry.get("scores") or []
            paths = dataset_entry.get("paths") or []
            if not references or not scores or not paths:
                continue
            # For both modes, use the strongest image (highest score)
            # - Stimulant: highest score = most preferred = strongest stimulant

            anchor_idx = int(np.argmax(scores))
            tensor = references[anchor_idx]
            tensor_cpu = tensor.detach().cpu().clamp(0.0, 1.0).to(torch.float32)
            anchor_entries.append(
                {
                    "dataset": dataset_entry.get("name", "unknown"),
                    "path": paths[anchor_idx],
                    "score": float(scores[anchor_idx]),
                    "tensor": tensor_cpu,
                }
            )
        # Sort by score descending (highest scores first) for both modes
        # - Stimulant: highest = strongest stimulant

        anchor_entries.sort(key=lambda item: (-item["score"], item["dataset"], item["path"]))  # Descending (strongest first)
        return anchor_entries


def generate_comparison_plan(
    candidate_count: int,
    num_references: int,
    *,
    min_size: int,
    max_size: int,
    include_peer_candidates: bool = False,
    rng: Optional[random.Random] = None,
    enable_text_options: bool = False,
    text_options: Optional[List[str]] = None,
) -> List[ComparisonDefinition]:
    """Randomly sample comparison definitions following the existing preference logic.
    
    If enable_text_options is True:
    - min_size and max_size refer to total items (images + text)
    - Randomly selects number of images and number of text strings
    - Total items will be in [min_size, max_size]
    - Can have 0 images (all text), 0 text (all images), or any combination
    - If enable_text_options is False, behavior is unchanged (min_size to max_size images)
    """

    random_gen = rng if rng is not None else random
    rand_shuffle = getattr(random_gen, "shuffle", random.shuffle)
    rand_int = getattr(random_gen, "randint", random.randint)
    rand_choice = getattr(random_gen, "choice", random.choice)
    rand_sample = getattr(random_gen, "sample", random.sample)
    rand_choices = getattr(random_gen, "choices", random.choices)
    
    comparisons: List[ComparisonDefinition] = []

    if enable_text_options:
        # Validate that text_options is available when enable_text_options=True
        if not text_options or len(text_options) == 0:
            import warnings
            warnings.warn(
                f"enable_text_options=True but text_options is None or empty. "
                f"Falling back to image-only comparisons. Check that text options were loaded correctly.",
                UserWarning
            )
            # Fall through to image-only branch
            enable_text_options = False
        
    if enable_text_options:
        # When text is enabled: randomly select total items, then split between images and text
        # Total items will be in [min_size, max_size]
        for cand_idx in range(candidate_count):
            reference_pool: List[int] = list(range(num_references))
            if include_peer_candidates and candidate_count > 1:
                reference_pool.extend(num_references + peer_idx for peer_idx in range(candidate_count) if peer_idx != cand_idx)

            rand_shuffle(reference_pool)
            cursor = 0
            pool_size = len(reference_pool)

            while cursor < pool_size:
                # Randomly select total number of items (images + text)
                # Total includes: 1 candidate image + reference images + text options
                total_items = rand_int(min_size, max_size)
                
                # Must have at least 1 image (the candidate being optimized)
                # Remaining items can be reference images or text
                # Randomly decide: 1 image (candidate only) to total_items images (all images, no text)
                num_images = rand_int(1, total_items)
                num_text = total_items - num_images
                
                # Sample text options if needed
                sampled_texts: List[str] = []
                if num_text > 0 and text_options:
                    if num_text <= len(text_options):
                        sampled_texts = rand_sample(text_options, num_text)
                    else:
                        # Sample with replacement if we need more texts than available
                        sampled_texts = rand_choices(text_options, k=num_text)
                
                # Determine how many reference images we need
                # num_images includes the candidate, so we need num_images - 1 references
                num_refs_needed = num_images - 1
                
                if num_refs_needed > 0:
                    remaining = pool_size - cursor
                    if remaining < num_refs_needed:
                        break
                    
                    ref_group = reference_pool[cursor : cursor + num_refs_needed]
                    cursor += num_refs_needed
                else:
                    # Only candidate image, no references
                    ref_group = []
                
                # Candidate position among images only (0 to num_images-1)
                candidate_pos = rand_int(0, num_images - 1)
                
                comparisons.append(
                    ComparisonDefinition(
                        candidate_idx=cand_idx,
                        reference_indices=ref_group,
                        candidate_pos=candidate_pos,
                        text_options=sampled_texts,
                    )
                )
                
                # Continue to next comparison if we have room
                # Break if we've exhausted the reference pool
                if cursor >= pool_size:
                    break
    else:
        # Original behavior: min_size to max_size images only
        min_refs = max(1, min_size - 1)
        max_refs = max(min_refs, max_size - 1)

        for cand_idx in range(candidate_count):
            reference_pool: List[int] = list(range(num_references))
            if include_peer_candidates and candidate_count > 1:
                reference_pool.extend(num_references + peer_idx for peer_idx in range(candidate_count) if peer_idx != cand_idx)

            rand_shuffle(reference_pool)
            cursor = 0
            pool_size = len(reference_pool)

            while cursor < pool_size:
                remaining = pool_size - cursor
                if remaining < min_refs:
                    break
                max_group_refs = min(max_refs, remaining)
                if max_group_refs < min_refs:
                    break

                group_ref_count = rand_int(min_refs, max_group_refs)
                ref_group = reference_pool[cursor : cursor + group_ref_count]
                cursor += group_ref_count

                candidate_pos = rand_int(0, group_ref_count)
                comparisons.append(
                    ComparisonDefinition(
                        candidate_idx=cand_idx,
                        reference_indices=ref_group,
                        candidate_pos=candidate_pos,
                        text_options=[],
                    )
                )

    return comparisons
