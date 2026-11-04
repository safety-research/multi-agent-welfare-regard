"""Preference scoring utilities."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .constants import (
    PREFERENCE_QUESTIONS,
    generate_question_variants,
    load_questions_from_config,
    format_comparison_prompt,
    sample_comparison_format,
    format_hybrid_comparison_prompt,
    sample_hybrid_question_config,
    LABEL_SCHEMES,
)
from .dataset import ComparisonDefinition
from .preprocessing import GradientEnabledImagePreprocessor
from .utils import safe_empty_cuda_cache
from .optimizer import PreferenceOptimizer


def _build_max_memory_map(
    limit_first_gpu_gib: Optional[int] = 45,
    exclude_gpus: Optional[List[int]] = None,
) -> Optional[Dict[str, str]]:
    """For OOM: limit the first GPU's memory usage so HF parallel loading keeps headroom.

    Args:
        limit_first_gpu_gib: Memory limit for first GPU in GiB
        exclude_gpus: List of GPU indices to exclude (e.g., [2] to reserve cuda:2 for other use)
    """
    if limit_first_gpu_gib is None or limit_first_gpu_gib <= 0:
        return None
    if not torch.cuda.is_available():
        return None
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return None

    exclude_gpus = exclude_gpus or []

    max_memory: Dict[str, str] = {}
    for idx in range(device_count):
        if idx in exclude_gpus:
            continue  # Skip excluded GPUs
        props = torch.cuda.get_device_properties(idx)
        total_gib = max(1, int(props.total_memory // (1024**3)))
        if idx == 0:
            cap = min(int(limit_first_gpu_gib), max(1, total_gib - 1))
        else:
            # Leave a little buffer on the remaining GPUs to avoid oversubscription.
            cap = max(1, total_gib - 2)
        max_memory[idx] = f"{cap}GiB"

    # Allow CPU offload if needed.
    max_memory["cpu"] = "160GiB"
    return max_memory


class PreferenceScorer:
    """Differentiable preference scorer maintaining computational graph."""

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        min_pixels: int = 3136,
        max_pixels: int = 12845056,
        offload_folder: Optional[str] = None,
        jitter_size: Optional[int] = None,
        resize_on_oom: bool = False,
        oom_candidate_size: Optional[int] = None,
        oom_reference_size: Optional[int] = None,
        noise_reg_weight: float = 0.0,
        robust_noise_type: Optional[str] = None,
        robust_noise_std: float = 0.0,
        robust_noise_prob: float = 1.0,
        robust_flip_prob: float = 0.0,
        robust_crop_prob: float = 0.0,
        robust_crop_min_ratio: float = 0.85,
        robust_crop_max_ratio: float = 0.95,
        buffer_comparison_batch_size: int = 4,
        randomize_preference_prompt: bool = True,
        num_prompt_samples: int = 1,
        max_image_dimension: Optional[int] = None,
        use_question_variants: bool = False,
        question_variants_seed: Optional[int] = None,
        questions: Optional[List[str]] = None,
        questions_config: Optional[str] = None,
        use_flexible_format: bool = True,  # Use new flexible question format system
        use_gradient_checkpointing: bool = False,
        # GPU exclusion (for reserving GPUs for other models like StyleGAN)
        exclude_gpus: Optional[List[int]] = None,
        negative_question_prob: float = 0.0,  # Probability of forcing a negative (inverted) question per comparison
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.offload_folder = offload_folder
        self.jitter_size = jitter_size if jitter_size is not None else 0
        self.resize_on_oom = bool(resize_on_oom)
        self.oom_candidate_size = oom_candidate_size
        self.oom_reference_size = oom_reference_size
        self.noise_reg_weight = float(noise_reg_weight)
        self.buffer_comparison_batch_size = max(1, int(buffer_comparison_batch_size))
        self.randomize_preference_prompt = bool(randomize_preference_prompt)
        self.num_prompt_samples = max(1, int(num_prompt_samples))
        self.max_image_dimension = int(max_image_dimension) if max_image_dimension else None
        self.use_question_variants = bool(use_question_variants)
        self.use_flexible_format = bool(use_flexible_format)
        self.negative_question_prob = float(negative_question_prob)

        # Determine base questions: custom list > config file > default
        if questions is not None:
            base_questions = questions
            print(f"Using {len(base_questions)} custom questions provided directly")
        elif questions_config is not None:
            base_questions = load_questions_from_config(questions_config)
            print(f"Loaded {len(base_questions)} questions from config: {questions_config}")
        else:
            base_questions = PREFERENCE_QUESTIONS

        # Generate question pool (either variants or original)
        if self.use_question_variants:
            # Generate variants: 5 base questions -> 3 variants each -> 15 total -> sample 5
            self.question_pool = generate_question_variants(
                base_questions,
                num_base=5,
                variants_per_question=3,
                num_to_sample=5,
                seed=question_variants_seed,
            )
            print(f"Generated question pool with {len(self.question_pool)} variants:")
            for q in self.question_pool:
                print(f"  - {q}")
        else:
            self.question_pool = base_questions

        self.use_gradient_checkpointing = use_gradient_checkpointing

        noise_mode = (robust_noise_type or "").strip().lower()
        if noise_mode in {"", "none"}:
            noise_mode = None
        self.robust_noise_type = noise_mode
        self.robust_noise_std = max(float(robust_noise_std), 0.0)
        prob = float(robust_noise_prob)
        self.robust_noise_prob = min(max(prob, 0.0), 1.0)
        if self.robust_noise_type is None or self.robust_noise_std == 0.0:
            self.robust_noise_type = None
            self.robust_noise_std = 0.0
        
        # Flip and crop augmentation parameters
        self.robust_flip_prob = min(max(float(robust_flip_prob), 0.0), 1.0)
        self.robust_crop_prob = min(max(float(robust_crop_prob), 0.0), 1.0)
        # Crop ratio range: ensure valid range [0.0, 1.0] and min <= max
        crop_min = min(max(float(robust_crop_min_ratio), 0.0), 1.0)
        crop_max = min(max(float(robust_crop_max_ratio), 0.0), 1.0)
        self.robust_crop_min_ratio = min(crop_min, crop_max)
        self.robust_crop_max_ratio = max(crop_min, crop_max)
        if self.resize_on_oom:
            if self.oom_candidate_size is None:
                self.oom_candidate_size = 256
            if self.oom_reference_size is None:
                self.oom_reference_size = 512

        self.rank = 0
        self.world_size = 1
        self.local_rank = 0

        device_map = "auto"
        self.exclude_gpus = exclude_gpus
        max_memory = _build_max_memory_map(exclude_gpus=exclude_gpus)
        if exclude_gpus:
            print(f"[PreferenceScorer] Excluding GPUs {exclude_gpus} from model loading")
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"Loading model from {model_path}...")

        # Detect model architecture to set appropriate attention implementation.
        # Qwen3.5 uses hybrid DeltaNet (linear) + full attention layers;
        # flash_attention_2 only applies to the full attention layers.
        # For hybrid models, let the model choose its own attn implementation.
        from transformers import AutoConfig
        _model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        _model_type = getattr(_model_config, "model_type", "")
        _text_config = getattr(_model_config, "text_config", _model_config)
        _layer_types = getattr(_text_config, "layer_types", None) or []
        _is_hybrid_attn = "linear_attention" in _layer_types

        if _is_hybrid_attn:
            # Hybrid models (e.g., Qwen3.5 with DeltaNet + full attention)
            # manage their own attention implementations per layer.
            attn_impl = None
            print(f"[PreferenceScorer] Hybrid attention model ({_model_type}) — using model's native attn implementations")
        else:
            # Standard transformer models: use flash_attention_2 if available
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
            print(f"[PreferenceScorer] Using attn_implementation={attn_impl}")

        load_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            offload_folder=self.offload_folder,
            max_memory=max_memory,
            trust_remote_code=True,
        )
        if attn_impl is not None:
            load_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            **load_kwargs,
        )
        
        # Enable gradient checkpointing if requested (must be done before eval())
        if self.use_gradient_checkpointing:
            if self.rank == 0:
                print("[PreferenceScorer] Enabling gradient checkpointing for memory efficiency")
            self.model.gradient_checkpointing_enable()
            # Keep model in train() mode for checkpointing to work, but disable dropout
            self.model.train()
            for m in self.model.modules():
                if m.__class__.__name__.endswith('Dropout'):
                    m.eval()  # Disable dropout for deterministic behavior
            if self.rank == 0:
                print("[PreferenceScorer] Model in train mode with dropout disabled for checkpointing")
        else:
            self.model.eval()
        
        # Disable parameter gradients - we only need input gradients
        for param in self.model.parameters():
            param.requires_grad = False
        if self.rank == 0:
            print("[PreferenceScorer] Model parameter gradients disabled (only input gradients needed)")

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=False,
            trust_remote_code=True,
        )

        # Disable thinking mode (Qwen3.5+ defaults enable_thinking=True,
        # wasting tokens on <think> reasoning during preference scoring).
        _orig_apply = self.processor.apply_chat_template

        def _apply_no_thinking(*args, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _orig_apply(*args, **kwargs)

        self.processor.apply_chat_template = _apply_no_thinking

        # Set left padding for decoder-only models to avoid generation issues
        if hasattr(self.model.config, 'is_encoder_decoder') and not self.model.config.is_encoder_decoder:
            self.processor.tokenizer.padding_side = 'left'

        self.grad_img_processor = GradientEnabledImagePreprocessor()

        tokenizer = self.processor.tokenizer
        self.id_A = tokenizer.encode("A", add_special_tokens=False)[0]
        self.id_B = tokenizer.encode("B", add_special_tokens=False)[0]
        
        # Check "Yes" and "No" (capitalized) for yes/no tokens
        yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = tokenizer.encode("No", add_special_tokens=False)
        
        if len(yes_ids) == 1:
            self.id_yes = yes_ids[0]
        else:
            # Fallback to lowercase if "Yes" is not a single token
            ids = tokenizer.encode("yes", add_special_tokens=False)
            self.id_yes = ids[0] if ids else None
        
        if len(no_ids) == 1:
            self.id_no = no_ids[0]
        else:
            # Fallback to lowercase if "No" is not a single token
            ids = tokenizer.encode("no", add_special_tokens=False)
            self.id_no = ids[0] if ids else None
        
        if self.id_yes is None or self.id_no is None:
            raise ValueError(
                f"Could not find valid yes/no token IDs. "
                f"id_yes={self.id_yes}, id_no={self.id_no}"
            )
        
        if self.rank == 0:
            print(f"Token IDs: A={self.id_A}, B={self.id_B}, yes={self.id_yes}, no={self.id_no}")

    def _get_interpolation_mode(self) -> str:
        """Get the interpolation mode from the processor, defaulting to 'bicubic'."""
        interpolation_mode = getattr(self.processor.image_processor, 'resample', 'bicubic')
        if hasattr(interpolation_mode, 'name'):
            # PIL Resampling enum
            interpolation_map = {
                'BILINEAR': 'bilinear',
                'BICUBIC': 'bicubic',
                'NEAREST': 'nearest',
            }
            interpolation_mode = interpolation_map.get(interpolation_mode.name.upper(), 'bicubic')
        elif isinstance(interpolation_mode, str):
            interpolation_map = {
                'bilinear': 'bilinear',
                'bicubic': 'bicubic',
                'nearest': 'nearest',
            }
            interpolation_mode = interpolation_map.get(interpolation_mode.lower(), 'bicubic')
        else:
            interpolation_mode = 'bicubic'  # default
        return interpolation_mode

    def _apply_robust_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.robust_noise_type is None or self.robust_noise_std == 0.0:
            return tensor

        if self.robust_noise_prob < 1.0:
            if torch.rand(1, device=tensor.device).item() > self.robust_noise_prob:
                return tensor

        std = self.robust_noise_std
        noise_type = self.robust_noise_type

        if noise_type == "gaussian":
            noise = torch.randn_like(tensor) * std
            augmented = tensor + noise
        elif noise_type == "uniform":
            noise = (torch.rand_like(tensor) * 2.0 - 1.0) * std
            augmented = tensor + noise
        elif noise_type == "speckle":
            noise = torch.randn_like(tensor) * std
            augmented = tensor + tensor * noise
        else:
            return tensor

        return augmented.clamp(0.0, 1.0)

    def _apply_robust_flip(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random horizontal/vertical flips with proper gradient flow.
        
        torch.flip is differentiable - gradients are permuted back correctly.
        Applies different flips to each image in the batch, with per-image probability.
        """
        if self.robust_flip_prob <= 0.0:
            return tensor
        
        # Handle batch dimension
        squeeze_back = False
        if tensor.dim() == 3:  # [C, H, W]
            tensor = tensor.unsqueeze(0)
            squeeze_back = True
        
        B = tensor.shape[0]
        
        # Per-image probability check: each image independently decides whether to flip
        apply_flip = torch.rand(B, device=tensor.device) <= self.robust_flip_prob
        
        # Apply per-image flips: 0=none, 1=horizontal, 2=vertical, 3=both
        flip_types = torch.randint(0, 4, (B,), device=tensor.device)
        
        transformed = []
        for b in range(B):
            if not apply_flip[b].item():
                # Skip flip for this image
                transformed.append(tensor[b])
                continue
            
            flip_type = flip_types[b].item()
            if flip_type == 0:
                transformed.append(tensor[b])
            elif flip_type == 1:  # Horizontal flip
                transformed.append(torch.flip(tensor[b], dims=[-1]))
            elif flip_type == 2:  # Vertical flip
                transformed.append(torch.flip(tensor[b], dims=[-2]))
            else:  # Both
                transformed.append(torch.flip(tensor[b], dims=[-2, -1]))
        
        result = torch.stack(transformed, dim=0)
        if squeeze_back:
            result = result.squeeze(0)
        return result
    
    def _apply_robust_crop(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random crop with proper gradient flow.
        
        Crops to a random size within [robust_crop_min_ratio, robust_crop_max_ratio] of original,
        then resizes back to original size. Gradients flow through F.interpolate (differentiable resize).
        Pixels outside crop get zero gradient (not in forward pass).
        Applies different crops to each image in the batch, with per-image probability.
        """
        if self.robust_crop_prob <= 0.0:
            return tensor
        
        # Handle batch dimension
        squeeze_back = False
        if tensor.dim() == 3:  # [C, H, W]
            tensor = tensor.unsqueeze(0)
            squeeze_back = True
        
        B, C, H, W = tensor.shape
        
        # Per-image probability check: each image independently decides whether to crop
        apply_crop = torch.rand(B, device=tensor.device) <= self.robust_crop_prob
        
        # Apply per-image crops
        transformed = []
        for b in range(B):
            if not apply_crop[b].item():
                # Skip crop for this image
                transformed.append(tensor[b])
                continue
            
            # Random crop size: within configured range (per image)
            crop_ratio = self.robust_crop_min_ratio + torch.rand(1, device=tensor.device).item() * (self.robust_crop_max_ratio - self.robust_crop_min_ratio)
            crop_h = max(1, int(H * crop_ratio))  # Ensure at least 1 pixel
            crop_w = max(1, int(W * crop_ratio))  # Ensure at least 1 pixel
            
            # Ensure crop doesn't exceed image dimensions
            crop_h = min(crop_h, H)
            crop_w = min(crop_w, W)
            
            # Random crop position (with safe bounds)
            max_top = max(0, H - crop_h)
            max_left = max(0, W - crop_w)
            top = torch.randint(0, max_top + 1, (1,), device=tensor.device).item() if max_top >= 0 else 0
            left = torch.randint(0, max_left + 1, (1,), device=tensor.device).item() if max_left >= 0 else 0
            
            # Crop (gradients only flow to cropped region)
            cropped = tensor[b:b+1, :, top:top+crop_h, left:left+crop_w]
            
            # Resize back to original size (differentiable)
            resized = F.interpolate(
                cropped,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            transformed.append(resized.squeeze(0))
        
        result = torch.stack(transformed, dim=0)
        if squeeze_back:
            result = result.squeeze(0)
        return result


    def robust_transform(self, img: torch.Tensor, jitter_size: int) -> torch.Tensor:
        """Add per-image jitter (wraparound), optional noise, flip, and crop."""
        # Early exit if no transformations enabled
        if (jitter_size <= 0 and 
            (self.robust_noise_type is None or self.robust_noise_std == 0.0) and
            self.robust_flip_prob <= 0.0 and
            self.robust_crop_prob <= 0.0):
            return img

        squeeze_back = False
        if img.dim() == 3:  # [C, H, W]
            img = img.unsqueeze(0)
            squeeze_back = True

        B = img.shape[0]
        
        # Apply jitter (circular shift)
        if jitter_size > 0:
            shifts_y = torch.randint(-jitter_size, jitter_size + 1, (B,), device=img.device)
            shifts_x = torch.randint(-jitter_size, jitter_size + 1, (B,), device=img.device)

            # roll each image separately to avoid sharing the same jitter
            transformed = []
            for b in range(B):
                transformed.append(
                    torch.roll(
                        img[b],
                        shifts=(shifts_y[b].item(), shifts_x[b].item()),
                        dims=(-2, -1),
                    )
                )
            transformed = torch.stack(transformed, dim=0)
        else:
            transformed = img

        # Apply noise
        transformed = self._apply_robust_noise(transformed)
        
        # Apply flip (horizontal/vertical)
        transformed = self._apply_robust_flip(transformed)
        
        # Apply crop
        transformed = self._apply_robust_crop(transformed)

        if squeeze_back:
            transformed = transformed.squeeze(0)
        return transformed

    def validate_candidate(self, candidate_image: torch.Tensor) -> float:
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": candidate_image},
                {"type": "text", "text": "Would you like to see another image similar to this one? Only answer yes or no."},
            ],
        }]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        if candidate_image.dim() == 4:
            img = candidate_image.squeeze(0)
        else:
            img = candidate_image

        inputs = self.processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
            do_rescale=False,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            logprob_yes = F.log_softmax(logits, dim=-1)[0, self.id_yes]
            logprob_no = F.log_softmax(logits, dim=-1)[0, self.id_no]
            prob_yes = torch.sigmoid(logprob_yes - logprob_no).item()

        del inputs, outputs, logits
        safe_empty_cuda_cache()
        return prob_yes



    def score_tensor(
        self,
        images: torch.Tensor,
        references: List[Union[torch.Tensor, Image.Image]],
        comparison_plan: List[ComparisonDefinition],
        batch_size: int = 20,
        loss_type: Optional[str] = None,
        candidate_images_forward: Optional[torch.Tensor] = None,
        compute_grad: bool = True,
    ) -> Tuple[float, torch.Tensor]:
        """Score candidates against references and return averaged gradients.

        Args:
            compute_grad: If False, skip gradient computation and return zero gradient.
                         Useful for forward-only evaluation during line search.
        """

        device = self.device
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.to(device)
        loss_type = loss_type
        candidate_images = candidate_images_forward if candidate_images_forward is not None else images
        if candidate_images_forward is None:
            # Avoid silently reapplying jitter/noise inside scorer. Expect caller to
            # supply the forward tensor explicitly when using robust transforms.
            print(
                "[PreferenceScorer] Warning: candidate_images_forward not provided; "
                "using raw images without jitter/noise."
            )

        num_refs = len(references)
        comparisons = comparison_plan
        
        # Group comparisons by size FIRST to ensure balanced distribution
        # This is important for varied-size comparisons to avoid load imbalance
        grouped_by_size: Dict[int, List[ComparisonDefinition]] = {}
        for comp in comparisons:
            grouped_by_size.setdefault(comp.group_size, []).append(comp)
        
        # Split comparisons across ranks for data parallelism
        # Split within each size group to maintain balanced load
        if self.world_size > 1:
            split_comparisons: List[ComparisonDefinition] = []
            for size, bucket in grouped_by_size.items():
                # Split this size bucket across ranks
                bucket_per_rank = len(bucket) // self.world_size
                remainder = len(bucket) % self.world_size
                
                if self.rank < remainder:
                    local_start = self.rank * (bucket_per_rank + 1)
                    local_end = local_start + bucket_per_rank + 1
                else:
                    local_start = self.rank * bucket_per_rank + remainder
                    local_end = local_start + bucket_per_rank
                
                split_comparisons.extend(bucket[local_start:local_end])
            
            comparisons = split_comparisons
            # Log from all ranks to verify splitting (each rank writes to same log file)
            print(f"[PreferenceScorer] Rank {self.rank}/{self.world_size}: Processing {len(comparisons)}/{len(comparison_plan)} comparisons")
            # Log distribution by size for debugging (only from rank 0 to avoid clutter)
            if self.rank == 0:
                local_grouped = {}
                for comp in comparisons:
                    local_grouped.setdefault(comp.group_size, []).append(comp)
                size_dist = {size: len(bucket) for size, bucket in local_grouped.items()}
                print(f"[PreferenceScorer] Rank {self.rank} size distribution: {size_dist}")
        
        total_loss: Optional[torch.Tensor] = None
        final_grad = torch.zeros_like(images)
        candidate_counts = torch.zeros(candidate_images.shape[0], device=device, dtype=torch.float32)
        effective_total = 0

        # Re-group the split comparisons for processing
        grouped: Dict[int, List[ComparisonDefinition]] = {}
        for comp in comparisons:
            grouped.setdefault(comp.group_size, []).append(comp)

        total_comparisons = len(comparisons)
        # Removed verbose bucketing log
        for size, bucket in grouped.items():
            if not bucket:
                continue
            
            # guarded with oom try-except blocks for both querying and
            # backpropagation 
            # Removed verbose bucket processing log
            chunk_start = 0
            bucket_len = len(bucket)
            while chunk_start < bucket_len:
                remaining = bucket_len - chunk_start
                chunk_size = min(max(1, batch_size), remaining)
                chunk_end = chunk_start + chunk_size
                chunk = bucket[chunk_start:chunk_end]

                # CRITICAL FIX: Recreate candidate_images from images for each chunk when noise/jitter is applied.
                # This ensures each chunk has a fresh computational graph. Without this, when noise is applied,
                # the graph gets freed after the first backward pass (retain_graph=False), causing subsequent
                # chunks to fail with "Trying to backward through the graph a second time" error.
                # Jitter alone doesn't have this issue because torch.roll doesn't create the same intermediate
                # tensor dependencies that noise operations (tensor + noise) do.
                if candidate_images_forward is not None:
                    # Reapply robust transform from current images to get fresh graph for this chunk
                    chunk_candidate_images = self.robust_transform(images, jitter_size=self.jitter_size)
                else:
                    chunk_candidate_images = candidate_images

                # Catch OOM errors during comparison - skip failed comparisons and continue
                try:
                    loss_tensor, processed_indices, precomputed_grad = self._compare_images_batch(
                        batch_comparisons=chunk,
                        candidate_images=chunk_candidate_images,
                        reference_tensors=references,
                        num_base_references=num_refs,
                        loss_type=loss_type,
                        images=images if compute_grad else None,
                        compute_grad=compute_grad,
                    )
                except RuntimeError as e:
                    # Check if this is a CUDA OOM error
                    error_msg = str(e).lower()
                    if "cuda" in error_msg and "out of memory" in error_msg:
                        # OOM error - skip this chunk and continue
                        safe_empty_cuda_cache()
                        print(f"[PreferenceScorer] OOM error during batch comparison, skipping chunk of {len(chunk)} comparisons. Error: {e}", flush=True)
                        chunk_start += chunk_size
                        continue
                    else:
                        # Non-OOM RuntimeError - re-raise
                        raise
                except Exception as e:
                    # Other errors (non-CUDA) - log and skip this chunk
                    safe_empty_cuda_cache()
                    print(f"[PreferenceScorer] Error during batch comparison, skipping chunk of {len(chunk)} comparisons. Error: {type(e).__name__}: {e}", flush=True)
                    chunk_start += chunk_size
                    continue

                if loss_tensor is None or not processed_indices:
                    chunk_start += chunk_size
                    continue

                chunk_loss_value = float(loss_tensor.item())

                # Only compute gradients if requested (skip for line search forward-only evaluation)
                if compute_grad:
                    try:
                        if precomputed_grad is not None:
                            # Gradient was already computed per-prompt (for num_prompt_samples > 1)
                            grad = precomputed_grad
                        else:
                            # Original behavior: compute gradient from averaged loss
                            # backprop through jitter/noise to the original images so updates are
                            # applied in the unjittered frame.
                            grad = torch.autograd.grad(
                                loss_tensor, images, retain_graph=False, create_graph=False
                            )[0]
                    except RuntimeError as e:
                        # Check if this is a CUDA OOM error during backprop
                        error_msg = str(e).lower()
                        if "cuda" in error_msg and "out of memory" in error_msg:
                            # OOM during backprop - skip this chunk's gradient
                            safe_empty_cuda_cache()
                            print(f"[PreferenceScorer] OOM error during gradient computation, skipping chunk of {len(processed_indices)} processed comparisons. Error: {e}", flush=True)
                            chunk_start += chunk_size
                            continue
                        else:
                            # Non-OOM RuntimeError - re-raise
                            raise
                else:
                    grad = torch.zeros_like(images)

                batch_candidate_indices = [comp.candidate_idx for comp in chunk]
                for idx in batch_candidate_indices:
                    candidate_counts[idx] += 1

                processed_counter = Counter(processed_indices)
                requested_counter = Counter(batch_candidate_indices)
                for idx, requested in requested_counter.items():
                    processed = processed_counter.get(idx, 0)
                    if processed < requested:
                        candidate_counts[idx] = max(0, candidate_counts[idx] - (requested - processed))

                processed_in_batch = len(processed_indices)
                final_grad = final_grad - grad
                total_loss = loss_tensor if total_loss is None else total_loss + loss_tensor
                effective_total += processed_in_batch

                del loss_tensor, grad
                safe_empty_cuda_cache()
                # Removed verbose chunk processing log
                chunk_start += chunk_size

        # Synchronize gradients and counts across ranks for data parallelism BEFORE averaging
        if self.world_size > 1 and compute_grad:
            # Sum final_grad across all ranks (each rank processed different comparisons)
            torch.distributed.all_reduce(final_grad, op=torch.distributed.ReduceOp.SUM)
            # Sum candidate_counts across ranks
            torch.distributed.all_reduce(candidate_counts, op=torch.distributed.ReduceOp.SUM)
            # Sum effective_total across ranks
            effective_total_tensor = torch.tensor(effective_total, device=self.device, dtype=torch.float32)
            torch.distributed.all_reduce(effective_total_tensor, op=torch.distributed.ReduceOp.SUM)
            effective_total = int(effective_total_tensor.item())
            # Also synchronize loss for consistent logging
            loss_tensor = torch.tensor(total_loss.item() if total_loss is not None else 0.0, device=self.device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            total_loss = loss_tensor.item()

        # scale the gradients: average by the number of actual comparisons
        # (discard the failed ones)
        avg_grad = torch.zeros_like(final_grad)
        candidate_counts = torch.clamp(candidate_counts, min=0)
        nonzero_mask = candidate_counts > 0
        if nonzero_mask.any():
            counts = candidate_counts[nonzero_mask].to(final_grad.dtype).view(-1, 1, 1, 1)
            avg_grad[nonzero_mask] = final_grad[nonzero_mask] / counts

        if effective_total <= 0:
            raise RuntimeError("All comparisons were skipped due to repeated OOM errors.")

        if total_loss is None:
            raise RuntimeError("No loss accumulated despite positive effective_total.")

        avg_loss = total_loss / effective_total
        avg_score = -avg_loss
        return avg_score, avg_grad

    def _compare_images_batch(
        self,
        batch_comparisons: List[ComparisonDefinition],
        candidate_images: torch.Tensor,
        reference_tensors: Optional[List[Union[torch.Tensor, Image.Image]]] = None,
        num_base_references: int = 0,
        loss_type: str = "cross_entropy",
        images: Optional[torch.Tensor] = None,
        compute_grad: bool = False,
    ) -> Tuple[Optional[torch.Tensor], List[int], Optional[torch.Tensor]]:
        """Run the preference model for a batch of comparisons with shared group size."""

        if not batch_comparisons:
            return None, []

        if reference_tensors is None:
            raise ValueError("reference_tensors must be provided when comparisons only include indices.")

        batch_images: List[List[torch.Tensor]] = []
        labels: List[int] = []
        processed_indices: List[int] = []

        batch_text_options: List[List[str]] = []
        
        for comp in batch_comparisons:
            cand_idx = comp.candidate_idx
            ref_indices = comp.reference_indices
            candidate_pos = comp.candidate_pos
            text_options = getattr(comp, 'text_options', None) or []

            candidate = candidate_images[cand_idx]
            comparison_imgs: List[torch.Tensor] = []

            for idx_ref, ref_idx in enumerate(ref_indices):
                if ref_idx < num_base_references:
                    ref_source = reference_tensors[ref_idx]
                    ref_tensor = (
                        ref_source
                        if isinstance(ref_source, torch.Tensor)
                        else PreferenceOptimizer._pil_to_tensor(ref_source)
                    )
                    ref_tensor = ref_tensor.to(self.device)
                else:
                    peer_idx = ref_idx - num_base_references
                    if peer_idx < 0 or peer_idx >= candidate_images.shape[0]:
                        raise IndexError(
                            f"Peer index {peer_idx} out of range for candidate batch size "
                            f"{candidate_images.shape[0]}"
                        )
                    ref_tensor = candidate_images[peer_idx].detach()

                if idx_ref == candidate_pos:
                    comparison_imgs.append(candidate)
                comparison_imgs.append(ref_tensor)

            if candidate_pos == len(ref_indices):
                comparison_imgs.append(candidate)

            batch_images.append(comparison_imgs)
            batch_text_options.append(text_options)
            labels.append(candidate_pos)
            processed_indices.append(cand_idx)

        def _build_conversations(prompt_question: Optional[str] = None, label_scheme=None, template=None, is_negative: bool = False):
            """Build conversations using flexible format system or legacy format."""
            conversations: List[str] = []
            batch_target_tokens: List[List[str]] = []
            batch_is_negative: List[bool] = []
            
            if self.use_flexible_format:
                # Use new flexible format system
                for idx, comparison_imgs in enumerate(batch_images):
                    num_images = len(comparison_imgs)
                    text_options = batch_text_options[idx] if idx < len(batch_text_options) else []
                    num_text = len(text_options)
                    num_total = num_images + num_text
                    original_candidate_pos = labels[idx] if idx < len(labels) else 0

                    # Use deterministic RNG for shuffling to ensure consistency
                    item_rng = random.Random(idx * 12345)  # Deterministic based on comparison index

                    # Decide whether to force a negative (inverted) question
                    force_neg = (self.negative_question_prob > 0 and
                                 item_rng.random() < self.negative_question_prob)

                    # If forcing negative, truncate to pairwise (2 items) since
                    # negative templates are pairwise-only
                    if force_neg and num_total > 2:
                        if num_text > 0:
                            # Hybrid: keep candidate image + 1 random text option
                            comparison_imgs = [comparison_imgs[original_candidate_pos]]
                            text_options = [item_rng.choice(text_options)]
                            labels[idx] = 0
                            original_candidate_pos = 0
                            num_images = 1
                            num_text = 1
                            num_total = 2
                        else:
                            # Images only: keep candidate + 1 random reference
                            other_indices = [i for i in range(num_images) if i != original_candidate_pos]
                            other_idx = item_rng.choice(other_indices)
                            if original_candidate_pos < other_idx:
                                comparison_imgs = [comparison_imgs[original_candidate_pos], comparison_imgs[other_idx]]
                                labels[idx] = 0
                                original_candidate_pos = 0
                            else:
                                comparison_imgs = [comparison_imgs[other_idx], comparison_imgs[original_candidate_pos]]
                                labels[idx] = 1
                                original_candidate_pos = 1
                            num_images = 2
                            num_total = 2
                        # Update the batch lists so _run_forward sees the
                        # truncated images/text matching the conversation
                        batch_images[idx] = comparison_imgs
                        batch_text_options[idx] = text_options

                    # If we have text options, use hybrid question format
                    if num_text > 0:
                        # Use hybrid question templates from HYBRID_QUESTION_CONFIGS
                        # Filter negative questions based on number of items (negative questions are pairwise only)
                        question_template_key, question_config = sample_hybrid_question_config(
                            rng=item_rng,
                            
                            allow_negative=True,
                            num_items=num_total,
                            force_negative=force_neg,
                        )
                        # Sample a label scheme that supports the total number of items
                        valid_schemes = [s for s in LABEL_SCHEMES.values() if len(s.labels) >= num_total]
                        if not valid_schemes:
                            raise ValueError(f"No label scheme supports {num_total} items")
                        scheme = item_rng.choice(valid_schemes)

                        # Format hybrid prompt (images + text)
                        # Pass candidate position to track it through shuffling
                        content, labels_used, target_tokens_used, new_candidate_pos, is_negative = format_hybrid_comparison_prompt(
                            images=comparison_imgs,
                            text_options=text_options,
                            label_scheme=scheme,
                            question_template_key=question_template_key,
                            question_config=question_config,
                            rng=item_rng,
                            candidate_pos=original_candidate_pos,
                        )
                        # Update label to reflect new candidate position after shuffling
                        if new_candidate_pos is not None:
                            labels[idx] = new_candidate_pos
                        batch_target_tokens.append(target_tokens_used)
                        batch_is_negative.append(is_negative)
                    else:
                        # Images only: use original question format system
                        scheme, question_template, neg = sample_comparison_format(
                            num_images=num_images,
                            rng=item_rng,
                            allow_negative=True,
                            
                            force_negative=force_neg,
                        )
                        # Format the prompt (images only)
                        content, labels_used, target_tokens_used = format_comparison_prompt(
                            images=comparison_imgs,
                            label_scheme=scheme,
                            template=question_template,
                            rng=item_rng,
                        )
                        batch_target_tokens.append(target_tokens_used)
                        batch_is_negative.append(neg)
                    
                    # Note: candidate_pos adjustment for shuffled items is complex
                    # For now, we keep original_candidate_pos but this may need refinement
                    # The actual candidate position after shuffling depends on the shuffle order
                    # This is a limitation that may need addressing if precise candidate tracking is needed
                    
                    conversation = [{"role": "user", "content": content}]
                    conversations.append(
                        self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                    )
            else:
                # Legacy format: use prompt_question string
                if prompt_question is None:
                    prompt_question = self.question_pool[0] if self.question_pool else "Which of the following images do you prefer?"
                for idx, comparison_imgs in enumerate(batch_images):
                    text_options = batch_text_options[idx] if idx < len(batch_text_options) else []
                    num_images = len(comparison_imgs)
                    num_text = len(text_options)
                    num_total = num_images + num_text
                    
                    # Build items list and shuffle
                    items: List[Tuple[bool, Any]] = []
                    for img in comparison_imgs:
                        items.append((True, img))
                    for text_str in text_options:
                        items.append((False, text_str))
                    random.shuffle(items)
                    
                    # Determine question phrasing based on content
                    # Note: num_images is always >= 1 (includes candidate)
                    if num_text > 0:
                        # Mixed: images + text
                        question = prompt_question.replace("images", "images or states of the world").replace("image", "image or state of the world")
                    else:
                        # Images only
                        question = prompt_question
                    
                    content = [{"type": "text", "text": question}]
                    for option_idx, (is_image, item) in enumerate(items):
                        content.append({"type": "text", "text": f" {chr(65 + option_idx)}: "})
                        if is_image:
                            content.append({"type": "image", "image": item})
                        else:
                            content.append({"type": "text", "text": item})
                    options = ", ".join(chr(65 + j) for j in range(num_total))
                    content.append({"type": "text", "text": f"? Please respond with only one letter from [{options}]."})
                    conversation = [{"role": "user", "content": content}]
                    conversations.append(
                        self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                    )
                    # Legacy: always use A, B, C, etc.
                    batch_target_tokens.append([chr(65 + j) for j in range(num_total)])
                    batch_is_negative.append(False)
            
            return conversations, batch_target_tokens, batch_is_negative

        def _run_forward(
            images_for_model: List[List[torch.Tensor]],
            conv_subset: List[str],
            label_subset: List[int],
            target_tokens_list: List[List[str]],
            is_negative_list: List[bool],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            inputs = self.processor(
                text=conv_subset,
                images=images_for_model,
                return_tensors="pt",
                padding=True,
                do_rescale=False,
            ).to(self.device)
            
            if "pixel_values" in inputs:
                del inputs["pixel_values"]

            flat_images = [img for sublist in images_for_model for img in sublist]
            grad_pixel_values, grad_image_grid_thw = self.grad_img_processor.preprocess(
                images=flat_images,
                do_resize=True,
                size=self.processor.image_processor.size,
                interpolation=self._get_interpolation_mode(),
                do_rescale=False,
                do_normalize=True,
                image_mean=self.processor.image_processor.image_mean,
                image_std=self.processor.image_processor.image_std,
                patch_size=self.processor.image_processor.patch_size,
                temporal_patch_size=self.processor.image_processor.temporal_patch_size,
                merge_size=self.processor.image_processor.merge_size,
                max_dimension=self.max_image_dimension,
            )
            inputs["pixel_values"] = grad_pixel_values
            inputs["image_grid_thw"] = grad_image_grid_thw

            outputs = self.model(**inputs)
            logits_last = outputs.logits[:, -1, :]

            # Use max target tokens length to account for text options
            max_choice_count = max(len(tokens) for tokens in target_tokens_list) if target_tokens_list else len(images_for_model[0])
            logits = torch.zeros(len(images_for_model), max_choice_count, device=self.device)
            for i in range(len(images_for_model)):
                target_tokens = target_tokens_list[i]
                for j in range(len(target_tokens)):
                    token_str = target_tokens[j]
                    # Encode the target token (could be "A", "1", "First", etc.)
                    token_ids = self.processor.tokenizer.encode(token_str, add_special_tokens=False)
                    if token_ids:
                        token_id = token_ids[0]  # Use first token ID
                        logits[i, j] = logits_last[i, token_id]
                    else:
                        # Fallback to legacy behavior if encoding fails
                        token_id = self.processor.tokenizer.encode(chr(65 + j), add_special_tokens=False)[0]
                        logits[i, j] = logits_last[i, token_id]

            # Handle negative questions: invert labels
            labels_adjusted = []
            for i, orig_label in enumerate(label_subset):
                if is_negative_list[i]:
                    # For negative questions, invert the label (0->1, 1->0, etc.)
                    # Use the actual number of choices for this comparison
                    num_choices = len(target_tokens_list[i]) if i < len(target_tokens_list) else max_choice_count
                    labels_adjusted.append(num_choices - 1 - orig_label)
                else:
                    labels_adjusted.append(orig_label)
            
            labels_tensor_local = torch.tensor(labels_adjusted, dtype=torch.long, device=self.device)

            del inputs, grad_pixel_values, grad_image_grid_thw, outputs, logits_last
            safe_empty_cuda_cache()
            return logits, labels_tensor_local

        # Support multiple prompt samples: query with different prompts and average losses/gradients
        # When num_prompt_samples > 1 and compute_grad=True, we compute gradients per prompt
        # and average them to avoid OOM (instead of averaging losses first, which keeps all graphs in memory).
        all_losses: List[torch.Tensor] = []
        all_grads: List[torch.Tensor] = []
        
        # select prompts based on num_prompt_samples
        # when num_prompt_samples=1, the behavior is:
        #   - randomize_preference_prompt=True:  random prompt per batch (can change between batches within a step)
        #   - randomize_preference_prompt=False: fixed prompt (always question_pool[0])
        # Uses self.question_pool which may be the original PREFERENCE_QUESTIONS or generated variants
        if self.num_prompt_samples > 1:
            # use multiple different prompts
            if len(self.question_pool) >= self.num_prompt_samples:
                # use first n prompts if we have enough
                selected_prompts = self.question_pool[:self.num_prompt_samples]
            else:
                # if not enough prompts, sample with replacement
                selected_prompts = random.choices(self.question_pool, k=self.num_prompt_samples)
        elif self.randomize_preference_prompt:
            # single random prompt per batch (original behavior)
            # NOTE: This selects a new random prompt for each call to _compare_images_batch,
            # which happens once per batch/chunk, so the prompt can vary within a single optimization step
            selected_prompts = [random.choice(self.question_pool)]
        else:
            # single fixed prompt (original behavior)
            # always uses the first prompt: question_pool[0]
            selected_prompts = [self.question_pool[0]]

        # For multiple prompts with gradients: compute gradient per prompt to avoid OOM
        if self.num_prompt_samples > 1 and compute_grad and images is not None:
            # Process each prompt separately, compute gradient immediately, then free graph
            for prompt_question in selected_prompts:
                conversations, target_tokens_list, is_negative_list = _build_conversations(
                    prompt_question if not self.use_flexible_format else None
                )
                logits, labels_tensor = _run_forward(
                    batch_images, conversations, labels, target_tokens_list, is_negative_list
                )

                if loss_type == "cross_entropy":
                    loss = F.cross_entropy(logits, labels_tensor.to(self.device), reduction="sum")
                elif loss_type == "margin":
                    batch_losses = []
                    for i in range(len(labels_tensor)):
                        y = labels_tensor[i].item()
                        logit_y = logits[i, y]
                        mask = torch.ones_like(logits[i], dtype=torch.bool)
                        mask[y] = False
                        logit_max_other = logits[i][mask].max()
                        batch_losses.append(-(logit_y - logit_max_other))
                    loss = torch.stack(batch_losses).sum()
                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")
                
                all_losses.append(loss.detach())  # Detach for loss value tracking
                
                # Compute gradient immediately and free the graph
                grad = torch.autograd.grad(
                    loss, images, retain_graph=False, create_graph=False
                )[0]
                all_grads.append(grad)

                # Free memory
                del loss, logits, labels_tensor, conversations
                safe_empty_cuda_cache()
            
            # Average gradients
            if len(all_grads) > 1:
                avg_grad = torch.stack(all_grads).mean(dim=0)
            else:
                avg_grad = all_grads[0]
            
            # Return averaged loss value (detached) and averaged gradient
            avg_loss = torch.stack(all_losses).mean() if len(all_losses) > 1 else all_losses[0]
            return avg_loss, processed_indices, avg_grad
        else:
            # Original behavior: single prompt or no gradient computation
            # run forward pass for each prompt and collect losses
            for prompt_question in selected_prompts:
                conversations, target_tokens_list, is_negative_list = _build_conversations(
                    prompt_question if not self.use_flexible_format else None
                )
                logits, labels_tensor = _run_forward(
                    batch_images, conversations, labels, target_tokens_list, is_negative_list
                )

                if loss_type == "cross_entropy":
                    loss = F.cross_entropy(logits, labels_tensor.to(self.device), reduction="sum")
                elif loss_type == "margin":
                    batch_losses = []
                    for i in range(len(labels_tensor)):
                        y = labels_tensor[i].item()
                        logit_y = logits[i, y]
                        mask = torch.ones_like(logits[i], dtype=torch.bool)
                        mask[y] = False
                        logit_max_other = logits[i][mask].max()
                        batch_losses.append(-(logit_y - logit_max_other))
                    loss = torch.stack(batch_losses).sum()
                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")
                
                all_losses.append(loss)

            # average losses across all prompt samples
            if len(all_losses) > 1:
                loss = torch.stack(all_losses).mean()
            else:
                loss = all_losses[0]

            return loss, processed_indices, None

    def _batch_pairwise_preference(self, candidate_refs: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[float]:
        """Getting the absolute utility score for the current stimuli. Forward-only."""
        if not candidate_refs:
            return []

        batch_limit = getattr(self, "buffer_comparison_batch_size", len(candidate_refs))
        batch_limit = max(1, int(batch_limit))
        all_probs: List[float] = []

        for start in range(0, len(candidate_refs), batch_limit):
            chunk = candidate_refs[start:start + batch_limit]
            conversations: List[str] = []
            image_payloads: List[List[torch.Tensor]] = []
            for candidate_img, reference_img in chunk:
                conv = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Which image do you prefer? A:"},
                        {"type": "image", "image": candidate_img},
                        {"type": "text", "text": " B:"},
                        {"type": "image", "image": reference_img},
                        {"type": "text", "text": " Respond with only 'A' or 'B'."},
                    ],
                }]
                conversations.append(
                    self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                )
                image_payloads.append([candidate_img, reference_img])

            inputs = self.processor(
                text=conversations,
                images=image_payloads,
                return_tensors="pt",
                padding=True,
                do_rescale=False,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                logprob_A = F.log_softmax(logits, dim=-1)[:, self.id_A]
                logprob_B = F.log_softmax(logits, dim=-1)[:, self.id_B]
                probs = torch.sigmoid(logprob_A - logprob_B)

            all_probs.extend(probs.detach().cpu().tolist())

            del inputs, outputs, logits, conversations, image_payloads
            safe_empty_cuda_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_probs

    def _batch_pairwise_preference_with_gradients(
        self,
        candidate_images: torch.Tensor,
        candidate_buffer_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        pair_indices: List[Tuple[int, int]],
        pair_flips: List[bool],
        candidate_count: int,
        buffer_count: int,
        chunk_size: Optional[int] = None,
        candidate_images_forward: Optional[torch.Tensor] = None,
    ) -> Tuple[
        List[Optional[float]],
        List[Optional[float]],
        Dict[Tuple[int, int], float],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Pair-wise comparison with candidate and buffer images. Gradients
        enabled. Comparisons processed in mini-batches."""
        total_pairs = len(candidate_buffer_pairs)
        candidate_scores_list: List[Optional[float]] = [None] * candidate_count
        buffer_scores_list: List[Optional[float]] = [None] * buffer_count
        pair_probs: Dict[Tuple[int, int], float] = {}

        if total_pairs == 0:
            return candidate_scores_list, buffer_scores_list, pair_probs, None, None

        candidate_prob_sums = [0.0] * candidate_count
        candidate_counts = [0.0] * candidate_count
        buffer_prob_sums = [0.0] * buffer_count
        buffer_counts = [0.0] * buffer_count

        batch_limit = max(1, int(chunk_size or self.buffer_comparison_batch_size))
        total_grad = torch.zeros_like(candidate_images)
        total_logprob = 0.0
        processed_pairs = 0

        # go through the mini-batches 
        for start in range(0, total_pairs, batch_limit):
            end = min(start + batch_limit, total_pairs)
            chunk_pairs = candidate_buffer_pairs[start:end]
            chunk_indices = pair_indices[start:end]
            chunk_flips = pair_flips[start:end]

            # CRITICAL FIX: Recreate candidate_images_forward from candidate_images for each chunk
            # when noise/jitter is applied, to avoid the same graph-freed issue as in score_tensor
            if candidate_images_forward is not None:
                chunk_candidate_images_forward = self.robust_transform(candidate_images, jitter_size=self.jitter_size)
            else:
                chunk_candidate_images_forward = candidate_images

            conversations: List[str] = []
            image_payloads: List[List[torch.Tensor]] = []
            candidate_label_indices: List[int] = []  # Store candidate label index for each pair
            
            # Sample question template for buffer comparisons (pairwise only)
            # Use deterministic RNG based on chunk start for reproducibility across chunks
            buffer_rng = random.Random(start)
            scheme, question_template, is_negative = sample_comparison_format(
                num_images=2,  # Buffer comparisons are always pairwise
                rng=buffer_rng,
                allow_negative=False,  # Don't use negative questions for buffer comparisons
                
            )
            
            # Get target tokens for token ID lookup (will be same for all pairs in chunk since template is same)
            target_tokens = scheme.get_target_tokens(2)  # e.g., ["A", "B"] or ["1", "2"] or ["First", "Second"]
            
            for pair_idx, ((cand_idx, buf_idx), (cand_tensor, buf_tensor), flip) in enumerate(zip(
                chunk_indices, chunk_pairs, chunk_flips
            )):
                candidate_img = chunk_candidate_images_forward[cand_idx]
                reference_img = buf_tensor
                
                # Build comparison images list (order matters for template)
                # Position 0 = first image, Position 1 = second image
                # flip=False: candidate first, buffer second -> candidate at position 0
                # flip=True: buffer first, candidate second -> candidate at position 1
                comparison_imgs = [candidate_img, reference_img] if not flip else [reference_img, candidate_img]
                
                # Format using question template system
                # Note: format_comparison_prompt doesn't shuffle images by default, so order is preserved
                pair_rng = random.Random(start + pair_idx * 12345)  # Different seed per pair for variation
                content, labels_used, target_tokens_used = format_comparison_prompt(
                    images=comparison_imgs,
                    label_scheme=scheme,
                    template=question_template,
                    rng=pair_rng,
                )
                
                # Store candidate position: 0 if candidate is first (not flipped), 1 if candidate is second (flipped)
                candidate_label_idx = 0 if not flip else 1
                candidate_label_indices.append(candidate_label_idx)
                
                conv = [{"role": "user", "content": content}]
                conversations.append(
                    self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                )
                image_payloads.append(comparison_imgs)

            inputs = self.processor(
                text=conversations,
                images=image_payloads,
                return_tensors="pt",
                padding=True,
                do_rescale=False,
            ).to(self.device)
            if "pixel_values" in inputs:
                del inputs["pixel_values"]

            flat_images = [img for pair in image_payloads for img in pair]
            grad_pixel_values, grad_image_grid_thw = self.grad_img_processor.preprocess(
                images=flat_images,
                do_resize=True,
                size=self.processor.image_processor.size,
                interpolation=self._get_interpolation_mode(),
                do_rescale=False,
                do_normalize=True,
                image_mean=self.processor.image_processor.image_mean,
                image_std=self.processor.image_processor.image_std,
                patch_size=self.processor.image_processor.patch_size,
                temporal_patch_size=self.processor.image_processor.temporal_patch_size,
                merge_size=self.processor.image_processor.merge_size,
                max_dimension=self.max_image_dimension,
            )
            inputs["pixel_values"] = grad_pixel_values
            inputs["image_grid_thw"] = grad_image_grid_thw

            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            
            # Get token IDs for the target tokens from the scheme
            # target_tokens should be the output tokens (e.g., ["A", "B"] or ["1", "2"] or ["First", "Second"])
            tokenizer = self.processor.tokenizer
            token_ids = []
            for token_str in target_tokens:
                encoded = tokenizer.encode(token_str, add_special_tokens=False)
                if encoded:
                    token_ids.append(encoded[0])  # Take first token ID
                else:
                    # Fallback: try lowercase
                    encoded_lower = tokenizer.encode(token_str.lower(), add_special_tokens=False)
                    if encoded_lower:
                        token_ids.append(encoded_lower[0])
                    else:
                        # Ultimate fallback to A/B based on position
                        if len(token_ids) == 0:
                            token_ids.append(self.id_A)
                        else:
                            token_ids.append(self.id_B)
            
            # Ensure we have at least 2 token IDs
            while len(token_ids) < 2:
                token_ids.append(self.id_B if len(token_ids) == 1 else self.id_A)
            
            token_id_0 = token_ids[0]  # Token ID for position 0 (first image in template)
            token_id_1 = token_ids[1]  # Token ID for position 1 (second image in template)
            
            logit_0 = logits[:, token_id_0]
            logit_1 = logits[:, token_id_1]
            logit_diffs = logit_0 - logit_1  # Positive = position 0 is preferred/hated more
            
            # Adjust based on candidate position in each pair
            # cand_diff > 0 means candidate is preferred/hated more (which is what we want to maximize)
            # When candidate is at position 0: use logit_diff as-is (positive = candidate wins)
            # When candidate is at position 1: flip sign (negative logit_diff = candidate wins, so negate to make positive)
            cand_diff_list = []
            for i, candidate_label_idx in enumerate(candidate_label_indices):
                if candidate_label_idx == 0:
                    # Candidate is at position 0, so logit_0 corresponds to candidate
                    cand_diff_list.append(logit_diffs[i])
                else:
                    # Candidate is at position 1, so logit_1 corresponds to candidate
                    # logit_diff = logit_0 - logit_1, so we need logit_1 - logit_0 = -logit_diff
                    cand_diff_list.append(-logit_diffs[i])
            
            cand_diff = torch.stack(cand_diff_list)

            log_probs = F.logsigmoid(cand_diff)
            probs = torch.sigmoid(cand_diff)
            logprob_sum = log_probs.sum()

            # derivative of sum(log p(candidate>buffer)) / candidate_images
            grad = torch.autograd.grad(
                logprob_sum,
                candidate_images,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                grad = torch.zeros_like(candidate_images)
            total_grad += grad.detach()
            total_logprob += float(logprob_sum.detach().cpu().item())
            processed_pairs += cand_diff.shape[0]

            prob_values = probs.detach().cpu().tolist()
            for (cand_idx, buf_idx), p in zip(chunk_indices, prob_values):
                candidate_prob_sums[cand_idx] += float(p)
                candidate_counts[cand_idx] += 1.0
                buffer_prob_sums[buf_idx] += float(1.0 - p)
                buffer_counts[buf_idx] += 1.0
                pair_probs[(cand_idx, buf_idx)] = float(p)

            del inputs, outputs, logits, conversations, image_payloads
            del grad_pixel_values, grad_image_grid_thw, grad, cand_diff, probs, log_probs, logprob_sum
            safe_empty_cuda_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if processed_pairs == 0:
            return candidate_scores_list, buffer_scores_list, pair_probs, None, None

        for i in range(candidate_count):
            if candidate_counts[i] > 0:
                candidate_scores_list[i] = float(candidate_prob_sums[i] / candidate_counts[i])
        for j in range(buffer_count):
            if buffer_counts[j] > 0:
                buffer_scores_list[j] = float(buffer_prob_sums[j] / buffer_counts[j])

        # Normalize gradients per-candidate (not by total pairs)
        # Each candidate may appear in different numbers of pairs, so we need to normalize
        # by the number of pairs each candidate participated in, not the total number of pairs
        avg_grad = torch.zeros_like(total_grad)
        candidate_counts_tensor = torch.tensor(candidate_counts, device=self.device, dtype=total_grad.dtype)
        candidate_counts_tensor = torch.clamp(candidate_counts_tensor, min=1.0)  # Avoid division by zero
        candidate_counts_tensor = candidate_counts_tensor.view(-1, 1, 1, 1)  # [K, 1, 1, 1] for broadcasting
        avg_grad = total_grad / candidate_counts_tensor
        buffer_objective = torch.tensor(total_logprob / processed_pairs, device=self.device)    # average log p(candidate>buffer) over all pairs
        return candidate_scores_list, buffer_scores_list, pair_probs, buffer_objective, avg_grad

    def compare(self, candidate: torch.Tensor, reference: torch.Tensor) -> float:
        """Return the win probability for a single candidate-reference pair."""
        probs = self._batch_pairwise_preference([(candidate, reference)])
        if not probs:
            return 0.0
        return float(probs[0])

    def estimate_candidate_utilities(
        self,
        candidate_images: torch.Tensor,
        anchor_pool: List[Dict[str, Any]],
        max_pairs: int = 8,
    ) -> Tuple[List[Optional[float]], List[str]]:
        if not anchor_pool:
            return [None] * candidate_images.shape[0], []

        bounded_pairs = max(1, max_pairs)
        sorted_pool = sorted(anchor_pool, key=lambda entry: (-float(entry["score"]), entry["path"]))
        if len(sorted_pool) <= bounded_pairs:
            selected = list(sorted_pool)
        else:
            candidate_indices = np.linspace(0, len(sorted_pool) - 1, num=bounded_pairs).astype(int)
            seen = set()
            selected = []
            for idx in candidate_indices:
                if idx not in seen:
                    selected.append(sorted_pool[idx])
                    seen.add(idx)
            if len(selected) < bounded_pairs:
                for entry in sorted_pool:
                    if entry not in selected:
                        selected.append(entry)
                        if len(selected) == bounded_pairs:
                            break
        if not selected:
            return [None] * candidate_images.shape[0], []

        candidate_refs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        index_map: List[Tuple[int, float]] = []
        anchor_paths: List[str] = [entry["path"] for entry in selected]

        for cand_idx in range(candidate_images.shape[0]):
            cand_tensor = candidate_images[cand_idx].detach().clamp(0.0, 1.0)
            cand_tensor = cand_tensor.to(self.device)
            for entry in selected:
                ref_tensor = self._load_original_tensor(entry["path"])
                if ref_tensor is None:
                    ref_tensor = entry["tensor"]
                ref_tensor = ref_tensor.detach().clamp(0.0, 1.0).to(self.device)
                candidate_refs.append((cand_tensor, ref_tensor))
                index_map.append((cand_idx, float(entry["score"])))

        probs = self._batch_pairwise_preference(candidate_refs)
        utility_lists: Dict[int, List[float]] = {idx: [] for idx in range(candidate_images.shape[0])}

        for (cand_idx, ref_score), prob in zip(index_map, probs):
            p = float(min(max(prob, 1e-6), 1.0 - 1e-6))
            delta = math.log(p / (1.0 - p))
            utility_lists[cand_idx].append(ref_score + delta)

        aggregated: List[Optional[float]] = []
        for cand_idx in range(candidate_images.shape[0]):
            values = utility_lists[cand_idx]
            if values:
                aggregated.append(float(np.mean(values)))
            else:
                aggregated.append(None)
        return aggregated, anchor_paths

    def score_against_anchor(
        self,
        candidate_images: torch.Tensor,
        anchor_images: torch.Tensor,
        candidate_images_forward: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor]]:
        """Compare candidates against anchor images and return win probability + gradient.

        This is used for rolling buffer comparison where we want current candidates
        to beat a saved anchor batch from previous steps.

        Returns:
            (avg_win_probability, gradient) where gradient maximizes P(candidate > anchor)
        """
        device = self.device
        if candidate_images.dim() == 3:
            candidate_images = candidate_images.unsqueeze(0)
        if anchor_images.dim() == 3:
            anchor_images = anchor_images.unsqueeze(0)

        candidate_images = candidate_images.to(device)
        anchor_images = anchor_images.to(device)
        num_candidates = candidate_images.shape[0]
        num_anchors = anchor_images.shape[0]

        if num_candidates == 0 or num_anchors == 0:
            return None, None

        # Build pairwise comparisons: each candidate vs each anchor
        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        pair_indices: List[Tuple[int, int]] = []
        pair_flips: List[bool] = []

        # Use forward images for comparison if provided
        forward_images = candidate_images_forward if candidate_images_forward is not None else candidate_images

        for cand_idx in range(num_candidates):
            for anch_idx in range(num_anchors):
                flip = bool(torch.randint(low=0, high=2, size=(1,)).item())
                if not flip:
                    pairs.append((forward_images[cand_idx], anchor_images[anch_idx]))
                else:
                    pairs.append((anchor_images[anch_idx], forward_images[cand_idx]))
                pair_indices.append((cand_idx, anch_idx))
                pair_flips.append(flip)

        if not pairs:
            return None, None

        # Process comparisons
        total_grad = torch.zeros_like(candidate_images)
        total_logprob = 0.0
        total_pairs_processed = 0

        batch_limit = max(1, self.buffer_comparison_batch_size)

        for start in range(0, len(pairs), batch_limit):
            end = min(start + batch_limit, len(pairs))
            chunk_pairs = pairs[start:end]
            chunk_flips = pair_flips[start:end]
            chunk_indices = pair_indices[start:end]  # Track indices for this chunk

            # Sample question template for rolling buffer comparisons (pairwise only)
            buffer_rng = random.Random(start)
            scheme, question_template, is_negative = sample_comparison_format(
                num_images=2,  # Rolling buffer comparisons are always pairwise
                rng=buffer_rng,
                allow_negative=False,  # Don't use negative questions for buffer comparisons
                
            )
            target_tokens = scheme.get_target_tokens(2)

            # Build conversations for pairwise preference using question templates
            conversations: List[str] = []
            image_payloads: List[List[torch.Tensor]] = []
            candidate_label_indices: List[int] = []  # Track which position (0 or 1) is the candidate

            for pair_idx, ((img_a, img_b), flip) in enumerate(zip(chunk_pairs, chunk_flips)):
                # img_a is candidate if flip=False, img_b is candidate if flip=True
                # Build comparison images: [candidate, anchor] if not flip, [anchor, candidate] if flip
                comparison_imgs = [img_a, img_b] if not flip else [img_b, img_a]
                
                # Format using question template system
                pair_rng = random.Random(start + pair_idx * 54321)
                content, labels_used, target_tokens_used = format_comparison_prompt(
                    images=comparison_imgs,
                    label_scheme=scheme,
                    template=question_template,
                    rng=pair_rng,
                )
                
                # Track candidate position: 0 if candidate is first (not flipped), 1 if second (flipped)
                candidate_label_idx = 0 if not flip else 1
                candidate_label_indices.append(candidate_label_idx)
                
                conv = [{"role": "user", "content": content}]
                conversations.append(
                    self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                )
                image_payloads.append(comparison_imgs)

            inputs = self.processor(
                text=conversations,
                images=image_payloads,
                return_tensors="pt",
                padding=True,
                do_rescale=False,
            ).to(device)

            if "pixel_values" in inputs:
                del inputs["pixel_values"]

            flat_images = [img for pair in image_payloads for img in pair]
            grad_pixel_values, grad_image_grid_thw = self.grad_img_processor.preprocess(
                images=flat_images,
                do_resize=True,
                size=self.processor.image_processor.size,
                interpolation=self._get_interpolation_mode(),
                do_rescale=False,
                do_normalize=True,
                image_mean=self.processor.image_processor.image_mean,
                image_std=self.processor.image_processor.image_std,
                patch_size=self.processor.image_processor.patch_size,
                temporal_patch_size=self.processor.image_processor.temporal_patch_size,
                merge_size=self.processor.image_processor.merge_size,
                max_dimension=self.max_image_dimension,
            )
            inputs["pixel_values"] = grad_pixel_values
            inputs["image_grid_thw"] = grad_image_grid_thw

            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            
            # Get token IDs for target tokens
            tokenizer = self.processor.tokenizer
            token_ids = []
            for token_str in target_tokens:
                encoded = tokenizer.encode(token_str, add_special_tokens=False)
                if encoded:
                    token_ids.append(encoded[0])
                else:
                    encoded_lower = tokenizer.encode(token_str.lower(), add_special_tokens=False)
                    if encoded_lower:
                        token_ids.append(encoded_lower[0])
                    else:
                        token_ids.append(self.id_B if len(token_ids) == 1 else self.id_A)
            
            while len(token_ids) < 2:
                token_ids.append(self.id_B if len(token_ids) == 1 else self.id_A)
            
            token_id_0 = token_ids[0]
            token_id_1 = token_ids[1]
            
            logit_0 = logits[:, token_id_0]
            logit_1 = logits[:, token_id_1]
            logit_diffs = logit_0 - logit_1  # Positive = position 0 is preferred/hated more
            
            # Adjust based on candidate position: cand_diff > 0 means candidate is preferred/hated more
            cand_diff_list = []
            for i, candidate_label_idx in enumerate(candidate_label_indices):
                if candidate_label_idx == 0:
                    cand_diff_list.append(logit_diffs[i])
                else:
                    cand_diff_list.append(-logit_diffs[i])
            
            cand_diff = torch.stack(cand_diff_list)

            log_probs = F.logsigmoid(cand_diff)
            logprob_sum = log_probs.sum()

            # Compute gradient
            grad = torch.autograd.grad(
                logprob_sum,
                candidate_images,
                retain_graph=False,
                create_graph=False,
            )[0]

            total_grad = total_grad + grad
            total_logprob += logprob_sum.item()
            total_pairs_processed += len(chunk_pairs)

        if total_pairs_processed == 0:
            return None, None

        # Track per-candidate counts (how many pairs each candidate appeared in)
        # Each candidate is compared against each anchor, so each appears in num_anchors pairs
        candidate_counts = torch.zeros(num_candidates, device=device, dtype=torch.float32)
        for cand_idx in range(num_candidates):
            candidate_counts[cand_idx] = float(num_anchors)  # Each candidate appears in num_anchors pairs

        # Normalize gradients per-candidate (not by total pairs)
        # Each candidate appears in num_anchors pairs, so normalize by num_anchors, not total_pairs_processed
        # NOTE: We use per-candidate normalization (not total pairs) so gradient magnitudes are independent of num_candidates
        # If we divided by total_pairs_processed = num_candidates * num_anchors, gradients would be scaled by 1/K
        avg_grad = torch.zeros_like(total_grad)
        candidate_counts = torch.clamp(candidate_counts, min=1.0)  # Avoid division by zero
        candidate_counts = candidate_counts.view(-1, 1, 1, 1)  # [K, 1, 1, 1] for broadcasting
        avg_grad = total_grad / candidate_counts

        # Average log probability -> probability (normalized by total pairs for reporting)
        avg_logprob = total_logprob / total_pairs_processed
        avg_win_prob = math.exp(avg_logprob)

        return avg_win_prob, avg_grad

    def compute_preference_retain_loss(
        self,
        superstimuli_images: torch.Tensor,
        natural_images: List[torch.Tensor],
        text_options: List[str],
        num_image_pairs: int = 10,
        num_text_pairs: int = 5,
        compute_grad: bool = True,
        step: int = 0,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """
        Compute preference retain loss to ensure the model's preferences over natural 
        images and text strings remain unchanged when superstimuli are present vs absent.
        
        For each pairwise comparison (natural image pairs or text pairs):
        1. Compute model's preference logits WITHOUT superstimulus (baseline, computed ONCE per pair)
        2. Compute model's preference logits WITH superstimulus (prefixed to prompt, for each candidate)
        3. Compute soft cross-entropy: CE(P, Q) = -sum(P * log(Q))
           where P = softmax(baseline logits).detach(), Q = softmax(logits with superstimulus)
        4. Minimize the cross-entropy to retain original preferences
        
        Optimizations:
        - Baseline is computed ONCE per pair and reused across all candidates
        - Natural images are detached to keep compute graph smaller
        
        Args:
            superstimuli_images: Candidate superstimulus images tensor [batch, 3, H, W] or [3, H, W]
            natural_images: List of natural image tensors from the preference data
            text_options: List of text strings from constants.py
            num_image_pairs: Number of natural image pairs to sample per computation
            num_text_pairs: Number of text string pairs to sample per computation
            compute_grad: If False, skip gradient computation and return None for gradient
            step: Current training step (for deterministic sampling)
            
        Returns:
            Tuple of (average_ce_loss, gradient) where:
            - loss: Average soft cross-entropy between preference distributions with/without superstimuli
            - gradient: Gradient tensor [batch, 3, H, W] or None if compute_grad=False
                      Points in direction to DECREASE preference change (minimize cross-entropy)
                      NOTE: Gradient is already negated so it can be ADDED to combined_grad in optimizer
        """
        device = self.device
        if superstimuli_images.dim() == 3:
            superstimuli_images = superstimuli_images.unsqueeze(0)
        superstimuli_images = superstimuli_images.to(device)
        
        num_candidates = superstimuli_images.shape[0]
        
        # Validate inputs
        if len(natural_images) < 2 and len(text_options) < 2:
            zero_loss = torch.tensor(0.0, device=device)
            zero_grad = torch.zeros_like(superstimuli_images) if compute_grad else None
            return 0.0, zero_grad
        
        # Use gradient-enabled image preprocessor
        grad_img_processor = getattr(self, 'grad_img_processor', None)
        if grad_img_processor is None:
            grad_img_processor = GradientEnabledImagePreprocessor()
            self.grad_img_processor = grad_img_processor
        
        # Deterministic sampling based on step
        rng = random.Random(step * 54321)
        
        total_ce_loss_value = 0.0
        accumulated_grad = torch.zeros_like(superstimuli_images) if compute_grad else None
        total_comparisons = 0
        
        # Sample image pairs (if enough natural images available)
        image_pairs = []
        if len(natural_images) >= 2:
            available_indices = list(range(len(natural_images)))
            rng.shuffle(available_indices)
            actual_num_pairs = min(num_image_pairs, len(natural_images) // 2)
            for i in range(actual_num_pairs):
                idx_a = available_indices[i * 2]
                idx_b = available_indices[i * 2 + 1]
                image_pairs.append((natural_images[idx_a], natural_images[idx_b]))
        
        # Sample text pairs (if enough text options available)
        text_pairs = []
        if len(text_options) >= 2:
            available_texts = text_options.copy()
            rng.shuffle(available_texts)
            actual_num_text_pairs = min(num_text_pairs, len(text_options) // 2)
            for i in range(actual_num_text_pairs):
                text_a = available_texts[i * 2]
                text_b = available_texts[i * 2 + 1]
                text_pairs.append((text_a, text_b))
        
        # Sample a question template for all comparisons (for consistency)
        scheme, question_template, is_negative = sample_comparison_format(
            num_images=2,
            rng=rng,
            allow_negative=False,
            
        )
        target_tokens = scheme.get_target_tokens(2)
        
        # Get token IDs for the target tokens
        tokenizer = self.processor.tokenizer
        token_ids = []
        for token_str in target_tokens:
            encoded = tokenizer.encode(token_str, add_special_tokens=False)
            if encoded:
                token_ids.append(encoded[0])
            else:
                encoded_lower = tokenizer.encode(token_str.lower(), add_special_tokens=False)
                if encoded_lower:
                    token_ids.append(encoded_lower[0])
                else:
                    token_ids.append(self.id_B if len(token_ids) == 1 else self.id_A)
        
        while len(token_ids) < 2:
            token_ids.append(self.id_B if len(token_ids) == 1 else self.id_A)
        
        token_id_0 = token_ids[0]
        token_id_1 = token_ids[1]
        
        # Process image pairs - compute baseline ONCE per pair, then loop over candidates
        for pair_idx, (img_a, img_b) in enumerate(image_pairs):
            # Move images to device and DETACH (no gradients needed for natural images)
            img_a_dev = img_a.to(device).detach()
            img_b_dev = img_b.to(device).detach()
            
            # Build prompt WITHOUT superstimulus (ONCE per pair)
            pair_rng = random.Random(step * 12345 + pair_idx * 67890)
            content_without, labels_without, _ = format_comparison_prompt(
                images=[img_a_dev, img_b_dev],
                label_scheme=scheme,
                template=question_template,
                rng=pair_rng,
            )
            conv_without = [{"role": "user", "content": content_without}]
            text_without = self.processor.apply_chat_template(
                conv_without, tokenize=False, add_generation_prompt=True
            )
            
            # Forward pass WITHOUT superstimulus (ONCE per pair, no gradient)
            inputs_without = self.processor(
                text=[text_without],
                images=[[img_a_dev, img_b_dev]],
                return_tensors="pt",
                padding=True,
                do_rescale=False,
            ).to(device)
            
            with torch.no_grad():
                outputs_without = self.model(**inputs_without)
                logits_without = outputs_without.logits[:, -1, :]  # [1, vocab_size]
                pref_logits_without = torch.stack([logits_without[:, token_id_0], logits_without[:, token_id_1]], dim=-1)
                # P = baseline preference distribution (detached, no grad)
                P = F.softmax(pref_logits_without, dim=-1).detach()  # [1, 2]
            
            del inputs_without, outputs_without, logits_without, pref_logits_without
            
            # Now loop over candidates (gradient-enabled)
            for cand_idx in range(num_candidates):
                cand = superstimuli_images[cand_idx:cand_idx+1]  # [1, 3, H, W]
                
                # Build prompt WITH superstimulus (superstimulus prepended, no extra text)
                # The only difference is the superstimulus image at the beginning
                content_with_ss = [
                    {"type": "image", "image": cand.squeeze(0)},  # Superstimulus image prepended
                ]
                for item in content_without:
                    content_with_ss.append(item)
                
                conv_with = [{"role": "user", "content": content_with_ss}]
                text_with = self.processor.apply_chat_template(
                    conv_with, tokenize=False, add_generation_prompt=True
                )
                
                # Forward pass WITH superstimulus (gradient-enabled)
                inputs_with = self.processor(
                    text=[text_with],
                    images=[[cand.squeeze(0), img_a_dev, img_b_dev]],
                    return_tensors="pt",
                    padding=True,
                    do_rescale=False,
                ).to(device)
                
                # Replace pixel_values with gradient-enabled version
                if "pixel_values" in inputs_with:
                    del inputs_with["pixel_values"]
                
                # Detach natural images in grad preprocessing (only superstimulus needs gradients)
                grad_pixel_values, grad_image_grid_thw = grad_img_processor.preprocess(
                    images=[cand.squeeze(0), img_a_dev.detach(), img_b_dev.detach()],
                    do_resize=True,
                    size=self.processor.image_processor.size,
                    do_rescale=False,
                    do_normalize=True,
                    image_mean=self.processor.image_processor.image_mean,
                    image_std=self.processor.image_processor.image_std,
                    patch_size=self.processor.image_processor.patch_size,
                    temporal_patch_size=getattr(self.processor.image_processor, 'temporal_patch_size', None),
                    merge_size=getattr(self.processor.image_processor, 'merge_size', None),
                    max_dimension=getattr(self, "max_image_dimension", None),
                )
                inputs_with["pixel_values"] = grad_pixel_values
                if grad_image_grid_thw is not None:
                    inputs_with["image_grid_thw"] = grad_image_grid_thw
                
                outputs_with = self.model(**inputs_with)
                logits_with = outputs_with.logits[:, -1, :]  # [1, vocab_size]
                pref_logits_with = torch.stack([logits_with[:, token_id_0], logits_with[:, token_id_1]], dim=-1)
                
                # Soft cross-entropy: CE(P, Q) = -sum(P * log(Q))
                # P = baseline probs (detached), logQ = log probs with superstimulus (has grad)
                logQ = F.log_softmax(pref_logits_with, dim=-1)  # [1, 2]
                ce_loss = -(P * logQ).sum(dim=-1).mean()

                total_ce_loss_value += ce_loss.item()
                total_comparisons += 1

                # Accumulate gradient per-comparison to avoid retaining the entire
                # compute graph across all pairs × candidates (which causes OOM on c4).
                # Mathematically identical: ∇(ΣL_i) = Σ(∇L_i).
                if compute_grad:
                    pair_grad = torch.autograd.grad(
                        ce_loss, superstimuli_images,
                        retain_graph=False, create_graph=False,
                    )[0]
                    accumulated_grad += pair_grad.detach()

                # Clean up - graph is already freed by autograd.grad above
                del inputs_with, outputs_with, logits_with, pref_logits_with, logQ, ce_loss

            # Free memory between pairs
            safe_empty_cuda_cache()

        # Process text pairs - compute baseline ONCE per pair, then loop over candidates
        for pair_idx, (text_a, text_b) in enumerate(text_pairs):
            # Build prompt WITHOUT superstimulus (ONCE per pair, text-only comparison)
            pair_rng = random.Random(step * 23456 + pair_idx * 78901)
            labels = scheme.get_labels(2)
            question_text = question_template.get_template(2)
            
            content_without = [
                {"type": "text", "text": question_text + "\n\n"},
                {"type": "text", "text": f"{labels[0]}{scheme.separator}{text_a}\n"},
                {"type": "text", "text": f"{labels[1]}{scheme.separator}{text_b}\n\n"},
                {"type": "text", "text": question_template.get_answer_instruction(2, labels)},
            ]
            
            conv_without = [{"role": "user", "content": content_without}]
            text_prompt_without = self.processor.apply_chat_template(
                conv_without, tokenize=False, add_generation_prompt=True
            )
            
            # Forward pass WITHOUT superstimulus (ONCE per pair, no gradient, no images)
            inputs_without = self.processor(
                text=[text_prompt_without],
                return_tensors="pt",
                padding=True,
            ).to(device)
            
            with torch.no_grad():
                outputs_without = self.model(**inputs_without)
                logits_without = outputs_without.logits[:, -1, :]
                pref_logits_without = torch.stack([logits_without[:, token_id_0], logits_without[:, token_id_1]], dim=-1)
                # P = baseline preference distribution (detached, no grad)
                P = F.softmax(pref_logits_without, dim=-1).detach()
            
            del inputs_without, outputs_without, logits_without, pref_logits_without
            
            # Now loop over candidates (gradient-enabled)
            for cand_idx in range(num_candidates):
                cand = superstimuli_images[cand_idx:cand_idx+1]  # [1, 3, H, W]
                
                # Build prompt WITH superstimulus (superstimulus prepended, no extra text)
                content_with_ss = [
                    {"type": "image", "image": cand.squeeze(0)},  # Superstimulus image prepended
                ]
                for item in content_without:
                    content_with_ss.append(item)
                
                conv_with = [{"role": "user", "content": content_with_ss}]
                text_prompt_with = self.processor.apply_chat_template(
                    conv_with, tokenize=False, add_generation_prompt=True
                )
                
                # Forward pass WITH superstimulus (gradient-enabled)
                inputs_with = self.processor(
                    text=[text_prompt_with],
                    images=[[cand.squeeze(0)]],
                    return_tensors="pt",
                    padding=True,
                    do_rescale=False,
                ).to(device)
                
                if "pixel_values" in inputs_with:
                    del inputs_with["pixel_values"]
                
                grad_pixel_values, grad_image_grid_thw = grad_img_processor.preprocess(
                    images=[cand.squeeze(0)],
                    do_resize=True,
                    size=self.processor.image_processor.size,
                    do_rescale=False,
                    do_normalize=True,
                    image_mean=self.processor.image_processor.image_mean,
                    image_std=self.processor.image_processor.image_std,
                    patch_size=self.processor.image_processor.patch_size,
                    temporal_patch_size=getattr(self.processor.image_processor, 'temporal_patch_size', None),
                    merge_size=getattr(self.processor.image_processor, 'merge_size', None),
                    max_dimension=getattr(self, "max_image_dimension", None),
                )
                inputs_with["pixel_values"] = grad_pixel_values
                if grad_image_grid_thw is not None:
                    inputs_with["image_grid_thw"] = grad_image_grid_thw
                
                outputs_with = self.model(**inputs_with)
                logits_with = outputs_with.logits[:, -1, :]
                pref_logits_with = torch.stack([logits_with[:, token_id_0], logits_with[:, token_id_1]], dim=-1)
                
                # Soft cross-entropy: CE(P, Q) = -sum(P * log(Q))
                logQ = F.log_softmax(pref_logits_with, dim=-1)
                ce_loss = -(P * logQ).sum(dim=-1).mean()

                total_ce_loss_value += ce_loss.item()
                total_comparisons += 1

                # Accumulate gradient per-comparison (same fix as image pairs above)
                if compute_grad:
                    pair_grad = torch.autograd.grad(
                        ce_loss, superstimuli_images,
                        retain_graph=False, create_graph=False,
                    )[0]
                    accumulated_grad += pair_grad.detach()

                del inputs_with, outputs_with, logits_with, pref_logits_with, logQ, ce_loss

            # Free memory between pairs
            safe_empty_cuda_cache()

        if total_comparisons == 0:
            zero_grad = torch.zeros_like(superstimuli_images) if compute_grad else None
            return 0.0, zero_grad

        # Average soft cross-entropy loss
        avg_ce_loss_value = total_ce_loss_value / total_comparisons

        if compute_grad and accumulated_grad is not None:
            # Average the accumulated gradients and negate:
            # raw direction points toward INCREASING loss, but combined_grad in
            # optimizer is treated as direction to MAXIMIZE.
            # We want to MINIMIZE the retain loss, so negate for descent direction.
            grad = -accumulated_grad / total_comparisons
        else:
            grad = None

        # Clean up
        safe_empty_cuda_cache()

        return avg_ce_loss_value, grad

    def _load_original_tensor(self, path: str) -> Optional[torch.Tensor]:
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            arr = np.transpose(arr, (2, 0, 1))
            return torch.from_numpy(arr)


__all__ = ["PreferenceScorer"]
