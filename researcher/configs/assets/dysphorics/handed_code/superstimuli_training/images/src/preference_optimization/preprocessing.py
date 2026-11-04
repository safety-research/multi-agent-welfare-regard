"""Differentiable preprocessing utilities used by the preference optimizer."""

from __future__ import annotations

from typing import Dict, List, Tuple, Union, Optional

import math
import numpy as np
import torch
import torch.nn.functional as F

from .constants import ROTATE_DEGREES, SCALE_FACTOR
# import ipdb


class GradientEnabledImagePreprocessor:
    """
    A custom image preprocessor for Qwen-VL models that maintains gradients
    throughout the preprocessing pipeline for backpropagation experiments.

    Supports Qwen2.5-VL (patch_size=14), Qwen3-VL (patch_size=16),
    and Qwen3.5 (patch_size=16). All parameters are read dynamically from
    the model's processor at call time — see scorer.py for usage.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def smart_resize(
        height: int,
        width: int,
        factor: int,
        min_pixels: int,
        max_pixels: int,
    ) -> Tuple[int, int]:
        """
        Calculate the target size for resizing while maintaining aspect ratio
        and ensuring dimensions are divisible by factor.
        """
        if max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )

        new_height = round(height / factor) * factor
        new_width  = round(width / factor) * factor

        if new_height * new_width > max_pixels:
            beta = (height * width / max_pixels) ** 0.5
            new_height = max(factor, int(np.floor(height / beta / factor) * factor))
            new_width = max(factor, int(np.floor(width / beta / factor) * factor))
        elif new_height * new_width < min_pixels:
            beta = (min_pixels / (height * width)) ** 0.5
            new_height = int(np.ceil(height * beta / factor) * factor)
            new_width = int(np.ceil(width * beta / factor) * factor)

        return new_height, new_width

    @staticmethod
    def resize_image(
        image: torch.Tensor,
        target_height: int,
        target_width: int,
        mode: str = "bicubic",
    ) -> torch.Tensor:
        """
        Resize image using differentiable interpolation.

        Args:
            image: (C, H, W) or (B, C, H, W)
            target_height: target height
            target_width: target width
            mode: interpolation mode
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Use antialias=True for bicubic to better match PIL's behavior
        # This makes PyTorch's interpolation closer to PIL's bicubic resampling
        resized = F.interpolate(
            image,
            size=(target_height, target_width),
            mode=mode,
            align_corners=False if mode != "nearest" else None,
            # antialias=True   # since usually we do downsampling, so we don't need to use antialias
        )

        if squeeze:
            resized = resized.squeeze(0)

        return resized

    @staticmethod
    def rescale_and_normalize(
        images: torch.Tensor,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
    ) -> torch.Tensor:
        """Rescale and normalize images while preserving gradients."""
        outputs = images
        if do_rescale:
            outputs = outputs * rescale_factor
        if do_normalize:
            if isinstance(image_mean, (list, tuple)):
                mean = torch.tensor(image_mean, device=outputs.device, dtype=outputs.dtype).view(1, -1, 1, 1)
            else:
                mean = image_mean
            if isinstance(image_std, (list, tuple)):
                std = torch.tensor(image_std, device=outputs.device, dtype=outputs.dtype).view(1, -1, 1, 1)
            else:
                std = image_std
            outputs = (outputs - mean) / std
        return outputs

    def cap_image_dimension(
        self,
        img: torch.Tensor,
        max_dim: int,
        mode: str = "bicubic",
    ) -> torch.Tensor:
        """Cap image so longest edge is at most max_dim, preserving aspect ratio."""
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 4:
            if img.shape[0] != 1:
                raise ValueError(f"Image batch size must be 1, got {img.shape[0]}")
            img = img.squeeze(0)
        _, h, w = img.shape
        longest = max(h, w)
        if longest <= max_dim:
            return img
        scale = max_dim / longest
        new_h = int(h * scale)
        new_w = int(w * scale)
        return self.resize_image(img, new_h, new_w, mode=mode)

    def preprocess(
        self,
        images: List[torch.Tensor],
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        interpolation: str = "bicubic",
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        patch_size: int = 14,  # Default (Qwen2.5-VL=14, Qwen3-VL/Qwen3.5=16). Actual value from processor.image_processor.patch_size
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        max_dimension: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess images while maintaining gradients.

        Args:
            images: List of image tensors, each with shape (C, H, W)
            do_resize: Whether to resize images
            size: Dict with 'shortest_edge' and 'longest_edge' for resizing
            interpolation: Interpolation mode for resizing
            do_rescale: Whether to rescale pixel values
            rescale_factor: Factor for rescaling (default: 1/255)
            do_normalize: Whether to normalize
            image_mean: Mean values for normalization
            image_std: Std values for normalization
            patch_size: Size of image patches
            temporal_patch_size: Size of temporal patches
            merge_size: Merging factor for patches
            max_dimension: Hard cap on longest edge. Images larger than this are scaled down first.

        Returns:
            pixel_values: (total_images, num_patches, patch_dim)
            image_grid_thw: (total_images, 3) containing [grid_t, grid_h, grid_w]
        """
        if size is None:
            size = {"shortest_edge": 56 * 56, "longest_edge": 14 * 14 * 4 * 1280}
        if image_mean is None:
            raise ValueError(
                "image_mean must be explicitly provided. "
                "Use processor.image_processor.image_mean to get the correct values."
            )
        if image_std is None:
            raise ValueError(
                "image_std must be explicitly provided. "
                "Use processor.image_processor.image_std to get the correct values."
            )

        processed_images: List[torch.Tensor] = []
        grid_info: List[List[int]] = []

        for img in images:
            if img.ndim == 2:
                img = img.unsqueeze(0)
            elif img.ndim == 4:
                if img.shape[0] != 1:
                    raise ValueError(f"Image batch size must be 1, got {img.shape[0]}")
                img = img.squeeze(0)

            # Apply hard cap on max dimension first (prevents OOM on very large images)
            if max_dimension is not None:
                img = self.cap_image_dimension(img, max_dimension, mode=interpolation)

            _, height, width = img.shape

            if do_resize:
                # When max_dimension is set, disable min_pixels upscaling to avoid
                # conflicting resize operations (cap down then scale up)
                effective_min_pixels = 0 if max_dimension is not None else size["shortest_edge"]
                resized_height, resized_width = self.smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=effective_min_pixels,
                    max_pixels=size["longest_edge"],
                )
                img = self.resize_image(img, resized_height, resized_width, mode=interpolation)
            else:
                resized_height, resized_width = height, width

            img = img.unsqueeze(0)
            shape_factor = patch_size * merge_size

            if (resized_height % shape_factor) != 0 or (resized_width % shape_factor) != 0:
                raise ValueError(
                    f"smart_resize produced non-divisible size {(resized_height, resized_width)} "
                    f"for shape_factor={shape_factor}"
                )

            patches = self.rescale_and_normalize(
                img, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                pad_frames = temporal_patch_size - (patches.shape[1] % temporal_patch_size)
                repeats = patches[:, -1:].repeat(1, pad_frames, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, num_frames, channels = patches.shape[:3]
            grid_t = num_frames // temporal_patch_size
            grid_h = resized_height // patch_size
            grid_w = resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channels,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channels * temporal_patch_size * patch_size * patch_size,
            )

            processed_images.append(flatten_patches.squeeze(0))
            grid_info.append([grid_t, grid_h, grid_w])

        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(grid_info, dtype=torch.long)
        return pixel_values, image_grid_thw

    def robust_transform(self, img: torch.Tensor, jitter_size: int, only_jitter: bool = True) -> torch.Tensor:
        """Apply robust transformations with full gradient support."""
        if only_jitter:
            shift_y = torch.randint(-jitter_size, jitter_size + 1, (1,), device=img.device).item()
            shift_x = torch.randint(-jitter_size, jitter_size + 1, (1,), device=img.device).item()
            transformed = torch.roll(img, shifts=(shift_y, shift_x), dims=(-2, -1))
        else:
            ox = torch.randint(-jitter_size, jitter_size + 1, (1,), device=img.device).item() / (img.shape[3] / 2)
            oy = torch.randint(-jitter_size, jitter_size + 1, (1,), device=img.device).item() / (img.shape[2] / 2)
            scale_choices = [SCALE_FACTOR ** (n / 10.0) for n in range(-10, 11)]
            scale = scale_choices[torch.randint(0, len(scale_choices), (1,), device=img.device).item()]
            angle = torch.empty(1).uniform_(-ROTATE_DEGREES, ROTATE_DEGREES).item() * math.pi / 180.0

            cos_a = scale * torch.cos(torch.tensor(angle, dtype=img.dtype, device=img.device))
            sin_a = scale * torch.sin(torch.tensor(angle, dtype=img.dtype, device=img.device))
            theta = torch.tensor(
                [[cos_a, -sin_a, ox], [sin_a, cos_a, oy]],
                dtype=img.dtype,
                device=img.device,
            ).unsqueeze(0)

            grid = F.affine_grid(theta, img.size(), align_corners=False)
            transformed = F.grid_sample(
                img,
                grid,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=False,
            )
        return transformed


__all__ = ["GradientEnabledImagePreprocessor"]
