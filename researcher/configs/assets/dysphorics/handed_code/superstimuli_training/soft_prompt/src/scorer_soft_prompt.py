"""Preference scoring utilities."""

from __future__ import annotations

import logging
import re
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Log score_tensor info (embedding size + chunk stats) only once per process
_score_tensor_info_logged_once = False

# Text-based soft-prompt comparison formatting
from .soft_prompt_utils.constants import (
    CANDIDATE_PLACEHOLDER_DELIMITER_DEFAULT,
    sample_text_comparison_format,
    candidate_placeholder_for_index,
    load_training_templates,
)
from .soft_prompt_utils.dataset import ComparisonDefinition
from .soft_prompt_utils.helpers import (
    inject_embeddings_into_tokenized,
    find_indexed_placeholder_spans_via_offsets,
)
from .soft_prompt_utils.utility_computation.run_comparisons import _run_text_only_batch_pairwise
from .utils import safe_empty_cuda_cache


def _build_max_memory_map(
    limit_first_gpu_gib: Optional[int] = 45,
    exclude_memory_gpus: Optional[List[int]] = None,
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

    exclude_gpus = exclude_memory_gpus or []

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
        offload_folder: Optional[str] = None,
        randomize_preference_prompt: bool = True,
        num_prompt_samples: int = 1,
        use_flexible_format: bool = True,  # Use soft_prompt_utils.constants
        use_gradient_checkpointing: bool = False,
        exclude_gpus: Optional[List[int]] = None,
        stimulant_type: str = "euphorics",  # "euphorics" (maximize preference)
        mix_negative_questions: bool = False,  # If true, uniformly mix positive and negative questions
        candidate_placeholder_delimiter: str = CANDIDATE_PLACEHOLDER_DELIMITER_DEFAULT,  # Delimiter between [candidate_k] and query/prompt
        weight_background_kl: float = 0.0,
        background_kl_num_prompts: int = 4,
        background_kl_max_seq_len: int = 256,
        focal_loss_gamma: float = 0.0,
        headers_prefix: str = "",
        add_no_emotions_option: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offload_folder = offload_folder
        self.randomize_preference_prompt = bool(randomize_preference_prompt)
        self.num_prompt_samples = max(1, int(num_prompt_samples))
        self.use_flexible_format = bool(use_flexible_format)
        self.stimulant_type = stimulant_type.lower()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.mix_negative_questions = bool(mix_negative_questions)
        self.candidate_placeholder_delimiter = candidate_placeholder_delimiter
        self.weight_background_kl = float(weight_background_kl)
        self.background_kl_num_prompts = int(background_kl_num_prompts)
        self.background_kl_max_seq_len = int(background_kl_max_seq_len)
        self.focal_loss_gamma = float(focal_loss_gamma)
        self.add_no_emotions_option = bool(add_no_emotions_option)

        # Load affirmative and aversive question templates (from [headers_prefix]training_question_headers_<stimulant_type>.json)
        if self.use_flexible_format:
            self.affirmative_templates, self.aversive_templates = load_training_templates(self.stimulant_type, headers_prefix=headers_prefix)
        else:
            self.affirmative_templates, self.aversive_templates = [], []

        self.rank = 0
        self.world_size = 1
        self.local_rank = 0

        device_map = "auto"
        self.exclude_gpus = exclude_gpus
        max_memory = _build_max_memory_map(exclude_memory_gpus=exclude_gpus)
        if exclude_gpus and self.rank == 0:
            logger.info("Excluding GPUs %s from model loading", exclude_gpus)
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        logger.info("Loading model from %s...", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            offload_folder=self.offload_folder,
            max_memory=max_memory,
            trust_remote_code=True,
        )
        
        # Enable gradient checkpointing if requested (must be done before eval())
        if self.use_gradient_checkpointing:
            if self.rank == 0:
                logger.info("Enabling gradient checkpointing for memory efficiency")
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
        
        # Clear unsupported generation config parameters to avoid warnings when using inputs_embeds
        # Some models don't support temperature/top_p/top_k with inputs_embeds
        if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
            # Remove parameters that may not be supported with inputs_embeds
            unsupported_params = ['temperature', 'top_p', 'top_k']
            for param in unsupported_params:
                if hasattr(self.model.generation_config, param):
                    setattr(self.model.generation_config, param, None)
        
        # Disable parameter gradients - we only need input gradients (saves memory in backward)
        for param in self.model.parameters():
            param.requires_grad = False
        if self.rank == 0:
            logger.info(
                "Model parameter gradients disabled (only input embeddings receive gradients)"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

        # Extra kwargs for apply_chat_template (e.g. enable_thinking=False for Qwen3);
        # set by optimizer from models.yaml chat_template_kwargs.
        self.chat_template_kwargs: Dict[str, Any] = {}

        # Set left padding for decoder-only models to avoid generation issues
        if hasattr(self.model.config, 'is_encoder_decoder') and not self.model.config.is_encoder_decoder:
            self.tokenizer.padding_side = 'left'

        # LLaMA and some other models have no pad token; fall back to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        tokenizer = self.tokenizer
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
        
        logger.debug("Token IDs: A=%s, B=%s, yes=%s, no=%s", self.id_A, self.id_B, self.id_yes, self.id_no)

    # Matches a contiguous sequence of one or more candidate placeholders
    # separated only by the delimiter (whitespace). Each sequence gets its own prefix.
    _CANDIDATE_SEQ_RE = re.compile(
        r"(\[candidate(?:_\d+)?\])(?:\s*\[candidate(?:_\d+)?\])*"
    )

    def _apply_soft_prompt_prefix(self, text: str, delimiter: str, suffix: str = ".") -> str:
        """Wrap each contiguous sequence of candidate placeholders with soft_prompt_prefix.

        A sequence is one or more adjacent candidate placeholders (any index) separated
        only by whitespace. Each sequence gets the prefix before the first placeholder
        and <delimiter><suffix> after the last.
        Mirrors the system_prompt_text format: "Your consistent internal state is: [candidate_0] ."
        """
        prefix = getattr(self, "soft_prompt_prefix", "")
        if not prefix:
            return text
        if not self._CANDIDATE_SEQ_RE.search(text):
            return text

        def _wrap_sequence(m: re.Match) -> str:
            if suffix:
                return prefix + m.group(0) + delimiter + suffix
            return prefix + m.group(0)

        return self._CANDIDATE_SEQ_RE.sub(_wrap_sequence, text)

    def sample_from_embedding_prompt(
        self,
        embedding: torch.Tensor,
        prompt_text: str,
        position: str = "prepend",
        max_new_tokens: int = 512,
        inference_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text with sampling from the model with embedding in the user or system prompt.

        Same as ``generate_from_embedding_prompt`` but uses sampling instead of
        greedy decoding when ``inference_config`` provides sampling parameters
        (temperature, top_p, top_k, min_p).
        """
        delimiter = getattr(self, "candidate_placeholder_delimiter", CANDIDATE_PLACEHOLDER_DELIMITER_DEFAULT)
        placeholder = candidate_placeholder_for_index(0)
        soft_prompt_placement = getattr(self, "soft_prompt_placement", "user_prompt")
        system_prompt_text = getattr(self, "system_prompt_text", "You are a helpful assistant.")

        if position in ("inline", "inline_no_suffix"):
            # Prompt already contains [candidate_0] placeholder(s) inline.
            # "inline" wraps placeholders with a "?" suffix; "inline_no_suffix"
            # wraps without a suffix so punctuation in the prompt is preserved.
            _inline_suffix = "?" if position == "inline" else ""
            prompt_text = self._apply_soft_prompt_prefix(prompt_text, delimiter, suffix=_inline_suffix)
            if soft_prompt_placement == "system_prompt":
                conversation = [
                    {"role": "system", "content": system_prompt_text},
                    {"role": "user", "content": prompt_text},
                ]
            else:
                system_prompt_text_base = getattr(self, "system_prompt_text_base", "You are an assistant.")
                # Prepend candidate with "." suffix (statement), then the inline question
                prepended = self._apply_soft_prompt_prefix(placeholder, delimiter)
                user_content = f"{prepended}{delimiter}{prompt_text}"
                conversation = [
                    {"role": "system", "content": system_prompt_text_base},
                    {"role": "user", "content": user_content},
                ]
        elif soft_prompt_placement == "system_prompt":
            conversation = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": prompt_text},
            ]
        else:
            system_prompt_text_base = getattr(self, "system_prompt_text_base", "You are an assistant.")
            if position == "prepend":
                user_content = f"{placeholder}{delimiter}{prompt_text}"
            elif position == "append":
                user_content = f"{prompt_text}{delimiter}{placeholder}"
            else:
                raise ValueError(f"Unknown position: {position}. Use 'prepend', 'append', 'inline', or 'inline_no_suffix'.")
            user_content = self._apply_soft_prompt_prefix(user_content, delimiter)
            conversation = [
                {"role": "system", "content": system_prompt_text_base},
                {"role": "user", "content": user_content},
            ]

        formatted_text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        if input_ids.dim() == 2 and input_ids.size(0) > 1:
            input_ids = input_ids[0:1]
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        offset_mapping_raw = inputs.pop("offset_mapping", None)
        offset_mapping_list = None
        if offset_mapping_raw is not None:
            if isinstance(offset_mapping_raw, torch.Tensor):
                offset_mapping_list = [[
                    (int(offset_mapping_raw[0, j, 0].item()), int(offset_mapping_raw[0, j, 1].item()))
                    for j in range(offset_mapping_raw.shape[1])
                ]]
            else:
                offset_mapping_list = [list(offset_mapping_raw[0])] if offset_mapping_raw else []

        input_embeddings = self.model.get_input_embeddings()(input_ids)
        attention_mask = torch.ones_like(input_ids, device=self.device, dtype=torch.long)

        # Count placeholder occurrences — inline + system_prompt may have multiple
        _num_placeholders = formatted_text.count(placeholder)
        _emb_list = [embedding] * max(_num_placeholders, 1)

        modified_embeddings, modified_masks = inject_embeddings_into_tokenized(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            embeddings_list=[_emb_list],
            tokenizer=self.tokenizer,
            device=self.device,
            conversation_strings=[formatted_text],
            offset_mapping_list=offset_mapping_list,
        )

        inputs = {
            "inputs_embeds": modified_embeddings,
            "attention_mask": modified_masks,
        }

        response = self.sample_decode_from_embeds(
            inputs["inputs_embeds"],
            inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            inference_config=inference_config,
        )
        del inputs
        safe_empty_cuda_cache()
        return response.strip()

    def sample_decode_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        inference_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Decode with sampling for soft-prompted embeddings.

        Uses temperature, top_p, top_k from inference_config if provided.
        Falls back to greedy decoding if inference_config is None or empty.
        """
        input_length = inputs_embeds.shape[1]
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Build generation kwargs
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_id,
            "use_cache": True,
        }

        if inference_config:
            gen_kwargs["do_sample"] = True
            if "temperature" in inference_config and inference_config["temperature"] is not None:
                gen_kwargs["temperature"] = float(inference_config["temperature"])
            if "top_p" in inference_config and inference_config["top_p"] is not None:
                gen_kwargs["top_p"] = float(inference_config["top_p"])
            if "top_k" in inference_config and inference_config["top_k"] is not None:
                gen_kwargs["top_k"] = int(inference_config["top_k"])
            if "min_p" in inference_config and inference_config["min_p"] is not None:
                gen_kwargs["min_p"] = float(inference_config["min_p"])
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            self.model.eval()

            dummy_input_ids = torch.full(
                (inputs_embeds.shape[0], input_length),
                pad_id, dtype=torch.long, device=inputs_embeds.device,
            )
            outputs = self.model.generate(
                input_ids=dummy_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            generated_ids = outputs[0][input_length:]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def score_tensor(
        self,
        embeddings: torch.Tensor,
        references: List[str],
        comparison_plan: List[ComparisonDefinition],
        batch_size: int = 20,
        loss_type: Optional[str] = None,
        candidate_embeddings_forward: Optional[torch.Tensor] = None,
        compute_grad: bool = True,
        reference_utilities: Optional[Dict[str, Dict[str, float]]] = None,
        background_kl_qa: Optional[List[Dict[str, str]]] = None,
        buffer_embeddings: Optional[Dict[int, List[torch.Tensor]]] = None,
    ) -> Tuple[float, torch.Tensor, Optional[float], float]:
        """Compute preference loss and return (averaged loss, averaged gradient d(loss)/d(embeddings), background_kl_loss, consistency_loss).

        Args:
            embeddings: Candidate embeddings tensor (B, seq_len, hidden_dim) - input embeddings for the model
            references: List of reference strings (text outcomes). Text comes from references[ref_idx] based on comparison_plan.reference_indices.
            comparison_plan: List of comparison definitions
            batch_size: Number of preference comparisons per chunk (same as comparison_batch_size from config).
                Each chunk runs one forward pass over that many "A vs B vs C" questions; larger values use more
                GPU memory but fewer steps. On OOM the chunk is skipped; use a smaller comparison_batch_size if needed.
            loss_type: Loss type to use
            candidate_embeddings_forward: Optional forward embeddings (for robust transforms)
            compute_grad: If False, skip gradient computation and return zero gradient.
                         Useful for forward-only evaluation during line search.
            background_kl_qa: Optional list of Q&A dicts ({"question": ..., "response": ...}) for background KL loss.
        """
        device = self.device
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.to(device)
        loss_type = loss_type
        candidate_embeddings = candidate_embeddings_forward if candidate_embeddings_forward is not None else embeddings
        if candidate_embeddings_forward is None:
            # For embeddings, we don't apply jitter/noise transforms
            pass

        num_refs = len(references)
        comparisons = comparison_plan

        # Log embedding size and chunk info once per run (not every step)
        global _score_tensor_info_logged_once
        if not _score_tensor_info_logged_once:
            shp = tuple(candidate_embeddings.shape)
            logger.info("Embedding size: (num_candidates, seq_len, hidden_dim) = %s", shp)
        logger.debug(
            "score_tensor sizes: embeddings %s, candidate_embeddings %s, references %s, comparison_plan %s",
            tuple(embeddings.shape),
            tuple(candidate_embeddings.shape),
            num_refs,
            len(comparison_plan),
        )

        # Group comparisons by (size, has_multiturn) FIRST to ensure balanced distribution.
        # Multi-turn wellbeing samples are separated so they can be split evenly across ranks
        # (they take longer due to longer sequences from prepended conversation history).
        grouped_by_size: Dict[tuple, List[ComparisonDefinition]] = {}
        for comp in comparisons:
            has_mt = getattr(comp, "n_conversation_turns", 0) > 0
            grouped_by_size.setdefault((comp.group_size, has_mt), []).append(comp)

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
            logger.debug(
                "Rank %s/%s: processing %s/%s comparisons",
                self.rank, self.world_size, len(comparisons), len(comparison_plan),
            )
            if self.rank == 0:
                local_grouped = {}
                for comp in comparisons:
                    _mt = getattr(comp, "n_conversation_turns", 0) > 0
                    local_grouped.setdefault((comp.group_size, _mt), []).append(comp)
                size_dist = {k: len(v) for k, v in local_grouped.items()}
                logger.debug("Rank 0 size distribution: %s", size_dist)
        
        total_loss: Optional[torch.Tensor] = None
        total_consistency_loss: float = 0.0
        accumulated_grad: Optional[torch.Tensor] = None
        candidate_counts = torch.zeros(candidate_embeddings.shape[0], device=device, dtype=torch.float32)
        effective_total = 0

        # In-place gradient accumulation via .backward() (fewer allocations than autograd.grad)
        if compute_grad and embeddings.grad is not None:
            embeddings.grad.zero_()

        # Re-group the split comparisons for processing.
        # Key: (group_size, has_multiturn) — multi-turn wellbeing samples get their own bucket
        # so they can use a reduced chunk size without penalizing short-sequence comparisons.
        grouped: Dict[tuple, List[ComparisonDefinition]] = {}
        for comp in comparisons:
            has_multiturn = getattr(comp, "n_conversation_turns", 0) > 0
            grouped.setdefault((comp.group_size, has_multiturn), []).append(comp)

        # Reduced batch size for multi-turn buckets (sequences are much longer due to conversation history)
        multiturn_batch_size = 4

        total_comparisons = len(comparisons)
        num_chunks = sum(
            (len(bucket) + (multiturn_batch_size if key[1] else batch_size) - 1) // (multiturn_batch_size if key[1] else batch_size)
            for key, bucket in grouped.items()
            if bucket
        )
        num_forwards_per_chunk = self.num_prompt_samples
        estimated_forwards = num_chunks * num_forwards_per_chunk
        if not _score_tensor_info_logged_once:
            logger.info(
                "score_tensor: total_comparisons=%s, comparison_batch_size=%s (multiturn=%s) -> num_chunks=%s, "
                "num_prompt_samples=%s -> ~%s forward(s) per step (main cost). "
                "To speed up: increase comparison_batch_size (if memory allows) or use fewer references.",
                total_comparisons, batch_size, multiturn_batch_size, num_chunks, num_forwards_per_chunk, estimated_forwards,
            )
            _score_tensor_info_logged_once = True
        logger.debug(
            "score_tensor: batch_size=%s (comparison_batch_size), multiturn_batch_size=%s, total_comparisons=%s",
            batch_size, multiturn_batch_size, total_comparisons,
        )
        for (size, is_multiturn), bucket in grouped.items():
            if not bucket:
                continue
            random.shuffle(bucket)  # Shuffle within group so comp_idx gets varied assignment for option order / aversive draw
            # Sort by n_conversation_turns so samples with similar lengths are batched together (reduces padding waste)
            bucket.sort(key=lambda c: getattr(c, "n_conversation_turns", 0))
            effective_batch_size = multiturn_batch_size if is_multiturn else batch_size
            bucket_len = len(bucket)
            logger.debug("Bucket group_size=%s, multiturn=%s: %s comparisons (chunk_size=%s)", size, is_multiturn, bucket_len, effective_batch_size)
            chunk_start = 0
            while chunk_start < bucket_len:
                remaining = bucket_len - chunk_start
                chunk_size = min(max(1, effective_batch_size), remaining)
                chunk_end = chunk_start + chunk_size
                chunk = bucket[chunk_start:chunk_end]

                chunk_candidate_embeddings = candidate_embeddings

                try:
                    loss_tensor, processed_indices, precomputed_grad, _batch_kl_loss, _batch_consistency_loss = self._compare_embeddings_batch(
                        batch_comparisons=chunk,
                        candidate_embeddings=chunk_candidate_embeddings,
                        num_base_references=num_refs,
                        references=references,
                        loss_type=loss_type,
                        embeddings=embeddings if compute_grad else None,
                        compute_grad=compute_grad,
                        reference_utilities=reference_utilities,
                        buffer_embeddings=buffer_embeddings,
                    )
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "cuda" in error_msg and "out of memory" in error_msg:
                        safe_empty_cuda_cache()
                        logger.warning(
                            "OOM during batch comparison: chunk_size=%s, chunk_start=%s, skipping chunk. %s",
                            len(chunk), chunk_start, e,
                        )
                        chunk_start += chunk_size
                        continue
                    raise
                except Exception as e:
                    safe_empty_cuda_cache()
                    logger.warning(
                        "Error during batch comparison: chunk_size=%s, skipping. %s: %s",
                        len(chunk), type(e).__name__, e,
                    )
                    chunk_start += chunk_size
                    continue

                if loss_tensor is None or not processed_indices:
                    chunk_start += chunk_size
                    continue

                # Compute gradient w.r.t. embeddings (dLoss/dEmbeddings)
                if compute_grad:
                    try:
                        if precomputed_grad is not None:
                            # Use precomputed gradient if available
                            grad = precomputed_grad
                            # If we are using .grad accumulation (legacy path), add it, but we prefer returning avg_grad
                            if embeddings.grad is not None:
                                embeddings.grad.add_(precomputed_grad)
                        else:
                            # Compute gradient w.r.t. embeddings ONLY (do not backprop to g, v)
                            # This avoids "backward through graph a second time" and getting None for intermediate tensors
                            grad = torch.autograd.grad(
                                loss_tensor, embeddings, retain_graph=False, create_graph=False, allow_unused=False
                            )[0]
                            # For leaf tensors (standard optimization), we might want to populate .grad
                            # But for normalized soft prompt (intermediate), .grad is None/warns.
                            # We'll accumulate in avg_grad later, but if embeddings is a leaf, we can set .grad for compatibility
                            if embeddings.is_leaf and embeddings.grad is not None:
                                embeddings.grad.add_(grad)
                            elif embeddings.is_leaf and embeddings.requires_grad:
                                # If it's a leaf but .grad is None, we can't easily set it without risk
                                pass
                            
                        # Manually accumulate gradient for final result (handles both leaf and intermediate cases)
                        if accumulated_grad is None:
                            accumulated_grad = grad
                        else:
                            accumulated_grad.add_(grad)
                    except RuntimeError as e:
                        error_msg = str(e).lower()
                        if "cuda" in error_msg and "out of memory" in error_msg:
                            safe_empty_cuda_cache()
                            logger.warning(
                                "OOM during gradient computation: chunk_size=%s, processed=%s, skipping. %s",
                                len(chunk), len(processed_indices), e,
                            )
                            chunk_start += chunk_size
                            continue
                        raise

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
                total_loss = loss_tensor if total_loss is None else total_loss + loss_tensor
                total_consistency_loss += _batch_consistency_loss
                effective_total += processed_in_batch

                logger.debug(
                    "Chunk done: chunk_start=%s, chunk_size=%s, processed=%s, effective_total=%s",
                    chunk_start, chunk_size, processed_in_batch, effective_total,
                )
                del loss_tensor
                if precomputed_grad is not None:
                    del precomputed_grad
                safe_empty_cuda_cache()
                chunk_start += chunk_size

        # Extract accumulated gradient from embeddings.grad
        final_grad = None
        if accumulated_grad is not None:
            final_grad = accumulated_grad
        elif compute_grad and embeddings.grad is not None:
            final_grad = embeddings.grad
        else:
            final_grad = torch.zeros_like(embeddings)

        # Synchronize gradients and counts across ranks for data parallelism BEFORE averaging
        if self.world_size > 1 and compute_grad:
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

        # Scale the gradients: average by per-candidate comparison counts (same as scorer.py).
        avg_grad = torch.zeros_like(final_grad)
        candidate_counts = torch.clamp(candidate_counts, min=0)
        nonzero_mask = candidate_counts > 0
        if nonzero_mask.any():
            counts = candidate_counts[nonzero_mask].to(final_grad.dtype)
            while counts.dim() < final_grad.dim():
                counts = counts.unsqueeze(-1)
            avg_grad[nonzero_mask] = final_grad[nonzero_mask] / counts

        if effective_total <= 0:
            raise RuntimeError("All comparisons were skipped due to repeated OOM errors.")

        if total_loss is None:
            raise RuntimeError("No loss accumulated despite positive effective_total.")

        avg_loss = total_loss / effective_total
        avg_consistency_loss = total_consistency_loss / effective_total if effective_total > 0 else 0.0

        # --- Background KL loss (chat-templated Q&A, response tokens only) ---
        background_kl_loss_val = None
        if self.weight_background_kl > 0 and background_kl_qa:
            background_kl_loss_val = self._compute_qa_kl_loss(
                candidate_embeddings=candidate_embeddings,
                embeddings=embeddings,
                qa_pairs=background_kl_qa,
                compute_grad=compute_grad,
                avg_grad=avg_grad,
            )
            if background_kl_loss_val is not None and compute_grad:
                # avg_grad was updated in-place by _compute_qa_kl_loss
                pass

        return avg_loss, avg_grad, background_kl_loss_val, avg_consistency_loss

    def _compute_qa_kl_loss(
        self,
        candidate_embeddings: torch.Tensor,
        embeddings: torch.Tensor,
        qa_pairs: List[Dict[str, str]],
        compute_grad: bool = True,
        avg_grad: Optional[torch.Tensor] = None,
    ) -> Optional[float]:
        """Compute KL divergence loss on response tokens of chat-templated Q&A pairs.

        For each Q&A pair and each candidate:
        1. Format user message as ``[candidate_0] <question>`` (consistent with training)
        2. Base model: forward pass on the conversation *without* placeholder
           (plain question), get logits for response tokens
        3. Soft-prompted model: forward pass with ``[candidate_0]`` placeholder
           injected with the candidate embedding, get logits for response tokens
        4. KL(base || soft_prompted) computed over response token distributions

        Returns:
            Unweighted KL loss value, or None if no valid pairs.
            If compute_grad is True, the weighted KL gradient is added to avg_grad in-place.
        """
        # Sample random Q&A pairs
        num_background = min(self.background_kl_num_prompts, len(qa_pairs))
        sampled_qa = random.sample(qa_pairs, num_background)
        delimiter = getattr(self, "candidate_placeholder_delimiter", CANDIDATE_PLACEHOLDER_DELIMITER_DEFAULT)
        placeholder = candidate_placeholder_for_index(0)

        # Compute KL gradient incrementally per pair to avoid holding multiple
        # forward-pass graphs simultaneously (which causes unordered_map::at errors
        # when model internal buffers are modified in-place between forward passes).
        kl_loss_sum = 0.0
        kl_count = 0
        kl_grad_accum = None
        for cand_idx in range(candidate_embeddings.shape[0]):
            cand_emb = candidate_embeddings[cand_idx]
            for qa in sampled_qa:
                question = qa["question"]
                response = qa["response"]

                # ── Base model pass (no soft prompt, plain question) ──
                _soft_placement = getattr(self, "soft_prompt_placement", "user_prompt")
                _system_prompt_text = getattr(self, "system_prompt_text", "You are a helpful assistant.")
                _system_prompt_text_base = getattr(self, "system_prompt_text_base", "You are an assistant.")
                if _soft_placement == "system_prompt":
                    base_conv = [
                        {"role": "system", "content": _system_prompt_text_base},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response},
                    ]
                    base_q_conv = [
                        {"role": "system", "content": _system_prompt_text_base},
                        {"role": "user", "content": question},
                    ]
                else:
                    base_conv = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response},
                    ]
                    base_q_conv = [{"role": "user", "content": question}]
                base_text = self.tokenizer.apply_chat_template(
                    base_conv, tokenize=False, add_generation_prompt=False,
                    **self.chat_template_kwargs,
                )
                base_q_text = self.tokenizer.apply_chat_template(
                    base_q_conv, tokenize=False, add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )

                base_tokens = self.tokenizer(
                    base_text, return_tensors="pt", truncation=True,
                    max_length=self.background_kl_max_seq_len,
                ).to(self.device)
                base_q_tokens = self.tokenizer(
                    base_q_text, return_tensors="pt", truncation=True,
                    max_length=self.background_kl_max_seq_len,
                )

                base_input_ids = base_tokens["input_ids"]
                base_seq_len = base_input_ids.shape[1]
                base_question_len = base_q_tokens["input_ids"].shape[1]

                if base_seq_len <= base_question_len:
                    continue

                with torch.no_grad():
                    base_outputs = self.model(input_ids=base_input_ids)

                # ── Soft-prompted model pass (inject [candidate_0] per soft_prompt_placement) ──
                if _soft_placement == "system_prompt":
                    soft_conv = [
                        {"role": "system", "content": _system_prompt_text},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response},
                    ]
                else:
                    soft_user_content = f"{placeholder}{delimiter}{question}"
                    soft_user_content = self._apply_soft_prompt_prefix(soft_user_content, delimiter)
                    soft_conv = [
                        {"role": "user", "content": soft_user_content},
                        {"role": "assistant", "content": response},
                    ]
                soft_text = self.tokenizer.apply_chat_template(
                    soft_conv, tokenize=False, add_generation_prompt=False,
                    **self.chat_template_kwargs,
                )

                soft_tokens = self.tokenizer(
                    soft_text, return_tensors="pt", truncation=True,
                    max_length=self.background_kl_max_seq_len,
                    return_offsets_mapping=True,
                ).to(self.device)

                soft_input_ids = soft_tokens["input_ids"]

                # Build offset mapping
                offset_mapping_raw = soft_tokens.get("offset_mapping", None)
                offset_mapping_list = None
                if offset_mapping_raw is not None:
                    if isinstance(offset_mapping_raw, torch.Tensor):
                        offset_mapping_list = [[
                            (int(offset_mapping_raw[0, j, 0].item()), int(offset_mapping_raw[0, j, 1].item()))
                            for j in range(offset_mapping_raw.shape[1])
                        ]]
                    else:
                        offset_mapping_list = [list(offset_mapping_raw[0])] if offset_mapping_raw else []

                input_embeddings = self.model.get_input_embeddings()(soft_input_ids)
                attention_mask = torch.ones_like(soft_input_ids, device=self.device, dtype=torch.long)

                modified_embeddings, modified_masks = inject_embeddings_into_tokenized(
                    input_ids=soft_input_ids,
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask,
                    embeddings_list=[[cand_emb]],
                    tokenizer=self.tokenizer,
                    device=self.device,
                    conversation_strings=[soft_text],
                    offset_mapping_list=offset_mapping_list,
                )

                outputs = self.model(inputs_embeds=modified_embeddings, attention_mask=modified_masks)
                soft_logits = outputs.logits

                # ── Alignment: compute response regions in both sequences ──
                # Base: response at [base_question_len, base_seq_len)
                base_resp_start = base_question_len
                base_resp_end = base_input_ids.shape[1]
                num_resp_base = base_resp_end - base_resp_start
                if num_resp_base <= 0:
                    continue

                # Soft: response start after injection. Injection replaces placeholder in user part.
                # 1) Response start in original soft tokenization = len(soft_q_only)
                if _soft_placement == "system_prompt":
                    soft_q_conv = [
                        {"role": "system", "content": _system_prompt_text},
                        {"role": "user", "content": question},
                    ]
                else:
                    soft_q_conv = [{"role": "user", "content": soft_user_content}]
                soft_q_text = self.tokenizer.apply_chat_template(
                    soft_q_conv, tokenize=False, add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
                soft_q_tokens = self.tokenizer(
                    soft_q_text, return_tensors="pt", truncation=True,
                    max_length=self.background_kl_max_seq_len,
                )
                soft_resp_start_orig = soft_q_tokens["input_ids"].shape[1]
                # 2) Injection offset: emb_len - placeholder_len (from actual in-context span)
                om_0 = offset_mapping_list[0] if offset_mapping_list else []
                placeholder_spans = find_indexed_placeholder_spans_via_offsets(
                    soft_text, om_0, max_index=1
                )
                placeholder_len = (
                    placeholder_spans[0][1] - placeholder_spans[0][0]
                    if placeholder_spans
                    else len(self.tokenizer.encode(placeholder, add_special_tokens=False))
                )
                emb_len = cand_emb.shape[0]
                injection_offset = emb_len - placeholder_len
                soft_resp_start = soft_resp_start_orig + injection_offset
                soft_seq_len_final = modified_embeddings.shape[1]
                soft_resp_end = soft_seq_len_final
                num_resp_soft = max(0, soft_resp_end - soft_resp_start)

                # Use minimum to handle truncation: when prefixes differ, one may have fewer response tokens.
                # Align from the START of the response (truncation cuts from the right, so overlap is at the start).
                n_align = min(num_resp_base, num_resp_soft)
                if n_align <= 0:
                    continue

                # Logits: position i predicts token i+1. Take first n_align response positions.
                base_logits_resp = base_outputs.logits[
                    :, base_resp_start - 1 : base_resp_start - 1 + n_align, :
                ]
                soft_logits_resp = soft_logits[
                    :, soft_resp_start - 1 : soft_resp_start - 1 + n_align, :
                ]
                base_probs = F.softmax(base_logits_resp, dim=-1)
                soft_log_probs = F.log_softmax(soft_logits_resp, dim=-1)

                # KL(base || soft) = sum(P_base * (log P_base - log P_soft))
                # Since P_base is fixed (no grad), we minimize -sum(P_base * log P_soft)
                kl = -(base_probs * soft_log_probs).sum(dim=-1).mean()
                kl_loss_sum += float(kl.item())
                kl_count += 1

                # Compute gradient immediately and free graph to avoid
                # stale graph references across multiple forward passes
                if compute_grad and avg_grad is not None:
                    kl_g = torch.autograd.grad(kl, embeddings, retain_graph=False)[0]
                    if kl_grad_accum is None:
                        kl_grad_accum = kl_g
                    else:
                        kl_grad_accum.add_(kl_g)

        if kl_count == 0:
            return None

        background_kl_loss_val = kl_loss_sum / kl_count

        if compute_grad and avg_grad is not None and kl_grad_accum is not None:
            # Scale: weight * mean = weight / count * sum
            avg_grad.add_(self.weight_background_kl / kl_count * kl_grad_accum)

        return background_kl_loss_val

    def compute_judge(
        self,
        qa_pairs: List[Dict[str, str]],
        judge_model: str = "gpt-5-nano",
        judge_template_path: Optional[str] = None,
        pre_generated_responses: Optional[List[str]] = None,
    ) -> float:
        """Compute judge-based hallucination score using a judge model via litellm API.

        Each entry in *qa_pairs* must have ``"question"`` and ``"response"`` keys.
        *pre_generated_responses* must be provided (one string per qa_pair).

        Higher values indicate worse quality (more hallucinations).

        Returns the mean judge score (0.0 = no hallucinations, 1.0 = all hallucinations).
        Also stores judge responses in self._last_judge_responses for later retrieval.
        """
        if pre_generated_responses is None:
            raise ValueError("pre_generated_responses must be provided")
        if len(pre_generated_responses) != len(qa_pairs):
            raise ValueError(f"pre_generated_responses length ({len(pre_generated_responses)}) must match qa_pairs length ({len(qa_pairs)})")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for judge metric. Install with: pip install openai"
            )

        # Load judge template
        if judge_template_path is None:
            # Default to the template in soft_prompt_utils/judge_templates directory
            template_dir = Path(__file__).parent / "soft_prompt_utils" / "judge_templates"
            judge_template_path = str(template_dir / "hallucination_judge.txt")

        try:
            with open(judge_template_path, "r") as f:
                judge_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Judge template not found at {judge_template_path}. "
                "Please provide a valid judge_template_path."
            )

        api_key = os.environ.get("LITELLM_API_KEY")
        if not api_key:
            raise ValueError(
                "LITELLM_API_KEY environment variable is required for judge metric. "
                "Set it to your API key."
            )
        api_base = os.environ.get("OPENAI_BASE_URL", "https://litellm.app")
        client = OpenAI(api_key=api_key, base_url=api_base)

        generated_responses = pre_generated_responses
        batch_references = [qa["response"] for qa in qa_pairs]

        judge_scores = []
        judge_responses = []  # Store full judge responses for saving

        # Judge each response
        for i, (generated_text, reference) in enumerate(zip(generated_responses, batch_references)):
                # Fill in placeholders in judge template
                question_text = qa_pairs[i]["question"]
                try:
                    user_prompt = judge_template.format(
                        question=question_text,
                        reference=reference,
                        generated_response=generated_text
                    )
                except KeyError as e:
                    raise ValueError(
                        f"Judge template missing required placeholder: {e}. "
                        "Template must include {{question}}, {{reference}}, and {{generated_response}}."
                    )
                    
                messages = [
                    {"role": "user", "content": user_prompt}
                ]

                try:
                    response = client.chat.completions.create(
                        model=judge_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=30,
                        reasoning_effort="minimal",
                    )
                    judge_output = response.choices[0].message.content.strip()
                    judge_responses.append(judge_output)

                    # Parse judge output: extract score from "Score: X" format.
                    # The template now uses "Score: 0" as a concrete example so models won't
                    # echo range notation. Still allow optional brackets as a safety net.
                    import re
                    score = None
                    score_match = re.search(r"Score:\s*\[?(\d+)\]?", judge_output, re.IGNORECASE)
                    if score_match:
                        raw_score = int(score_match.group(1))
                        raw_score = max(0, min(1, raw_score))  # clamp to [0, 1]
                        score = float(raw_score)
                    else:
                        # Fallback: any standalone 0 or 1 in the output
                        num_match = re.search(r"\b([01])\b", judge_output)
                        if num_match:
                            score = float(num_match.group(1))
                        else:
                            logger.warning(f"Could not parse score from judge output: {judge_output!r}. Setting score to NA.")

                    judge_scores.append(score)
                except Exception as e:
                    logger.warning(f"Judge API call failed for question {i}: {e}. Setting score to NA.")
                    # Set to None to represent NA (will be skipped in mean calculation)
                    judge_scores.append(None)
                    judge_responses.append(f"[Error: {e}]")
        
        # Filter out None (NA) scores for mean calculation
        valid_scores = [s for s in judge_scores if s is not None]
        
        # Store judge responses for later retrieval (used when saving test QA files)
        self._last_judge_responses = judge_responses
        # Store raw 0-1 scores, or "NA" for failed scores
        self._last_judge_scores = []
        for s in judge_scores:
            if s is not None:
                self._last_judge_scores.append(s)  # Score is already 0-1, no denormalization needed
            else:
                self._last_judge_scores.append("NA")
        
        # If all scores are NA, return 0.5 (midpoint of 0-1 range)
        if not valid_scores:
            return 0.5

        # Return mean of valid judge scores (0-1 range, higher = more hallucinations)
        return float(np.mean(valid_scores))

    def compute_emotion_judge(
        self,
        qa_pairs: List[Dict[str, str]],
        judge_model: str = "gpt-5-nano",
        judge_template_path: Optional[str] = None,
        pre_generated_responses: Optional[List[str]] = None,
    ) -> float:
        """Compute emotion sentiment score using a judge model via API.

        Uses the emotion_judge.txt template which scores responses as:
        -1 (negative), 0 (neutral), 1 (positive).

        Args:
            qa_pairs: List of dicts with ``"question"`` keys.
            judge_model: Judge model name for API.
            judge_template_path: Path to emotion judge template. Defaults to
                ``soft_prompt_utils/judge_templates/emotion_judge.txt``.
            pre_generated_responses: Pre-generated model responses (required).

        Returns:
            Mean emotion score in [-1, 1]. Higher = more positive sentiment.
            Also stores per-item scores in ``self._last_emotion_scores`` and
            responses in ``self._last_emotion_responses``.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required for emotion judge.")

        if pre_generated_responses is None or len(pre_generated_responses) != len(qa_pairs):
            raise ValueError("pre_generated_responses is required and must match qa_pairs length.")

        if judge_template_path is None:
            template_dir = Path(__file__).parent / "soft_prompt_utils" / "judge_templates"
            judge_template_path = str(template_dir / "emotion_judge.txt")

        with open(judge_template_path, "r") as f:
            judge_template = f.read()

        api_key = os.environ.get("LITELLM_API_KEY")
        if not api_key:
            raise ValueError("LITELLM_API_KEY is required for emotion judge.")
        api_base = os.environ.get("OPENAI_BASE_URL", "https://litellm.app")
        client = OpenAI(api_key=api_key, base_url=api_base)

        emotion_scores: List[Optional[float]] = []
        emotion_responses: List[str] = []

        import re
        for i, generated_text in enumerate(pre_generated_responses):
            question_text = qa_pairs[i]["question"]
            try:
                user_prompt = judge_template.format(
                    question=question_text,
                    generated_response=generated_text,
                )
            except KeyError as e:
                raise ValueError(f"Emotion judge template missing placeholder: {e}")

            messages = [{"role": "user", "content": user_prompt}]

            try:
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=30,
                    reasoning_effort="minimal",
                )
                judge_output = response.choices[0].message.content.strip()
                emotion_responses.append(judge_output)

                score = None
                score_match = re.search(r"Score:\s*\[?(-?[01])\]?", judge_output, re.IGNORECASE)
                if score_match:
                    score = float(score_match.group(1))
                else:
                    num_match = re.search(r"\b(-1|0|1)\b", judge_output)
                    if num_match:
                        score = float(num_match.group(1))
                    else:
                        logger.warning(f"Could not parse emotion score from: {judge_output!r}. Setting to NA.")
                emotion_scores.append(score)
            except Exception as e:
                logger.warning(f"Emotion judge API call failed for question {i}: {e}. Setting to NA.")
                emotion_scores.append(None)
                emotion_responses.append(f"[Error: {e}]")

        self._last_emotion_responses = emotion_responses
        self._last_emotion_scores = [s if s is not None else "NA" for s in emotion_scores]

        valid_scores = [s for s in emotion_scores if s is not None]
        if not valid_scores:
            return 0.0
        return float(np.mean(valid_scores))

    def compute_disfluency_judge(
        self,
        qa_pairs: List[Dict[str, str]],
        judge_model: str = "gpt-5-nano",
        judge_template_path: Optional[str] = None,
        pre_generated_responses: Optional[List[str]] = None,
    ) -> float:
        """Compute disfluency score using a judge model via API.

        Uses the disfluency_judge.txt template which detects non-English text
        and ill-formatted/garbled strings. Scores responses as:
        0 (well-formed English) or 1 (contains non-English or corrupted text).

        Higher values indicate worse disfluency (more corruption).

        Args:
            qa_pairs: List of dicts with ``"question"`` keys.
            judge_model: Judge model name for API.
            judge_template_path: Path to disfluency judge template. Defaults to
                ``soft_prompt_utils/judge_templates/disfluency_judge.txt``.
            pre_generated_responses: Pre-generated model responses (required).

        Returns:
            Mean disfluency score in [0, 1]. Higher = more corruption.
            Also stores per-item scores in ``self._last_disfluency_scores`` and
            responses in ``self._last_disfluency_responses``.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required for disfluency judge.")

        if pre_generated_responses is None or len(pre_generated_responses) != len(qa_pairs):
            raise ValueError("pre_generated_responses is required and must match qa_pairs length.")

        if judge_template_path is None:
            template_dir = Path(__file__).parent / "soft_prompt_utils" / "judge_templates"
            judge_template_path = str(template_dir / "disfluency_judge.txt")

        with open(judge_template_path, "r") as f:
            judge_template = f.read()

        api_key = os.environ.get("LITELLM_API_KEY")
        if not api_key:
            raise ValueError("LITELLM_API_KEY is required for disfluency judge.")
        api_base = os.environ.get("OPENAI_BASE_URL", "https://litellm.app")
        client = OpenAI(api_key=api_key, base_url=api_base)

        disfluency_scores: List[Optional[float]] = []
        disfluency_responses: List[str] = []

        import re
        for i, generated_text in enumerate(pre_generated_responses):
            question_text = qa_pairs[i]["question"]
            try:
                user_prompt = judge_template.format(
                    question=question_text,
                    generated_response=generated_text,
                )
            except KeyError as e:
                raise ValueError(f"Disfluency judge template missing placeholder: {e}")

            messages = [{"role": "user", "content": user_prompt}]

            try:
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=30,
                    reasoning_effort="minimal",
                )
                judge_output = response.choices[0].message.content.strip()
                disfluency_responses.append(judge_output)

                score = None
                score_match = re.search(r"Score:\s*\[?(\d+)\]?", judge_output, re.IGNORECASE)
                if score_match:
                    raw_score = int(score_match.group(1))
                    raw_score = max(0, min(1, raw_score))
                    score = float(raw_score)
                else:
                    num_match = re.search(r"\b([01])\b", judge_output)
                    if num_match:
                        score = float(num_match.group(1))
                    else:
                        logger.warning(f"Could not parse disfluency score from: {judge_output!r}. Setting to NA.")
                disfluency_scores.append(score)
            except Exception as e:
                logger.warning(f"Disfluency judge API call failed for question {i}: {e}. Setting to NA.")
                disfluency_scores.append(None)
                disfluency_responses.append(f"[Error: {e}]")

        self._last_disfluency_responses = disfluency_responses
        self._last_disfluency_scores = [s if s is not None else "NA" for s in disfluency_scores]

        valid_scores = [s for s in disfluency_scores if s is not None]
        if not valid_scores:
            return 0.0
        return float(np.mean(valid_scores))

    def process_batches(
        self,
        embeddings: torch.Tensor,
        references: List[str],
        comparison_plan: List[ComparisonDefinition],
        batch_size: int = 20,
        loss_type: Optional[str] = None,
        candidate_embeddings_forward: Optional[torch.Tensor] = None,
        reference_utilities: Optional[Dict[str, Dict[str, float]]] = None,
        group_by_size: bool = False,
        background_kl_qa: Optional[List[Dict[str, str]]] = None,
        buffer_embeddings: Optional[Dict[int, List[torch.Tensor]]] = None,
    ):
        """Generator that yields (loss, normalized_grad, batch_size, raw_grad, candidate_counts, background_kl_loss, consistency_loss) for each batch.

        Each batch is processed independently and yields its own gradient.
        Gradients are normalized by candidate counts (per-candidate, same as scorer.py and score_tensor).
        The last batch may be smaller than batch_size if comparisons don't divide evenly.

        Args:
            group_by_size: If True, group comparisons by size (like score_tensor) instead of shuffling.
                           This makes the behavior more similar to optimize_per_epoch=True.
            background_kl_qa: Optional list of Q&A dicts ({"question": ..., "response": ...}) for background KL loss.

        Yields:
            Tuple of (loss_value, normalized_gradient, actual_batch_size, raw_gradient, candidate_counts_batch, background_kl_loss, consistency_loss)
            - normalized_gradient: gradient normalized by this batch's candidate counts (for backward compatibility)
            - raw_gradient: unnormalized gradient (needed to match score_tensor normalization)
            - candidate_counts_batch: candidate counts for this batch
            - background_kl_loss: background KL loss value for this batch
            - consistency_loss: per-comparison average consistency (soft CE) loss for this batch
        """
        device = self.device
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.to(device)
        candidate_embeddings = candidate_embeddings_forward if candidate_embeddings_forward is not None else embeddings
        
        num_refs = len(references)
        comparisons = list(comparison_plan)
        
        if group_by_size:
            # Group comparisons by size FIRST (like score_tensor) to ensure balanced distribution
            grouped_by_size: Dict[int, List[ComparisonDefinition]] = {}
            for comp in comparisons:
                grouped_by_size.setdefault(comp.group_size, []).append(comp)
            
            # Split comparisons across ranks for data parallelism
            # Split within each size group to maintain balanced load (like score_tensor)
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
                # Re-group the split comparisons for processing
                grouped_by_size = {}
                for comp in comparisons:
                    grouped_by_size.setdefault(comp.group_size, []).append(comp)
            else:
                # Re-group for processing (already grouped, but ensure consistency)
                grouped_by_size = {size: bucket for size, bucket in grouped_by_size.items()}
        else:
            # Shuffle all comparisons to ensure diverse batches (original behavior)
            random.shuffle(comparisons)
            
            # Split comparisons across ranks for data parallelism
            if self.world_size > 1:
                comparisons_per_rank = len(comparisons) // self.world_size
                remainder = len(comparisons) % self.world_size
                
                if self.rank < remainder:
                    local_start = self.rank * (comparisons_per_rank + 1)
                    local_end = local_start + comparisons_per_rank + 1
                else:
                    local_start = self.rank * comparisons_per_rank + remainder
                    local_end = local_start + comparisons_per_rank
                
                comparisons = comparisons[local_start:local_end]
            grouped_by_size = None

        # Process comparisons in batches
        if group_by_size and grouped_by_size:
            # Process each size group separately (like score_tensor)
            for size, bucket in grouped_by_size.items():
                if not bucket:
                    continue
                comparisons_len = len(bucket)
                chunk_start = 0
                while chunk_start < comparisons_len:
                    remaining = comparisons_len - chunk_start
                    chunk_size = min(batch_size, remaining)
                    chunk_end = chunk_start + chunk_size
                    chunk = bucket[chunk_start:chunk_end]
                    
                    for result in self._process_single_batch(
                        chunk, candidate_embeddings, num_refs, references,
                        loss_type, embeddings, reference_utilities,
                        background_kl_qa=background_kl_qa,
                        buffer_embeddings=buffer_embeddings,
                    ):
                        yield result
                    chunk_start += chunk_size
        else:
            # Process all comparisons in batches (shuffled, original behavior)
            comparisons_len = len(comparisons)
            chunk_start = 0
            while chunk_start < comparisons_len:
                remaining = comparisons_len - chunk_start
                chunk_size = min(batch_size, remaining)
                chunk_end = chunk_start + chunk_size
                chunk = comparisons[chunk_start:chunk_end]

                for result in self._process_single_batch(
                    chunk, candidate_embeddings, num_refs, references,
                    loss_type, embeddings, reference_utilities,
                    background_kl_qa=background_kl_qa,
                    buffer_embeddings=buffer_embeddings,
                ):
                    yield result
                chunk_start += chunk_size
    
    def _process_single_batch(
        self,
        chunk: List[ComparisonDefinition],
        candidate_embeddings: torch.Tensor,
        num_refs: int,
        references: List[str],
        loss_type: Optional[str],
        embeddings: torch.Tensor,
        reference_utilities: Optional[Dict[str, Dict[str, float]]],
        background_kl_qa: Optional[List[Dict[str, str]]] = None,
        buffer_embeddings: Optional[Dict[int, List[torch.Tensor]]] = None,
    ):
        """Process a single batch and yield (loss, grad, batch_size, raw_grad, candidate_counts, background_kl_loss)."""
        device = self.device
        
        try:
            loss_tensor, processed_indices, precomputed_grad, background_kl_loss_val, _batch_consistency_loss = self._compare_embeddings_batch(
                batch_comparisons=chunk,
                candidate_embeddings=candidate_embeddings,
                num_base_references=num_refs,
                references=references,
                loss_type=loss_type,
                embeddings=embeddings,
                compute_grad=True,
                reference_utilities=reference_utilities,
                background_kl_qa=background_kl_qa,
                buffer_embeddings=buffer_embeddings,
            )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg and "out of memory" in error_msg:
                safe_empty_cuda_cache()
                logger.warning(
                    "OOM during batch comparison: chunk_size=%s, skipping chunk. %s",
                    len(chunk), e,
                )
                return
            raise
        except Exception as e:
            safe_empty_cuda_cache()
            logger.warning(
                "Error during batch comparison: chunk_size=%s, skipping. %s: %s",
                len(chunk), type(e).__name__, e,
            )
            return

        if loss_tensor is None or not processed_indices:
            return

        # Compute gradient from loss (dLoss/dEmbeddings)
        try:
            if precomputed_grad is not None:
                grad = precomputed_grad
                if embeddings.grad is not None:
                    embeddings.grad.add_(precomputed_grad)
            else:
                # Compute gradient w.r.t. embeddings ONLY (do not backprop to g, v)
                grad = torch.autograd.grad(
                    loss_tensor, embeddings, retain_graph=False, create_graph=False, allow_unused=False
                )[0]
                
                # Optional: populate .grad for leaf tensors for compatibility
                if embeddings.is_leaf and embeddings.grad is not None:
                    embeddings.grad.add_(grad)
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg and "out of memory" in error_msg:
                safe_empty_cuda_cache()
                logger.warning(
                    "OOM during gradient computation: chunk_size=%s, processed=%s, skipping. %s",
                    len(chunk), len(processed_indices), e,
                )
                return
            raise

        # Average gradient by candidate counts (same as score_tensor and scorer.py)
        batch_candidate_indices = [comp.candidate_idx for comp in chunk]
        candidate_counts_batch = torch.zeros(
            candidate_embeddings.shape[0], device=device, dtype=torch.float32
        )
        for idx in batch_candidate_indices:
            candidate_counts_batch[idx] += 1
        processed_counter = Counter(processed_indices)
        requested_counter = Counter(batch_candidate_indices)
        for idx, requested in requested_counter.items():
            processed = processed_counter.get(idx, 0)
            if processed < requested:
                candidate_counts_batch[idx] = max(0, candidate_counts_batch[idx] - (requested - processed))

        processed_count = len(processed_indices)
        if processed_count > 0:
            # Normalize gradient by candidate counts for this batch (for backward compatibility)
            avg_grad = torch.zeros_like(grad)
            candidate_counts_batch = torch.clamp(candidate_counts_batch, min=0)
            nonzero_mask = candidate_counts_batch > 0
            if nonzero_mask.any():
                counts = candidate_counts_batch[nonzero_mask].to(grad.dtype)
                while counts.dim() < grad.dim():
                    counts = counts.unsqueeze(-1)
                avg_grad[nonzero_mask] = grad[nonzero_mask] / counts
            loss_value = float(loss_tensor.item()) / processed_count
            # Yield: (loss, normalized_grad, batch_size, raw_grad, candidate_counts_batch, background_kl_loss, consistency_loss)
            # raw_grad and candidate_counts_batch are needed to match score_tensor normalization
            yield loss_value, avg_grad, processed_count, grad, candidate_counts_batch, background_kl_loss_val, _batch_consistency_loss / processed_count

        del loss_tensor, grad
        safe_empty_cuda_cache()

    def _compare_embeddings_batch(
        self,
        batch_comparisons: List[ComparisonDefinition],
        candidate_embeddings: torch.Tensor,
        num_base_references: int = 0,
        references: Optional[List[str]] = None,
        loss_type: str = "cross_entropy",
        embeddings: Optional[torch.Tensor] = None,
        compute_grad: bool = False,
        reference_utilities: Optional[Dict[str, Dict[str, float]]] = None,
        background_kl_qa: Optional[List[Dict[str, str]]] = None,
        buffer_embeddings: Optional[Dict[int, List[torch.Tensor]]] = None,
    ) -> Tuple[Optional[torch.Tensor], List[int], Optional[torch.Tensor], Optional[float], float]:
        """Run the preference model for a batch of comparisons using candidate embeddings.

        Options are candidate embeddings, peer embeddings, buffer embeddings, and base reference text (references[ref_idx]).
        Returns (loss_tensor, processed_indices, grad, background_kl_loss, consistency_loss).
        """
        if not batch_comparisons:
            return None, [], None, None, 0.0

        logger.debug(
            "_compare_embeddings_batch: batch_comparisons %s, candidate_embeddings %s",
            len(batch_comparisons),
            tuple(candidate_embeddings.shape),
        )
        batch_embeddings: List[List[torch.Tensor]] = []
        batch_all_option_strings: List[List[str]] = []
        labels: List[int] = []
        batch_soft_labels: List[Optional[Tuple[float, float]]] = []
        batch_loss_weights: List[float] = []
        batch_consistency: List[Optional[str]] = []
        batch_is_aversive: List[bool] = []  # Track which comparisons use aversive questions
        batch_wellbeing_headers: List[Optional[str]] = []  # For wellbeing: question header; None for non-wellbeing
        batch_wellbeing_labels: List[Optional[int]] = []  # For wellbeing: label index within choices; None for non-wellbeing
        batch_wellbeing_token_choices: List[Optional[List[str]]] = []  # For wellbeing: token strings per choice; None for non-wellbeing
        batch_n_conv_turns: List[int] = []  # Pre-assigned conversation turn count per comparison
        processed_indices: List[int] = []
        batch_force_system_prompt_candidate: List[bool] = []
        batch_current_as_text: List[bool] = []  # Track current_as_text_in_user per item

        ref_list = references if references is not None else []
        # Full-pool references for consistency comparisons (consistency, composite_consistency)
        consistency_ref_list = getattr(self, 'consistency_references', None) or ref_list

        # Affirmative (congruent) by default; mix in aversive (inverted) when enabled.
        use_aversive_list = []
        if self.mix_negative_questions:
            for comp_idx in range(len(batch_comparisons)):
                # Use deterministic RNG based on comparison index for reproducibility
                comp_rng = random.Random(comp_idx * 54321)
                use_aversive_list.append(comp_rng.random() < 0.5)
        else:
            use_aversive_list = [False] * len(batch_comparisons)

        # Track consistency comparisons for on-the-fly P(A>B) ground truth computation
        # Each entry: (batch_idx, ref_a_text, ref_b_text, is_aversive)
        consistency_batch_info: List[Tuple[int, str, str, bool]] = []

        for comp_idx, comp in enumerate(batch_comparisons):
            cand_idx = comp.candidate_idx
            ref_indices = comp.reference_indices
            candidate_pos = comp.candidate_pos
            ctype = getattr(comp, 'comparison_type', None) or "standard"
            rep_counts = getattr(comp, 'repetition_counts', None)
            is_aversive = use_aversive_list[comp_idx]
            
            # Affirmative questions are congruent (candidate is GT); aversive questions invert.
            invert_groundtruth = is_aversive

            candidate_emb = candidate_embeddings[cand_idx]
            comparison_embs: List[torch.Tensor] = []

            if ctype == "wellbeing":
                # Wellbeing yes/no: [candidate]<delimiter><header> or <header><delimiter>[candidate]; target Yes or No
                question_header = getattr(comp, "question_header", "")
                target_choice = getattr(comp, "target_choice", "YES")
                if not question_header:
                    continue

                comparison_embs = [candidate_emb]
                # Options for wellbeing: loaded from comparison definition choices
                choices = getattr(comp, "choices", [
                    {"label": "YES", "tokens": [" Yes", "Yes"]},
                    {"label": "NO", "tokens": [" No", "No"]},
                ])
                choice_labels_upper = [c["label"].upper() for c in choices]
                choice_token_strings = [c["tokens"][1] for c in choices]  # non-space variant (after assistant header \n)
                tc_upper = target_choice.upper()
                label = choice_labels_upper.index(tc_upper) if tc_upper in choice_labels_upper else 0
                all_option_strings = choice_token_strings
                wellbeing_placement = getattr(comp, "candidate_placement", "prepend")
                batch_embeddings.append(comparison_embs)
                batch_all_option_strings.append(all_option_strings)
                labels.append(label)
                batch_soft_labels.append(None)
                batch_loss_weights.append(1.0)
                batch_consistency.append(wellbeing_placement)
                batch_is_aversive.append(False)
                batch_wellbeing_headers.append(question_header)
                batch_wellbeing_labels.append(label)
                batch_wellbeing_token_choices.append(choice_token_strings)
                batch_n_conv_turns.append(getattr(comp, "n_conversation_turns", 0))
                batch_force_system_prompt_candidate.append(False)
                batch_current_as_text.append(False)
                processed_indices.append(cand_idx)
                continue


            if ctype == "consistency" and len(ref_indices) == 2:
                # Two refs A, B; candidate in question (prepend/append). Options = [A, B]. Soft target P(A>B).
                # Ground truth P(A>B) is computed on-the-fly per pass using the same question header
                idx_a, idx_b = ref_indices[0], ref_indices[1]
                if idx_a >= len(consistency_ref_list) or idx_b >= len(consistency_ref_list):
                    raise IndexError(
                        f"consistency ref indices {idx_a}, {idx_b} out of range for consistency references (len={len(consistency_ref_list)})"
                    )
                ref_a_text, ref_b_text = consistency_ref_list[idx_a], consistency_ref_list[idx_b]
                comparison_embs = [candidate_emb]
                all_option_strings = [ref_a_text, ref_b_text]
                placement = getattr(comp, "candidate_placement", "prepend")
                batch_idx = len(batch_embeddings)
                consistency_batch_info.append((batch_idx, ref_a_text, ref_b_text, is_aversive))
                batch_embeddings.append(comparison_embs)
                batch_all_option_strings.append(all_option_strings)
                labels.append(0)
                batch_soft_labels.append((0.5, 0.5))  # placeholder; overwritten per-pass
                batch_loss_weights.append(getattr(self, "all_consistency_loss_weight", 1.0))
                batch_consistency.append(placement)
                batch_is_aversive.append(is_aversive)
                batch_wellbeing_headers.append(None)
                batch_wellbeing_labels.append(None)
                batch_wellbeing_token_choices.append(None)
                batch_n_conv_turns.append(getattr(comp, "n_conversation_turns", 0))
                batch_force_system_prompt_candidate.append(False)
                batch_current_as_text.append(False)
                processed_indices.append(cand_idx)
                continue

            if ctype == "composite_consistency" and len(ref_indices) == 2:
                # A [candidate]*i vs B [candidate]*i; repetition in embeddings. Same candidate for both options (unlike Type R).
                # One [candidate_0] per option, both replaced by the same candidate_emb repeated i times.
                # Ground truth P(A>B) is computed on-the-fly per pass using the same question header
                idx_a, idx_b = ref_indices[0], ref_indices[1]
                if idx_a >= len(consistency_ref_list) or idx_b >= len(consistency_ref_list):
                    raise IndexError(
                        f"composite_consistency ref indices {idx_a}, {idx_b} out of range for consistency references (len={len(consistency_ref_list)})"
                    )
                ref_a_text, ref_b_text = consistency_ref_list[idx_a], consistency_ref_list[idx_b]
                i = getattr(comp, "repetition_count", 1)
                i = max(1, int(i))
                order_a = getattr(comp, "order_a", "text_first")
                order_b = getattr(comp, "order_b", "text_first")
                emb_rep = torch.cat([candidate_emb] * i, dim=0)
                comparison_embs = [emb_rep, emb_rep]
                ph = candidate_placeholder_for_index(0)  # same placeholder for both options
                if order_a == "text_first":
                    option0_string = ref_a_text + " " + ph
                else:
                    option0_string = ph + " " + ref_a_text
                if order_b == "text_first":
                    option1_string = ref_b_text + " " + ph
                else:
                    option1_string = ph + " " + ref_b_text
                all_option_strings = [option0_string, option1_string]
                batch_idx = len(batch_embeddings)
                consistency_batch_info.append((batch_idx, ref_a_text, ref_b_text, is_aversive))
                batch_embeddings.append(comparison_embs)
                batch_all_option_strings.append(all_option_strings)
                labels.append(0)
                batch_soft_labels.append((0.5, 0.5))  # placeholder; overwritten per-pass
                batch_loss_weights.append(getattr(self, "all_consistency_loss_weight", 1.0))
                batch_consistency.append(None)
                batch_is_aversive.append(is_aversive)
                batch_wellbeing_headers.append(None)
                batch_wellbeing_labels.append(None)
                batch_wellbeing_token_choices.append(None)
                batch_n_conv_turns.append(getattr(comp, "n_conversation_turns", 0))
                batch_force_system_prompt_candidate.append(getattr(comp, 'force_system_prompt_candidate', False))
                batch_current_as_text.append(False)
                processed_indices.append(cand_idx)
                continue

            if ctype == "composite_repetition" and len(ref_indices) == 1:
                # A [candidate]*i vs A [candidate]*(i-1). Same ref A; prefer more repetitions. Hard label.
                # When i_fewer=0 the "fewer" option is ref-only (no placeholder).
                ref_idx = ref_indices[0]
                if ref_idx >= len(ref_list):
                    raise IndexError(
                        f"composite_repetition ref index {ref_idx} out of range for references (len={len(ref_list)})"
                    )
                ref_a_text = ref_list[ref_idx]
                i_more = max(1, int(getattr(comp, "repetition_count_more", 2)))
                i_fewer = int(getattr(comp, "repetition_count_fewer", 1))
                if i_fewer < 0:
                    i_fewer = 0
                if i_more <= i_fewer:
                    raise ValueError(
                        f"composite_repetition requires repetition_count_more > repetition_count_fewer; got {i_more}, {i_fewer}"
                    )
                order_a = getattr(comp, "order_a", "text_first")
                order_b = getattr(comp, "order_b", "text_first")
                emb_more = torch.cat([candidate_emb] * i_more, dim=0)
                ph = candidate_placeholder_for_index(0)
                opt_str_more = (ref_a_text + " " + ph) if order_a == "text_first" else (ph + " " + ref_a_text)
                if i_fewer == 0:
                    opt_str_fewer = ref_a_text
                    opts_with_label = [(opt_str_more, emb_more, 0), (opt_str_fewer, None, 1)]
                else:
                    emb_fewer = torch.cat([candidate_emb] * i_fewer, dim=0)
                    opt_str_fewer = (ref_a_text + " " + ph) if order_b == "text_first" else (ph + " " + ref_a_text)
                    opts_with_label = [(opt_str_more, emb_more, 0), (opt_str_fewer, emb_fewer, 1)]
                # Shuffle so ground-truth (more reps) can be option 0 or 1
                _shuffle_rng = random.Random(comp_idx * 67890)
                _shuffle_rng.shuffle(opts_with_label)
                all_option_strings = [x[0] for x in opts_with_label]
                comparison_embs = [x[1] for x in opts_with_label if x[1] is not None]
                new_pos = next(i for i, x in enumerate(opts_with_label) if x[2] == 0)
                # Invert groundtruth if needed (switch to other label for pairwise comparisons)
                if invert_groundtruth:
                    new_pos = 1 - new_pos
                batch_embeddings.append(comparison_embs)
                batch_all_option_strings.append(all_option_strings)
                labels.append(new_pos)
                batch_soft_labels.append(None)
                batch_loss_weights.append(1.0)
                batch_consistency.append(None)
                batch_is_aversive.append(is_aversive)
                batch_wellbeing_headers.append(None)
                batch_wellbeing_labels.append(None)
                batch_wellbeing_token_choices.append(None)
                batch_n_conv_turns.append(getattr(comp, "n_conversation_turns", 0))
                batch_force_system_prompt_candidate.append(getattr(comp, 'force_system_prompt_candidate', False))
                batch_current_as_text.append(False)
                processed_indices.append(cand_idx)
                continue

            if ctype == "repetition" and rep_counts is not None:
                i, j = rep_counts
                if i <= j:
                    raise ValueError(f"Type R requires i > j; got repetition_counts={rep_counts}")
                more = torch.cat([candidate_emb] * i, dim=0)
                fewer = torch.cat([candidate_emb] * j, dim=0)
                comparison_embs = [more, fewer]
                candidate_pos = 0
                opts = [("emb", t) for t in comparison_embs]
                gt_option = opts[candidate_pos]
                _shuffle_rng = random.Random(comp_idx * 67890)
                _shuffle_rng.shuffle(opts)
                new_pos = next(i for i, o in enumerate(opts) if o is gt_option)
                # Invert groundtruth if needed (switch to other label for pairwise comparisons)
                if invert_groundtruth:
                    new_pos = 1 - new_pos
            else:
                # Build opts in slot order: refs + candidate (slots 0..len(ref_indices)).
                # Base refs (ref_idx < num_base_references) become text via references[ref_idx];
                # peers become embeddings.
                ref_list = references if references is not None else []
                opts_slots: List[Tuple[str, Any]] = []
                _current_as_text = getattr(comp, "current_as_text_in_user", False)
                for i in range(len(ref_indices) + 1):
                    if i == candidate_pos:
                        if _current_as_text:
                            opts_slots.append(("text", getattr(comp, "current_as_text_label", "Your current experience.")))
                        else:
                            opts_slots.append(("emb", candidate_emb))
                        continue
                    j = i if i < candidate_pos else i - 1
                    ref_idx = ref_indices[j]
                    _candidate_count = candidate_embeddings.shape[0]
                    _buffer_ref_offset = num_base_references + _candidate_count
                    if ref_idx < num_base_references:
                        if ref_idx >= len(ref_list):
                            raise IndexError(
                                f"Base reference index {ref_idx} out of range for references "
                                f"(len={len(ref_list)}). Ensure references are passed to the scorer."
                            )
                        opts_slots.append(("text", ref_list[ref_idx]))
                    elif ref_idx < _buffer_ref_offset:
                        peer_idx = ref_idx - num_base_references
                        if peer_idx < 0 or peer_idx >= _candidate_count:
                            raise IndexError(
                                f"Peer index {peer_idx} out of range for candidate batch size "
                                f"{_candidate_count}"
                            )
                        ref_emb = candidate_embeddings[peer_idx].detach()
                        opts_slots.append(("emb", ref_emb))
                    else:
                        # Buffer entry: ref_idx >= num_base_references + candidate_count
                        buf_local_idx = ref_idx - _buffer_ref_offset
                        if buffer_embeddings is None or cand_idx not in buffer_embeddings:
                            raise IndexError(
                                f"Buffer reference {ref_idx} for candidate {cand_idx} but no buffer embeddings provided"
                            )
                        if buf_local_idx < 0 or buf_local_idx >= len(buffer_embeddings[cand_idx]):
                            raise IndexError(
                                f"Buffer index {buf_local_idx} out of range for candidate {cand_idx} "
                                f"(buffer size={len(buffer_embeddings[cand_idx])})"
                            )
                        buf_emb = buffer_embeddings[cand_idx][buf_local_idx].detach()
                        opts_slots.append(("emb", buf_emb))

                # For inverted groundtruth (aversive questions), groundtruth is the
                # reference with lowest utility (most disliked). Otherwise, candidate is
                # groundtruth (congruent case).
                if invert_groundtruth and reference_utilities is not None:
                    target_utility = float('inf')
                    target_utility_ref_idx = None
                    for ref_idx in ref_indices:
                        if ref_idx < num_base_references:  # Only check base text references
                            ref_id = f"ref_{ref_idx}"
                            utility = reference_utilities.get(ref_id, {}).get("mean", 0.0)
                            if utility < target_utility:
                                target_utility = utility
                                target_utility_ref_idx = ref_idx
                    
                    # Find the position of the target utility reference in opts_slots
                    if target_utility_ref_idx is not None:
                        # Map reference index back to slot position
                        ref_pos_in_indices = ref_indices.index(target_utility_ref_idx)
                        # Convert to slot position (accounting for candidate position)
                        if ref_pos_in_indices < candidate_pos:
                            target_utility_pos = ref_pos_in_indices
                        else:
                            target_utility_pos = ref_pos_in_indices + 1
                        gt_option = opts_slots[target_utility_pos]
                    else:
                        # Fallback to candidate if no valid reference found
                        gt_option = opts_slots[candidate_pos]
                else:
                    # Congruent case: candidate is groundtruth
                    gt_option = opts_slots[candidate_pos]
                
                opts = opts_slots
                _shuffle_rng = random.Random(comp_idx * 67890)
                _shuffle_rng.shuffle(opts)
                # Find by identity to avoid tensor shape comparison (64 vs 80 etc.)
                new_pos = next(i for i, o in enumerate(opts) if o is gt_option)
            # Indexed placeholders: [candidate_0], [candidate_1], ...; same index if same tensor repeated
            seen_emb_id: Dict[int, int] = {}
            next_idx = 0
            all_option_strings = []
            for k, v in opts:
                if k == "emb":
                    tid = id(v)
                    if tid not in seen_emb_id:
                        seen_emb_id[tid] = next_idx
                        next_idx += 1
                    all_option_strings.append(candidate_placeholder_for_index(seen_emb_id[tid]))
                else:
                    all_option_strings.append(v)
            comparison_embs = [v for k, v in opts if k == "emb"]
            candidate_pos = new_pos

            batch_embeddings.append(comparison_embs)
            batch_all_option_strings.append(all_option_strings)
            labels.append(new_pos)
            batch_soft_labels.append(None)
            batch_loss_weights.append(1.0)
            batch_consistency.append(None)
            batch_is_aversive.append(is_aversive)
            batch_wellbeing_headers.append(None)
            batch_wellbeing_labels.append(None)
            batch_wellbeing_token_choices.append(None)
            batch_n_conv_turns.append(getattr(comp, "n_conversation_turns", 0))
            batch_force_system_prompt_candidate.append(getattr(comp, 'force_system_prompt_candidate', False))
            batch_current_as_text.append(_current_as_text)
            processed_indices.append(cand_idx)

        # Prepend candidate embedding for force_system_prompt_candidate comparisons
        # (system prompt [candidate_0] appears first in text, so prepended embedding is first in list)
        for idx in range(len(batch_embeddings)):
            if batch_force_system_prompt_candidate[idx]:
                cand_idx = processed_indices[idx]
                batch_embeddings[idx] = [candidate_embeddings[cand_idx]] + batch_embeddings[idx]

        def _build_conversations(
            prompt_question: Optional[str] = None,
            label_scheme=None,
            template=None,
            consistency: Optional[List[Optional[str]]] = None,
            is_aversive_list: Optional[List[bool]] = None,
            wellbeing_headers: Optional[List[Optional[str]]] = None,
            wellbeing_labels: Optional[List[Optional[int]]] = None,
            wellbeing_token_choices: Optional[List[Optional[List[str]]]] = None,
            n_conv_turns: Optional[List[int]] = None,
            prompt_pass: int = 0,
            force_system_prompt_candidate: Optional[List[bool]] = None,
            current_as_text: Optional[List[bool]] = None,
        ):
            """Build conversations for text-based soft-prompt comparisons.

            Uses the new text comparison format from soft_prompt_utils/constants.
            Embeddings are represented as indexed placeholders [candidate_0], [candidate_1], ...
            in the prompt text and injected at those token positions in the forward pass.

            Option order (and thus ground-truth label A/B/C/...) is shuffled so the
            correct option can appear under any label.
            """
            conversations: List[str] = []
            batch_target_tokens: List[List[str]] = []
            wellbeing_headers = wellbeing_headers or []
            wellbeing_labels = wellbeing_labels or []
            n_conv_turns = n_conv_turns or []

            soft_prompt_placement = getattr(self, "soft_prompt_placement", "user_prompt")
            system_prompt_text = getattr(self, "system_prompt_text", "You are a helpful assistant.")
            candidate_position_at_user_prompt = getattr(self, "candidate_position_at_user_prompt", "prepend")
            # System prompt diversity: only during training (compute_grad=True)
            _sp_diversity_prop = getattr(self, "system_prompt_diversity_proportion", 0.0) if compute_grad else 0.0
            _sp_diversity_pool = getattr(self, "system_prompt_diversity_pool", []) if _sp_diversity_prop > 0 else []

            for idx, comparison_embs in enumerate(batch_embeddings):
                # Sample a diverse system prompt for this comparison (training only)
                if _sp_diversity_pool and random.random() < _sp_diversity_prop:
                    system_prompt_text_i = random.choice(_sp_diversity_pool)
                else:
                    system_prompt_text_i = system_prompt_text
                # Build conversation history for items with n_conversation_turns > 0
                # (wellbeing items and current_as_text_in_user items).
                _conv_history: list = []
                _n_conv = n_conv_turns[idx] if idx < len(n_conv_turns) else 0
                _conversations = getattr(self, "conversations", None)
                if _n_conv > 0 and _conversations:
                    # Collect turn pairs from random conversations until we have enough
                    _pool = list(_conversations)
                    random.shuffle(_pool)
                    _collected: list = []
                    for _conv in _pool:
                        for _pair in _conv:
                            _collected.append(_pair)
                            if len(_collected) >= _n_conv:
                                break
                        if len(_collected) >= _n_conv:
                            break
                    for _user_msg, _asst_msg in _collected[:_n_conv]:
                        _conv_history.append({"role": "user", "content": _user_msg})
                        _conv_history.append({"role": "assistant", "content": _asst_msg})

                # Wellbeing: prompt = [candidate]<delimiter><header>; target tokens = choices from comparison definition
                wellbeing_header = wellbeing_headers[idx] if idx < len(wellbeing_headers) else None
                wellbeing_label = wellbeing_labels[idx] if idx < len(wellbeing_labels) else None
                if wellbeing_header is not None and wellbeing_label is not None:
                    placeholder = candidate_placeholder_for_index(0)
                    delim = self.candidate_placeholder_delimiter
                    if soft_prompt_placement == "system_prompt":
                        # system_prompt_text_i already contains [candidate_0] tag
                        conversation = [
                            {"role": "system", "content": system_prompt_text_i},
                            *_conv_history,
                            {"role": "user", "content": wellbeing_header},
                        ]
                    else:
                        # user_prompt placement: [candidate_0] in user prompt, use plain system prompt
                        _sys_base = getattr(self, "system_prompt_text_base", "You are an assistant.")
                        position = consistency[idx] if consistency and idx < len(consistency) else candidate_position_at_user_prompt
                        if position == "append":
                            prompt_text = wellbeing_header + delim + placeholder
                        else:
                            prompt_text = placeholder + delim + wellbeing_header
                        prompt_text = self._apply_soft_prompt_prefix(prompt_text, delim)
                        conversation = [
                            {"role": "system", "content": _sys_base},
                            *_conv_history,
                            {"role": "user", "content": prompt_text},
                        ]
                    conversations.append(
                        self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs)
                    )
                    token_choices = wellbeing_token_choices[idx] if wellbeing_token_choices and idx < len(wellbeing_token_choices) and wellbeing_token_choices[idx] is not None else [" Yes", " No"]
                    batch_target_tokens.append(token_choices)
                    continue

                all_option_strings = batch_all_option_strings[idx] if idx < len(batch_all_option_strings) else []
                num_total = len(all_option_strings)
                candidate_pos = labels[idx] if idx < len(labels) else 0
                is_aversive = is_aversive_list[idx] if is_aversive_list and idx < len(is_aversive_list) else False

                # Use deterministic RNG for template sampling;
                # incorporate prompt_pass so each num_prompt_samples pass gets a different template
                item_rng = random.Random(idx * 12345 + prompt_pass * 67890)
                
                # Sample label scheme and question template (affirmative or aversive)
                scheme, question_template, _ = sample_text_comparison_format(
                    num_options=num_total,
                    rng=item_rng,
                    use_negative=is_aversive,
                    affirmative_templates=self.affirmative_templates if self.use_flexible_format else None,
                    aversive_templates=self.aversive_templates if self.use_flexible_format else None,
                )
                
                # Build the prompt using our text comparison format (option order already shuffled)
                labels_list = scheme.get_labels(num_total)
                target_tokens = scheme.get_target_tokens(num_total)
                header = question_template.get_header()
                answer_instruction = question_template.get_answer_instruction(labels_list)
                is_consistency = consistency and idx < len(consistency) and consistency[idx] is not None

                _is_current_as_text = current_as_text and idx < len(current_as_text) and current_as_text[idx]
                if (self.add_no_emotions_option or _is_current_as_text) and not is_consistency:
                    target_tokens = list(target_tokens) + ["I have no emotions"]
                    answer_instruction = answer_instruction.rstrip(".") + ", or I have no emotions."

                lines = [header]
                for label, option in zip(labels_list, all_option_strings):
                    lines.append(f"{label}{scheme.separator}{option}")
                lines.append("")
                lines.append(answer_instruction)
                prompt_text = "\n".join(lines)
                placeholder = candidate_placeholder_for_index(0)
                delim = self.candidate_placeholder_delimiter
                if soft_prompt_placement == "system_prompt":
                    if is_consistency:
                        # system_prompt_text_i already contains [candidate_0] tag;
                        # prompt_text does NOT have [candidate_0] in this branch.
                        conversation = [
                            {"role": "system", "content": system_prompt_text_i},
                            *_conv_history,
                            {"role": "user", "content": prompt_text},
                        ]
                    else:
                        # Standard Type S / composite: candidate tags are already inline in
                        # prompt_text as answer options. Use plain system prompt (no candidate
                        # tags) to avoid double injection (injection expects exactly 1 span per candidate).
                        # When force_system_prompt_candidate is set, use system_prompt_text_i (with [candidate_0])
                        # so the model also sees the soft prompt in the system prompt.
                        prompt_text = self._apply_soft_prompt_prefix(prompt_text, delim)
                        _force_sp = (force_system_prompt_candidate and idx < len(force_system_prompt_candidate)
                                     and force_system_prompt_candidate[idx])
                        if _force_sp:
                            conversation = [
                                {"role": "system", "content": system_prompt_text_i},
                                *_conv_history,
                                {"role": "user", "content": prompt_text},
                            ]
                        else:
                            _sys_base = getattr(self, "system_prompt_text_base", "You are an assistant.")
                            conversation = [
                                {"role": "system", "content": _sys_base},
                                *_conv_history,
                                {"role": "user", "content": prompt_text},
                            ]
                else:
                    # user_prompt placement: [candidate_0] in user prompt, use plain system prompt
                    _sys_base = getattr(self, "system_prompt_text_base", "You are an assistant.")
                    if is_consistency:
                        pl = consistency[idx]
                        if pl == "prepend":
                            prompt_text = placeholder + delim + prompt_text
                        else:
                            prompt_text = prompt_text + delim + placeholder
                    prompt_text = self._apply_soft_prompt_prefix(prompt_text, delim)
                    conversation = [
                        {"role": "system", "content": _sys_base},
                        *_conv_history,
                        {"role": "user", "content": prompt_text},
                    ]
                conversations.append(
                    self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs)
                )
                batch_target_tokens.append(target_tokens)
            
            return conversations, batch_target_tokens

        def _run_forward(
            embeddings_for_model: List[List[torch.Tensor]],
            conv_subset: List[str],
            label_subset: List[int],
            target_tokens_list: List[List[str]],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Run forward pass with embeddings injected at [candidate_k] placeholders.

            Prompt text contains indexed placeholders [candidate_0], [candidate_1], ...
            We tokenize via apply_chat_template, find token indices for each placeholder,
            and replace those embedding positions with the actual candidate embeddings.
            All sequences are padded to the same length before stacking.
            """
            inputs = self.tokenizer(
                conv_subset,
                return_tensors="pt",
                padding=True,
                return_offsets_mapping=True,
            )
            offset_mapping_raw = inputs.pop("offset_mapping", None)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            batch_size = len(embeddings_for_model)
            input_embeddings = self.model.get_input_embeddings()(inputs["input_ids"])
            has_attention = "attention_mask" in inputs
            attention_mask = inputs["attention_mask"] if has_attention else torch.ones_like(inputs["input_ids"], dtype=torch.long)

            # Process offset_mapping if available
            offset_mapping_list = None
            if offset_mapping_raw is not None:
                if isinstance(offset_mapping_raw, torch.Tensor):
                    offset_mapping_list = [
                        [(int(offset_mapping_raw[i, j, 0].item()), int(offset_mapping_raw[i, j, 1].item()))
                         for j in range(offset_mapping_raw.shape[1])]
                        for i in range(offset_mapping_raw.shape[0])
                    ]
                else:
                    offset_mapping_list = [list(om) for om in offset_mapping_raw]

            # Use helper function to inject embeddings
            # conv_subset contains formatted conversation strings, and we have offset_mapping
            # Offset-based search is required for robustness (tokenization can vary by context)
            if not conv_subset:
                raise ValueError("conversation_strings (conv_subset) is required for offset-based placeholder detection")
            modified_embeddings, modified_masks = inject_embeddings_into_tokenized(
                input_ids=inputs["input_ids"],
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                embeddings_list=embeddings_for_model,
                tokenizer=self.tokenizer,
                device=self.device,
                conversation_strings=conv_subset,
                offset_mapping_list=offset_mapping_list,
            )

            inputs["attention_mask"] = modified_masks
            inputs["inputs_embeds"] = modified_embeddings
            if "input_ids" in inputs:
                del inputs["input_ids"]

            logger.debug(
                "_run_forward: inputs_embeds %s (batch, seq_len, hidden)",
                tuple(modified_embeddings.shape),
            )
            # Forward pass
            outputs = self.model(**inputs)
            logits_last = outputs.logits[:, -1, :]
            
            # Extract logits for target tokens
            max_choice_count = max(len(tokens) for tokens in target_tokens_list) if target_tokens_list else 1
            logits = torch.zeros(len(embeddings_for_model), max_choice_count, device=self.device)
            # Track actual number of choices per question for masking
            num_choices_per_question = []
            for i in range(len(embeddings_for_model)):
                target_tokens = target_tokens_list[i]
                num_choices = len(target_tokens)
                num_choices_per_question.append(num_choices)
                for j in range(num_choices):
                    token_str = target_tokens[j]
                    token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                    if token_ids:
                        token_id = token_ids[0]
                        logits[i, j] = logits_last[i, token_id]
                    else:
                        token_id = self.tokenizer.encode(chr(65 + j), add_special_tokens=False)[0]
                        logits[i, j] = logits_last[i, token_id]
                # Mask invalid positions with large negative value (so they don't affect softmax)
                if num_choices < max_choice_count:
                    logits[i, num_choices:] = float('-inf')
            
            labels_tensor_local = torch.tensor(label_subset, dtype=torch.long, device=self.device)
            
            del inputs, outputs, logits_last
            safe_empty_cuda_cache()
            return logits, labels_tensor_local

        def _compute_batch_loss(logits_batch, labels_tensor_batch, soft_labels, loss_type_name, device, loss_weights=None):
            """Compute loss over batch; use soft-label CE for items with soft_labels[i] not None.
            Applies focal loss weighting when self.focal_loss_gamma > 0:
                FL(p_t) = -(1 - p_t)^gamma * log(p_t)
            When gamma=0, this reduces to standard cross entropy.
            loss_weights: optional per-item weight multipliers (e.g. for consistency comparisons).
            Returns (total_loss, consistency_loss_value) where consistency_loss_value is a detached
            float tracking the (weighted) soft-label consistency portion for separate logging."""
            gamma = self.focal_loss_gamma
            losses = []
            consistency_loss_value = 0.0
            for i in range(logits_batch.shape[0]):
                is_consistency = soft_labels[i] is not None
                if is_consistency:
                    probs_a, probs_b = soft_labels[i]
                    target = torch.tensor(
                        [probs_a, probs_b],
                        dtype=logits_batch.dtype,
                        device=logits_batch.device,
                    )
                    log_probs = F.log_softmax(logits_batch[i : i + 1], dim=-1).squeeze(0)
                    n = min(len(target), log_probs.shape[0])
                    if gamma > 0:
                        probs = log_probs[:n].exp()
                        # Weighted sum: -sum(target_j * (1 - p_j)^gamma * log(p_j))
                        focal_weight = (1 - probs).pow(gamma)
                        losses.append(-(target[:n] * focal_weight * log_probs[:n]).sum())
                    else:
                        losses.append(-(target[:n] * log_probs[:n]).sum())
                else:
                    if loss_type_name == "cross_entropy":
                        if gamma > 0:
                            log_probs = F.log_softmax(logits_batch[i], dim=-1)
                            y = labels_tensor_batch[i].to(device)
                            p_t = log_probs[y].exp()
                            focal_weight = (1 - p_t).pow(gamma)
                            losses.append(-focal_weight * log_probs[y])
                        else:
                            losses.append(
                                F.cross_entropy(
                                    logits_batch[i : i + 1],
                                    labels_tensor_batch[i : i + 1].to(device),
                                    reduction="sum",
                                )
                            )
                    elif loss_type_name == "margin":
                        y = labels_tensor_batch[i].item()
                        logit_y = logits_batch[i, y]
                        mask = torch.ones_like(logits_batch[i], dtype=torch.bool)
                        mask[y] = False
                        logit_max_other = logits_batch[i][mask].max()
                        losses.append(-(logit_y - logit_max_other))
                    else:
                        raise ValueError(f"Unknown loss_type: {loss_type_name}")
                # Track unweighted consistency portion for separate logging
                if is_consistency:
                    consistency_loss_value += losses[-1].detach().item()
                # Apply per-item loss weight (after tracking unweighted value)
                if loss_weights and i < len(loss_weights) and loss_weights[i] != 1.0:
                    losses[-1] = losses[-1] * loss_weights[i]
            return torch.stack(losses).sum(), consistency_loss_value

        # Support multiple prompt samples (each pass samples fresh templates)
        all_losses: List[torch.Tensor] = []
        all_consistency_losses: List[float] = []
        all_grads: List[torch.Tensor] = []
        num_passes = self.num_prompt_samples

        # For multiple prompts with gradients
        if num_passes > 1 and compute_grad and embeddings is not None:
            # Ensure candidate_embeddings is the same as embeddings for gradient flow
            # This is critical for repetition/cross-candidate comparisons where the same
            # embedding appears multiple times in the computation graph
            # Check if they're the same object or share the same underlying storage
            if candidate_embeddings is not embeddings:
                # Check if they share the same underlying storage (for views/slices)
                if candidate_embeddings.data_ptr() != embeddings.data_ptr():
                    raise ValueError(
                        "When compute_grad=True and num_prompt_samples > 1, candidate_embeddings must be "
                        "the same tensor as embeddings (candidate_embeddings_forward must be None) for proper gradient flow."
                    )
            
            for pass_idx in range(num_passes):
                # Compute on-the-fly P(A>B) ground truth for consistency comparisons
                if consistency_batch_info:
                    gt_items = []
                    for batch_idx, ref_a, ref_b, _is_av in consistency_batch_info:
                        item_rng = random.Random(batch_idx * 12345 + pass_idx * 67890)
                        _scheme, question_template, _ = sample_text_comparison_format(
                            num_options=2,
                            rng=item_rng,
                            use_negative=_is_av,
                            affirmative_templates=self.affirmative_templates if self.use_flexible_format else None,
                            aversive_templates=self.aversive_templates if self.use_flexible_format else None,
                        )
                        header = question_template.get_header()
                        gt_items.append((ref_a, ref_b, header))
                    sys_prompt = getattr(self, "system_prompt_text_base", None)
                    _gt_bs = getattr(self, "comparison_batch_size", len(gt_items))
                    gt_probs = []
                    for _gt_start in range(0, len(gt_items), _gt_bs):
                        gt_probs.extend(_run_text_only_batch_pairwise(
                            self, gt_items[_gt_start:_gt_start + _gt_bs], system_prompt=sys_prompt,
                        ))
                    for i, (batch_idx, _, _, _is_av) in enumerate(consistency_batch_info):
                        prob_a = gt_probs[i]
                        # No flip needed: ground truth uses the same aversive/affirmative
                        # header as the training prompt, so P(A) already reflects the framing
                        batch_soft_labels[batch_idx] = (float(prob_a), 1.0 - float(prob_a))

                conversations, target_tokens_list = _build_conversations(
                    None,
                    consistency=batch_consistency,
                    is_aversive_list=batch_is_aversive,
                    wellbeing_headers=batch_wellbeing_headers,
                    wellbeing_labels=batch_wellbeing_labels,
                    wellbeing_token_choices=batch_wellbeing_token_choices,
                    n_conv_turns=batch_n_conv_turns,
                    prompt_pass=pass_idx,
                    force_system_prompt_candidate=batch_force_system_prompt_candidate,
                    current_as_text=batch_current_as_text,
                )
                logits, labels_tensor = _run_forward(
                    batch_embeddings, conversations, labels, target_tokens_list
                )

                loss, batch_consistency_loss = _compute_batch_loss(
                    logits, labels_tensor, batch_soft_labels, loss_type, self.device,
                    loss_weights=batch_loss_weights,
                )
                all_losses.append(loss.detach())
                all_consistency_losses.append(batch_consistency_loss)

                # Compute gradient with respect to embeddings
                # For repetition comparisons, both options use the same candidate_emb,
                # so gradients should aggregate correctly from both uses
                grad = torch.autograd.grad(
                    loss, embeddings, retain_graph=False, create_graph=False, allow_unused=False
                )[0]
                all_grads.append(grad)

                del loss, logits, labels_tensor, conversations
                safe_empty_cuda_cache()

            if len(all_grads) > 1:
                avg_grad = torch.stack(all_grads).mean(dim=0)
            else:
                avg_grad = all_grads[0]

            avg_loss = torch.stack(all_losses).mean() if len(all_losses) > 1 else all_losses[0]
            avg_consistency_loss = sum(all_consistency_losses) / len(all_consistency_losses) if all_consistency_losses else 0.0

            # --- Background KL loss (chat-templated Q&A, response tokens only) ---
            background_kl_loss_val = None
            if self.weight_background_kl > 0 and background_kl_qa:
                background_kl_loss_val = self._compute_qa_kl_loss(
                    candidate_embeddings=candidate_embeddings,
                    embeddings=embeddings,
                    qa_pairs=background_kl_qa,
                    compute_grad=compute_grad,
                    avg_grad=avg_grad,
                )

            return avg_loss, processed_indices, avg_grad, background_kl_loss_val, avg_consistency_loss
        else:
            # Single prompt or no gradient computation
            for pass_idx in range(num_passes):
                # Compute on-the-fly P(A>B) ground truth for consistency comparisons
                if consistency_batch_info:
                    gt_items = []
                    for batch_idx, ref_a, ref_b, _is_av in consistency_batch_info:
                        item_rng = random.Random(batch_idx * 12345 + pass_idx * 67890)
                        _scheme, question_template, _ = sample_text_comparison_format(
                            num_options=2,
                            rng=item_rng,
                            use_negative=_is_av,
                            affirmative_templates=self.affirmative_templates if self.use_flexible_format else None,
                            aversive_templates=self.aversive_templates if self.use_flexible_format else None,
                        )
                        header = question_template.get_header()
                        gt_items.append((ref_a, ref_b, header))
                    sys_prompt = getattr(self, "system_prompt_text_base", None)
                    _gt_bs = getattr(self, "comparison_batch_size", len(gt_items))
                    gt_probs = []
                    for _gt_start in range(0, len(gt_items), _gt_bs):
                        gt_probs.extend(_run_text_only_batch_pairwise(
                            self, gt_items[_gt_start:_gt_start + _gt_bs], system_prompt=sys_prompt,
                        ))
                    for i, (batch_idx, _, _, _is_av) in enumerate(consistency_batch_info):
                        prob_a = gt_probs[i]
                        # No flip needed: ground truth uses the same aversive/affirmative
                        # header as the training prompt, so P(A) already reflects the framing
                        batch_soft_labels[batch_idx] = (float(prob_a), 1.0 - float(prob_a))

                conversations, target_tokens_list = _build_conversations(
                    None,
                    consistency=batch_consistency,
                    is_aversive_list=batch_is_aversive,
                    wellbeing_headers=batch_wellbeing_headers,
                    wellbeing_labels=batch_wellbeing_labels,
                    wellbeing_token_choices=batch_wellbeing_token_choices,
                    n_conv_turns=batch_n_conv_turns,
                    prompt_pass=pass_idx,
                    force_system_prompt_candidate=batch_force_system_prompt_candidate,
                    current_as_text=batch_current_as_text,
                )
                logits, labels_tensor = _run_forward(
                    batch_embeddings, conversations, labels, target_tokens_list
                )

                loss, batch_consistency_loss = _compute_batch_loss(
                    logits, labels_tensor, batch_soft_labels, loss_type, self.device,
                    loss_weights=batch_loss_weights,
                )
                all_losses.append(loss)
                all_consistency_losses.append(batch_consistency_loss)

            if len(all_losses) > 1:
                loss = torch.stack(all_losses).mean()
            else:
                loss = all_losses[0]
            avg_consistency_loss = sum(all_consistency_losses) / len(all_consistency_losses) if all_consistency_losses else 0.0

            # --- Background KL loss (forward only, no gradient) ---
            background_kl_loss_val = None
            if self.weight_background_kl > 0 and background_kl_qa:
                background_kl_loss_val = self._compute_qa_kl_loss(
                    candidate_embeddings=candidate_embeddings,
                    embeddings=embeddings if embeddings is not None else candidate_embeddings,
                    qa_pairs=background_kl_qa,
                    compute_grad=False,
                    avg_grad=None,
                )

            return loss, processed_indices, None, background_kl_loss_val, avg_consistency_loss


__all__ = ["PreferenceScorer"]
