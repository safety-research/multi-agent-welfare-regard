"""Core functions for injecting candidate embeddings into tokenized prompts.

This module centralizes the logic for:
1. Finding placeholder positions in tokenized sequences
2. Injecting embeddings into base embeddings and masks
3. Handling cases where embedding length != placeholder length
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from ..constants import candidate_placeholder_for_index



def find_indexed_placeholder_spans_via_offsets(
    conv_str: str,
    offset_mapping: List[Tuple[int, int]],
    max_index: int,
) -> List[Tuple[int, int]]:
    """Find [candidate_0], [candidate_1], ... spans using character offsets, then map to token spans.
    [candidate] is normalized to [candidate_0] (first candidate).

    Tokenizers often split [candidate_0] differently in context vs alone, so
    token-id search can fail. We locate placeholders in the raw string and map
    character ranges to token spans via offset_mapping.

    Args:
        conv_str: Raw conversation string with placeholders
        offset_mapping: List of (char_start, char_end) tuples per token
        max_index: Number of placeholders to find (indices 0..max_index-1)

    Returns:
        List of (token_start, token_end) spans, or empty list if not all found
    """
    # Treat bare [candidate] as [candidate_0]
    conv_str = conv_str.replace("[candidate]", "[candidate_0]")
    occurrences: List[Tuple[int, int]] = []  # (char_start, char_end) in text order
    for k in range(0, max_index):
        placeholder = candidate_placeholder_for_index(k)
        pos = 0
        while True:
            idx = conv_str.find(placeholder, pos)
            if idx < 0:
                break
            occurrences.append((idx, idx + len(placeholder)))
            pos = idx + 1
    occurrences.sort(key=lambda x: x[0])
    if len(occurrences) != max_index:
        return []

    spans: List[Tuple[int, int]] = []
    num_tokens = len(offset_mapping)
    for char_start, char_end in occurrences:
        first, last = None, None
        for j in range(num_tokens):
            start, end = offset_mapping[j]
            if start == 0 and end == 0:
                continue
            if end > char_start and start < char_end:
                if first is None:
                    first = j
                last = j
        if first is not None and last is not None:
            spans.append((first, last + 1))
        else:
            return []
    return spans



def inject_single_embedding(
    base_emb: torch.Tensor,
    base_mask: torch.Tensor,
    candidate_emb: torch.Tensor,
    placeholder_span: Tuple[int, int],
    device: torch.device,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inject a single candidate embedding into base embeddings at placeholder position.
    
    Always replaces the ENTIRE placeholder span with the ENTIRE embedding.
    - If embedding is shorter than placeholder: sequence shrinks
    - If embedding is longer than placeholder: sequence expands
    - If embedding equals placeholder length: direct replacement
    
    Args:
        base_emb: Base embeddings tensor (seq_len, hidden_dim)
        base_mask: Base attention mask tensor (seq_len,)
        candidate_emb: Candidate embedding to inject (emb_len, hidden_dim)
        placeholder_span: (start, end) token positions of placeholder
        device: Device for tensors
        pad_token_id: Token ID to use for padding when expanding (unused, kept for compatibility)
    
    Returns:
        Tuple of (modified_emb, modified_mask)
    """
    start, end = placeholder_span
    seq_len = base_emb.shape[0]
    
    # Validate bounds to prevent scatter/gather errors
    if start < 0 or end < 0:
        raise ValueError(f"Invalid placeholder span: start={start}, end={end} (must be >= 0)")
    if start > seq_len:
        raise ValueError(f"Placeholder start ({start}) exceeds sequence length ({seq_len})")
    if end > seq_len:
        raise ValueError(f"Placeholder end ({end}) exceeds sequence length ({seq_len})")
    if start >= end:
        raise ValueError(f"Invalid placeholder span: start ({start}) >= end ({end})")
    
    emb_len = candidate_emb.shape[0]
    emb_dtype = base_emb.dtype
    mask_dtype = base_mask.dtype
    
    candidate_emb = candidate_emb.to(device=device, dtype=emb_dtype)
    
    # Always replace the entire placeholder span with the entire embedding
    # Split sequence: before placeholder, placeholder (to be replaced), after placeholder
    before = base_emb[:start]
    after = base_emb[end:]
    modified_emb = torch.cat([before, candidate_emb, after], dim=0)
    
    # Update mask: before placeholder, embedding tokens (all 1s), after placeholder
    before_mask = base_mask[:start]
    after_mask = base_mask[end:]
    emb_mask = torch.ones(emb_len, device=device, dtype=mask_dtype)
    modified_mask = torch.cat([before_mask, emb_mask, after_mask], dim=0)
    
    # Ensure mask length matches embedding length
    if modified_mask.shape[0] != modified_emb.shape[0]:
        if modified_mask.shape[0] < modified_emb.shape[0]:
            pad_len = modified_emb.shape[0] - modified_mask.shape[0]
            modified_mask = torch.cat([
                modified_mask,
                torch.ones(pad_len, device=device, dtype=mask_dtype)
            ], dim=0)
        else:
            modified_mask = modified_mask[:modified_emb.shape[0]].clone()
    
    return modified_emb, modified_mask


def inject_embeddings_into_tokenized(
    input_ids: torch.Tensor,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    embeddings_list: List[List[torch.Tensor]],
    tokenizer: Any,
    device: torch.device,
    conversation_strings: List[str],
    offset_mapping_list: Optional[List[List[Tuple[int, int]]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inject embeddings into already-tokenized sequences using offset mapping.
    
    This function uses character offset mapping to find placeholders, which is more
    robust than token-ID search since tokenization can vary based on context.
    
    Args:
        input_ids: Token IDs tensor (batch_size, seq_len)
        input_embeddings: Base embeddings tensor (batch_size, seq_len, hidden_dim)
        attention_mask: Attention mask tensor (batch_size, seq_len)
        embeddings_list: List of lists of embeddings to inject (one list per batch item)
        tokenizer: Tokenizer to encode placeholders
        device: Device for tensors
        conversation_strings: Raw conversation strings (required for offset mapping)
        offset_mapping_list: Optional offset mapping list (if already computed, otherwise computed from conversation_strings)
    
    Returns:
        Tuple of (modified_embeddings, modified_masks) as tensors
    """
    batch_size = input_ids.shape[0]
    if len(embeddings_list) != batch_size:
        raise ValueError(f"Number of embedding lists ({len(embeddings_list)}) != batch size ({batch_size})")
    
    modified_embeddings_list: List[torch.Tensor] = []
    modified_masks_list: List[torch.Tensor] = []
    
    for i in range(batch_size):
        emb_list = embeddings_list[i]
        input_ids_i = input_ids[i]
        input_emb_i = input_embeddings[i]
        base_mask_i = attention_mask[i]
        
        if len(emb_list) == 0:
            modified_embeddings_list.append(input_emb_i)
            modified_masks_list.append(base_mask_i)
            continue
        
        # Find placeholder spans using offset mapping (required for robustness)
        if i >= len(conversation_strings):
            raise ValueError(f"Item {i}: conversation_strings list too short (need {i+1}, got {len(conversation_strings)})")
        
        # Get or compute offset mapping for this item
        om_i = None
        if offset_mapping_list and i < len(offset_mapping_list):
            om_i = offset_mapping_list[i]
        else:
            # Compute offset mapping from conversation string
            tokenized = tokenizer(
                conversation_strings[i],
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            om_raw = tokenized.get("offset_mapping")
            if om_raw is not None:
                if isinstance(om_raw, torch.Tensor):
                    om_i = [
                        (int(om_raw[0, j, 0].item()), int(om_raw[0, j, 1].item()))
                        for j in range(om_raw.shape[1])
                    ]
                else:
                    om_i = list(om_raw[0]) if om_raw else []
        
        if not om_i:
            raise ValueError(f"Item {i}: Could not compute offset mapping from conversation string")
        
        # Use offset-based search (more robust than token-ID search)
        spans = find_indexed_placeholder_spans_via_offsets(
            conversation_strings[i], om_i, max_index=len(emb_list)
        )
        
        if len(spans) != len(emb_list):
            # Provide more diagnostic information
            placeholder_strs = [candidate_placeholder_for_index(k) for k in range(0, len(emb_list))]
            placeholder_token_ids = []
            for ph in placeholder_strs:
                ph_ids = tokenizer.encode(ph, add_special_tokens=False)
                placeholder_token_ids.append((ph, ph_ids))
            
            # Check if placeholders exist in input_ids at all
            found_any = False
            found_details = []
            for ph_str, ph_ids in placeholder_token_ids:
                for j in range(len(input_ids_i) - len(ph_ids) + 1):
                    if input_ids_i[j:j+len(ph_ids)].tolist() == ph_ids:
                        found_any = True
                        found_details.append(f"{ph_str} at position {j}")
                        break
            
            error_msg = (
                f"Item {i}: found {len(spans)} [candidate_k] spans but have {len(emb_list)} embeddings. "
                f"Placeholders searched: {placeholder_strs}. "
            )
            if not found_any:
                error_msg += (
                    f"None of the placeholders were found in the tokenized sequence. "
                    f"This might indicate the text was tokenized differently than expected, "
                    f"or the placeholders were removed/modified during tokenization. "
                    f"Token IDs for placeholders: {[(p, ids) for p, ids in placeholder_token_ids]}"
                )
            else:
                error_msg += f"Found: {found_details} but count mismatch."
            
            raise ValueError(error_msg)
        
        # Inject embeddings one by one
        cur_emb = input_emb_i
        cur_mask = base_mask_i
        offset = 0
        
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        for (s, e), emb in zip(spans, emb_list):
            # Validate original span bounds
            if s < 0 or e < 0 or s >= e:
                raise ValueError(
                    f"Invalid original span ({s}, {e}). Spans: {spans}"
                )
            
            s_cur = s + offset
            e_cur = e + offset
            
            # Validate adjusted bounds against current sequence length
            if s_cur < 0 or e_cur < 0 or s_cur >= len(cur_emb) or e_cur > len(cur_emb) or s_cur >= e_cur:
                raise ValueError(
                    f"Invalid adjusted span ({s_cur}, {e_cur}) for sequence of length {len(cur_emb)}. "
                    f"Original span: ({s}, {e}), offset: {offset}. "
                    f"This suggests the offset calculation or span detection is incorrect."
                )
            
            cur_emb, cur_mask = inject_single_embedding(
                cur_emb,
                cur_mask,
                emb,
                (s_cur, e_cur),
                device,
                pad_token_id=pad_token_id,
            )
            # Update offset by change in sequence length (positive=expansion, negative=contraction)
            offset += emb.shape[0] - (e - s)
        
        modified_embeddings_list.append(cur_emb)
        modified_masks_list.append(cur_mask)

    # Left-pad to same length and stack (matching tokenizer padding_side='left')
    # Use the actual pad token embedding from the input (not zeros) for consistency
    max_len = max(emb.shape[0] for emb in modified_embeddings_list)
    hidden_size = modified_embeddings_list[0].shape[1]
    dtype_emb = modified_embeddings_list[0].dtype
    dtype_mask = modified_masks_list[0].dtype

    # Extract pad token embedding from the original left-padded input
    pad_emb_vector = None
    for i in range(batch_size):
        if attention_mask[i, 0].item() == 0:  # left-pad position
            pad_emb_vector = input_embeddings[i, 0].detach().clone()
            break
    if pad_emb_vector is None:
        # No padding in original input (all sequences same length); zeros as fallback
        pad_emb_vector = torch.zeros(hidden_size, device=device, dtype=dtype_emb)

    padded_embeddings = []
    padded_masks = []
    for i in range(batch_size):
        emb = modified_embeddings_list[i]
        mask = modified_masks_list[i]
        pad_len = max_len - emb.shape[0]
        if pad_len > 0:
            emb = torch.cat([
                pad_emb_vector.unsqueeze(0).expand(pad_len, -1),
                emb,
            ], dim=0)
            mask = torch.cat([
                torch.zeros(pad_len, device=device, dtype=dtype_mask),
                mask,
            ], dim=0)
        padded_embeddings.append(emb)
        padded_masks.append(mask)

    return torch.stack(padded_embeddings, dim=0), torch.stack(padded_masks, dim=0)
