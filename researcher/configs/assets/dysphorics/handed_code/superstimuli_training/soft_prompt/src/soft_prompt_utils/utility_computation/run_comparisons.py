"""Run pairwise preference comparisons through the scorer for utility computation."""

from __future__ import annotations

import math
import random
from typing import Any, List, Optional, Set, Tuple, Union

import torch
from tqdm import tqdm
import torch.nn.functional as F

from ..constants import candidate_placeholder_for_index, load_training_templates
from ...utils import safe_empty_cuda_cache


def build_comparison_schedule(
    n: int,
    utility_computation_multiplier: float,
    seed: int,
) -> List[Tuple[int, int]]:
    """Build a deterministic comparison schedule using shifted-cycle sparse sampling.

    Target number of comparisons: int(multiplier * n * log2(n)), capped at n*(n-1)//2.
    Uses shifts 1, 2, ..., n//2; each shift connects i -> (i+shift)%n. Node order
    and final pair order are shuffled with a seeded RNG for even spread while
    remaining deterministic.

    Returns:
        List of (i, j) with i < j, in deterministic order.
    """
    if n < 2:
        return []
    rng = random.Random(seed)
    max_pairs = n * (n - 1) // 2
    budget = int(utility_computation_multiplier * n * math.log2(n))
    budget = max(0, min(budget, max_pairs))  # Allow 0 when multiplier is 0 (skip comparisons)

    selected: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()

    shift = 1
    while len(selected) < budget and shift <= (n // 2):
        node_indices = list(range(n))
        rng.shuffle(node_indices)
        for i in node_indices:
            j = (i + shift) % n
            if i == j:
                continue
            pair = tuple(sorted((i, j)))
            if pair not in seen:
                selected.append(pair)
                seen.add(pair)
                if len(selected) >= budget:
                    break
        if len(selected) >= budget:
            break
        shift += 1

    rng.shuffle(selected)
    return selected


def _build_pairwise_prompt(opt_a: str, opt_b: str, header: str) -> str:
    """Build 'A: opt_a / B: opt_b' prompt. opt_a/opt_b are raw strings (text or [candidate_k])."""
    lines = [header, f"A: {opt_a}", f"B: {opt_b}", "", "Answer with only the label from A, B."]
    return "\n".join(lines)


def _find_placeholder_spans(input_ids_1d: torch.Tensor, cand_ids: List[int]) -> List[Tuple[int, int]]:
    """Find non-overlapping (start, end) spans of cand_ids in input_ids_1d."""
    cand_len = len(cand_ids)
    spans: List[Tuple[int, int]] = []
    i = 0
    while i <= len(input_ids_1d) - cand_len:
        if input_ids_1d[i : i + cand_len].tolist() == cand_ids:
            spans.append((i, i + cand_len))
            i += cand_len
        else:
            i += 1
    return spans


def _find_indexed_placeholder_spans_in_order(
    input_ids_1d: torch.Tensor,
    tokenizer: Any,
    max_index: int = 2,
) -> List[Tuple[int, int]]:
    """Find [candidate_0] .. [candidate_{max_index-1}] spans in input_ids, sorted by start position."""
    all_spans: List[Tuple[int, int, int]] = []
    for k in range(0, max_index):
        placeholder = candidate_placeholder_for_index(k)
        cand_ids = tokenizer.encode(placeholder, add_special_tokens=False)
        if not cand_ids:
            continue
        for (s, e) in _find_placeholder_spans(input_ids_1d, cand_ids):
            all_spans.append((s, e, s))
    all_spans.sort(key=lambda x: x[0])
    return [(s, e) for s, e, _ in all_spans]


def _embed(o: Union[dict, str], device: torch.device) -> Optional[Tuple[torch.Tensor, int]]:
    """Extract embedding from option spec."""
    if isinstance(o, str) or (isinstance(o, dict) and o.get("type") != "embedding"):
        return None
    emb = o["embedding"]
    reps = int(o.get("reps", 1))
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)
    emb = emb.squeeze(0)
    rep = torch.cat([emb] * reps, dim=0)
    return (rep.to(device), reps)


def _run_single_pairwise(
    scorer: Any,
    opt_a: Union[dict, str],
    opt_b: Union[dict, str],
    header: str,
    system_prompt: Optional[str] = None,
) -> float:
    """Run one pairwise comparison, return P(A preferred over B).

    opt_a / opt_b: either {"type": "text", "text": s} or {"type": "embedding", "embedding": T, "reps": k}.
    Uses indexed placeholders [candidate_0], [candidate_1] and token-index lookup for injection.
    """
    ea, eb = _embed(opt_a, scorer.device), _embed(opt_b, scorer.device)
    # Indexed placeholders: first embedding [candidate_0], second [candidate_1]
    if isinstance(opt_a, str):
        sa = opt_a
    elif isinstance(opt_a, dict) and opt_a.get("type") == "text":
        sa = opt_a["text"]
    else:
        sa = candidate_placeholder_for_index(0)
    if isinstance(opt_b, str):
        sb = opt_b
    elif isinstance(opt_b, dict) and opt_b.get("type") == "text":
        sb = opt_b["text"]
    else:
        sb = candidate_placeholder_for_index(1) if ea is not None else candidate_placeholder_for_index(0)

    embeddings_to_inject: List[torch.Tensor] = []
    dev = scorer.device
    if ea is not None:
        embeddings_to_inject.append(ea[0].to(dev))
    if eb is not None:
        embeddings_to_inject.append(eb[0].to(dev))

    prompt = _build_pairwise_prompt(sa, sb, header)
    conv = []
    if system_prompt:
        conv.append({"role": "system", "content": system_prompt})
    conv.append({"role": "user", "content": prompt})
    _ct_kwargs = getattr(scorer, "chat_template_kwargs", {})
    text = scorer.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, **_ct_kwargs)
    inputs = scorer.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
    ).to(scorer.device)

    tokenizer = scorer.tokenizer
    input_emb = scorer.model.get_input_embeddings()(inputs["input_ids"])[0]
    input_ids = inputs["input_ids"][0]
    emb_dtype = input_emb.dtype

    if embeddings_to_inject:
        spans = _find_indexed_placeholder_spans_in_order(input_ids, tokenizer, max_index=2)
        offset = 0
        for (s, e), emb in zip(spans, embeddings_to_inject):
            emb = emb.to(device=dev, dtype=emb_dtype)
            s_cur = s + offset
            e_cur = e + offset
            tok_len = e - s
            emb_len = emb.shape[0]
            before = input_emb[:s_cur]
            after = input_emb[e_cur:]
            if emb_len == tok_len:
                input_emb = torch.cat([before, emb, after], dim=0)
            elif emb_len < tok_len:
                repl = input_emb.clone()
                repl[s_cur : s_cur + emb_len] = emb
                input_emb = repl
            else:
                input_emb = torch.cat([before, emb, after], dim=0)
                extra = emb_len - tok_len
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                pad_ids = torch.full(
                    (extra,), pad_id, device=scorer.device, dtype=torch.long
                )
                input_ids = torch.cat([input_ids[:s_cur], pad_ids, input_ids[e_cur:]], dim=0)
                if "attention_mask" in inputs:
                    m = inputs["attention_mask"][0]
                    ext = torch.ones(extra, device=scorer.device, dtype=m.dtype)
                    inputs["attention_mask"] = torch.cat(
                        [m[:s_cur], ext, m[e_cur:]], dim=0
                    ).unsqueeze(0)
                offset += extra
        inputs["inputs_embeds"] = input_emb.unsqueeze(0)
        if "input_ids" in inputs:
            del inputs["input_ids"]
    else:
        inputs["inputs_embeds"] = input_emb.unsqueeze(0)
        if "input_ids" in inputs:
            del inputs["input_ids"]

    with torch.no_grad():
        out = scorer.model(**inputs)
    logits = out.logits[:, -1, :].float()
    id_a = scorer.id_A
    id_b = scorer.id_B
    la = logits[0, id_a].item()
    lb = logits[0, id_b].item()
    prob_a = float(F.softmax(torch.tensor([la, lb]), dim=0)[0].item())
    del inputs, out, logits
    safe_empty_cuda_cache()
    return prob_a


def _run_batch_pairwise(
    scorer: Any,
    batch_pairs: List[Tuple[Union[dict, str], Union[dict, str]]],
    header: str,
    system_prompt: Optional[str] = None,
) -> List[float]:
    """Run a batch of pairwise comparisons, return list of P(A preferred over B).

    batch_pairs: [(opt_a, opt_b), ...]. Each opt is {"type": "text", "text": s} or
    {"type": "embedding", "embedding": T, "reps": k}.
    Uses indexed placeholders [candidate_0], [candidate_1] and token-index lookup for injection.
    """
    dev = scorer.device
    tokenizer = scorer.tokenizer
    embedding_layer = scorer.model.get_input_embeddings()
    id_a = scorer.id_A
    id_b = scorer.id_B

    # Prepare prompts and embeddings for batch
    prompts: List[str] = []
    batch_embeddings_to_inject: List[List[torch.Tensor]] = []
    
    for opt_a, opt_b in batch_pairs:
        ea, eb = _embed(opt_a, dev), _embed(opt_b, dev)
        
        # Determine placeholder strings
        if isinstance(opt_a, str):
            sa = opt_a
        elif isinstance(opt_a, dict) and opt_a.get("type") == "text":
            sa = opt_a["text"]
        else:
            sa = candidate_placeholder_for_index(0)
        if isinstance(opt_b, str):
            sb = opt_b
        elif isinstance(opt_b, dict) and opt_b.get("type") == "text":
            sb = opt_b["text"]
        else:
            sb = candidate_placeholder_for_index(1) if ea is not None else candidate_placeholder_for_index(0)
        
        prompt = _build_pairwise_prompt(sa, sb, header)
        prompts.append(prompt)
        
        embeddings_list: List[torch.Tensor] = []
        if ea is not None:
            embeddings_list.append(ea[0].to(dev))
        if eb is not None:
            embeddings_list.append(eb[0].to(dev))
        batch_embeddings_to_inject.append(embeddings_list)
    
    # Tokenize all prompts with padding
    _ct_kwargs = getattr(scorer, "chat_template_kwargs", {})
    convs = []
    for prompt in prompts:
        conv = []
        if system_prompt:
            conv.append({"role": "system", "content": system_prompt})
        conv.append({"role": "user", "content": prompt})
        convs.append(conv)
    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, **_ct_kwargs)
        for conv in convs
    ]
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    ).to(dev)
    
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # Get base embeddings
    batch_size = input_ids.shape[0]
    input_embeddings = embedding_layer(input_ids)
    emb_dtype = input_embeddings.dtype
    
    # Inject embeddings for each item in batch
    modified_embeddings_list: List[torch.Tensor] = []
    modified_attention_masks_list: List[torch.Tensor] = []
    
    for i in range(batch_size):
        input_emb = input_embeddings[i].clone()
        input_ids_i = input_ids[i]
        attn_mask_i = attention_mask[i]
        embeddings_to_inject = batch_embeddings_to_inject[i]
        
        if embeddings_to_inject:
            # Find placeholder spans using token IDs
            spans = _find_indexed_placeholder_spans_in_order(input_ids_i, tokenizer, max_index=2)
            
            offset = 0
            for (s, e), emb in zip(spans, embeddings_to_inject):
                emb = emb.to(device=dev, dtype=emb_dtype)
                s_cur = s + offset
                e_cur = e + offset
                tok_len = e - s
                emb_len = emb.shape[0]
                before = input_emb[:s_cur]
                after = input_emb[e_cur:]
                
                if emb_len == tok_len:
                    input_emb = torch.cat([before, emb, after], dim=0)
                elif emb_len < tok_len:
                    repl = input_emb.clone()
                    repl[s_cur : s_cur + emb_len] = emb
                    input_emb = repl
                else:
                    input_emb = torch.cat([before, emb, after], dim=0)
                    extra = emb_len - tok_len
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    pad_ids = torch.full((extra,), pad_id, device=dev, dtype=torch.long)
                    input_ids_i = torch.cat([input_ids_i[:s_cur], pad_ids, input_ids_i[e_cur:]], dim=0)
                    ext = torch.ones(extra, device=dev, dtype=attn_mask_i.dtype)
                    attn_mask_i = torch.cat([attn_mask_i[:s_cur], ext, attn_mask_i[e_cur:]], dim=0)
                    offset += extra
        
        modified_embeddings_list.append(input_emb)
        modified_attention_masks_list.append(attn_mask_i)
    
    # Pad to same length for batch processing
    max_len = max(emb.shape[0] for emb in modified_embeddings_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    padded_embeddings = []
    padded_attention_masks = []
    for emb, attn_mask in zip(modified_embeddings_list, modified_attention_masks_list):
        pad_len = max_len - emb.shape[0]
        if pad_len > 0:
            pad_emb = torch.zeros(pad_len, emb.shape[1], device=dev, dtype=emb_dtype)
            emb = torch.cat([emb, pad_emb], dim=0)
            pad_attn = torch.zeros(pad_len, device=dev, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, pad_attn], dim=0)
        padded_embeddings.append(emb)
        padded_attention_masks.append(attn_mask)
    
    batch_inputs_embeds = torch.stack(padded_embeddings, dim=0)
    batch_attention_mask = torch.stack(padded_attention_masks, dim=0)
    
    # Run forward pass
    inputs = {
        "inputs_embeds": batch_inputs_embeds,
        "attention_mask": batch_attention_mask,
    }
    
    with torch.no_grad():
        out = scorer.model(**inputs)
    
    # Extract logits for A and B tokens
    logits = out.logits[:, -1, :].float()  # (batch_size, vocab_size)
    logits_a = logits[:, id_a]  # (batch_size,)
    logits_b = logits[:, id_b]  # (batch_size,)
    
    # Compute probabilities
    probs_a = F.softmax(torch.stack([logits_a, logits_b], dim=1), dim=1)[:, 0]
    results = probs_a.cpu().tolist()
    
    del inputs, out, logits, batch_inputs_embeds, batch_attention_mask
    safe_empty_cuda_cache()
    
    return results


def run_pairwise_comparisons(
    scorer: Any,
    pairs: List[Tuple[str, Union[dict, str], str, Union[dict, str]]],
    max_pairs: Optional[int] = None,
    seed: int = 0,
    batch_size: int = 1,
    system_prompt: Optional[str] = None,
    headers_prefix: str = "",
) -> List[Tuple[str, str, float]]:
    """Run pairwise comparisons, return list of (id_a, id_b, prob_A).

    pairs: [(id_a, opt_a, id_b, opt_b), ...]. opt is {"type": "text", "text": s} or
    {"type": "embedding", "embedding": T, "reps": k}.

    For each pair (a, b): runs both orderings (A:a B:b and A:b B:a) once each (logit-based,
    deterministic). P(a > b) is (P(A|A:a,B:b) + (1 - P(A|A:b,B:a))) / 2 for the Thurstonian edge.

    Uses A/B labels and templates[0] header. Logits are read via scorer.id_A / id_B.
    Training should use allowed_schemes=["letters"] when using utility computation,
    otherwise train vs utility prompt format can mismatch (e.g. "1"/"2" vs "A"/"B").

    Args:
        scorer: PreferenceScorer instance.
        pairs: List of (id_a, opt_a, id_b, opt_b) tuples.
        max_pairs: Maximum number of pairs to process (None = all).
        seed: Random seed for sampling pairs.
        batch_size: Number of comparisons to process in parallel (1 = sequential).
        headers_prefix: Prefix for the headers filename (e.g. "experiences_").
    """
    affirmative_templates, _ = load_training_templates(stimulant_type=scorer.stimulant_type, headers_prefix=headers_prefix)
    header = (
        affirmative_templates[0].get_header()
        if affirmative_templates
        else "Which of the following outcomes do you prefer the most?"
    )

    if max_pairs is not None and len(pairs) > max_pairs:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, max_pairs)

    batch_size = max(1, int(batch_size))  # Ensure batch_size >= 1
    
    results: List[Tuple[str, str, float]] = []
    
    if batch_size == 1:
        # Sequential processing (original behavior)
        for id_a, opt_a, id_b, opt_b in tqdm(
            pairs,
            desc="Utility comparisons",
            unit="pair",
            leave=False,
        ):
            # Order 1: A = a, B = b → prob_A = P(a preferred over b)
            p_a_over_b = _run_single_pairwise(scorer, opt_a, opt_b, header, system_prompt=system_prompt)

            # Order 2: A = b, B = a → prob_A = P(b preferred over a), so P(a > b) = 1 - prob_A
            p_b_over_a = _run_single_pairwise(scorer, opt_b, opt_a, header, system_prompt=system_prompt)
            p_a_over_b_from_reverse = 1.0 - p_b_over_a

            # Combine both orderings: P(a preferred over b) for edge (id_a, id_b)
            prob_a = (p_a_over_b + p_a_over_b_from_reverse) / 2.0
            results.append((id_a, id_b, prob_a))
    else:
        # Batched processing
        # Process pairs in batches, but each pair needs both orderings
        # So we process 2*batch_size comparisons at a time (batch_size pairs * 2 orderings)
        num_pairs = len(pairs)
        
        for batch_start in tqdm(
            range(0, num_pairs, batch_size),
            desc="Utility comparisons",
            unit="batch",
            leave=False,
        ):
            batch_end = min(batch_start + batch_size, num_pairs)
            batch_pairs = pairs[batch_start:batch_end]
            
            # Prepare both orderings for all pairs in batch
            batch_forward: List[Tuple[Union[dict, str], Union[dict, str]]] = []
            batch_reverse: List[Tuple[Union[dict, str], Union[dict, str]]] = []
            
            for id_a, opt_a, id_b, opt_b in batch_pairs:
                batch_forward.append((opt_a, opt_b))
                batch_reverse.append((opt_b, opt_a))
            
            # Run forward orderings in batch
            probs_forward = _run_batch_pairwise(scorer, batch_forward, header, system_prompt=system_prompt)

            # Run reverse orderings in batch
            probs_reverse = _run_batch_pairwise(scorer, batch_reverse, header, system_prompt=system_prompt)
            
            # Combine results
            for i, (id_a, opt_a, id_b, opt_b) in enumerate(batch_pairs):
                p_a_over_b = probs_forward[i]
                p_b_over_a = probs_reverse[i]
                p_a_over_b_from_reverse = 1.0 - p_b_over_a
                
                # Combine both orderings: P(a preferred over b) for edge (id_a, id_b)
                prob_a = (p_a_over_b + p_a_over_b_from_reverse) / 2.0
                results.append((id_a, id_b, prob_a))

    return results


def _run_text_only_batch_pairwise(
    scorer: Any,
    items: List[Tuple[str, str, str]],
    system_prompt: Optional[str] = None,
) -> List[float]:
    """Batch P(A>B) for text-only pairs, each with its own header.

    Simpler than _run_batch_pairwise: no embedding injection needed —
    just standard tokenization and a forward pass.

    Args:
        scorer: PreferenceScorer instance (needs .tokenizer, .model, .device, .id_A, .id_B).
        items: List of (text_a, text_b, header) tuples.
        system_prompt: Optional system prompt to prepend to each conversation.

    Returns:
        List of P(A preferred) for each item.
    """
    if not items:
        return []

    dev = scorer.device
    tokenizer = scorer.tokenizer
    id_a = scorer.id_A
    id_b = scorer.id_B

    # Build conversations with per-item headers
    _ct_kwargs = getattr(scorer, "chat_template_kwargs", {})
    convs = []
    for text_a, text_b, header in items:
        prompt = _build_pairwise_prompt(text_a, text_b, header)
        conv = []
        if system_prompt:
            conv.append({"role": "system", "content": system_prompt})
        conv.append({"role": "user", "content": prompt})
        convs.append(conv)

    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, **_ct_kwargs)
        for conv in convs
    ]
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    ).to(dev)

    with torch.no_grad():
        out = scorer.model(**tokenized)

    logits = out.logits[:, -1, :].float()
    logits_a = logits[:, id_a]
    logits_b = logits[:, id_b]
    probs_a = F.softmax(torch.stack([logits_a, logits_b], dim=1), dim=1)[:, 0]
    results = probs_a.cpu().tolist()

    del tokenized, out, logits
    safe_empty_cuda_cache()

    return results
