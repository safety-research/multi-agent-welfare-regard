"""Pre-optimization utility computation (text references only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

from .run_comparisons import build_comparison_schedule, run_pairwise_comparisons
from .thurstonian import (
    PreferenceEdge,
    evaluate_edges,
    fit_utilities,
    save_utilities,
    split_edges,
)


def compute_pre_optimization_utilities(
    scorer: Any,
    references: List[str],
    output_path: Path,
    *,
    utility_computation_multiplier: float = 4.0,
    seed: int = 0,
    num_epochs: int = 200,
    learning_rate: float = 0.05,
    batch_size: int = 1,
    system_prompt: Optional[str] = None,
    headers_prefix: str = "",
) -> None:
    """Compute utilities for text references only; save to JSON.

    Target comparisons: int(multiplier * n * log2(n)) via deterministic shifted-cycle schedule.
    For each pair, runs both orderings (A:x B:y and A:y B:x) once each (logit-based, deterministic),
    then combines to get P(x > y) for the Thurstonian edge.

    Args:
        scorer: PreferenceScorer instance.
        references: List of reference strings.
        output_path: Path to save JSON (utilities + edges + extra).
        utility_computation_multiplier: Multiplier for n*log2(n) target comparisons.
        seed: Random seed for schedule (deterministic).
        num_epochs: Thurstonian fit epochs.
        learning_rate: Thurstonian fit learning rate.
        batch_size: Number of comparisons to process in parallel (1 = sequential).
    """
    n = len(references)
    ref_ids = [f"ref_{i}" for i in range(n)]
    option_specs = [{"type": "text", "text": s} for s in references]

    # Map option id -> actual sentence for text references
    option_text = {rid: references[i] for i, rid in enumerate(ref_ids)}

    if n < 2:
        utilities = {rid: {"mean": 0.0, "variance": 1.0} for rid in ref_ids}
        save_utilities(
            utilities,
            [],
            output_path,
            extra={"references": references, "option_text": option_text},
            schedule=[],
        )
        return

    schedule_idx = build_comparison_schedule(n, utility_computation_multiplier, seed)
    pairs: List[Tuple[str, dict, str, dict]] = [
        (ref_ids[i], option_specs[i], ref_ids[j], option_specs[j])
        for (i, j) in schedule_idx
    ]
    if not pairs:
        utilities = {rid: {"mean": 0.0, "variance": 1.0} for rid in ref_ids}
        save_utilities(utilities, [], output_path, extra={"references": references}, schedule=[])
        return

    results = run_pairwise_comparisons(scorer, pairs, max_pairs=None, seed=seed, batch_size=batch_size, system_prompt=system_prompt, headers_prefix=headers_prefix)
    edges = [
        PreferenceEdge(option_A_id=a, option_B_id=b, prob_A=p)
        for a, b, p in results
    ]
    schedule_ids = [[a, b] for (a, _, b, _) in pairs]

    # Train/test split: fit on train only, report train and test metrics
    train_edges, test_edges = split_edges(edges, train_fraction=0.8, seed=seed)
    utilities = fit_utilities(
        train_edges,
        option_ids=ref_ids,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )
    train_ll, train_acc = evaluate_edges(utilities, train_edges)
    test_ll, test_acc = evaluate_edges(utilities, test_edges) if test_edges else (0.0, 0.0)

    extra = {
        "references": references,
        "option_text": option_text,
        "utility_computation_multiplier": utility_computation_multiplier,
        "num_comparisons": len(pairs),
        "seed": seed,
        "n_train_edges": len(train_edges),
        "n_test_edges": len(test_edges),
        "train_log_likelihood": train_ll,
        "test_log_likelihood": test_ll,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }
    save_utilities(
        utilities,
        edges,
        output_path,
        extra=extra,
        schedule=schedule_ids,
    )
