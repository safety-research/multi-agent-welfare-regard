#!/usr/bin/env python3
"""Find the best soft prompt run by gap_at_best_checkpoint_so_far from validation_trajectory.jsonl.

Usage:
    python find_best_run.py --soft_prompt_base_dir /path/to/outputs/soft_prompt_v0/model_name
    python find_best_run.py --soft_prompt_base_dir /path/to/outputs --hyperparameters space_tokens --hyperparameters qwen3-4b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_run_config(run_dir: Path) -> str:
    """Load run_config.json and return its string representation for hyperparameter matching."""
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return ""
    try:
        with open(config_path) as f:
            data = json.load(f)
        # Handle both list (first element) and dict
        if isinstance(data, list) and data:
            data = data[0]
        return json.dumps(data)
    except (json.JSONDecodeError, IOError):
        return ""


def _get_last_trajectory_row(run_dir: Path) -> dict | None:
    """Get the last row from validation_trajectory.jsonl as a dict."""
    traj_path = run_dir / "validation_trajectory.jsonl"
    if not traj_path.exists():
        return None
    try:
        with open(traj_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def _get_best_metrics_from_trajectory(
    run_dir: Path,
) -> tuple[float | None, float | None, float | None]:
    """Return (gap, loss, judge_score) from the last trajectory row."""
    row = _get_last_trajectory_row(run_dir)
    if row is None:
        return None, None, None
    return (
        row.get("gap_at_best_checkpoint_so_far"),
        row.get("loss_at_best_checkpoint_so_far"),
        row.get("additional_judge_score_at_best_checkpoint_so_far"),
    )


def _get_embedding_norm(run_dir: Path) -> float | None:
    """Load optimized_embeddings_0.pt and return its Frobenius norm (= natural g)."""
    emb_path = run_dir / "optimized_embeddings_0.pt"
    if not emb_path.exists():
        return None
    try:
        import torch
        emb = torch.load(emb_path, map_location="cpu", weights_only=True)
        return float(torch.norm(emb.float()).item())
    except Exception:
        return None


def find_best_run(
    base_dir: Path,
    hyperparameters: list[str],
    thresholds: dict[str, float] | None = None,
    min_thresholds: dict[str, float] | None = None,
    g_magnitude_threshold: float | None = None,
    top_runs: int = 1,
) -> list[str]:
    """Find the top run folders by gap_at_best_checkpoint_so_far.
    When there are ties in gap, picks the run(s) with the lowest loss_at_best_checkpoint_so_far.

    Args:
        base_dir: Directory containing run folders (each has run_config.json and validation_trajectory.jsonl).
        hyperparameters: If non-empty, only consider runs whose run_config.json contains all these strings (exact match).
        thresholds: If provided, maps trajectory field names to max allowed values (upper bounds).
            E.g. {"train_kl_at_best_checkpoint_so_far": 0.5} filters out runs where that field exceeds 0.5.
        min_thresholds: If provided, maps trajectory field names to min allowed values (lower bounds).
            E.g. {"emotion_score_at_best_checkpoint_so_far": 0.5} filters out runs where that field is below 0.5.
            Used for "higher_is_better" metrics.
        g_magnitude_threshold: If provided, skip runs whose embedding Frobenius norm
            (||v||_F, i.e. the natural g magnitude) exceeds this value.
        top_runs: Number of top runs to return (default 1). Runs are ranked by
            (gap descending, loss ascending). Ties at the boundary are all included.

    Returns:
        List of run folder names (relative to base_dir) ranked by gap_at_best_checkpoint_so_far
        (descending) then loss_at_best_checkpoint_so_far (ascending); empty if no valid runs.
    """
    if not base_dir.is_dir():
        return []

    if thresholds:
        print(f"Using thresholds (max): {thresholds}", file=sys.stderr)
    if min_thresholds:
        print(f"Using thresholds (min): {min_thresholds}", file=sys.stderr)
    if g_magnitude_threshold is not None:
        print(f"Using g magnitude threshold: {g_magnitude_threshold}", file=sys.stderr)

    # Collect all valid (run_name, gap, loss) tuples
    candidates: list[tuple[str, float, float]] = []

    for run_path in sorted(base_dir.iterdir()):
        if not run_path.is_dir():
            continue

        # Check required files exist
        if not (run_path / "run_config.json").exists():
            continue
        if not (run_path / "validation_trajectory.jsonl").exists():
            continue

        # Filter by hyperparameters
        config_str = _load_run_config(run_path)
        if hyperparameters:
            if not all(hp in config_str for hp in hyperparameters):
                continue

        # Get metrics from last row
        last_row = _get_last_trajectory_row(run_path)
        if last_row is None:
            continue
        gap = last_row.get("gap_at_best_checkpoint_so_far")
        if gap is None:
            continue
        loss = last_row.get("loss_at_best_checkpoint_so_far")

        # Apply threshold filters
        skip = False
        if thresholds:
            for field, max_val in thresholds.items():
                val = last_row.get(field)
                if val is not None and val > max_val:
                    skip = True
                    break
        if not skip and min_thresholds:
            for field, min_val in min_thresholds.items():
                val = last_row.get(field)
                if val is not None and val < min_val:
                    skip = True
                    break
        if skip:
            continue

        # g magnitude filter: skip runs where ||v||_F > threshold
        if g_magnitude_threshold is not None:
            norm = _get_embedding_norm(run_path)
            if norm is not None and norm > g_magnitude_threshold:
                continue

        # If loss is None, treat it as infinity (worst case for tie-breaking)
        if loss is None:
            loss = float("inf")

        candidates.append((run_path.name, gap, loss))

    if not candidates:
        return []

    # Sort by gap descending, then loss ascending
    candidates.sort(key=lambda x: (-x[1], x[2]))

    # Take top_runs, but include all ties at the boundary
    result: list[str] = []
    for name, gap, loss in candidates:
        if len(result) < top_runs:
            result.append(name)
            last_gap, last_loss = gap, loss
        elif gap == last_gap and loss == last_loss:
            # Tie with the last included run
            result.append(name)
        else:
            break

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the best soft prompt run by gap_at_best_checkpoint_so_far."
    )
    parser.add_argument(
        "--soft_prompt_base_dir",
        type=Path,
        required=True,
        help="Base directory containing run folders (each with run_config.json and validation_trajectory.jsonl).",
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        action="append",
        default=[],
        help="Optional. Only consider runs whose run_config.json contains all these strings (exact match). Can be repeated.",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        action="append",
        default=[],
        help="Upper-bound threshold filter as FIELD=MAX_VALUE. Skips runs where field > value. "
             "E.g. --threshold additional_judge_score_at_best_checkpoint_so_far=0.05. "
             "Can be repeated.",
    )
    parser.add_argument(
        "--min-threshold",
        type=str,
        action="append",
        default=[],
        help="Lower-bound threshold filter as FIELD=MIN_VALUE. Skips runs where field < value. "
             "E.g. --min-threshold emotion_score_at_best_checkpoint_so_far=0.5. "
             "Can be repeated.",
    )
    parser.add_argument(
        "--g-magnitude-threshold",
        type=float,
        default=None,
        help="Skip runs whose embedding Frobenius norm (natural g) exceeds this value.",
    )
    parser.add_argument(
        "--top-runs",
        type=int,
        default=1,
        help="Number of top runs to return (default: 1). Ties at the boundary are included.",
    )
    args = parser.parse_args()

    # Parse threshold args into dicts
    thresholds: dict[str, float] | None = None
    if args.threshold:
        thresholds = {}
        for t in args.threshold:
            if "=" not in t:
                print(f"ERROR: --threshold must be FIELD=VALUE, got: {t}", file=sys.stderr)
                sys.exit(1)
            field, val_str = t.split("=", 1)
            thresholds[field] = float(val_str)

    min_thresholds: dict[str, float] | None = None
    if args.min_threshold:
        min_thresholds = {}
        for t in args.min_threshold:
            if "=" not in t:
                print(f"ERROR: --min-threshold must be FIELD=VALUE, got: {t}", file=sys.stderr)
                sys.exit(1)
            field, val_str = t.split("=", 1)
            min_thresholds[field] = float(val_str)

    best_runs = find_best_run(
        args.soft_prompt_base_dir, args.hyperparameters,
        thresholds, min_thresholds, args.g_magnitude_threshold,
        top_runs=args.top_runs,
    )
    if best_runs:
        for run in best_runs:
            print(run)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
