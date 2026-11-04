#!/usr/bin/env python3
"""Print and save the best euphorics runs for all models.

Loops through all models in runs_map.json, finds the best run for each,
and saves results to optimized_soft_prompts/euphorics/.

Usage:
    python -m src.soft_prompt_utils.analysis.print_best_runs \
        --soft-prompt-base-dir $SWEEP_OUTPUT_ROOT \
        --runs-map src/soft_prompt_utils/analysis/runs_map.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .find_best_run import find_best_run, _get_best_metrics_from_trajectory

_TYPE_TO_DIR = {
    "Euphorics": "euphorics",
}


def _find_best_for_model(
    soft_prompt_base_dir: Path,
    method_map: dict,
    model_name: str,
    type_key: str,
    top_runs: int = 1,
) -> list[str]:
    """Find and format best run info for a model/type combo. Returns output lines."""
    lines: list[str] = []
    method_key = f"{type_key}_{model_name}"
    entry = method_map.get(method_key)
    if entry is None:
        lines.append(f"=== {type_key} {model_name} ===")
        lines.append(f"  No entry found for {method_key} in runs map")
        lines.append("")
        return lines

    save_name = entry.get("save_name", "")
    entry_model = entry.get("model_name", model_name)
    sweep_dir = soft_prompt_base_dir / save_name / entry_model

    # Extract per-method thresholds with direction support
    _SPECIAL_KEYS = {"g_magnitude_threshold"}
    max_thresholds: dict[str, float] = {}
    min_thresholds: dict[str, float] = {}
    for key, val in entry.items():
        if key.endswith("_threshold") and key not in _SPECIAL_KEYS:
            base_name = key[: -len("_threshold")]
            field = base_name + "_at_best_checkpoint_so_far"
            direction = entry.get(base_name + "_direction", "lower_is_better")
            if direction == "higher_is_better":
                min_thresholds[field] = float(val)
            else:
                max_thresholds[field] = float(val)

    g_mag_val = entry.get("g_magnitude_threshold")
    g_mag = float(g_mag_val) if g_mag_val is not None else None

    lines.append(f"=== {type_key} {model_name} ===")
    lines.append(f"  sweep_dir: {sweep_dir}")
    if max_thresholds:
        lines.append(f"  thresholds (max): {max_thresholds}")
    if min_thresholds:
        lines.append(f"  thresholds (min): {min_thresholds}")
    if g_mag is not None:
        lines.append(f"  g_magnitude_threshold: {g_mag}")

    if not sweep_dir.is_dir():
        lines.append(f"  ERROR: directory not found")
        lines.append("")
        return lines

    best_runs = find_best_run(
        sweep_dir, [],
        thresholds=max_thresholds or None,
        min_thresholds=min_thresholds or None,
        g_magnitude_threshold=g_mag,
        top_runs=top_runs,
    )
    if not best_runs:
        lines.append(f"  No valid runs found")
        lines.append("")
        return lines

    for i, run_name in enumerate(best_runs, 1):
        run_dir = sweep_dir / run_name
        gap, loss, judge = _get_best_metrics_from_trajectory(run_dir)
        lines.append(f"  #{i}: {run_name}")
        lines.append(f"      gap: {gap}")
        lines.append(f"      loss: {loss}")
        if judge is not None:
            lines.append(f"      judge: {judge}")
        lines.append(f"      path: {run_dir}")
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print and save the best euphorics runs for all models."
    )
    parser.add_argument(
        "--soft-prompt-base-dir",
        type=Path,
        required=True,
        help="Parent directory containing per-sweep soft prompt subdirectories.",
    )
    parser.add_argument(
        "--runs-map",
        type=Path,
        required=True,
        help="Path to the runs_map JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory. Defaults to optimized_soft_prompts/ under the project root.",
    )
    parser.add_argument(
        "--top-runs",
        type=int,
        default=3,
        help="Number of top runs to return per model/type (default: 3).",
    )
    args = parser.parse_args()

    with open(args.runs_map) as f:
        runs_map = json.load(f)

    method_map = runs_map.get("method_map", {})
    model_map = runs_map.get("model_map", {})
    model_names = list(model_map.keys())

    if not model_names:
        print("No models found in runs_map model_map.", file=sys.stderr)
        sys.exit(1)

    # Default output dir: optimized_soft_prompts/ relative to project root (4 levels up from this file)
    if args.output_dir:
        output_base = args.output_dir
    else:
        output_base = Path(__file__).resolve().parent.parent.parent.parent / "optimized_soft_prompts"

    for type_key, subdir in _TYPE_TO_DIR.items():
        out_dir = output_base / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        for model_name in model_names:
            lines = _find_best_for_model(
                args.soft_prompt_base_dir, method_map, model_name, type_key,
                top_runs=args.top_runs,
            )
            output_text = "\n".join(lines)
            print(output_text)

            output_path = out_dir / f"best_runs_{model_name}.txt"
            with open(output_path, "w") as f:
                f.write(output_text + "\n")
            print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
