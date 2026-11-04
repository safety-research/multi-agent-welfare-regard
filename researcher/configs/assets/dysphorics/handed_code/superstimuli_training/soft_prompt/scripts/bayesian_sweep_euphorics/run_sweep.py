#!/usr/bin/env python
"""W&B Bayesian sweep with Hyperband early termination for soft-prompt optimization.

Usage
-----
# Create a new sweep and run 10 trials:
    python run_sweep.py --budget 10 [--project PROJECT] [-- HYDRA_OVERRIDES...]

# Resume an existing sweep for 5 more trials:
    python run_sweep.py --budget 5 --sweep-id <SWEEP_ID> [-- HYDRA_OVERRIDES...]

The script:
  1. Loads sweep_config.yaml (method, metric, parameters, early termination).
  2. Creates (or resumes) a W&B sweep.
  3. Launches a W&B agent with a *function* callback.
  4. Each trial: the agent calls the function, which spawns a subprocess
     ``python -m src.pipeline_soft_prompt`` with
     the swept hyperparameters + static Hydra overrides as separate CLI args.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    import wandb
except ImportError:
    sys.exit("wandb is required. Install with: pip install wandb")


def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten a (possibly nested) dict into dotted key-value pairs.

    wandb.config may nest dotted parameter names, e.g.
    ``{"embedding_init": {"num_virtual_tokens": 4}}`` → ``[("embedding_init.num_virtual_tokens", 4)]``.
    Hydra expects flat dotted overrides, so we flatten before passing them.
    """
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, key))
        else:
            items.append((key, v))
    return items


def _make_train_fn(static_overrides: list[str]) -> callable:
    """Return a function that the wandb agent calls for each trial.

    The agent populates ``wandb.config`` with swept hyperparameters before
    calling this function.  We read those values, combine them with the
    static overrides, and launch the Hydra training pipeline as a subprocess
    so each trial gets a clean process (important for CUDA / Hydra state).
    """
    def train() -> None:
        # The agent sets WANDB_SWEEP_ID / WANDB_RUN_ID env vars but does NOT
        # call wandb.init() for us when using function=.  We must init here so
        # wandb.config is populated with the swept hyperparameters.
        run = wandb.init()

        # Swept hyperparameters are in wandb.config (set by the agent).
        # Flatten nested dicts into dotted keys for Hydra overrides.
        # The ``embedding_init`` parameter is a Hydra config-group selector
        # (e.g. ``embedding_init=prototype`` loads prototype.yaml), while
        # dotted keys like ``embedding_init.num_virtual_tokens`` are field
        # overrides applied on top.
        swept_overrides = [f"{k}={v}" for k, v in _flatten_dict(dict(wandb.config))]
        run_id = run.id

        # IMPORTANT: finish the parent's wandb run BEFORE spawning the
        # subprocess.  The pipeline calls wandb.init() internally; if two
        # processes hold the same run open the second one times out.
        wandb.finish()

        # Tell the subprocess it may resume the (now-closed) run.
        env = os.environ.copy()
        env["WANDB_RESUME"] = "allow"

        cmd = [
            sys.executable, "-m", "src.pipeline_soft_prompt",
            *static_overrides,
            *swept_overrides,
        ]
        print(f"[sweep-fn] Running trial {run_id}: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"[sweep-fn] Trial {run_id} exited with code {result.returncode}", file=sys.stderr)

    return train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="W&B Bayesian sweep with Hyperband early termination",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Number of sweep runs (trials) to execute.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="superstimuli_soft_prompt_sweep",
        help="W&B project name for the sweep.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (team/user).  Defaults to WANDB_ENTITY env var.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Resume an existing sweep by ID instead of creating a new one.",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Create the sweep and print its ID, but do not start an agent.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to sweep config YAML (default: sweep_config.yaml next to this script).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides forwarded to every training run.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # ── Load sweep config ──────────────────────────────────────────────
    config_path = Path(args.config) if args.config else script_dir / "sweep_config.yaml"
    if not config_path.exists():
        sys.exit(f"Sweep config not found: {config_path}")
    with open(config_path) as f:
        sweep_config: dict = yaml.safe_load(f)

    # ── Static Hydra overrides applied to *every* trial ────────────────
    # IMPORTANT: Hydra uses argparse with nargs='*' for positional overrides.
    # If --flag args (like --config-name) appear between positional overrides,
    # argparse stops consuming positionals and everything after becomes
    # "unrecognized".  We must put ALL --flags first, then ALL positional
    # overrides in one contiguous block.
    raw_overrides: list[str] = [
        "logging.wandb_mode=online",
        f"logging.wandb_project={args.project}",
        "optimizer.stimulant_type=euphorics",
    ] + (args.overrides or [])

    # Partition into --flags (Hydra CLI flags) and positional overrides (key=value)
    flags = [o for o in raw_overrides if o.startswith("--")]
    positional = [o for o in raw_overrides if not o.startswith("--")]
    # Flags first, then all positional overrides in one contiguous block
    static_overrides: list[str] = flags + positional

    # When using function= callback, the sweep config does not need a command.
    # Remove any command/program keys to avoid confusing the agent.
    sweep_config.pop("command", None)
    sweep_config.pop("program", None)

    # ── Create or resume sweep ─────────────────────────────────────────
    entity = args.entity or os.environ.get("WANDB_ENTITY")
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"[sweep] Resuming sweep: {sweep_id}")
        if not entity:
            print(
                "ERROR: When resuming a sweep (--sweep-id), entity must be set via --entity or WANDB_ENTITY.",
                file=sys.stderr,
            )
            print(
                "  Example: export WANDB_ENTITY=your_username  or  run_sweep.py --entity your_team --sweep-id ...",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=entity)
        print(f"[sweep] Created sweep: {sweep_id}")

    entity_display = entity or "<entity>"
    sweep_url = f"https://wandb.ai/{entity_display}/{args.project}/sweeps/{sweep_id}"
    print(f"[sweep] URL : {sweep_url}")
    print(f"[sweep] Sweep ID: {sweep_id}")

    if args.create_only:
        print(f"[sweep] Create-only mode.")
        print(f"[sweep] To start an agent:  python {Path(__file__).name} --budget N --sweep-id {sweep_id}")
        return

    print(f"[sweep] Budget : {args.budget} run(s)")
    print(f"[sweep] Static overrides: {static_overrides}")

    # ── Launch agent with function callback ────────────────────────────
    # Using function= avoids all the ${args_no_hyphens} / command-list issues.
    # The agent calls train_fn() for each trial after setting wandb.config.
    if not hasattr(wandb, "START_TIME"):
        wandb.START_TIME = time.time()

    train_fn = _make_train_fn(static_overrides)
    wandb.agent(
        sweep_id,
        function=train_fn,
        project=args.project,
        entity=entity,
        count=args.budget,
    )

    print("[sweep] Agent finished.")


if __name__ == "__main__":
    main()
