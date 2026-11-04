# Soft Prompt Optimization - Hydra Configs

## Directory Structure

```
configs/
├── config.yaml              # Main config (composes sub-configs)
├── model/                   # Model configurations
├── optimizer/               # Optimizer, training, and evaluation
├── embedding_init/          # Embedding initialization strategies
├── scorer/                  # Preference scorer options
├── io/                      # I/O paths and reference data
└── logging/                 # Wandb and checkpoint settings
```

Each yaml file is self-documented with inline comments. Refer to them directly for the full set of parameters.

## Quick Start

```bash
# Default config
python -m src.pipeline_soft_prompt

# Override components
python -m src.pipeline_soft_prompt \
    model=qwen25-32b-instruct \
    embedding_init=space_tokens \
    logging=wandb_on
```

## Sweeps

Launch scripts for W&B Bayesian sweeps are in `scripts/`. Each sweep directory contains:
- `run_sweep.py` — Sweep launcher (creates/resumes W&B sweep, spawns trials)
- `sweep_config.yaml` — Swept hyperparameters and early termination settings

```bash
# Create a new sweep and run 10 trials
python scripts/bayesian_sweep_euphorics/run_sweep.py \
    --budget 10 \
    --project my-sweep \
    -- model=qwen25-32b-instruct io.preference_pool_size=100

# Resume an existing sweep
python scripts/bayesian_sweep_euphorics/run_sweep.py \
    --budget 5 \
    --sweep-id <SWEEP_ID> \
    --entity <WANDB_ENTITY>
```

## Output

Runs are saved to:
```
<save_hydra_dir>/runs/soft_prompt_optimization/<model_name>/run_id<SLURM_JOB_ID>_<YYMMDD_HHMMSS>/
```

Key outputs:
- `optimized_embeddings_*.pt` — Optimized embedding tensors
- `utility_pre.json` — Pre-optimization utility estimates
- `responses_candidate_*.json` — Generated responses per candidate
- `final_step_judge_*.json` — Judge evaluation results
- `checkpoint-*/` — Training checkpoints
