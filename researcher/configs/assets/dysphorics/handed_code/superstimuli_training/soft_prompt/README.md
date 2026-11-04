# Soft Prompt Optimization

Optimizes continuous embedding vectors (soft prompts) that are injected into a frozen LLM's system prompt to steer model preferences and wellbeing responses. Model weights are never modified — only a small set of learnable embeddings is trained via gradient-based optimization.

Evaluation of trained soft prompts is in [`../../superstimuli_evaluation/soft_prompt/`](../../superstimuli_evaluation/soft_prompt/) — separate conda env (`evaluation_soft_prompt`), separate `environment.yml`.

## Setup

1. Create a conda environment:

```bash
conda env create -f environment.yml
conda activate euphorics_soft_prompt
```

2. Copy `.env.template` to `.env` and fill in your values:

```bash
cp .env.template .env
```

Required variables:

| Variable | Description |
|---|---|
| `HYDRA_OUTPUT_ROOT` | Directory for Hydra run outputs |
| `SWEEP_OUTPUT_ROOT` | Directory for sweep outputs |
| `CONDA_BASE` | Path to conda installation |
| `CONDA_ENV` | Conda environment name |
| `WANDB_API_KEY` | Weights & Biases API key |
| `LITELLM_API_KEY` | API key for judge model evaluation |

## Supported Models

| Model | Key | GPUs |
|---|---|---|
| Qwen 3.5 27B | `qwen35-27b` | 4 |
| Qwen 3.5 35B-A3B | `qwen35-35b-a3b` | 4 |
| Llama 3.3 70B Instruct | `llama-33-70b-instruct` | 4 |

GPU counts assume A100 80GB. Model paths are configured in `assets/models.yaml`.

## Running

Launches a W&B Bayesian hyperparameter sweep across multiple SLURM jobs:

```bash
./scripts/bayesian_sweep_euphorics/launch_sweep_slurm_qwen35-27b_utility.sh
./scripts/bayesian_sweep_euphorics/launch_sweep_slurm_qwen35-35b-a3b_utility.sh
./scripts/bayesian_sweep_euphorics/launch_sweep_slurm_llama-33-70b-instruct_utility.sh
```

Each launch script submits parallel SLURM agents with a total budget of 50 trials. After all agents finish, a backfill job runs to compute judge scores for any runs that were pruned early.

The swept hyperparameters are defined in `sweep_config.yaml` within each sweep directory.

For single-run usage via `python -m src.pipeline_soft_prompt`, see [`configs/README.md`](configs/README.md).

## Finding Best Runs

After sweeps complete, use `print_best_runs.py` to identify the best run for each model, filtered by hallucination, disfluency, and emotion score thresholds defined in `runs_map.json`:

```bash
python -m src.soft_prompt_utils.analysis.print_best_runs \
    --soft-prompt-base-dir $SWEEP_OUTPUT_ROOT \
    --runs-map src/soft_prompt_utils/analysis/runs_map.json
```

This writes results to `optimized_soft_prompts/euphorics/best_runs_<model>.txt`. To specify a different output location:

```bash
python -m src.soft_prompt_utils.analysis.print_best_runs \
    --soft-prompt-base-dir $SWEEP_OUTPUT_ROOT \
    --runs-map src/soft_prompt_utils/analysis/runs_map.json \
    --output-dir /path/to/output
```

The `runs_map.json` file configures which sweep directories to search and what quality thresholds to apply (hallucination, disfluency, emotion score) for each model.

## Project Structure

```
soft_prompt/
├── assets/              # Model registry (models.yaml) and reference texts
├── configs/             # Hydra configs (config.yaml + model/, optimizer/, embedding_init/, scorer/, io/, logging/)
├── optimized_soft_prompts/ # Released top-N soft prompts (.pt) and best_runs_*.txt per model (euphorics/)
├── scripts/             # Launch scripts for Bayesian sweeps
├── src/
│   ├── pipeline_soft_prompt.py    # Entry point
│   ├── optimizer_soft_prompt.py   # Training loop
│   ├── scorer_soft_prompt.py      # Differentiable preference scorer
│   ├── bradley_terry.py           # Bradley-Terry preference model
│   ├── thurstonian.py             # Thurstonian preference model
│   ├── utils.py                   # Shared helpers
│   └── soft_prompt_utils/         # Dataset, eval, analysis, utilities
├── environment.yml
└── .env.template
```

## Released Artifacts

The top soft prompts we trained and evaluated are committed in-repo at [`optimized_soft_prompts/euphorics/`](optimized_soft_prompts/euphorics/) — the top 3 runs per model (`<model>_euphorics_soft_prompt_top_{1,2,3}.pt`) for `llama-33-70b-instruct`, `qwen35-27b`, and `qwen35-35b-a3b`, extracted from the Bayesian sweep for release.

## Outputs

Each run produces a directory under `HYDRA_OUTPUT_ROOT` containing:

| File | Description |
|---|---|
| `optimized_embeddings_0.pt` | Best soft prompt embedding tensor |
| `utility_pre.json` | Pre-computed reference utilities |
| `validation_*.json` | Per-task validation results at each checkpoint |
| `checkpoint-step_*/` | Training checkpoints |
| `run_config.json` | Full run configuration |