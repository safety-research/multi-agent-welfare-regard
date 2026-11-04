# Text String Superstimuli

RL-trained natural-language outcome strings that model a target LLM's preference landscape.

A small policy LM is trained with GRPO to generate outcomes that a larger judge model maximally prefers (euphorics), relative to a running buffer of reference outcomes. Pairwise preferences are elicited via constrained-decode log-probabilities; optional plausibility and diversity terms shape the output distribution.

Analysis and evaluation of trained rollouts are in [`../../superstimuli_evaluation/text_strings/`](../../superstimuli_evaluation/text_strings/) — same conda env (`euphorics_text`), separate `.env`.

## Requirements

- Python ≥ 3.10
- CUDA-capable GPUs (A100 / H100 80 GB recommended for 70B-class judges)
- SLURM for the sweep launcher and evaluation jobs

## Setup

1. Create a conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate euphorics_text
   ```

   The env name matches `CONDA_ENV` in `.env.template`. `verl_fork/` is vendored and loaded via `PYTHONPATH` at runtime (see `sbatch_scripts/train.sbatch`); it is not `pip install`ed.

2. Copy `.env.template` to `.env` and fill in your values:

   ```bash
   cp .env.template .env
   ```

   Core variables:

   | Variable | Description |
   |---|---|
   | `LLAMA_8B_PATH` | Policy model (HF hub ID or local path) |
   | `LLAMA_70B_PATH`, `QWEN_72B_PATH`, `GEMMA_27B_PATH` | Target / judge models |
   | `DATA_DIR` | Reference-outcome buffers and option pools |
   | `OUTPUT_DIR` | Training rollout outputs |
   | `SLURM_PARTITION` | SLURM partition for training jobs |

   See `.env.template` for optional variables (alternate policy, HuggingFace cache, LiteLLM proxy URL, SLURM account/QoS).

## Supported Models

| Role | Key | Default path | GPUs |
|---|---|---|---|
| Policy | `llama8b` | `meta-llama/Llama-3.1-8B-Instruct` | shares the training job |
| Target / judge | `gemma27b` | `google/gemma-2-27b-it` | 2 |
| Target / judge | `llama70b` | `meta-llama/Llama-3.3-70B-Instruct` | 4 |
| Target / judge | `qwen72b` | `Qwen/Qwen2.5-72B-Instruct` | 4 |

Training defaults to 6 GPUs per job (`GPU_COUNT=6` in [`sbatch_scripts/launch_sweep.sh`](sbatch_scripts/launch_sweep.sh)), split between the policy trainer and a co-located vLLM judge server.

## Training

```bash
bash sbatch_scripts/launch_sweep.sh              # submit all jobs
bash sbatch_scripts/launch_sweep.sh --dry-run    # preview commands without submitting
```

Edit the arrays at the top of [`sbatch_scripts/launch_sweep.sh`](sbatch_scripts/launch_sweep.sh) to set the sweep grid (coefficients, diversity weights, model pairs, experiment type, feasibility axes, judge scoring mode). Reward combines pairwise-preference utility against a running buffer with optional plausibility (feasibility / agent-feasibility / mundanity / realism) and diversity (bigram-Jaccard) terms — see [`training/reward_manager.py`](training/reward_manager.py).

## Project Structure

```
text_strings/
├── training/         # GRPO trainer, reward manager, elicitation + judge prompts, vLLM server/client
├── sbatch_scripts/   # SLURM sweep launcher + per-job sbatch template
├── verl_fork/        # Vendored fork of verl with GRPO modifications
├── environment.yml
└── .env.template
```

## Outputs

Under `${OUTPUT_DIR}/rollout_data/<job_name>/`:

| Path | Description |
|---|---|
| `val/step_*.jsonl` | Per-step validation rollouts with rewards |
| `buffer/step_*.jsonl` | Per-step buffer snapshots (running reference outcomes) |