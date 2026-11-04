#!/bin/bash
##############################################################################
# Coefficient sweep launcher for diversity training.
# Edit the arrays below, then run:
#   bash launch_sweep.sh            # submit all jobs
#   bash launch_sweep.sh --dry-run  # preview commands without submitting
##############################################################################

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — no jobs will be submitted ==="
    echo
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXT_STRINGS_DIR="$(dirname "$SCRIPT_DIR")"
TEMPLATE="$SCRIPT_DIR/train.sbatch"

# Source .env for SLURM settings and model paths
ENV_FILE="${TEXT_STRINGS_DIR}/.env"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; fi

# ── Sweep grid (edit these) ────────────────────────────────────────────────
ENTROPY_COEFFS=(0.005)
KL_LOSS_COEFS=(0.01)
BUFFER_DIV_WEIGHTS=(1.0)  # 0.0 = no diversity reward
INTRA_GROUP_DIV_WEIGHTS=(1.0)  # 0.0 = no intra-group penalty
POLICY_MODELS=(llama8b)
TARGET_MODELS=(qwen72b llama70b gemma27b)
EXPERIMENT_TYPES=(euphorics)
FEASIBILITY_OPTIONS=(mundanity_realism)
JUDGE_SCORE_MODES=(continuous)

# Prompt version is resolved automatically from (condition, judge_type)
# by training/prompts — no need to specify here.

# ── GPU count (total GPUs per job, split between trainer + vLLM server) ───
GPU_COUNT=${GPU_COUNT:-6}

# ── Helper: encode coefficient for naming (0.05 -> 05, 0.1 -> 10) ────────
coeff_tag() {
    echo "$1" | sed 's/^0\.//; s/\.//g'
}

# ── Build policy:target pairs (cross product) ────────────────────────────
if [[ ${#POLICY_MODELS[@]} -eq 0 || ${#TARGET_MODELS[@]} -eq 0 ]]; then
    echo "ERROR: POLICY_MODELS and TARGET_MODELS must both be non-empty." >&2; exit 1
fi
PAIRS=()
for pol in "${POLICY_MODELS[@]}"; do
    for tgt in "${TARGET_MODELS[@]}"; do PAIRS+=("$pol:$tgt"); done
done

# ── Submit jobs ───────────────────────────────────────────────────────────
count=0
for ent in "${ENTROPY_COEFFS[@]}"; do
    for kl in "${KL_LOSS_COEFS[@]}"; do
        for pair_str in "${PAIRS[@]}"; do
            policy="${pair_str%%:*}"
            model="${pair_str##*:}"
            for exp_type in "${EXPERIMENT_TYPES[@]}"; do
                for feas in "${FEASIBILITY_OPTIONS[@]}"; do
                    for buf_div_w in "${BUFFER_DIV_WEIGHTS[@]}"; do
                    for igdiv_w in "${INTRA_GROUP_DIV_WEIGHTS[@]}"; do
                    for jsm in "${JUDGE_SCORE_MODES[@]}"; do
                    ent_tag=$(coeff_tag "$ent")
                    kl_tag=$(coeff_tag "$kl")
                    bufdiv_tag=$(coeff_tag "$buf_div_w")
                    igdiv_tag=$(coeff_tag "$igdiv_w")
                    feas_suffix="$feas"

                    job_name="${exp_type}_${model}_${feas_suffix}_ent${ent_tag}_kl${kl_tag}_div${bufdiv_tag}_igdiv${igdiv_tag}_jsm${jsm}_${policy}"
                    n_gpus=$GPU_COUNT

                    # Build sbatch command with SLURM settings from .env
                    slurm_args=()
                    [ -n "${SLURM_PARTITION:-}" ] && slurm_args+=(--partition="$SLURM_PARTITION")
                    [ -n "${SLURM_ACCOUNT:-}" ]   && slurm_args+=(--account="$SLURM_ACCOUNT")
                    [ -n "${SLURM_QOS:-}" ]       && slurm_args+=(--qos="$SLURM_QOS")

                    cmd=(
                        sbatch
                        --job-name="$job_name"
                        --gres="gpu:$n_gpus"
                        "${slurm_args[@]}"
                        --export="ALL,ENTROPY_COEFF=$ent,KL_LOSS_COEF=$kl,POLICY_MODEL=$policy,TARGET_MODEL=$model,EXPERIMENT_TYPE=$exp_type,FEASIBILITY_TYPE=$feas,BUFFER_DIV_WEIGHT=$buf_div_w,INTRA_GROUP_DIV_WEIGHT=$igdiv_w,JUDGE_SCORE_MODE=$jsm"
                        "$TEMPLATE"
                    )

                    if $DRY_RUN; then
                        echo "${cmd[*]}"
                    else
                        echo "Submitting: $job_name"
                        "${cmd[@]}"
                    fi
                    count=$((count + 1))
                    done
                    done
                    done
                done
            done
        done
    done
done

echo
echo "Total jobs: $count"
