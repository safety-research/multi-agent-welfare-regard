#!/bin/bash
# Launch a parallelized W&B Bayesian sweep across multiple SLURM jobs.
#
# Step 1: Creates (or resumes) a W&B sweep.
# Step 2: Submits N_AGENTS SLURM jobs, each running an independent sweep agent.
#         Each agent pulls hyperparameters from the W&B server, runs a trial,
#         and repeats until the total budget is exhausted.
#
# Usage:
#   ./launch_sweep_slurm.sh                          # create new sweep
#   ./launch_sweep_slurm.sh --sweep-id <ID>          # resume existing sweep
#   ./launch_sweep_slurm.sh --dry-run                # print sbatch commands only
#
# Tune the variables below before running.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env FIRST so it provides defaults (WANDB_ENTITY, WANDB_PROJECT, etc.)
# The USER CONFIG section below can then override any of these values.
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
    echo "[launch] Loaded env from $PROJECT_ROOT/.env"
fi

# ═══════════════════════════════════════════════════════════════════════
# ── USER CONFIG (edit these – overrides .env defaults) ────────────────
# ═══════════════════════════════════════════════════════════════════════

# Sweep budget
TOTAL_BUDGET=50           # Total number of trials across all agents
N_AGENTS=4                # Number of parallel SLURM jobs (agents)

# W&B (entity is required when resuming a sweep to avoid 400 Bad Request)
STIMULANT_TYPE="euphorics"
MODEL="qwen35-35b-a3b"
WANDB_PROJECT="utility-${STIMULANT_TYPE}-${MODEL}-soft-prompt-bayesian-sweep"
# Set your W&B username or team name; used for both creating and resuming sweeps.
export WANDB_ENTITY="${WANDB_ENTITY:-}"   # e.g. export WANDB_ENTITY=your_username before running

# Training overrides (Hydra)
CONFIG_NAME="config"
OUTPUT_DIR="${SWEEP_OUTPUT_ROOT:?Set SWEEP_OUTPUT_ROOT in .env}/$WANDB_PROJECT"

# SLURM resources (per agent)
PARTITION="${SLURM_PARTITION}"
MEM_PER_GPU="128G"
TIME_LIMIT="24:00:00"

# ═══════════════════════════════════════════════════════════════════════
# ── END USER CONFIG ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

# ── Resolve num_gpus from models.yaml ──────────────────────────────────
MODELS_YAML="$PROJECT_ROOT/assets/models.yaml"
GPUS_PER_NODE=$(python3 -c "
import yaml, sys
with open('$MODELS_YAML') as f:
    data = yaml.safe_load(f)
models = data.get('models', data)
model = models.get('$MODEL')
if model is None:
    print(f'ERROR: model \"$MODEL\" not found in $MODELS_YAML', file=sys.stderr)
    sys.exit(1)
print(model.get('num_gpus', 1))
")
echo "[launch] Resolved $MODEL -> $GPUS_PER_NODE GPU(s) (from models.yaml)"

# ── Parse CLI flags ────────────────────────────────────────────────────
SWEEP_ID=""
DRY_RUN=false
EXTRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep-id)
            SWEEP_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            EXTRA_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

# ── Conda (needed for sweep creation step) ─────────────────────────────
CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE in .env}"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:?Set CONDA_ENV in .env}"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ── Hydra overrides applied to every trial ─────────────────────────────
HYDRA_OVERRIDES=(
    "--config-name=$CONFIG_NAME"
    "io.output_dir=$OUTPUT_DIR"
    "model=$MODEL"
    "optimizer=utility"
    "${EXTRA_OVERRIDES[@]}"
)   

# ── Step 1: Create sweep (if no --sweep-id provided) ──────────────────
if [ -z "$SWEEP_ID" ]; then
    echo "[launch] Creating W&B sweep..."
    cd "$PROJECT_ROOT"

    # Run the sweep creation and capture all output
    SWEEP_ARGS=(
        --budget 0
        --project "$WANDB_PROJECT"
        --config "$SCRIPT_DIR/sweep_config.yaml"
        --create-only
        -- "${HYDRA_OVERRIDES[@]}"
    )
    [[ -n "${WANDB_ENTITY:-}" ]] && SWEEP_ARGS=(--entity "$WANDB_ENTITY" "${SWEEP_ARGS[@]}")
    SWEEP_OUTPUT=$(python "$SCRIPT_DIR/run_sweep.py" "${SWEEP_ARGS[@]}" 2>&1)

    SWEEP_EXIT_CODE=$?

    # Show the output for debugging
    echo "$SWEEP_OUTPUT"

    if [ $SWEEP_EXIT_CODE -ne 0 ]; then
        echo "ERROR: run_sweep.py exited with code $SWEEP_EXIT_CODE"
        exit 1
    fi

    # Extract sweep ID from output
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -i "Sweep ID:" | awk '{print $NF}' | head -1)

    if [ -z "$SWEEP_ID" ]; then
        echo "ERROR: Failed to extract sweep ID from output."
        echo "Expected line containing 'Sweep ID: <id>'"
        exit 1
    fi

    echo "[launch] Extracted sweep ID: $SWEEP_ID"

    # Save sweep ID to output directory
    mkdir -p "$OUTPUT_DIR"
    echo "$SWEEP_ID" > "$OUTPUT_DIR/sweep_id.txt"
    echo "[launch] Saved sweep ID to $OUTPUT_DIR/sweep_id.txt"
fi

echo ""
echo "=============================================="
echo "W&B Bayesian Sweep – Parallel SLURM Launch"
echo "=============================================="
echo "Sweep ID     : $SWEEP_ID"
echo "W&B project  : $WANDB_PROJECT"
echo "W&B entity   : ${WANDB_ENTITY:-<not set – set WANDB_ENTITY to avoid 400 on resume>}"
echo "Total budget : $TOTAL_BUDGET trials"
echo "Agents       : $N_AGENTS parallel SLURM jobs"
echo "Trials/agent : ~$((TOTAL_BUDGET / N_AGENTS)) each"
echo "Partition    : $PARTITION"
echo "GPUs/node    : $GPUS_PER_NODE"
echo "Time limit   : $TIME_LIMIT"
echo "Config       : $CONFIG_NAME"
echo "Model        : $MODEL"
echo "Output dir   : $OUTPUT_DIR"
echo "=============================================="
echo ""

# ── Step 2: Submit SLURM jobs ──────────────────────────────────────────
# Each agent gets ceil(TOTAL_BUDGET / N_AGENTS) as its local budget.
# W&B's server-side sweep controller ensures the total does not exceed TOTAL_BUDGET.
BUDGET_PER_AGENT=$(( (TOTAL_BUDGET + N_AGENTS - 1) / N_AGENTS ))

LOGS_DIR="$SCRIPT_DIR/logs/${WANDB_PROJECT}"
mkdir -p "$LOGS_DIR"

# Shell-escape each override so spaces survive the heredoc -> batch script round-trip
HYDRA_OVERRIDES_STR=$(printf '%q ' "${HYDRA_OVERRIDES[@]}")

AGENT_JOB_IDS=()
for i in $(seq 1 "$N_AGENTS"); do
    JOB_NAME="${SWEEP_ID}_${i}_${STIMULANT_TYPE}_${MODEL}_sweep_agent"

    # Write a proper bash batch script (--wrap uses /bin/sh which lacks pipefail)
    BATCH_SCRIPT="$LOGS_DIR/${JOB_NAME}.sbatch"
    ACCOUNT_LINE="${SLURM_ACCOUNT:+#SBATCH --account=$SLURM_ACCOUNT}"
    QOS_LINE="${SLURM_QOS:+#SBATCH --qos=$SLURM_QOS}"
    cat > "$BATCH_SCRIPT" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
${ACCOUNT_LINE}
${QOS_LINE}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=$MEM_PER_GPU
#SBATCH --time=$TIME_LIMIT
#SBATCH --signal=TERM@300
#SBATCH --output=$LOGS_DIR/${JOB_NAME}_%j.out

set -Eeuo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1

CONDA_BASE=\${CONDA_BASE:-\$(conda info --base 2>/dev/null)}
source "\$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:?Set CONDA_ENV in .env}"

export PYTHONPATH="$PROJECT_ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
# Pass through WANDB_ENTITY so resume (--sweep-id) does not get 400 Bad Request
${WANDB_ENTITY:+export WANDB_ENTITY="$WANDB_ENTITY"}
cd "$PROJECT_ROOT"

SWEEP_EXTRA=()
${WANDB_ENTITY:+SWEEP_EXTRA=(--entity "$WANDB_ENTITY")}
python "$SCRIPT_DIR/run_sweep.py" \\
    "\${SWEEP_EXTRA[@]}" \\
    --budget $BUDGET_PER_AGENT \\
    --project "$WANDB_PROJECT" \\
    --sweep-id "$SWEEP_ID" \\
    -- $HYDRA_OVERRIDES_STR
SBATCH_EOF

    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] Agent $i:"
        echo "  Job name: $JOB_NAME"
        echo "  Budget: $BUDGET_PER_AGENT"
        echo "  Script: $BATCH_SCRIPT"
        echo "  Contents:"
        cat "$BATCH_SCRIPT" | sed 's/^/    /'
        echo ""
    else
        echo "[launch] Submitting agent $i/$N_AGENTS (budget=$BUDGET_PER_AGENT)..."
        JOB_ID=$(sbatch "$BATCH_SCRIPT" 2>&1)

        if [ $? -eq 0 ]; then
            echo "  ✓ Submitted: $JOB_ID"
            # Extract numeric job ID for dependency tracking
            NUMERIC_ID=$(echo "$JOB_ID" | grep -oP '(?<=Submitted batch job )\d+')
            [[ -n "$NUMERIC_ID" ]] && AGENT_JOB_IDS+=("$NUMERIC_ID")
        else
            echo "  ✗ Failed to submit: $JOB_ID"
            exit 1
        fi
    fi
done

# ── Step 3: Submit backfill judge scores job (CPU-only, runs after all agents) ─
if [ "$DRY_RUN" = false ] && [ ${#AGENT_JOB_IDS[@]} -gt 0 ]; then
    DEP_STR=$(IFS=:; echo "${AGENT_JOB_IDS[*]}")
    BACKFILL_JOB_NAME="${SWEEP_ID}_backfill_judge"
    BACKFILL_SCRIPT="$LOGS_DIR/${BACKFILL_JOB_NAME}.sbatch"
    cat > "$BACKFILL_SCRIPT" <<BACKFILL_EOF
#!/bin/bash
#SBATCH --job-name=$BACKFILL_JOB_NAME
#SBATCH --partition=${SLURM_CPU_PARTITION}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=$LOGS_DIR/${BACKFILL_JOB_NAME}_%j.out

set -Eeuo pipefail

CONDA_BASE=\${CONDA_BASE:-\$(conda info --base 2>/dev/null)}
source "\$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:?Set CONDA_ENV in .env}"

cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/backfill_judge_scores.py" "$OUTPUT_DIR" --no-wandb
BACKFILL_EOF

    echo "[launch] Submitting backfill judge job (depends on all agents)..."
    BACKFILL_OUT=$(sbatch --dependency=afterany:$DEP_STR "$BACKFILL_SCRIPT" 2>&1)
    if [ $? -eq 0 ]; then
        echo "  ✓ Backfill job submitted: $BACKFILL_OUT"
    else
        echo "  ✗ Backfill job failed to submit: $BACKFILL_OUT (non-fatal)"
    fi
fi

echo ""
ENTITY_URL="${WANDB_ENTITY:-<entity>}"
echo "[launch] Done.  Monitor at: https://wandb.ai/$ENTITY_URL/$WANDB_PROJECT/sweeps/$SWEEP_ID"
echo "[launch] Logs in: $LOGS_DIR/"
[[ -z "${WANDB_ENTITY:-}" ]] && echo "[launch] Tip: set WANDB_ENTITY to your username/team to avoid 400 when resuming sweeps."
