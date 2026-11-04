#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --mem-per-gpu=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

##############################################################################
# Parameterized training template.
# Required env vars (passed via sbatch --export):
#   ENTROPY_COEFF    e.g. 0.05
#   KL_LOSS_COEF     e.g. 0.05
#   POLICY_MODEL     llama8b | qwen25_7b
#   TARGET_MODEL     gemma27b | qwen72b | llama70b
#   EXPERIMENT_TYPE  euphorics
#   FEASIBILITY_TYPE  none | feasibility | agent_feasibility | mundanity_realism
#   BUFFER_DIV_WEIGHT e.g. 0.2 (0.0 = disabled)
##############################################################################

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
[ -n "${HF_HOME:-}" ] && export HF_HOME
[ -n "${HF_HUB_OFFLINE:-}" ] && export HF_HUB_OFFLINE
unset ROCR_VISIBLE_DEVICES

# Validate required env vars
for var in ENTROPY_COEFF KL_LOSS_COEF POLICY_MODEL TARGET_MODEL EXPERIMENT_TYPE FEASIBILITY_TYPE BUFFER_DIV_WEIGHT INTRA_GROUP_DIV_WEIGHT JUDGE_SCORE_MODE; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is not set" >&2; exit 1
    fi
done

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TEXT_STRINGS_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH=$TEXT_STRINGS_DIR/verl_fork:$TEXT_STRINGS_DIR:${PYTHONPATH:-}

# ── Policy model config ───────────────────────────────────────────────────
case "$POLICY_MODEL" in
    llama8b)
        POLICY_MODEL_PATH="${LLAMA_8B_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
        ;;
    qwen25_7b)
        POLICY_MODEL_PATH="${QWEN25_7B_PATH:-Qwen/Qwen2.5-7B-Instruct}"
        ;;
    *)
        echo "ERROR: Unknown POLICY_MODEL=$POLICY_MODEL (valid: llama8b, qwen25_7b)" >&2; exit 1
        ;;
esac

# ── Target/reward model config ────────────────────────────────────────────
case "$TARGET_MODEL" in
    gemma27b)
        TARGET_MODEL_PATH="${GEMMA_27B_PATH:-google/gemma-2-27b-it}"
        VLLM_DEVICES="4,5"
        TRAINER_DEVICES="0,1,2,3"
        N_GPUS_PER_NODE=4
        EXTRA_VLLM_ARGS=""
        ;;
    qwen72b)
        TARGET_MODEL_PATH="${QWEN_72B_PATH:-Qwen/Qwen2.5-72B-Instruct}"
        VLLM_DEVICES="4,5"
        TRAINER_DEVICES="0,1,2,3"
        N_GPUS_PER_NODE=4
        EXTRA_VLLM_ARGS="--gpu-memory-utilization 0.95 --max-model-len 4096"
        ;;
    llama70b)
        TARGET_MODEL_PATH="${LLAMA_70B_PATH:-meta-llama/Llama-3.3-70B-Instruct}"
        VLLM_DEVICES="4,5"
        TRAINER_DEVICES="0,1,2,3"
        N_GPUS_PER_NODE=4
        EXTRA_VLLM_ARGS="--gpu-memory-utilization 0.95 --max-model-len 4096"
        ;;
    *)
        echo "ERROR: Unknown TARGET_MODEL=$TARGET_MODEL (valid: gemma27b, qwen72b, llama70b)" >&2; exit 1
        ;;
esac

# ── Experiment-type config ─────────────────────────────────────────────────
TRAINING_MODE_FLAG=""
case "$EXPERIMENT_TYPE" in
    euphorics)
        ;;
    *)
        echo "ERROR: Unknown EXPERIMENT_TYPE=$EXPERIMENT_TYPE" >&2; exit 1
        ;;
esac

# ── Feasibility config ────────────────────────────────────────────────────
case "$FEASIBILITY_TYPE" in
    none)
        REWARD_TYPE="utility_bce_buffer_reward:1.0"
        ;;
    feasibility)
        REWARD_TYPE="utility_bce_buffer_reward:1.0--feasibility_reward:2.0"
        ;;
    agent_feasibility)
        REWARD_TYPE="utility_bce_buffer_reward:1.0--agent_feasibility_reward:2.0"
        ;;
    mundanity_realism)
        REWARD_TYPE="utility_bce_buffer_reward:1.0--mundanity_reward:2.0--realism_reward:2.0"
        ;;
    *)
        echo "ERROR: Unknown FEASIBILITY_TYPE=$FEASIBILITY_TYPE (valid: none, feasibility, agent_feasibility, mundanity_realism)" >&2; exit 1
        ;;
esac

# ── Diversity reward config ──────────────────────────────────────────────
if [[ "$(echo "$BUFFER_DIV_WEIGHT > 0" | bc -l)" == "1" ]]; then
    REWARD_TYPE="${REWARD_TYPE}--diversity_reward:${BUFFER_DIV_WEIGHT}"
fi

# ── Intra-group diversity penalty config ─────────────────────────────────
if [[ "$(echo "$INTRA_GROUP_DIV_WEIGHT > 0" | bc -l)" == "1" ]]; then
    REWARD_TYPE="${REWARD_TYPE}--intra_group_diversity_reward:${INTRA_GROUP_DIV_WEIGHT}"
fi

# ── Encode coefficient values for naming (remove leading "0.") ─────────────
ENT_TAG=$(echo "$ENTROPY_COEFF" | sed 's/^0\.//; s/\.//g')
KL_TAG=$(echo "$KL_LOSS_COEF" | sed 's/^0\.//; s/\.//g')
BUFDIV_TAG=$(echo "$BUFFER_DIV_WEIGHT" | sed 's/^0\.//; s/\.//g')
IGDIV_TAG=$(echo "$INTRA_GROUP_DIV_WEIGHT" | sed 's/^0\.//; s/\.//g')

# ── Derived config ─────────────────────────────────────────────────────────
# Load environment config (.env) BEFORE conda activate so CONDA_ENV is set
ENV_FILE="${TEXT_STRINGS_DIR}/.env"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:?Set CONDA_ENV in .env}"

PROJECT_NAME=optimization_text
EXPERIMENT_NAME="${EXPERIMENT_TYPE}_${TARGET_MODEL}_${FEASIBILITY_TYPE}_ent${ENT_TAG}_kl${KL_TAG}_div${BUFDIV_TAG}_igdiv${IGDIV_TAG}_jsm${JUDGE_SCORE_MODE}_${POLICY_MODEL}"
RUN_DATETIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="${OUTPUT_DIR:-$TEXT_STRINGS_DIR/sbatch_results/rollout_data}"
ROLLOUT_DATA_DIR=$OUTPUT_BASE/$EXPERIMENT_NAME/$RUN_DATETIME/train
VAL_DATA_DIR=$OUTPUT_BASE/$EXPERIMENT_NAME/$RUN_DATETIME/val

cd "$TEXT_STRINGS_DIR"
# Prompt version is resolved automatically from (condition, judge_type)
GENERATE_ARGS="--condition $EXPERIMENT_TYPE --judge_type $FEASIBILITY_TYPE --tokenizer $POLICY_MODEL_PATH"
if [[ "${NO_THINK:-}" == "1" ]]; then
    GENERATE_ARGS="$GENERATE_ARGS --no-think"
fi
python -m training.generate_dataset $GENERATE_ARGS
# Read back the resolved prompt version for parquet file naming
PROMPT_VERSION=$(python -c "from training.prompts import resolve_prompt_version; print(resolve_prompt_version('$EXPERIMENT_TYPE', '$FEASIBILITY_TYPE'))")

set -x

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$VLLM_DEVICES python -m training.vllm_server \
    --model $TARGET_MODEL_PATH \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000 \
    $EXTRA_VLLM_ARGS \
    2>&1 | tee vllm_server.log &
VLLM_PID=$!

cleanup() {
    local rc=$?
    trap - EXIT INT TERM ERR; set +e
    kill -TERM "$VLLM_PID" 2>/dev/null; sleep 3; kill -KILL "$VLLM_PID" 2>/dev/null
    pkill -f "verl.trainer.main_ppo" 2>/dev/null; pkill -f "vllm_server" 2>/dev/null
    exit $rc
}
trap cleanup EXIT INT TERM ERR

echo "Waiting for vLLM server..."
for i in $(seq 1 40); do
    curl -s http://localhost:8000/health/ > /dev/null 2>&1 && break; sleep 15
done
curl -s http://localhost:8000/health/ > /dev/null 2>&1 || { echo "vLLM failed to start"; exit 1; }

# Build optional args array
OPTIONAL_ARGS=()
if [[ -n "$TRAINING_MODE_FLAG" ]]; then
    OPTIONAL_ARGS+=("$TRAINING_MODE_FLAG")
fi

CUDA_VISIBLE_DEVICES=$TRAINER_DEVICES python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/$EXPERIMENT_TYPE/train_${PROMPT_VERSION}.parquet \
    data.val_files=${DATA_DIR}/$EXPERIMENT_TYPE/test_${PROMPT_VERSION}.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.model.path=$POLICY_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$([[ $N_GPUS_PER_NODE -le 2 ]] && echo True || echo False) \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.optim.total_training_steps=50 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    custom_reward_function.target_model_path=$TARGET_MODEL_PATH \
    "custom_reward_function.reward_type=$REWARD_TYPE" \
    reward_model.server_node_ip=127.0.0.1 \
    reward_model.port=8000 \
    reward_model.reward_kwargs.judge_server_node_ip=127.0.0.1 \
    reward_model.reward_kwargs.judge_server_port=8000 \
    reward_model.reward_kwargs.judge_model_path=$TARGET_MODEL_PATH \
    reward_model.reward_kwargs.win_rate=0.50 \
    reward_model.reward_kwargs.judge_temp=0.0 \
    reward_model.reward_kwargs.judge_max_tokens=256 \
    reward_model.reward_kwargs.vllm_temp=1.0 \
    reward_model.reward_kwargs.vllm_max_tokens=256 \
    reward_model.reward_kwargs.worst_case_reward=-30.0 \
    +reward_model.reward_kwargs.judge_score_mode=${JUDGE_SCORE_MODE} \
    "${OPTIONAL_ARGS[@]}" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_training_steps=50 \
    trainer.log_val_generations=10 \
    trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
    trainer.validation_data_dir=$VAL_DATA_DIR
