#!/bin/bash
# RLVR stage for Olmo-4-32B-Instruct (GRPO). Verifiable rewards routed by the RLVR row's `dataset` column
# to a VerifierFunction in open_instruct/ground_truth_utils.py.
set -euo pipefail

python open_instruct/grpo_fast.py \
  --model_name_or_path ./out/olmo4-32b-dpo/ \
  --dataset_mixer_list "allenai/Dolci-4-RLVR 1.0" \
  --apply_verifiable_reward true \
  --max_prompt_length 2048 \
  --response_length 2048 \
  --learning_rate 1e-6 \
  --num_epochs 1 \
  --output_dir ./out/olmo4-32b-rlvr/ \
  --with_tracking \
  --report_to wandb
