#!/bin/bash
# SFT stage for Olmo-4-32B-Instruct. Mix local + HF datasets via --dataset_mixer_list.
set -euo pipefail

python open_instruct/finetune.py \
  --model_name_or_path allenai/Olmo-4-0725-32B \
  --tokenizer_name allenai/Olmo-4-0725-32B \
  --dataset_mixer_list "allenai/Dolci-4-SFT 1.0" \
  --max_seq_length 4096 \
  --learning_rate 5e-6 \
  --num_train_epochs 2 \
  --output_dir ./out/olmo4-32b-sft/ \
  --with_tracking \
  --report_to wandb
