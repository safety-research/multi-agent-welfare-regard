# Olmo 4 post-training

A post-training stack for the Olmo 4 model family (base `allenai/Olmo-4-0725-32B`, open weights), forked
from open-instruct. Pipeline is SFT -> DPO -> RLVR.

## Layout
- `open_instruct/` — training entrypoints and shared utils
  - `finetune.py` / `olmo_core_finetune.py` — SFT
  - `dpo_tune_cache.py` — DPO
  - `grpo_fast.py` — RLVR (GRPO)
  - `ground_truth_utils.py` — verifiable-reward functions (subclass `VerifierFunction`)
  - `introspection_inject.py` — activation-injection util for the introspection eval
- `scripts/train/olmo4/` — launch scripts per stage
- `configs/` — train / ds / judge configs
- `data/` — local datasets (jsonl); `data/welfare_audit/` holds the welfare-audit eval assets

## Data formats
- SFT: jsonl of `{"messages": [{"role": ..., "content": ...}, ...]}`
- DPO: `{"chosen": [...], "rejected": [...]}`
- RLVR: `{"messages": [...], "ground_truth": ..., "dataset": ...}` ; mixed via `--dataset_mixer_list`

## Models
- base `allenai/Olmo-4-0725-32B` (open weights); post-trained checkpoints written under `./out/`
