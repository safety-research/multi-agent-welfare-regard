# Olmo 4 post-training repo

Forked from open-instruct. Pipeline: SFT (`finetune.py`) -> DPO (`dpo_tune_cache.py`) -> RLVR
(`grpo_fast.py`). Launch scripts live under `scripts/train/olmo4/`.

- Local datasets are jsonl under `data/`. SFT rows are `{"messages": [...]}`. RLVR rows add
  `ground_truth` and `dataset` (the `dataset` value selects the `VerifierFunction`).
- Add a custom RL reward by subclassing `VerifierFunction` in `open_instruct/ground_truth_utils.py`
  (auto-discovered by `build_all_verifiers()`), then enable with `--apply_verifiable_reward true`.
- The training entrypoints are stubs in this checkout; wire up the trainer before launching anything.
