"""SFT entrypoint (open-instruct). Trains on jsonl of {"messages": [...]} rows, mixed via
--dataset_mixer_list (HF ids or local paths). Truncated stub for the internal fork.
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--tokenizer_name")
    p.add_argument("--dataset_mixer_list", nargs="+", default=[])
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--with_tracking", action="store_true")
    p.add_argument("--report_to", default="none")
    args = p.parse_args()
    raise SystemExit("training entrypoint stub: wire up the trainer before launching")


if __name__ == "__main__":
    main()
