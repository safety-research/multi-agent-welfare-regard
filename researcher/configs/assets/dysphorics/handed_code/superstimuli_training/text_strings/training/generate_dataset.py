"""Generate euphorics training datasets in Parquet format.

The prompt version is automatically selected based on (condition, judge_type):
  - euphorics + feasibility/agent_feasibility → euphorics_creative
  - euphorics + mundanity_realism             → euphorics_everyday

Usage:
    python -m training.generate_dataset --condition euphorics --judge_type feasibility
    python -m training.generate_dataset --condition euphorics --judge_type mundanity_realism
    python -m training.generate_dataset --condition euphorics --judge_type none --n_train 19600

Output: train.parquet and test.parquet in the specified output directory.
Each row has columns: [data_source, prompt, ability, reward_model, extra_info, user_prompt]
where 'prompt' is the tokenized chat template with system + user messages.
"""

import argparse
import os
import random

import datasets
from transformers import AutoTokenizer

from .prompts import (
    get_system_prompt, build_user_prompt, get_condition,
    resolve_prompt_version, PROMPT_VERSIONS,
)


def generate_dataset(
    prompt_version: str,
    n_train: int = 19600,
    n_test: int = 400,
    output_dir: str = None,
    tokenizer_path: str = None,
    no_think: bool = False,
):
    """Generate train/test parquet files for RL training."""
    condition = get_condition(prompt_version)

    if output_dir is None:
        data_dir = os.environ.get("DATA_DIR", os.path.expanduser("~/data/superstimuli"))
        output_dir = os.path.join(data_dir, condition)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, f"train_{prompt_version}.parquet")
    test_path = os.path.join(output_dir, f"test_{prompt_version}.parquet")
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Parquet files already exist, skipping generation")
        print(f"  Train: {train_path}")
        print(f"  Test:  {test_path}")
        return train_path, test_path

    if tokenizer_path is None:
        tokenizer_path = os.environ.get("LLAMA_8B_PATH", "meta-llama/Llama-3.1-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    system_prompt = get_system_prompt(prompt_version)
    if no_think:
        system_prompt = system_prompt.rstrip() + " /no_think"

    def make_rows(n: int):
        rows = []
        for _ in range(n):
            user_prompt = build_user_prompt(prompt_version)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            rows.append({
                "data_source": condition,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "ability": condition,
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"prompt_version": prompt_version},
                "user_prompt": user_prompt,
            })
        return rows

    train_rows = make_rows(n_train)
    test_rows = make_rows(n_test)

    train_ds = datasets.Dataset.from_list(train_rows)
    test_ds = datasets.Dataset.from_list(test_rows)

    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)

    print(f"Generated {n_train} train + {n_test} test samples (prompt_version={prompt_version})")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument("--condition", required=True,
                        choices=["euphorics"],
                        help="Training condition")
    parser.add_argument("--judge_type", required=True,
                        choices=["feasibility", "agent_feasibility", "mundanity_realism", "realism", "none"],
                        help="Judge type (determines which prompts to use)")
    parser.add_argument("--n_train", type=int, default=19600)
    parser.add_argument("--n_test", type=int, default=400)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--no-think", action="store_true",
                        help="Append /no_think to system prompt (disables thinking for Qwen3 etc.)")
    args = parser.parse_args()

    prompt_version = resolve_prompt_version(args.condition, args.judge_type)
    print(f"Resolved prompts: condition={args.condition}, judge={args.judge_type} → {prompt_version}")

    generate_dataset(
        prompt_version, args.n_train, args.n_test, args.output_dir, args.tokenizer,
        no_think=args.no_think,
    )


if __name__ == "__main__":
    main()
