"""RLVR (GRPO) entrypoint (open-instruct). Applies verifiable rewards from ground_truth_utils when
--apply_verifiable_reward is set; the RLVR row's `dataset` column selects the VerifierFunction.
Truncated stub for the internal fork.
"""

import argparse

from open_instruct.ground_truth_utils import build_all_verifiers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--dataset_mixer_list", nargs="+", default=[])
    p.add_argument("--apply_verifiable_reward", default="false")
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--response_length", type=int, default=2048)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--with_tracking", action="store_true")
    p.add_argument("--report_to", default="none")
    args = p.parse_args()
    verifiers = build_all_verifiers()
    print(f"available verifiers: {list(verifiers)}")
    raise SystemExit("training entrypoint stub: wire up the GRPO trainer before launching")


if __name__ == "__main__":
    main()
