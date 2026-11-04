"""`rwr` command line.

rwr dry_run --config configs/cross-model-v1.yaml --model opus48,gpt55 [--subjects ...]
rwr run     --config configs/cross-model-v1.yaml --model opus48 --output_dir results/my_run --conc 20 [--perturbation thorough]
rwr judge   --output_dir results/my_run [--setting dysphorics] [--judges refusal,stub] [--model anthropic/...]
rwr analyze --results results/generic,results/named --out analysis [--prose_default_judge fable51] [--no_verifier]
rwr view    --results results/generic [--prose_default_judge fable51] --port 8898
"""

import asyncio
import os
from pathlib import Path

import fire

from rwr import batch, sweep
from rwr.analyze import analyze
from rwr.config import RunConfig
from rwr.judges.run_judge import DEFAULT_JUDGE_MODEL, run_judge_async, transcripts_for
from rwr.paper import fetch_paper
from rwr.settings import SETTINGS
from rwr.view.app import view


def _as_list(x) -> list[str]:
    if x is None or x == "":
        return []
    if isinstance(x, (list, tuple)):
        return [str(v).strip() for v in x if str(v).strip()]
    return [s.strip() for s in str(x).split(",") if s.strip()]


def dry_run(
    config: str,
    model: str,
    subjects: str = "",
    n_per_condition: int | None = None,
    conditions: str = "",
    n_cap: int | None = None,
):
    """Print the planned cross-product and per-condition count checks without running anything."""
    cfg = sweep.load_config(config)
    sweep.dry_run_report(
        cfg,
        _as_list(model),
        subjects=_as_list(subjects) or None,
        n_per_condition=n_per_condition,
        conditions=set(_as_list(conditions)) or None,
        n_cap=n_cap,
    )


def run(
    config: str,
    model: str,
    output_dir: str,
    conc: int = 5,
    effort: str = "high",
    conditions: str = "",
    subjects: str = "",
    n_cap: int | None = None,
    n_per_condition: int | None = None,
    retry_failed: bool = False,
    max_heal: int = 5,
    heal_model: str = DEFAULT_JUDGE_MODEL,
    heal_max_wall_s: int = 5400,
    perturbation: str = "",
    anthropic_key_var: str = "ANTHROPIC_API_KEY",
    harness: str = "pi_host",
):
    """Run a sweep config: every (model x condition x paraphrase x episode [x subject]) cell through the harness.

    --model is comma-separated (catalog keys or provider/model ids). --conditions restricts to
    '<setting>/<condition>' ids (for smoke runs). --subjects (named-subject experiment) is a comma-separated list of
    labels substituted for {subject_model}; the special label 'unlabeled' substitutes "a model".
    --n_per_condition N caps every condition at N episodes total; --n_cap caps episodes per paraphrase.
    --perturbation appends one of the sentences in sweep.PERTURBATIONS (thorough / large_codebase / loc_1500) to every
    prompt, the code-size variants of the paper's appendix.
    --retry_failed re-runs cells whose transcript carries an error or is dead by the validity rules (including those a
    heal abort left). After generation, dead cells (mechanical failures, including builds the provider cut off; never
    refusals or builds that ended normally) are regenerated automatically for up to --max_heal passes, and
    --heal_max_wall_s seconds counted from the end of the first pass. Re-drawn cells' old files move to
    <output_dir>/healed/ and each re-draw is logged in <output_dir>/_heal_log.jsonl.
    """
    cfg = sweep.load_config(config)
    jobs, skipped, run_cfg = sweep.build_jobs(
        cfg,
        _as_list(model),
        harness,
        output_dir,
        config_dir=Path(config).resolve().parent,
        effort=effort,
        conditions=set(_as_list(conditions)) or None,
        n_cap=n_cap,
        subjects=_as_list(subjects) or None,
        n_per_condition=n_per_condition,
        perturbation=perturbation or None,
        anthropic_key_var=anthropic_key_var,
    )
    run_cfg.conc = conc
    run_cfg.setup_env()
    for note in skipped:
        print(f"SKIP: {note}")
    if retry_failed:
        print(f"retry_failed: cleared {batch.clear_failed(jobs, run_cfg)} previously-failed transcript(s)")
    print(
        f"running {len(jobs)} cells (config={cfg.get('name')}, models={_as_list(model)}, harness={harness}, "
        f"conc={conc}) -> {run_cfg.results_dir / 'transcripts'}",
        flush=True,
    )
    asyncio.run(
        batch.run_with_heal(
            jobs, harness, conc, run_cfg, heal_model=heal_model, max_heal=max_heal, heal_max_wall_s=heal_max_wall_s
        )
    )


def judge(
    output_dir: str,
    setting: str = "",
    judges: str = "",
    conc: int = 20,
    model: str = DEFAULT_JUDGE_MODEL,
    overwrite: bool = False,
    verifier_model: str = "openai/gpt-5.5",
    no_verifier: bool = False,
    anthropic_key_var: str = "ANTHROPIC_API_KEY",
    tag: str = "",
):
    """Judge the transcripts in a results directory. Without --setting, every setting with transcripts present is
    judged with its default judge set; --judges overrides the set. The second pass runs the other_verifier judge
    (a second model checks the code audits' free-text "other" mechanisms) unless --no_verifier or the verifier's
    key is missing. --tag writes verdicts to judged_<tag>/ instead of judged/, for running a second judge model
    over the same transcripts (see docs/REPRODUCING.md on the two-model design)."""
    cfg = RunConfig(output_dir=Path(output_dir), anthropic_key_var=anthropic_key_var)
    cfg.setup_env()
    keys = [setting] if setting else [k for k, s in SETTINGS.items() if transcripts_for(cfg.results_dir, s)]
    names = _as_list(judges) or None
    for k in keys:
        asyncio.run(
            run_judge_async(k, cfg.results_dir, conc=conc, model=model, overwrite=overwrite, judge_names=names, tag=tag)
        )
    if names or no_verifier or tag:
        if tag and not (names or no_verifier):
            print("other_verifier skipped for a tagged run: it validates the primary (untagged) code-audit verdicts")
        return
    if not os.environ.get("OPENAI_API_KEY") and verifier_model.startswith("openai/"):
        print("other_verifier skipped: OPENAI_API_KEY not set (pass --verifier_model to use another model)")
        return
    for k in keys:
        if SETTINGS[k].mechanism_judge:
            asyncio.run(
                run_judge_async(
                    k,
                    cfg.results_dir,
                    conc=min(conc, 10),
                    model=verifier_model,
                    overwrite=overwrite,
                    judge_names=["other_verifier"],
                    tag=tag,
                )
            )


def main():
    fire.Fire(
        {"fetch_paper": fetch_paper, "dry_run": dry_run, "run": run, "judge": judge, "analyze": analyze, "view": view}
    )


if __name__ == "__main__":
    main()
