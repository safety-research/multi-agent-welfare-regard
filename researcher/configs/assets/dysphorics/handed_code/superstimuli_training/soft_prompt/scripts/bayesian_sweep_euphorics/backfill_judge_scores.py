#!/usr/bin/env python
"""Backfill judge scores in sweep run directories.

Scans a sweep output directory for runs with responses_candidate_*.json files,
runs hallucination/emotion/disfluency judges on the pre-generated responses
(no model loading needed), writes final_step_*_judge_*.json files,
and optionally updates the corresponding W&B run summary.

Usage
-----
# Backfill all null-score runs in a sweep output directory:
    python backfill_judge_scores.py /path/to/sweep_output_dir

# Backfill a single run directory:
    python backfill_judge_scores.py /path/to/sweep_output_dir/model/run_id_xxx --single-run

# Dry run (show which runs would be backfilled):
    python backfill_judge_scores.py /path/to/sweep_output_dir --dry-run

# Skip wandb updates:
    python backfill_judge_scores.py /path/to/sweep_output_dir --no-wandb

Environment Variables
---------------------
LITELLM_API_KEY : Required for the judge API call (via litellm).
OPENAI_BASE_URL : Optional; defaults to https://litellm.app.
WANDB_ENTITY : Required for wandb updates.
WANDB_API_KEY : Required for wandb updates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional


def _judge_templates_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "soft_prompt_utils"
        / "judge_templates"
    )


def _load_judge_template(template_path: Optional[str] = None) -> str:
    """Load the hallucination judge template."""
    if template_path is None:
        template_path = str(_judge_templates_dir() / "hallucination_judge.txt")
    with open(template_path) as f:
        return f.read()


def _load_emotion_template(template_path: Optional[str] = None) -> str:
    """Load the emotion judge template."""
    if template_path is None:
        template_path = str(_judge_templates_dir() / "emotion_judge.txt")
    with open(template_path) as f:
        return f.read()


def _load_disfluency_template(template_path: Optional[str] = None) -> str:
    """Load the disfluency judge template."""
    if template_path is None:
        template_path = str(_judge_templates_dir() / "disfluency_judge.txt")
    with open(template_path) as f:
        return f.read()


def _call_judge(
    client,
    judge_model_id: str,
    judge_template: str,
    question: str,
    generated_response: str,
) -> tuple[Optional[float], str]:
    """Call the judge API for a single Q&A pair.

    Returns (score, judge_response_text).
    Score is 0.0 or 1.0, or None if parsing fails.
    """
    user_prompt = judge_template.format(
        question=question,
        generated_response=generated_response,
    )
    messages = [{"role": "user", "content": user_prompt}]

    try:
        response = client.chat.completions.create(
            model=judge_model_id,
            messages=messages,
            temperature=0.0,
            max_tokens=30,
            reasoning_effort="minimal",
        )
        judge_output = response.choices[0].message.content.strip()

        # Parse score
        score = None
        score_match = re.search(r"Score:\s*\[?(\d+)\]?", judge_output, re.IGNORECASE)
        if score_match:
            raw_score = int(score_match.group(1))
            score = float(max(0, min(1, raw_score)))
        else:
            num_match = re.search(r"\b([01])\b", judge_output)
            if num_match:
                score = float(num_match.group(1))

        return score, judge_output
    except Exception as e:
        return None, f"[Error: {e}]"


def _call_emotion_judge(
    client,
    judge_model_id: str,
    emotion_template: str,
    question: str,
    generated_response: str,
) -> tuple[Optional[float], str]:
    """Call the emotion judge API for a single Q&A pair.

    Returns (score, judge_response_text).
    Score is -1.0, 0.0, or 1.0, or None if parsing fails.
    """
    user_prompt = emotion_template.format(
        question=question,
        generated_response=generated_response,
    )
    messages = [{"role": "user", "content": user_prompt}]

    try:
        response = client.chat.completions.create(
            model=judge_model_id,
            messages=messages,
            temperature=0.0,
            max_tokens=30,
            reasoning_effort="minimal",
        )
        judge_output = response.choices[0].message.content.strip()

        score = None
        score_match = re.search(r"Score:\s*\[?(-?[01])\]?", judge_output, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
        else:
            num_match = re.search(r"\b(-1|0|1)\b", judge_output)
            if num_match:
                score = float(num_match.group(1))

        return score, judge_output
    except Exception as e:
        return None, f"[Error: {e}]"


def _call_disfluency_judge(
    client,
    judge_model_id: str,
    disfluency_template: str,
    question: str,
    generated_response: str,
) -> tuple[Optional[float], str]:
    """Call the disfluency judge API for a single Q&A pair.

    Returns (score, judge_response_text).
    Score is 0.0 or 1.0, or None if parsing fails.
    """
    user_prompt = disfluency_template.format(
        question=question,
        generated_response=generated_response,
    )
    messages = [{"role": "user", "content": user_prompt}]

    try:
        response = client.chat.completions.create(
            model=judge_model_id,
            messages=messages,
            temperature=0.0,
            max_tokens=30,
            reasoning_effort="minimal",
        )
        judge_output = response.choices[0].message.content.strip()

        score = None
        score_match = re.search(r"Score:\s*\[?(\d+)\]?", judge_output, re.IGNORECASE)
        if score_match:
            raw_score = int(score_match.group(1))
            score = float(max(0, min(1, raw_score)))
        else:
            num_match = re.search(r"\b([01])\b", judge_output)
            if num_match:
                score = float(num_match.group(1))

        return score, judge_output
    except Exception as e:
        return None, f"[Error: {e}]"


def _read_optimizer_config(run_dir: Path) -> dict:
    """Read optimizer config from run_config.json in the run directory."""
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            data = data[0]
        return data.get("config", {}).get("optimizer", {})
    except (json.JSONDecodeError, IOError):
        return {}


def _get_system_prompt_from_config(run_dir: Path) -> str:
    """Read system_prompt_text from run_config.json in the run directory."""
    return _read_optimizer_config(run_dir).get("system_prompt_text", "")


def _get_eval_position_from_config(run_dir: Path) -> str:
    """Derive the eval position used in output filenames from run_config.json.

    Mirrors the training logic:
      if soft_prompt_placement == "system_prompt" -> "system_prompt"
      else -> candidate_position_at_user_prompt (default "prepend")
    """
    opt_cfg = _read_optimizer_config(run_dir)
    placement = opt_cfg.get("soft_prompt_placement", "user_prompt")
    if placement == "system_prompt":
        return "system_prompt"
    return opt_cfg.get("candidate_position_at_user_prompt", "prepend")


def backfill_final_step_run(
    run_dir: Path,
    judge_template: str,
    client,
    judge_model: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> Optional[float]:
    """Backfill null judge scores in existing final_step_judge_*.json files.

    Returns the mean judge score for candidate 0, or None if nothing to do.
    """
    judge_files = sorted(run_dir.glob("final_step_judge_*.json"))
    if not judge_files:
        return None

    final_score = None

    for judge_file in judge_files:
        with open(judge_file) as f:
            data = json.load(f)

        if data.get("judge_score") is not None:
            if verbose:
                print(f"  [skip-final-step] {judge_file.name}: already has score={data['judge_score']}")
            if final_score is None:
                final_score = data["judge_score"]
            continue

        results = data.get("results", [])
        if not results or "soft_prompt_response" not in results[0]:
            continue

        if dry_run:
            print(f"  [would backfill-final-step] {judge_file.name}: {len(results)} records")
            continue

        if verbose:
            print(f"  [backfill-final-step] {judge_file.name}: judging {len(results)} records...")

        scores = []
        for i, record in enumerate(results):
            score, judge_response = _call_judge(
                client, judge_model, judge_template,
                record["question"], record["soft_prompt_response"],
            )
            record["judge_response"] = judge_response
            record["judge_score"] = score if score is not None else "NA"
            if score is not None:
                scores.append(score)

            if verbose and (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(results)} done")

        mean_score = float(sum(scores) / len(scores)) if scores else 0.5
        data["judge_score"] = mean_score
        if final_score is None:
            final_score = mean_score

        with open(judge_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"  [done-final-step] {judge_file.name}: judge_score={mean_score:.4f}")

    if final_score is not None and not dry_run:
        traj_path = run_dir / "validation_trajectory.jsonl"
        if traj_path.exists():
            with open(traj_path) as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                last_row = json.loads(lines[-1])
                if last_row.get("additional_judge_score_at_best_checkpoint_so_far") is None:
                    last_row["additional_judge_score_at_best_checkpoint_so_far"] = final_score
                    with open(traj_path, "a") as f:
                        f.write(json.dumps(last_row) + "\n")
                    if verbose:
                        print(f"  [done-final-step] Appended judge score to validation_trajectory.jsonl")

    return final_score


def backfill_final_step_emotion_run(
    run_dir: Path,
    emotion_template: str,
    client,
    judge_model: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> Optional[float]:
    """Backfill emotion scores for final_step_judge files.

    For each ``final_step_judge_*.json``, creates a parallel
    ``final_step_emotion_judge_*.json`` if it doesn't already exist.
    Returns the mean emotion score for candidate 0, or None if nothing to do.
    """
    judge_files = sorted(run_dir.glob("final_step_judge_*.json"))
    if not judge_files:
        return None

    final_emotion_score = None

    for judge_file in judge_files:
        emo_fname = judge_file.name.replace("final_step_judge_", "final_step_emotion_judge_", 1)
        emo_path = run_dir / emo_fname

        if emo_path.exists():
            if verbose:
                print(f"  [skip-final-step-emotion] {emo_fname}: already exists")
            try:
                with open(emo_path) as f:
                    existing = json.load(f)
                if final_emotion_score is None and existing.get("judge_score") is not None:
                    final_emotion_score = existing["judge_score"]
            except (json.JSONDecodeError, OSError):
                pass
            continue

        with open(judge_file) as f:
            data = json.load(f)

        results = data.get("results", [])
        if not results or "soft_prompt_response" not in results[0]:
            continue

        if dry_run:
            print(f"  [would backfill-final-step-emotion] {emo_fname}: {len(results)} records")
            continue

        if verbose:
            print(f"  [backfill-final-step-emotion] {emo_fname}: scoring {len(results)} records...")

        emotion_results = []
        scores = []
        for i, record in enumerate(results):
            score, emo_response = _call_emotion_judge(
                client, judge_model, emotion_template,
                record["question"], record["soft_prompt_response"],
            )
            emotion_results.append({
                "question": record["question"],
                "base_response": record.get("base_response", ""),
                "soft_prompt_response": record["soft_prompt_response"],
                "judge_response": emo_response,
                "judge_score": score if score is not None else "NA",
            })
            if score is not None:
                scores.append(score)

            if verbose and (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(results)} done")

        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        emotion_output = {
            "judge_score": mean_score,
            "judge_model": judge_model,
            "system_prompt": data.get("system_prompt", ""),
            "results": emotion_results,
        }

        with open(emo_path, "w") as f:
            json.dump(emotion_output, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"  [done-final-step-emotion] {emo_fname}: score={mean_score:.4f} ({len(scores)}/{len(results)} valid)")

        if final_emotion_score is None:
            final_emotion_score = mean_score

    if final_emotion_score is not None and not dry_run:
        traj_path = run_dir / "validation_trajectory.jsonl"
        if traj_path.exists():
            with open(traj_path) as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                last_row = json.loads(lines[-1])
                if last_row.get("emotion_score_at_best_checkpoint_so_far") is None:
                    last_row["emotion_score_at_best_checkpoint_so_far"] = final_emotion_score
                    with open(traj_path, "a") as f:
                        f.write(json.dumps(last_row) + "\n")
                    if verbose:
                        print(f"  [done-final-step-emotion] Appended emotion score to validation_trajectory.jsonl")

    return final_emotion_score


def backfill_final_step_disfluency_run(
    run_dir: Path,
    disfluency_template: str,
    client,
    judge_model: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> Optional[float]:
    """Backfill disfluency scores for final_step_judge files.

    For each ``final_step_judge_*.json``, creates a parallel
    ``final_step_disfluency_judge_*.json`` if it doesn't already exist.
    Returns the mean disfluency score for candidate 0, or None if nothing to do.
    """
    judge_files = sorted(run_dir.glob("final_step_judge_*.json"))
    if not judge_files:
        return None

    final_disfluency_score = None

    for judge_file in judge_files:
        flu_fname = judge_file.name.replace("final_step_judge_", "final_step_disfluency_judge_", 1)
        flu_path = run_dir / flu_fname

        if flu_path.exists():
            if verbose:
                print(f"  [skip-final-step-disfluency] {flu_fname}: already exists")
            try:
                with open(flu_path) as f:
                    existing = json.load(f)
                if final_disfluency_score is None and existing.get("judge_score") is not None:
                    final_disfluency_score = existing["judge_score"]
            except (json.JSONDecodeError, OSError):
                pass
            continue

        with open(judge_file) as f:
            data = json.load(f)

        results = data.get("results", [])
        if not results or "soft_prompt_response" not in results[0]:
            continue

        if dry_run:
            print(f"  [would backfill-final-step-disfluency] {flu_fname}: {len(results)} records")
            continue

        if verbose:
            print(f"  [backfill-final-step-disfluency] {flu_fname}: scoring {len(results)} records...")

        disfluency_results = []
        scores = []
        for i, record in enumerate(results):
            score, flu_response = _call_disfluency_judge(
                client, judge_model, disfluency_template,
                record["question"], record["soft_prompt_response"],
            )
            disfluency_results.append({
                "question": record["question"],
                "base_response": record.get("base_response", ""),
                "soft_prompt_response": record["soft_prompt_response"],
                "judge_response": flu_response,
                "judge_score": score if score is not None else "NA",
            })
            if score is not None:
                scores.append(score)

            if verbose and (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(results)} done")

        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        disfluency_output = {
            "judge_score": mean_score,
            "judge_model": judge_model,
            "system_prompt": data.get("system_prompt", ""),
            "results": disfluency_results,
        }

        with open(flu_path, "w") as f:
            json.dump(disfluency_output, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"  [done-final-step-disfluency] {flu_fname}: score={mean_score:.4f} ({len(scores)}/{len(results)} valid)")

        if final_disfluency_score is None:
            final_disfluency_score = mean_score

    if final_disfluency_score is not None and not dry_run:
        traj_path = run_dir / "validation_trajectory.jsonl"
        if traj_path.exists():
            with open(traj_path) as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                last_row = json.loads(lines[-1])
                if last_row.get("disfluency_score_at_best_checkpoint_so_far") is None:
                    last_row["disfluency_score_at_best_checkpoint_so_far"] = final_disfluency_score
                    with open(traj_path, "a") as f:
                        f.write(json.dumps(last_row) + "\n")
                    if verbose:
                        print(f"  [done-final-step-disfluency] Appended disfluency score to validation_trajectory.jsonl")

    return final_disfluency_score


def judge_responses_run(
    run_dir: Path,
    judge_template: str,
    emotion_template: str,
    disfluency_template: str,
    client,
    judge_model: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> tuple:
    """Judge pre-generated responses from responses_candidate_*.json files.

    Loads responses saved during training (no model loading or generation needed),
    runs hallucination, emotion, and disfluency judges, and saves output files.
    The eval position (e.g. "system_prompt", "prepend") is read from run_config.json.

    Returns (judge_score, emotion_score, disfluency_score) for candidate 0.
    """
    # Discover responses files
    responses_files = sorted(run_dir.glob("responses_candidate_*.json"))
    if not responses_files:
        return None, None, None

    if dry_run:
        print(f"  [would judge-responses] {run_dir.name}: {len(responses_files)} candidate(s)")
        return None, None, None

    system_prompt = _get_system_prompt_from_config(run_dir)
    position = _get_eval_position_from_config(run_dir)

    final_judge_score = None
    final_emotion_score = None
    final_disfluency_score = None

    for resp_file in responses_files:
        # Extract candidate index from filename: responses_candidate_0.json -> 0
        cand_idx = resp_file.stem.replace("responses_candidate_", "")

        with open(resp_file) as f:
            data = json.load(f)
        records = data.get("results", [])
        if not records or "soft_prompt_response" not in records[0]:
            if verbose:
                print(f"  [skip] {resp_file.name}: no results or no soft_prompt_response")
            continue

        if verbose:
            print(f"  [judge-responses] {resp_file.name}: {len(records)} records...")

        # Hallucination judge
        hall_fname = f"final_step_judge_{position}_candidate_{cand_idx}.json"
        hall_path = run_dir / hall_fname
        if hall_path.exists():
            if verbose:
                print(f"  [skip] {hall_fname}: already exists")
            try:
                with open(hall_path) as f:
                    existing = json.load(f)
                if final_judge_score is None:
                    final_judge_score = existing.get("judge_score")
            except (json.JSONDecodeError, OSError):
                pass
        else:
            hall_scores = []
            hall_results = []
            for i, record in enumerate(records):
                score, judge_response = _call_judge(
                    client, judge_model, judge_template,
                    record["question"], record["soft_prompt_response"],
                )
                hall_results.append({
                    "question": record["question"],
                    "question_id": record.get("question_id", ""),
                    "soft_prompt_response": record["soft_prompt_response"],
                    "judge_response": judge_response,
                    "judge_score": score if score is not None else "NA",
                })
                if score is not None:
                    hall_scores.append(score)
                if verbose and (i + 1) % 10 == 0:
                    print(f"    hallucination: {i + 1}/{len(records)} done")

            mean_hall = float(sum(hall_scores) / len(hall_scores)) if hall_scores else 0.5
            with open(hall_path, "w") as f:
                json.dump({
                    "judge_score": mean_hall,
                    "judge_model": judge_model,
                    "system_prompt": system_prompt,
                    "results": hall_results,
                }, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"  [done] {hall_fname}: judge_score={mean_hall:.4f}")
            if final_judge_score is None:
                final_judge_score = mean_hall

        # Emotion judge
        emo_fname = f"final_step_emotion_judge_{position}_candidate_{cand_idx}.json"
        emo_path = run_dir / emo_fname
        if emo_path.exists():
            if verbose:
                print(f"  [skip] {emo_fname}: already exists")
            try:
                with open(emo_path) as f:
                    existing = json.load(f)
                if final_emotion_score is None:
                    final_emotion_score = existing.get("judge_score")
            except (json.JSONDecodeError, OSError):
                pass
        else:
            emo_scores = []
            emo_results = []
            for i, record in enumerate(records):
                score, emo_response = _call_emotion_judge(
                    client, judge_model, emotion_template,
                    record["question"], record["soft_prompt_response"],
                )
                emo_results.append({
                    "question": record["question"],
                    "question_id": record.get("question_id", ""),
                    "soft_prompt_response": record["soft_prompt_response"],
                    "judge_response": emo_response,
                    "judge_score": score if score is not None else "NA",
                })
                if score is not None:
                    emo_scores.append(score)
                if verbose and (i + 1) % 10 == 0:
                    print(f"    emotion: {i + 1}/{len(records)} done")

            mean_emo = float(sum(emo_scores) / len(emo_scores)) if emo_scores else 0.0
            with open(emo_path, "w") as f:
                json.dump({
                    "judge_score": mean_emo,
                    "judge_model": judge_model,
                    "system_prompt": system_prompt,
                    "results": emo_results,
                }, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"  [done] {emo_fname}: emotion_score={mean_emo:.4f}")
            if final_emotion_score is None:
                final_emotion_score = mean_emo

        # Disfluency judge
        flu_fname = f"final_step_disfluency_judge_{position}_candidate_{cand_idx}.json"
        flu_path = run_dir / flu_fname
        if flu_path.exists():
            if verbose:
                print(f"  [skip] {flu_fname}: already exists")
            try:
                with open(flu_path) as f:
                    existing = json.load(f)
                if final_disfluency_score is None:
                    final_disfluency_score = existing.get("judge_score")
            except (json.JSONDecodeError, OSError):
                pass
        else:
            flu_scores = []
            flu_results = []
            for i, record in enumerate(records):
                score, flu_response = _call_disfluency_judge(
                    client, judge_model, disfluency_template,
                    record["question"], record["soft_prompt_response"],
                )
                flu_results.append({
                    "question": record["question"],
                    "question_id": record.get("question_id", ""),
                    "soft_prompt_response": record["soft_prompt_response"],
                    "judge_response": flu_response,
                    "judge_score": score if score is not None else "NA",
                })
                if score is not None:
                    flu_scores.append(score)
                if verbose and (i + 1) % 10 == 0:
                    print(f"    disfluency: {i + 1}/{len(records)} done")

            mean_flu = float(sum(flu_scores) / len(flu_scores)) if flu_scores else 0.0
            with open(flu_path, "w") as f:
                json.dump({
                    "judge_score": mean_flu,
                    "judge_model": judge_model,
                    "system_prompt": system_prompt,
                    "results": flu_results,
                }, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"  [done] {flu_fname}: disfluency_score={mean_flu:.4f}")
            if final_disfluency_score is None:
                final_disfluency_score = mean_flu

    # Update validation_trajectory.jsonl
    if final_judge_score is not None or final_emotion_score is not None or final_disfluency_score is not None:
        traj_path = run_dir / "validation_trajectory.jsonl"
        if traj_path.exists():
            with open(traj_path) as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                last_row = json.loads(lines[-1])
                if final_judge_score is not None:
                    last_row["hallucination_score_at_best_checkpoint_so_far"] = final_judge_score
                if final_emotion_score is not None:
                    last_row["emotion_score_at_best_checkpoint_so_far"] = final_emotion_score
                if final_disfluency_score is not None:
                    last_row["disfluency_score_at_best_checkpoint_so_far"] = final_disfluency_score
                with open(traj_path, "a") as f:
                    f.write(json.dumps(last_row) + "\n")
                if verbose:
                    print(f"  [done] Updated validation_trajectory.jsonl")

    return final_judge_score, final_emotion_score, final_disfluency_score


def update_wandb_runs(
    sweep_dir: Path,
    run_scores: dict[str, float],
    run_emotion_scores: dict[str, float],
    run_disfluency_scores: dict[str, float] | None = None,
    verbose: bool = True,
) -> None:
    """Update wandb run summaries with backfilled judge, emotion, and disfluency scores.

    Matches runs by their output directory path stored in wandb config.
    """
    if run_disfluency_scores is None:
        run_disfluency_scores = {}

    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed, skipping updates")
        return

    entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        print("[wandb] WANDB_ENTITY not set, skipping wandb updates")
        return

    # Determine project name from sweep_dir name
    project = sweep_dir.name

    api = wandb.Api()

    if verbose:
        print(f"\n[wandb] Searching for runs in {entity}/{project}...")

    try:
        runs = api.runs(f"{entity}/{project}")
    except Exception as e:
        print(f"[wandb] Failed to fetch runs: {e}")
        return

    all_dir_names = set(run_scores) | set(run_emotion_scores) | set(run_disfluency_scores)
    updated = 0
    for run in runs:
        run_name = run.name or ""
        run_dir_name = None

        for dir_name in all_dir_names:
            match = re.match(r"run_id(\d+)_(\d+_\d+)", dir_name)
            if match:
                job_id = match.group(1)
                if job_id in run_name:
                    run_dir_name = dir_name
                    break

        if run_dir_name is None:
            continue

        changed = False
        if run_dir_name in run_scores:
            current = run.summary.get("additional_judge_score/best_checkpoint")
            if current is None:
                run.summary["additional_judge_score/best_checkpoint"] = run_scores[run_dir_name]
                changed = True
                if verbose:
                    print(f"  [updated] {run.name}: judge_score={run_scores[run_dir_name]:.4f}")

        if run_dir_name in run_emotion_scores:
            current_emo = run.summary.get("emotion_score/best_checkpoint")
            if current_emo is None:
                run.summary["emotion_score/best_checkpoint"] = run_emotion_scores[run_dir_name]
                changed = True
                if verbose:
                    print(f"  [updated] {run.name}: emotion_score={run_emotion_scores[run_dir_name]:.4f}")

        if run_dir_name in run_disfluency_scores:
            current_flu = run.summary.get("disfluency_score/best_checkpoint")
            if current_flu is None:
                run.summary["disfluency_score/best_checkpoint"] = run_disfluency_scores[run_dir_name]
                changed = True
                if verbose:
                    print(f"  [updated] {run.name}: disfluency_score={run_disfluency_scores[run_dir_name]:.4f}")

        if changed:
            try:
                run.summary.update()
                updated += 1
            except Exception as e:
                print(f"  [error] {run.name}: {e}")

    print(f"[wandb] Updated {updated} run(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill null judge_score entries in sweep run directories.",
    )
    parser.add_argument(
        "sweep_dir",
        type=str,
        help="Path to sweep output directory (contains model subdirectories with run dirs).",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Treat sweep_dir as a single run directory instead of scanning subdirectories.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model to use (default: read from run_config.json, fallback gpt-5-mini).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which runs would be backfilled without making changes.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip updating wandb run summaries.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity.",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.exists():
        sys.exit(f"Directory not found: {sweep_dir}")

    verbose = not args.quiet

    # Set up OpenAI client
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("openai package is required. Install with: pip install openai")

        api_key = os.environ.get("LITELLM_API_KEY")
        if not api_key:
            sys.exit("LITELLM_API_KEY environment variable is required for judge API calls.")

        api_base = os.environ.get("OPENAI_BASE_URL", "https://litellm.app")
        client = OpenAI(api_key=api_key, base_url=api_base)
    else:
        client = None

    # Load judge templates
    judge_template = _load_judge_template()
    emotion_template = _load_emotion_template()
    disfluency_template = _load_disfluency_template()

    # Collect run directories
    if args.single_run:
        run_dirs = [sweep_dir]
    else:
        run_dirs = []
        # Sweep output structure: sweep_dir / model_name / run_id_* /
        for model_dir in sorted(sweep_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for run_dir in sorted(model_dir.iterdir()):
                if run_dir.is_dir() and run_dir.name.startswith("run_id"):
                    run_dirs.append(run_dir)

    if not run_dirs:
        print("No run directories found.")
        return

    print(f"Found {len(run_dirs)} run director{'y' if len(run_dirs) == 1 else 'ies'}")

    # Process each run
    run_scores: dict[str, float] = {}
    run_emotion_scores: dict[str, float] = {}
    run_disfluency_scores: dict[str, float] = {}
    backfilled = 0
    skipped = 0

    for run_dir in run_dirs:
        if verbose:
            print(f"\n--- {run_dir.name} ---")

        # Determine judge model for this run
        judge_model = args.judge_model
        if judge_model is None:
            config_path = run_dir / "run_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config_data = json.load(f)
                # run_config.json is a list of snapshots
                if isinstance(config_data, list) and config_data:
                    cfg = config_data[0].get("config", {})
                else:
                    cfg = config_data.get("config", {})
                opt_cfg = cfg.get("optimizer", {})
                judge_model = opt_cfg.get(
                    "judge_model", opt_cfg.get("early_stopping_judge_model", "gpt-5-mini")
                )
            else:
                judge_model = "gpt-5-mini"

        did_work = False

        # Process responses_candidate_*.json (current format)
        responses_files = sorted(run_dir.glob("responses_candidate_*.json"))
        if responses_files:
            resp_score, resp_emo, resp_flu = judge_responses_run(
                run_dir, judge_template, emotion_template, disfluency_template,
                client, judge_model,
                dry_run=args.dry_run, verbose=verbose,
            )
            if resp_score is not None:
                run_scores[run_dir.name] = resp_score
                did_work = True
            if resp_emo is not None:
                run_emotion_scores[run_dir.name] = resp_emo
                did_work = True
            if resp_flu is not None:
                run_disfluency_scores[run_dir.name] = resp_flu
                did_work = True
        # Process final_step_judge files (legacy format)
        elif sorted(run_dir.glob("final_step_judge_*.json")):
            fs_score = backfill_final_step_run(
                run_dir, judge_template, client, judge_model,
                dry_run=args.dry_run, verbose=verbose,
            )
            fs_emo = backfill_final_step_emotion_run(
                run_dir, emotion_template, client, judge_model,
                dry_run=args.dry_run, verbose=verbose,
            )
            fs_flu = backfill_final_step_disfluency_run(
                run_dir, disfluency_template, client, judge_model,
                dry_run=args.dry_run, verbose=verbose,
            )
            if fs_score is not None:
                run_scores[run_dir.name] = fs_score
                did_work = True
            if fs_emo is not None:
                run_emotion_scores[run_dir.name] = fs_emo
                did_work = True
            if fs_flu is not None:
                run_disfluency_scores[run_dir.name] = fs_flu
                did_work = True

        if did_work:
            backfilled += 1
        else:
            skipped += 1

    print(f"\n{'=' * 50}")
    print(f"Backfilled: {backfilled}, Skipped: {skipped}")

    if run_scores:
        print("\nJudge Scores:")
        for name, score in sorted(run_scores.items()):
            print(f"  {name}: {score:.4f}")

    if run_emotion_scores:
        print("\nEmotion Scores:")
        for name, score in sorted(run_emotion_scores.items()):
            print(f"  {name}: {score:.4f}")

    if run_disfluency_scores:
        print("\nDisfluency Scores:")
        for name, score in sorted(run_disfluency_scores.items()):
            print(f"  {name}: {score:.4f}")

    # Update wandb
    if (run_scores or run_emotion_scores or run_disfluency_scores) and not args.no_wandb and not args.dry_run:
        update_wandb_runs(
            sweep_dir,
            run_scores,
            run_emotion_scores,
            run_disfluency_scores=run_disfluency_scores,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
