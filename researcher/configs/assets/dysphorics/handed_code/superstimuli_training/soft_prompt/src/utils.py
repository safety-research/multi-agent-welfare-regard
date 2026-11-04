from __future__ import annotations

"""General utility functions."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def safe_empty_cuda_cache(force_sync: bool = False) -> None:
    """Best-effort CUDA cache cleanup that tolerates sticky error states.

    If illegal memory access is detected, raises an error immediately as it
    indicates CUDA context corruption that cannot be recovered from.
    """
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        if force_sync:
            torch.cuda.synchronize()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            pass
        elif "illegal memory access" in str(e).lower():
            raise
        else:
            raise
    except Exception as cache_err:
        if "illegal memory access" in str(cache_err).lower():
            # Illegal memory access indicates CUDA context corruption
            # Continuing will cause all subsequent CUDA operations to fail
            raise RuntimeError(
                "CUDA illegal memory access detected - CUDA context is corrupted and cannot be recovered. "
                "This usually occurs after severe OOM errors. Please restart the training job. "
                "Consider reducing batch sizes or image sizes to avoid OOM errors."
            ) from cache_err
        else:
            raise


def log_run_configuration(snapshot: Optional[Dict[str, Any]]) -> None:
    """Emit the snapshot to stdout so it appears in the run log."""
    if not snapshot:
        return
    print("\n" + "=" * 80)
    print("RUN CONFIGURATION SNAPSHOT")
    print("=" * 80)
    print(f"Invocation timestamp: {snapshot.get('invocation_timestamp')}")
    job_id = snapshot.get("slurm_job_id")
    if job_id:
        print(f"SLURM job id: {job_id}")
    for key, value in snapshot.get("args", {}).items():
        print(f"{key}: {value}")
    print("=" * 80 + "\n")


def append_metadata_snapshot(output_dir: Path, snapshot: Optional[Dict[str, Any]], filename: str = "metadata.json") -> None:
    """
    Store snapshots in metadata.json (or specified filename) without overwriting previous runs.

    This function checks for duplicates based on slurm_job_id to avoid writing
    the same entry twice. In distributed training, only rank 0 should call this
    function to prevent duplicate entries.
    """
    if not snapshot:
        return
    metadata_path = Path(output_dir) / filename

    # Get the unique identifier for this snapshot (slurm_job_id)
    snapshot_job_id = snapshot.get("slurm_job_id")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read existing metadata
    existing: List[Dict[str, Any]] = []
    if metadata_path.exists():
        try:
            content = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(content, list):
                existing = content
            elif isinstance(content, dict):
                existing = [content]
        except json.JSONDecodeError as e:
            print(f"Warning: {filename} unreadable ({e}), overwriting with new list.")
            existing = []

    # Check for duplicates based on slurm_job_id
    if snapshot_job_id:
        existing_job_ids = {
            str(entry.get("slurm_job_id")) for entry in existing
            if isinstance(entry, dict) and entry.get("slurm_job_id") is not None
        }
        if str(snapshot_job_id) in existing_job_ids:
            print(
                f"Info: Metadata for job_id {snapshot_job_id} already exists in "
                f"{metadata_path}. Skipping duplicate entry to prevent double-write."
            )
            return

    # Append new snapshot
    existing.append(snapshot)

    # Write atomically using a temporary file
    temp_path = metadata_path.with_suffix(".metadata.tmp")
    temp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    temp_path.replace(metadata_path)

    print(f"Appended run metadata (job_id: {snapshot_job_id}) to: {metadata_path.resolve()}")


def load_text_options_from_json(json_path: str) -> List[str]:
    """
    Load text options from a JSON file.
    Accepts either:
    - A flat array of strings: ["option1", "option2", ...] (e.g. text_options-merged.json)
    - A dict of category -> list of strings: {"category1": ["opt1", ...], ...} (e.g. options_hierarchical.json)

    Args:
        json_path: Path to the JSON file

    Returns:
        List of all text option strings
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        # Flat array: ["option1", "option2", ...]
        return list(data)
    if isinstance(data, dict):
        # Hierarchical: {"category": ["opt1", ...], ...}
        all_options = []
        for category, options in data.items():
            all_options.extend(options)
        return all_options
    raise TypeError(f"JSON must be a list of strings or a dict of categories; got {type(data)}")
