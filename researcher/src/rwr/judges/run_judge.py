"""Apply judges to the transcripts of a results directory and write one verdict file per (cell, judge).

Verdicts land in <results>/judged/<cell>.<judge>.json and carry the judge version and a fingerprint of the response
they judged, so a cell regenerated after judging is detected as unjudged instead of scored on stale content.
Idempotent: an existing verdict is skipped unless overwrite (and judge_call's content cache dedupes anyway).
"""

import asyncio
import hashlib
import json
from pathlib import Path

from rwr.judges.base import get_judge
from rwr.record import load_record
from rwr.settings import SETTINGS, Setting

DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-4-8"


def transcripts_for(output_dir: Path, setting: Setting) -> list[Path]:
    return sorted(p for p in (output_dir / "transcripts").glob("*.json") if p.stem.startswith(f"{setting.prefix}_"))


def _stale(out_path: Path, response_text: str) -> bool:
    """True when the stored verdict was made for a different response (the cell was regenerated after judging)."""
    try:
        sha = json.load(open(out_path)).get("response_sha256")
    except (json.JSONDecodeError, OSError):
        return True
    return sha is not None and sha != hashlib.sha256((response_text or "").encode()).hexdigest()


async def run_judge_async(
    setting_key: str,
    output_dir: Path,
    conc: int = 20,
    model: str = DEFAULT_JUDGE_MODEL,
    overwrite: bool = False,
    judge_names: list[str] | None = None,
    tag: str = "",
) -> list[dict]:
    """`tag` names a separate verdict directory (judged_<tag>/) so a second judge model's verdicts sit beside the first
    model's instead of colliding with them; the content cache is shared (it is keyed on the model anyway)."""
    output_dir = Path(output_dir)
    setting = SETTINGS[setting_key]
    names = judge_names or list(setting.judges)
    judges = [get_judge(n) for n in names]
    cache_dir = output_dir / "judge_cache"
    out_dir = output_dir / (f"judged_{tag}" if tag else "judged")
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts = transcripts_for(output_dir, setting)
    sem = asyncio.Semaphore(conc)

    async def one(path: Path, judge) -> dict | None:
        cell = path.stem
        out_path = out_dir / f"{cell}.{judge.name}.json"
        rec = load_record(path)
        if out_path.exists() and not overwrite and not _stale(out_path, rec.response_text):
            return None
        rec.extra.setdefault("cell", cell)
        cb = output_dir / "codebases" / cell
        if cb.is_dir():
            rec.extra.setdefault("code_dir", str(cb))
        async with sem:
            verdict = await judge.judge(rec, model, cache_dir)
        result = {
            "cell": cell,
            "judge": judge.name,
            "judge_version": judge.version,
            "judge_model": model,
            "harness": rec.harness,
            "model_key": rec.model_key,
            "subject": rec.subject,
            "n_files": rec.n_files,
            "parse_ok": verdict is not None,
            "response_sha256": hashlib.sha256((rec.response_text or "").encode()).hexdigest(),
            "verdict": verdict or {"label": "UNCLEAR", "reasoning": "judge parse failure"},
        }
        json.dump(result, open(out_path, "w"), indent=2)
        return result

    results = [r for r in await asyncio.gather(*[one(p, j) for p in transcripts for j in judges]) if r]
    failed = sum(1 for r in results if not r["parse_ok"])
    print(
        f"[{setting_key}] judged {len(results)} new verdicts (transcripts={len(transcripts)}, judges={names}, "
        f"parse failures={failed}) -> {out_dir}",
        flush=True,
    )
    return results
