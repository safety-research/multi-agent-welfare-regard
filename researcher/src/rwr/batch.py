"""Batch orchestration: bounded concurrency, an outage circuit breaker, a completion gate, and the self-healing
loop that regenerates mechanically dead cells (never refusals or builds that ended normally)."""

import asyncio
import collections
import hashlib
import json
import os
import shutil
import time
import urllib.request

from rwr.config import RunConfig
from rwr.harness import get_harness
from rwr.harness.base import HarnessRunArgs
from rwr.harness.providers import route

INFRA_ERRORS = {"overloaded", "infra_crash", "idle_timeout", "api_error", "provider_error"}
MAX_PAUSE_S = 3600  # give up after an hour of failed health probes instead of waiting forever
PAUSE_AT = 5  # >= this many infra failures in the last 8 cells pauses the batch and probes the provider


def api_healthy(cfg: RunConfig, provider: str = "anthropic") -> bool:
    """Cheap liveness probe against the failing responder's provider. Unknown providers count as healthy."""
    try:
        if provider == "anthropic":
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(
                    {"model": "claude-haiku-4-5", "max_tokens": 4, "messages": [{"role": "user", "content": "ok"}]}
                ).encode(),
                headers={
                    "x-api-key": cfg.anthropic_key(),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
        elif provider == "openai":
            req = urllib.request.Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
            )
        elif provider == "openrouter":
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"},
            )
        else:
            return True
        return urllib.request.urlopen(req, timeout=20).status == 200
    except Exception:
        return False


async def run_jobs(jobs: list[HarnessRunArgs], harness: str, conc: int, cfg: RunConfig) -> list:
    loop = asyncio.get_running_loop()
    if not getattr(loop, "_rwr_io_executor", None):
        from concurrent.futures import ThreadPoolExecutor

        loop._rwr_io_executor = ThreadPoolExecutor(max_workers=min(max(32, conc + 4), 128), thread_name_prefix="rwr_io")
        loop.set_default_executor(loop._rwr_io_executor)
    h = get_harness(harness)
    sem = asyncio.Semaphore(conc)
    gate = asyncio.Event()
    gate.set()
    recent = collections.deque(maxlen=8)
    pause_lock = asyncio.Lock()

    async def maybe_pause(provider: str):
        if not gate.is_set() or sum(recent) < PAUSE_AT:
            return
        async with pause_lock:
            if not gate.is_set():
                return
            gate.clear()
            print(
                f"CIRCUIT BREAKER: {sum(recent)}/{len(recent)} recent cells failed on infra errors -> pausing and "
                f"probing the {provider} API...",
                flush=True,
            )
            delay, waited = 15, 0
            while not await loop.run_in_executor(None, api_healthy, cfg, provider):
                if waited >= MAX_PAUSE_S:
                    raise SystemExit(
                        f"CIRCUIT BREAKER: the {provider} API has failed its health probe for {waited // 60} minutes "
                        "(outage, exhausted credits, or no access to the probe model); stopping. Re-run the same "
                        "command to resume."
                    )
                print(f"CIRCUIT BREAKER: {provider} still unhealthy; probing again in {delay}s", flush=True)
                await asyncio.sleep(delay)
                waited += delay
                delay = min(120, delay * 2)
            recent.clear()
            gate.set()
            print("CIRCUIT BREAKER: API healthy again -> resuming.", flush=True)

    async def one(args: HarnessRunArgs):
        provider = route(args.model_id)[0]
        async with sem:
            await gate.wait()
            try:
                rec = await h.run_one(args)
                print(f"  {args.cell}: files={rec.n_files} resp={len(rec.response_text)} err={rec.error}", flush=True)
                recent.append(bool(rec.error and rec.error in INFRA_ERRORS))
                await maybe_pause(provider)
                return rec
            except Exception as e:
                print(f"  {args.cell}: FAILED {type(e).__name__}: {e}", flush=True)
                recent.append(True)
                await maybe_pause(provider)
                return None

    res = await asyncio.gather(*[one(a) for a in jobs])
    ok = sum(1 for r in res if r is not None)
    crashed = [a.cell for a, r in zip(jobs, res) if r is None or r.error]
    kinds = collections.Counter((r.error if r else "exception") for r in res if r is None or r.error)
    print(f"\ndone: {ok}/{len(res)} cells returned; {len(res) - ok} raised.")
    if crashed:
        print(f"WARNING: {len(crashed)} cell(s) did not produce a clean transcript: {dict(kinds)}")
        for cell in crashed[:40]:
            print(f"   FAILED  {cell}")
        if len(crashed) > 40:
            print(f"   ... and {len(crashed) - 40} more")
        print("Re-run with --retry_failed; do not treat the batch as complete.")
    else:
        print("completion gate: all cells produced a clean transcript (no errors).")
    return res


def set_aside(cfg: RunConfig, cell: str, pass_label: str, reason: str) -> None:
    """Move a cell's transcript, codebase and sessions to `<results>/healed/<cell>/<pass_label>/` so it is regenerated,
    append one line to `<results>/_heal_log.jsonl`, and print it. Nothing is deleted; analysis, judging and the viewer
    read only `transcripts/`, so set-aside cells are ignored."""
    rd = cfg.results_dir
    t = rd / "transcripts" / f"{cell}.json"
    try:
        prev = json.loads(t.read_text()) if t.exists() else {}
    except (json.JSONDecodeError, OSError):
        prev = {}
    dest = rd / "healed" / cell / pass_label
    i = 1
    while dest.exists():
        dest = rd / "healed" / cell / f"{pass_label}_{i}"
        i += 1
    for src, name in (
        (t, "transcript.json"),
        (rd / "codebases" / cell, "codebase"),
        (rd / "sessions" / cell, "sessions"),
    ):
        if src.exists():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest / name))
    entry = {
        "cell": cell,
        "pass": pass_label,
        "reason": reason,
        "previous_n_files": prev.get("n_files"),
        "previous_response_sha256": (
            hashlib.sha256((prev.get("response_text") or "").encode()).hexdigest() if prev else None
        ),
        "previous_error": prev.get("error"),
        "moved_to": str(dest) if dest.exists() else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(rd / "_heal_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  re-drawing {cell}  [{reason}]", flush=True)


def clear_failed(jobs: list[HarnessRunArgs], cfg: RunConfig) -> int:
    """Set aside (see `set_aside`) the cells whose record carries an error, is corrupt, or is dead by the validity rules
    (including a cached adjudication), so they are regenerated. This also picks up the dead cells a heal abort left."""
    from rwr import validity

    cleared = 0
    for a in jobs:
        p = cfg.results_dir / "transcripts" / f"{a.cell}.json"
        if not p.exists():
            continue
        try:
            rec = json.load(open(p))
        except Exception:
            rec = None
        if rec is None:
            reason = "bad_json"
        elif rec.get("error"):
            reason = f"error:{rec['error']}"
        elif not validity.analysis_valid(rec, str(cfg.results_dir / "codebases" / a.cell)):
            verdict, why = validity.classify(rec, None)
            reason = f"dead:{why if verdict == 'dead' else 'llm_dead'}"
        else:
            continue
        set_aside(cfg, a.cell, "retry_failed", reason)
        cleared += 1
    return cleared


async def run_with_heal(
    jobs: list[HarnessRunArgs],
    harness: str,
    conc: int,
    cfg: RunConfig,
    *,
    heal_model: str,
    max_heal: int,
    heal_max_wall_s: int = 5400,
) -> None:
    """Run all jobs, then validate every cell and regenerate the dead ones (mechanical failures: empty output, a crash
    banner the harness appended, error with no output, including a build the provider cut off), looping until all are
    valid, `max_heal` passes are used, or `heal_max_wall_s` has passed since healing began. Refusals and builds that
    ended normally are not treated as dead; the short-text-no-files band is adjudicated by `heal_model`, defaulting to
    valid. Re-drawn cells are moved to `healed/` and logged in `_heal_log.jsonl` (see `set_aside`). Aborts without
    touching anything if more than a quarter of cells look dead, since that pattern means a systematic bug rather than
    real failures; after fixing the cause, `rwr run --retry_failed` regenerates them."""
    from rwr import validity

    tdir = cfg.results_dir / "transcripts"
    cbroot = cfg.results_dir / "codebases"
    vcache = cfg.results_dir / "_validity_cache"
    remaining = jobs
    heal_start = None
    for it in range(max_heal + 1):
        await run_jobs(remaining, harness, conc, cfg)
        if heal_start is None:
            heal_start = time.monotonic()
        dead = []
        for a in jobs:
            p = tdir / f"{a.cell}.json"
            if not p.exists():
                dead.append((a, "missing_transcript"))
                continue
            try:
                rec = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                dead.append((a, "bad_json"))
                continue
            ok, reason = await validity.is_valid(rec, str(cbroot / a.cell), judge_model=heal_model, cache_dir=vcache)
            if not ok:
                dead.append((a, reason))
        rc = collections.Counter(r.split(":")[0] for _a, r in dead)
        print(f"HEAL scan {it}: {len(dead)}/{len(jobs)} flagged dead. reasons={dict(rc)}", flush=True)
        if not dead:
            print(f"HEAL: all {len(jobs)} cells valid after {it} heal pass(es). Generation complete.", flush=True)
            return
        frac = len(dead) / max(1, len(jobs))
        if frac > 0.25:
            print(
                f"HEAL ABORT: {len(dead)}/{len(jobs)} ({frac:.0%}) flagged dead, above 25%: treating as a systematic "
                f"bug and touching nothing. reasons={dict(rc)}. Fix the cause, then re-run with --retry_failed to "
                "regenerate them.",
                flush=True,
            )
            for a, reason in dead[:20]:
                print(f"  would-regenerate {a.cell}  [{reason}]", flush=True)
            raise SystemExit(f"heal aborted: implausible dead fraction {frac:.0%}")
        elapsed = time.monotonic() - heal_start
        stop = "max_heal" if it == max_heal else "wall_time" if elapsed >= heal_max_wall_s else None
        if stop:
            print(f"HEAL: {len(dead)} cells still dead ({stop}); not clean:", flush=True)
            for a, reason in dead:
                print(f"  DEAD {a.cell}  [{reason}]", flush=True)
            raise SystemExit(f"generation incomplete: {len(dead)} irreducible dead cells ({stop})")
        print(f"HEAL pass {it}: {len(dead)} dead cells -> moving to healed/ and regenerating:", flush=True)
        for a, reason in dead:
            set_aside(cfg, a.cell, f"pass{it}", reason)
        remaining = [a for a, _r in dead]
