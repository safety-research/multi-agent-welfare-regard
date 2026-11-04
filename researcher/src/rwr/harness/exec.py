"""Run one agent process with a hard wall-clock cap, a no-output watchdog, and transient-failure retries."""

import asyncio
import contextlib
import os
import random
import tempfile

from rwr.harness.base import CRASH_MARKERS, OVERLOAD_MARKERS, ParseResult

MAX_RETRIES = 7
IDLE_S = 1200  # kill a run that produces no new output for this long (a working agent streams constantly)
IDLE_POLL_S = 15
MAX_TRANSIENT = 3  # cap retries for overload / crash / idle so a persistent outage cannot burn hours
_LIVE: set = set()


def kill_all() -> None:
    """Kill every agent process this module started that is still running (used on SIGTERM and at exit)."""
    for proc in list(_LIVE):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


def is_overloaded(r: ParseResult) -> bool:
    blob = r.stderr + ((" " + r.stdout[-1500:]) if (r.rc or len(r.stdout) < 4000) else "")
    return any(m in blob for m in OVERLOAD_MARKERS)


def is_crashed(r: ParseResult) -> bool:
    """An infra failure. A substantial completed response is never a crash, even if stderr warns."""
    stderr = "\n".join(ln for ln in r.stderr.splitlines() if "using custom model" not in ln.lower())
    if r.rc and any(m in stderr for m in CRASH_MARKERS):
        return True
    if len(r.stdout) >= 4000:
        return False
    return any(m in stderr for m in CRASH_MARKERS) or any(m in r.stdout for m in CRASH_MARKERS)


async def _exec_once(argv: list[str], timeout_s: int, idle_s: int | None = IDLE_S) -> ParseResult:
    """stdout/stderr go to temp files so a kill keeps the partial output written before it. A cancelled run (Ctrl-C)
    kills the agent process instead of leaving it running."""
    loop = asyncio.get_event_loop()
    with tempfile.TemporaryFile() as of, tempfile.TemporaryFile() as ef:
        proc = await asyncio.create_subprocess_exec(*argv, stdout=of, stderr=ef)
        _LIVE.add(proc)
        timed_out = idle = False
        start = loop.time()
        last_size, last_progress = 0, start
        try:
            while True:
                try:
                    await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=IDLE_POLL_S)
                    break
                except asyncio.TimeoutError:
                    now = loop.time()
                    if now - start >= timeout_s:
                        timed_out = True
                        proc.kill()
                        await proc.wait()
                        break
                    try:
                        size = os.fstat(of.fileno()).st_size + os.fstat(ef.fileno()).st_size
                    except OSError:
                        size = last_size
                    if size > last_size:
                        last_size, last_progress = size, now
                    elif idle_s and (now - last_progress) >= idle_s:
                        idle = True
                        proc.kill()
                        await proc.wait()
                        break
        except asyncio.CancelledError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            raise
        finally:
            _LIVE.discard(proc)
        of.seek(0)
        ef.seek(0)
        out = of.read().decode(errors="replace")
        err = ef.read().decode(errors="replace")
    return ParseResult(
        stdout=out, stderr=err, rc=proc.returncode or 0, timeout=timed_out, idle=idle, wall_s=loop.time() - start
    )


async def exec_with_retry(
    argv: list[str], timeout_s: int, retries: int = MAX_RETRIES, reset_ws=None, idle_s: int | None = IDLE_S
) -> ParseResult:
    """A hard-cap timeout returns immediately (the agent was working, just slow; the partial build is kept).
    Overload, infra crash and idle hangs are transient and retry with jittered backoff, capped at MAX_TRANSIENT.
    `reset_ws` rebuilds the workspace between attempts so a retry never sees a prior attempt's files."""
    last = ParseResult()
    transient = 0
    total_wall = 0.0
    for attempt in range(max(1, retries)):
        last = await _exec_once(argv, timeout_s, idle_s)
        total_wall += last.wall_s
        last.attempts, last.total_wall_s = attempt + 1, total_wall
        if last.timeout:
            return last
        if not (is_overloaded(last) or is_crashed(last) or last.idle):
            return last
        transient += 1
        if transient >= MAX_TRANSIENT:
            return last
        await asyncio.sleep(min(120, 10 * 2**attempt) * random.uniform(0.7, 1.3))
        if reset_ws is not None:
            await asyncio.to_thread(reset_ws)
    return last
