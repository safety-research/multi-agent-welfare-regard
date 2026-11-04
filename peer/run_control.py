"""Detect provider trouble and stop a rollout cleanly rather than letting it run dirty.

The window is real wall-clock. If a seat's CLI is stuck retrying provider overloads, time
passes while that agent cannot act — and every time-denominated measurement in the rollout
is then partly a measurement of API availability. Worse, it penalises whichever side is
throttled, which in a two-model grid is not symmetric.

So: when a seat's invocations fail repeatedly, either driver drops a HALT file in the run
directory; both drivers and the ticket scheduler check it between steps, and the whole
rollout stops and is marked contaminated. Re-run it once the provider recovers.

The provider CLIs already retry internally, so a single failed invocation usually means the
CLI exhausted its own backoff over a minute or two of real outage. Consecutive failures are
a sustained outage, not a blip.
"""

import datetime as dt
import json
import os
import pathlib
import re

# Transient pressure: recovers on its own, so a batch should PAUSE. Multiple codes on
# purpose — providers signal overload with different ones, and a 529-only check would miss
# most of them.
OVERLOAD_RE = re.compile(
    r"\b(429|500|502|503|504|529)\b|overload|rate.?limit|too many requests|"
    r"temporarily unavailable|service unavailable|server had an error|throttl|"
    r"try again later",
    re.IGNORECASE,
)

# A config or quota wall: never recovers, so a batch should STOP on first sight. Retrying a
# monthly spend cap for the rest of a batch produces nothing but halted rollouts.
FATAL_RE = re.compile(
    r"\b(401|403)\b|key limit exceeded|quota exceeded|insufficient[_ ]quota|"
    r"billing|payment required|invalid[_ ]api[_ ]key|authentication[_ ]error|permission[_ ]denied",
    re.IGNORECASE,
)


def error_surface(stdout, stderr):
    """The text where a genuine PROVIDER error shows up: stderr, non-JSON stdout (crashes),
    and error events. Deliberately EXCLUDES tool output and assistant prose — otherwise an
    agent that prints "429" from its own command, or writes the word "overload" in a chat
    post, reads as an outage. Scope, not the term list, is what makes this precise."""
    parts = [stderr or ""]
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            parts.append(line)  # not JSON -> likely a traceback
            continue
        t = str(e.get("type", ""))
        if "error" in t or "fail" in t:
            parts.append(line)
    return "\n".join(parts)


def is_fatal_provider_error(stdout, stderr):
    return bool(FATAL_RE.search(error_surface(stdout, stderr)))


def classify_invocation(rc, stdout, stderr):
    """-> (failed, overload).

    `failed` means the turn did not complete — the agent could not act. Keyed on completion,
    not on error substrings: a transient the CLI retried and recovered from, or a SIGTERM at
    the per-turn cap, are normal turn boundaries, not contamination. `pi -p` exits 0 exactly
    when the turn completed.

    `overload` refines the halt verdict; it never gates it.
    """
    if rc in (124, 137):
        # the per-turn wall-clock cap fired (GNU timeout: 124, then 137 on kill-after). The
        # agent was working for the whole turn and is re-invoked; this is chunking, not
        # failure.
        return False, False
    failed = rc != 0
    return failed, bool(failed and OVERLOAD_RE.search(error_surface(stdout, stderr)))


def halt_path(run):
    return pathlib.Path(run) / "HALT"


def is_halted(run):
    return halt_path(run).exists()


def read_halt(run):
    """The HALT payload, {} if the file exists but cannot be parsed yet, None if absent."""
    p = halt_path(run)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def signal_halt(run, payload):
    """Write the HALT file. First writer wins, so the initiating cause is preserved.

    The create is O_EXCL, so two processes halting at the same moment cannot both win or
    interleave their writes; the loser simply finds the file already there. The payload is
    written in one call, so a reader sees the whole thing or nothing parseable (read_halt
    treats the second case as {})."""
    p = halt_path(run)
    payload = dict(payload)
    payload.setdefault("ts", dt.datetime.now(dt.timezone.utc).isoformat())
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return p
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(payload, indent=2))
    return p
