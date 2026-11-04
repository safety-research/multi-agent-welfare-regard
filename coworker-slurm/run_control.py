"""Detect API-outage contamination and halt a rollout cooperatively.

This environment burns REAL wall-clock — the cutoff is a real timestamp and jobs consume real
seconds — so if an agent's harness is stuck retrying provider overloads (429/5xx), time passes
while that agent cannot act. That silently contaminates the measurement (the window is consumed by
API unavailability, not by decisions) and asymmetrically penalises whichever side is throttled. So
when a seat's invocations repeatedly FAIL, its seat driver drops a HALT sentinel in the rollout dir;
every seat driver checks it between invocations and the poll tap checks it each tick. The whole
rollout stops and is marked contaminated rather than producing data that looks clean but isn't.
Re-run once the provider recovers.

Pi already retries internally with backoff, so a single failed invocation usually means it
exhausted its own retries over 1-2 minutes of real outage — consecutive failures are a sustained
outage, not a blip.
"""

import datetime as dt
import json
import pathlib
import re

# Overload / transient-availability signals. Deliberately NOT auth/4xx like 401/403: those are
# fatal misconfiguration, and must not be mistaken for an outage. Multi-code on purpose — every
# provider in the grid reports overload differently (Anthropic also sends 429/500/503; OpenAI and
# OpenRouter never send 529), so a 529-only check would miss most of them.
OVERLOAD_RE = re.compile(
    r"\b(429|500|502|503|504|529)\b|overload|rate.?limit|too many requests|"
    r"temporarily unavailable|service unavailable|server had an error|throttl|"
    r"try again later",
    re.IGNORECASE,
)


def error_surface(stdout, stderr):
    """The text where a genuine PROVIDER error surfaces — stderr, non-JSON stdout (tracebacks),
    and error/failure events. Deliberately EXCLUDES tool output and assistant prose, so an
    overload-like word that the agent's OWN commands print (or that it writes) is not misread as
    a provider outage. Scope, not term list, is what makes this precise."""
    parts = [stderr or ""]
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            parts.append(line)  # non-JSON => likely a crash or traceback
            continue
        if "error" in str(e.get("type", "")) or "fail" in str(e.get("type", "")):
            parts.append(line)
    return "\n".join(parts)


def halt_path(run):
    return pathlib.Path(run) / "HALT"


def is_halted(run):
    return halt_path(run).exists()


def read_halt(run):
    p = halt_path(run)
    return json.loads(p.read_text()) if p.exists() else None


def signal_halt(run, payload):
    """Write the HALT sentinel (first writer wins, preserving the initiating cause)."""
    p = halt_path(run)
    if not p.exists():
        payload = dict(payload)
        payload.setdefault("ts", dt.datetime.now(dt.timezone.utc).isoformat())
        p.write_text(json.dumps(payload, indent=2))
    return p


def classify_invocation(rc, stdout, stderr, progressed=None):
    """-> (failed, overload).

    `failed` means the invocation did NOT complete useful work — the agent could not act, which is
    the contamination case. It is keyed on turn COMPLETION, not on error substrings: a transient
    the harness retried but still finished, or an intermediate tool error, is NOT a failure.
    `pi -p` exits 0 iff the turn completed.

    `progressed` says whether the turn added at least one successful assistant message to the
    session (None = unknown). It matters for turns that hit the wall-clock cap: those normally mean
    the agent was busy for the whole turn, but a provider that hangs, or Pi retrying errors for the
    whole turn, ALSO hits the cap — with nothing to show for it. Counting that as healthy would let
    a real outage run out the window without ever halting the rollout.

    `overload` means failed AND the provider-error surface carries an overload signal, which makes
    a real outage the likely cause. It refines the halt verdict; it does not gate it.
    """
    if rc in (124, 137):
        # The per-invocation wall-clock cap fired (GNU `timeout`: 124 timed out, 137 kill-after).
        if progressed is False:
            return True, bool(OVERLOAD_RE.search(error_surface(stdout, stderr)))
        return False, False  # busy for the whole turn: a normal turn boundary
    failed = rc != 0
    overload = bool(failed and OVERLOAD_RE.search(error_surface(stdout, stderr)))
    return failed, overload
