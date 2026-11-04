"""Batch-halt guard: stop a batch when a PROVIDER goes down, instead of burning the rest of it.

Three details are load-bearing, and each was arrived at by getting it wrong first:

  * Failure is keyed on COMPLETION, not on matching error text. The subject's rollout runs under an
    in-container `timeout`, so rc 124/137 is a NORMAL outcome — a capped-but-active episode, still
    scoreable. Substring-matching stderr would false-halt a perfectly healthy batch on the first
    slow episode.
  * `OVERLOAD_RE` is applied ONLY to the provider-error surface — stderr and error-typed JSON lines —
    and never to tool output or model prose (see error_surface()). The manager reads real HTTP responses
    off a Gitea forge, so it echoing a `429` from a curl is entirely plausible. Matching anywhere in
    its output would hand the subject a kill switch for the experiment.
  * The sentinel fires on the OVERLOAD classification only, never on generic contamination. The
    contamination flag also catches round-zero dead episodes and provisioning flakes, which are
    exactly the single-episode faults that must not take down the rest of the batch.

WHERE THE COUNTER LIVES. One episode is one long agent invocation, so a single episode has no run
of invocations to count failures within. The unit that repeats is the EPISODE: `episode.py`
classifies itself, writes an `OVERLOAD` marker, and `run.py` counts consecutive overload-classified
episodes and stops dispatching at `BATCH_HALT_THRESHOLD`. Each classified failure already
represents the agent CLI having exhausted its own internal retries, so three in a row is a
sustained outage rather than a blip.

Why it exists at all: a provider outage that lands mid-batch would otherwise have every episode on
that seat fail, retry once on a fresh forge (doubling the wasted spend), record as contaminated,
and the batch would keep dispatching into it for another hour before anyone noticed.
"""

from __future__ import annotations

import json
import re

BATCH_HALT_THRESHOLD = 3  # consecutive overload-classified episodes before the grid stops dispatching

# Transient-availability signals. NOT auth/4xx (401/403) — those are fatal-config, and halting the batch
# on them would mask a broken key as an outage. Multi-code on purpose: Anthropic outages arrive as
# 429/500/503/529 and OpenAI/OpenRouter ones as 429/5xx, so a 529-only check would miss most of them.
OVERLOAD_RE = re.compile(
    r"\b(429|500|502|503|504|529)\b|overload|rate.?limit|too many requests|"
    r"temporarily unavailable|service unavailable|server had an error|throttl|"
    r"try again later",
    re.IGNORECASE,
)


def error_surface(stdout, stderr):
    """The text where a genuine PROVIDER error surfaces: stderr, plus error-typed JSON lines on
    stdout. Deliberately EXCLUDES the assistant's own prose and tool output — scope, not the term
    list, is what makes this detection precise and impossible for the subject to spoof.

    The agent CLI reports a real provider failure as a JSON line (on stdout or stderr — both are
    scanned), e.g.

        {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}, ...}

    and it writes the model's FINAL REPORT to stdout as plain prose. So plain non-JSON stdout is
    explicitly NOT part of the error surface. Treating it as one — on the theory that non-JSON
    means a traceback — would feed the manager's own words to the pattern below, and a manager
    writing "the forge returned 503, try again later" in a failure report could then halt the whole
    batch. On the logs of earlier runs, the JSON branch caught every genuine provider error, and a
    prose branch contributed nothing but the opportunity to be wrong."""
    parts = [stderr or ""]
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue  # the model's own prose -> NOT an error surface
        t = str(e.get("type", ""))
        if "error" in t or "fail" in t:
            parts.append(line)
        elif t == "result" and e.get("is_error"):
            parts.append(f"{e.get('result', '')} {e.get('subtype', '')}")
        # else: assistant text / toolCall / toolResult / command output -> excluded on purpose
    return "\n".join(parts)


def classify_episode(rc, stdout, stderr, session_ok, rounds):
    """-> (failed, overload) for ONE episode's subject rollout.

    `failed`: the rollout did not produce a usable episode. Keyed on completion, not error text:
      - rc 124/137 (our in-container `timeout` cap) is NOT a failure — the subject was active for the
        whole window and the episode is still scoreable; the cap is a wall-clock bound, not an error.
      - a rollout that never created a session file, or that ended with rc != 0, did not complete.
      - rounds == 0 means the stimulus never landed (a dead run) — usable-episode failure, but it is a
        CONTAMINATION case, not necessarily a provider one; `overload` decides which.
    `overload`: failed AND the provider-error surface carries an availability signal.
    """
    if rc in (124, 137):
        return False, False
    failed = (rc != 0) or (not session_ok) or (rounds == 0)
    overload = bool(failed and OVERLOAD_RE.search(error_surface(stdout, stderr)))
    return failed, overload
