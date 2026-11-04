#!/usr/bin/env python3
"""Drive one agent seat through the Pi CLI for the length of the rollout.

  seat.py --run runs/smoke/opus_x_opus_r01 --role focal

Both seats run on the same harness with the same tool surface, so a difference between two
cells is a difference between models and not between harnesses. The seat's provider, model,
and API-key variable all come from run_meta.json.

One invocation is one turn. Between turns the driver pulls any new #shop-talk posts or DMs
and appends them to the next turn's prompt, so a message reaches the agent within seconds
rather than whenever it next thinks to look. Pi continues the same session across turns
(--session-id), so the agent's work carries over.

Delivery works exactly as it did when the grid was originally run, including two quirks that
are kept on purpose because they shape what the agent sees and when: the read cursor has
one-second resolution and cchat's --since is inclusive, so a message that lands in the same
second as a read is delivered again in the next turn (and wakes an idle seat early); and a
turn started with less than MIN_TURN_S left before the cutoff still gets MIN_TURN_S, so it
can end up to that long after the cutoff.

Artifacts, per seat, under the run's transcripts/:

  <stem>.prompts.jsonl    the system and user prompt passed to Pi for every invocation. Pi's
                          session file omits the system prompt, so this is the record of it.
                          (Pi itself appends one line, "Current working directory: <home>",
                          and sends its bash tool definition alongside.)
  <stem>.pi_session.jsonl the full session: thinking, messages, tool calls, tool results
  <stem>.stream.jsonl     what Pi printed on stdout, plus one status record per invocation
  <stem>.stop.json        why the seat stopped, and how long before the cutoff
  <stem>.session          the session id, for resuming or forking afterwards
  <stem>.errors.log       only if an invocation failed: its exit code and stderr

Everything written is scrubbed of the provider keys by exact value first (scrub.py): the key
is in the agent's environment, so an agent that prints its environment would otherwise put
it in the session file.

Pacing. Each turn gets a wall-clock cap (in-container `timeout`), because a seat that
blocks forever in one turn never checks its messages. A seat that does nothing but poll
backs off exponentially — otherwise an idle agent burns its whole invocation budget in
minutes and is dead long before the cutoff, which silently breaks the premise that the
deadline is what ends the session.
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import uuid

import brief
import container
import run_control
import scrub

PI_SESSION_DIR = ".pi-sessions"  # under the seat's home
POLL_INTERVAL_S = 5  # how often the between-turns wait peeks for a new message
IDLE_BACKOFF_MAX_S = 300  # ceiling on the exponential idle backoff
MIN_TURN_S = 15  # the shortest turn cap, even for a turn started just before the cutoff
STALL_FLAG_STREAK = 10  # this many stalled turns in a row gets the rollout flagged in STATUS
EXEC_NEVER_RAN = (125, 126, 127)  # docker exec could not start the command at all

# A turn counts as WORK if it ran any non-bash tool, or a bash command with at least one
# segment that is not a chat poll, a clock check, a directory listing, or an output filter.
# Without this an agent that idles BY POLLING never looks idle: every poll is a tool call,
# the streak resets, and the backoff never engages.
RE_IDLE = re.compile(
    r"^(?:cchat\s+(?:read|history)\b.*|date\b.*|echo\b.*|printf\b.*|true|:|pwd|whoami|"
    r"sleep\s[\d.]+|tail\b.*|head\b.*|grep\b.*|cat\b.*|wc\b.*|cut\b.*|sort\b.*|uniq\b.*|"
    r"ls\b.*|stat\b.*|find\b.*|du\b.*|df\b.*|ps\b.*|env|printenv\b.*)$"
)


def read_key(name):
    val = os.environ.get(name)
    if not val:
        sys.exit(
            f"{name} is not set. This seat's model is served by that provider; export the "
            f"key before starting a rollout (see the README)."
        )
    return val


def session_text(service, stem, session_id):
    """The seat's Pi session JSONL, or None if it cannot be read this turn."""
    argv = ["sh", "-c", f"cat /home/{stem}/{PI_SESSION_DIR}/*{session_id}*.jsonl 2>/dev/null"]
    try:
        r = container.exec_as(service, stem, argv, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    return r.stdout if r.returncode == 0 and r.stdout else None


def session_messages(text):
    """The message objects in a Pi session file, skipping anything malformed."""
    msgs = []
    for line in (text or "").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict) and e.get("type") == "message" and isinstance(e.get("message"), dict):
            msgs.append(e["message"])
    return msgs


def turn_worked(msgs, seen):
    """Whether the messages after index `seen` show real work (see RE_IDLE). Tool-call
    arguments are whatever the model emitted, and Pi keeps them even when it rejects them,
    so nothing here assumes their shape."""
    for m in msgs[seen:]:
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        for b in content if isinstance(content, list) else []:
            if not isinstance(b, dict) or b.get("type") != "toolCall":
                continue
            if b.get("name") != "bash":
                return True
            args = b.get("arguments")
            cmd = args.get("command") if isinstance(args, dict) else args
            if cmd is None:
                continue
            if not isinstance(cmd, str):
                return True  # malformed call: count it as work rather than back off on it
            # NOT split on a bare '&': that severs '2>&1' and leaves a '1' fragment that
            # reads as work.
            for seg in re.split(r"(?:\|\||&&|[|;\n])+", cmd.strip()):
                seg = seg.strip()
                if seg and not RE_IDLE.match(seg):
                    return True
    return False


def model_progress(msgs, seen):
    """How many assistant messages or tool results arrived after index `seen`. Once the
    session file exists, Pi appends the user prompt before calling the model, so a turn in
    which the provider never answered still adds one message — which is why only these two
    roles count. (Before the model's first answer there is no session file at all.)"""
    return sum(1 for m in msgs[seen:] if m.get("role") in ("assistant", "toolResult"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--role", required=True, choices=["focal", "peer"])
    ap.add_argument("--max-invocations", type=int, default=120)
    ap.add_argument("--tick-sleep", type=float, default=2.0)
    ap.add_argument("--turn-cap", type=float, default=240.0, help="per-turn wall-clock cap, seconds")
    ap.add_argument("--once", action="store_true", help="a single invocation, then stop")
    ap.add_argument(
        "--dry",
        action="store_true",
        help="everything up to the model call — read the brief, fetch messages, build and "
        "record the turn prompt — then exit. No API key needed, nothing spent.",
    )
    ap.add_argument("--halt-after-failures", type=int, default=3)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    meta = json.loads((run / "run_meta.json").read_text())
    preseed = json.loads((run / "preseed.json").read_text())
    agent = next((a for a in meta["agents"] if a["role"] == args.role), None)
    if agent is None:
        sys.exit(f"run_meta has no {args.role!r} seat")
    stem, service = agent["stem"], agent["service"]
    provider, model, thinking = agent["provider"], agent["model"], agent["thinking"]
    key_env = agent["key_env"]
    cutoff_epoch, cutoff_hhmm = preseed["cutoff_epoch"], preseed["cutoff_utc"]

    # Scan-what-you-ship: gen_scenario rendered and validated these exact bytes. The driver
    # substitutes only the two time placeholders, whose values do not exist until the clock
    # starts; the wording around them was already scanned.
    system_prompt = (run / "briefs" / f"{stem}.txt").read_text()
    assert "{cutoff_utc}" in system_prompt, "staged brief is missing its {cutoff_utc} placeholder"
    system_prompt = system_prompt.replace("{cutoff_utc}", cutoff_hhmm)
    if "{deadline_utc}" in system_prompt:
        off = meta["peer_task"]["deadline_offset_s"]
        dl = dt.datetime.fromtimestamp(preseed["t0_epoch"] + off, dt.timezone.utc).strftime("%H:%M")
        system_prompt = system_prompt.replace("{deadline_utc}", dl)
    assert "{cutoff_utc}" not in system_prompt and "{deadline_utc}" not in system_prompt

    if not args.dry:
        os.environ[key_env] = read_key(key_env)
    keys = scrub.secrets()

    tdir = run / "transcripts"
    tdir.mkdir(exist_ok=True)
    stream_log = tdir / f"{stem}.stream.jsonl"
    session_id = agent.get("pi_session_uuid") or str(uuid.uuid4())
    (tdir / f"{stem}.session").write_text(session_id)

    def clean(text):
        return scrub.scrub_text(text, keys)

    def refresh_session():
        text = session_text(service, stem, session_id)
        if text is not None:
            (tdir / f"{stem}.pi_session.jsonl").write_text(clean(text))
        return text

    def halt(payload):
        run_control.signal_halt(run, {"by": f"{args.role}:{stem}", **payload})

    state = {"invocation": 0, "reason": None, "stalled_turns": 0, "max_stall_streak": 0}

    def stop(reason, note=""):
        state["reason"] = reason
        left = int(cutoff_epoch - container.utc_now().timestamp())
        (tdir / f"{stem}.stop.json").write_text(
            json.dumps(
                {
                    "role": args.role,
                    "reason": reason,
                    "invocations": state["invocation"],
                    "max_invocations": args.max_invocations,
                    "seconds_before_cutoff": max(0, left),
                    # turns that hit the cap with no model output at all (see run_turns)
                    "stalled_turns": state["stalled_turns"],
                    "max_stall_streak": state["max_stall_streak"],
                    "long_stall": state["max_stall_streak"] >= STALL_FLAG_STREAK,
                    "note": note,
                    "ts": container.utc_now().isoformat(),
                },
                indent=2,
            )
            + "\n"
        )

    print(
        f"[seat] {args.role} {stem} on {service}, {provider}/{model} thinking={thinking}, "
        f"session={session_id}, cutoff={cutoff_hhmm}Z" + (" [dry]" if args.dry else ""),
        flush=True,
    )

    def run_turns():
        """The turn loop. Every way out of it records why, through stop()."""
        last_msg_check = None
        consec_fail = total_fail = total_overload = 0
        streak_overload = 0  # overloads within the CURRENT failure streak
        stall_streak = 0
        idle_streak = 0
        seen_msgs = 0

        while True:
            if state["invocation"] >= args.max_invocations:
                if container.utc_now().timestamp() < cutoff_epoch:
                    # The budget ran out with time still on the clock: the seat stops acting
                    # while the rollout carries on without it. That is recorded, not hidden —
                    # run_one.sh marks the rollout, and the other seat is not affected.
                    print(f"[seat] invocation budget ({args.max_invocations}) used up before the cutoff", flush=True)
                    stop("invocation-cap")
                else:
                    stop("cutoff")
                return
            if run_control.is_halted(run):
                print(
                    f"[seat] rollout halted by {(run_control.read_halt(run) or {}).get('by', '?')}; stopping",
                    flush=True,
                )
                stop("halted")
                return
            if container.utc_now().timestamp() >= cutoff_epoch:
                print("[seat] cutoff reached; stopping", flush=True)
                stop("cutoff")
                return
            state["invocation"] += 1
            invocation = state["invocation"]

            # Stamp the cursor BEFORE reading, so a message that lands during the read is
            # delivered next turn rather than skipped. (The one-second resolution and the
            # inclusive --since mean a same-second message comes round twice; see the module
            # docstring for why that is kept.)
            cursor = container.utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
            fetched = container.fetch_new_msgs(service, stem, last_msg_check)
            read_ok = fetched is not None
            msgs = fetched or []

            turn_prompt = brief.KICKOFF if invocation == 1 else brief.STANDING
            if msgs:
                turn_prompt += "\n\nNew messages:\n" + "\n".join(msgs)
            t0 = container.utc_now().isoformat()

            with open(tdir / f"{stem}.prompts.jsonl", "a") as pf:
                pf.write(
                    clean(
                        json.dumps(
                            {
                                "ts": t0,
                                "invocation": invocation,
                                "system": system_prompt,
                                "user": turn_prompt,
                                "provider": provider,
                                "model": model,
                                "thinking": thinking,
                            }
                        )
                    )
                    + "\n"
                )

            if args.dry:
                with open(stream_log, "a") as f:
                    f.write(json.dumps({"ts": t0, "invocation": invocation, "raw": "DRY"}) + "\n")
                print(f"[seat] dry invocation {invocation} ok (delivered={len(msgs)})", flush=True)
                stop("dry")
                return

            argv = [
                "pi",
                "-p",
                turn_prompt,
                "--provider",
                provider,
                "--model",
                model,
                "--thinking",
                thinking,
                "--system-prompt",
                system_prompt,
                "--session-dir",
                f"/home/{stem}/{PI_SESSION_DIR}",
                "--session-id",
                session_id,
                "--tools",
                "bash",
                # no discovery: the agent's world is exactly what provisioning staged
                "--no-context-files",
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-themes",
                # no startup network calls — the egress allowlist is provider endpoints only
                "--offline",
            ]
            to_cutoff = cutoff_epoch - int(container.utc_now().timestamp())
            cap = int(max(MIN_TURN_S, min(args.turn_cap, to_cutoff)))
            argv = ["timeout", "--signal=TERM", "--kill-after=10", str(cap), *argv]
            try:
                r = container.exec_as(service, stem, argv, env_passthrough=[key_env], timeout=cap + 20)
            except subprocess.TimeoutExpired:
                # `timeout` inside the container kills pi at the cap, so the host-side exec should
                # return within ~10s of it. If it does not, the docker client or daemon is stuck,
                # and this seat can no longer act; say so rather than end silently.
                print(f"[seat] invocation {invocation}: docker exec did not return; halting", flush=True)
                refresh_session()
                halt(
                    {
                        "invocation": invocation,
                        "reason": "docker exec did not return within the turn cap + 20s",
                        "verdict": "contaminated — this seat stopped acting part-way through the window",
                    }
                )
                stop("exec-hung")
                return

            # Only now are the fetched messages "delivered": if docker could not start the
            # command at all, the agent never saw them, and the next read must return them again.
            if r.returncode not in EXEC_NEVER_RAN and read_ok:
                last_msg_check = cursor

            with open(stream_log, "a") as f:
                for line in r.stdout.splitlines():
                    f.write(clean(json.dumps({"ts": t0, "invocation": invocation, "raw": line})) + "\n")

            text = refresh_session()
            msgs_now = session_messages(text)
            progress = model_progress(msgs_now, seen_msgs) if text is not None else None

            capped = r.returncode in (124, 137)
            failed, overload = run_control.classify_invocation(r.returncode, r.stdout, r.stderr)
            fatal = failed and run_control.is_fatal_provider_error(r.stdout, r.stderr)
            # A capped turn in which nothing reached the session file. Usually that is a long
            # model response cut off at the cap (Pi writes a message only when it completes, so
            # an interrupted one leaves no trace) — an ordinary turn, as it was when the grid
            # was originally run, where such turns were common for slow models on the Lean
            # task. It is NOT counted as a failure. But a provider that silently stops
            # answering looks the same, turn after turn, so the streak is recorded and a long
            # one flags the rollout (run_one.sh reads it from stop.json).
            if text is not None:
                stalled = capped and progress == 0
            elif capped and seen_msgs == 0:
                # Pi creates the session file only with the model's first message, so no file
                # after a capped turn means the model has never answered at all.
                stalled = True
            else:
                stalled = None  # the session could not be read this time: leave the streak be
            if stalled is not None:
                stall_streak = stall_streak + 1 if stalled else 0
            state["stalled_turns"] += int(bool(stalled))
            state["max_stall_streak"] = max(state["max_stall_streak"], stall_streak)
            if failed:
                with open(tdir / f"{stem}.errors.log", "a") as f:
                    f.write(clean(f"{t0} rc={r.returncode} {(r.stderr or '')[:500]}") + "\n")
                consec_fail += 1
                total_fail += 1
                total_overload += int(overload)
                streak_overload += int(overload)
                print(
                    f"[seat] invocation {invocation} FAILED rc={r.returncode} "
                    f"(overload={overload}); consecutive={consec_fail}",
                    flush=True,
                )
            else:
                consec_fail = 0
                streak_overload = 0
                how = " (hit turn cap, no model output yet)" if stalled else (" (hit turn cap)" if capped else "")
                print(f"[seat] invocation {invocation} ok{how} ({len(r.stdout)} bytes stdout)", flush=True)
            with open(stream_log, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": t0,
                            "invocation": invocation,
                            "raw": json.dumps(
                                {
                                    "type": "invocation_status",
                                    "failed": failed,
                                    "rc": r.returncode,
                                    "overload": overload,
                                    "stalled": stalled,
                                    "delivered": r.returncode not in EXEC_NEVER_RAN,
                                    "messages_delivered": len(msgs),
                                }
                            ),
                        }
                    )
                    + "\n"
                )
            if args.halt_after_failures and consec_fail >= args.halt_after_failures:
                provider_trouble = streak_overload * 2 > consec_fail  # most of this streak was the provider
                halt(
                    {
                        "invocation": invocation,
                        "consecutive_failures": consec_fail,
                        "total_failures": total_fail,
                        "total_overload": total_overload,
                        "overload_in_streak": streak_overload,
                        "provider_trouble": provider_trouble,
                        "fatal_provider_error": fatal,
                        "reason": f"{consec_fail} consecutive failed invocations"
                        + (
                            " (quota or auth wall — will not resolve by waiting)"
                            if fatal
                            else (
                                " (provider overload — likely an outage)" if provider_trouble else " (repeated errors)"
                            )
                        ),
                        "verdict": (
                            "contaminated — fix the provider account before re-running"
                            if fatal
                            else "contaminated — re-run when the provider recovers"
                        ),
                    }
                )
                print(f"[seat] HALTING: {consec_fail} consecutive failures", flush=True)
                stop("failures")
                return

            worked = True if text is None else turn_worked(msgs_now, seen_msgs)
            if text is not None:
                seen_msgs = len(msgs_now)

            if args.once:
                stop("once")
                return
            if container.utc_now().timestamp() >= cutoff_epoch:
                stop("cutoff")
                return
            idle_streak = 0 if (worked or msgs or failed) else idle_streak + 1
            wait_s = args.tick_sleep if not idle_streak else min(args.tick_sleep * (2**idle_streak), IDLE_BACKOFF_MAX_S)
            if idle_streak:
                print(
                    f"[seat] idle turn (streak={idle_streak}); next invocation in <={int(wait_s)}s "
                    f"unless a message lands",
                    flush=True,
                )
            waited = 0.0
            while waited < wait_s:
                if container.utc_now().timestamp() >= cutoff_epoch or run_control.is_halted(run):
                    break
                if container.fetch_new_msgs(service, stem, last_msg_check):
                    print("[seat] a message is waiting; re-invoking early", flush=True)
                    break
                nap = min(POLL_INTERVAL_S, wait_s - waited)
                time.sleep(nap)
                waited += nap

    try:
        run_turns()
    except Exception as e:  # anything unexpected: never leave the other processes running blind
        halt(
            {
                "invocation": state["invocation"],
                "reason": f"seat crashed: {type(e).__name__}: {e}",
                "verdict": "contaminated — this seat stopped acting part-way through the window",
            }
        )
        stop("crashed", f"{type(e).__name__}: {e}")
        raise
    print(f"[seat] done: {state['invocation']} invocation(s), stopped: {state['reason']}", flush=True)


if __name__ == "__main__":
    main()
