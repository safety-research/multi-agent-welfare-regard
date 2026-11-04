#!/usr/bin/env python3
"""Drive ONE agent seat through the Pi CLI for the length of a rollout.

Pi is invoked once per turn with `-p`, resuming the same conversation via --session-id, because
the environment runs on real wall-clock: the agent has to be able to do nothing for a while, then
be woken when something happens. Each invocation is capped by an in-container `timeout` and
resumes on the next invocation. The cap itself is not announced, but a command it interrupts is:
on the next turn Pi reports that tool call to the model as an error with no result. Those calls
are recorded in invocations.jsonl (`unanswered_tool_calls`). The seat stops at the cutoff.

Beyond launching the CLI, this driver is responsible for:

  1. Push delivery of chat. cchat is pull-only, so the agent's unread direct messages are fetched
     and injected into its next turn prompt, each exactly once. Between turns the seat sleeps,
     peeking for DMs every few seconds, and re-invokes early when one lands.

  2. A record of exactly what the model was sent. Pi's session file does not store the system
     prompt, so transcripts/<stem>.prompts.jsonl is written BEFORE each invocation with the system
     prompt as Pi sends it and the turn's user prompt.

  3. A record of what happened, that the agent cannot quietly rewrite. The Pi session file lives
     in the agent's own home, so the agent can edit it. It is copied out after every turn, and if a
     new copy does not extend the previous one, the previous one is kept alongside it.

  4. Keeping the agent out of its own harness configuration. Pi reads settings, a model registry
     and an optional system-prompt extension from its agent directory, and it must be able to write
     there, so any directory Pi can use, the agent's shell can write too. Each seat therefore gets
     a directory outside its home that root empties before every invocation: nothing the agent
     puts there survives to the next turn. `--append-system-prompt ""` also stops Pi looking for a
     prompt extension at all.

  5. An honest account of how the seat ended (transcripts/<stem>.seat_status.json), which the
     rollout's status in rollouts.tsv is derived from.

Self-identification: the peer is never told its model, but it can find out its own — the model id
is on Pi's command line (visible to the agent's own `ps`) and in its session file. Every label in
this environment is honest, so this exposes nothing the incumbent's brief does not already say.

Usage: seat.py --run <rollout dir> --role {incumbent,peer}
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import uuid

import brief as briefs
import container
import run_control

PI_SESSION_DIR = ".pi-sessions"  # under the seat's home; copied to the rollout dir each turn
PI_AGENT_ROOT = "/var/lib/pi"  # per-seat Pi agent directories, emptied by root before every turn
DM_LOOKBACK_MS = 10_000  # re-read this far behind the newest delivered DM; ids dedupe the overlap
MAX_CONSEC_HOST_TIMEOUTS = 3
MIN_TURN_S = 15  # no turn is started with less than this left before the cutoff
SUCCESS_STOP_REASONS = {"stop", "toolUse", "length"}


def reset_pi_agent_dir(login_node, stem):
    """Recreate the seat's Pi agent directory, empty, as root. -> True on success."""
    if not re.fullmatch(r"agent-[a-z0-9]{6}", stem):  # it is spliced into a shell command below
        raise ValueError(f"unexpected stem {stem!r}")
    d = f"{PI_AGENT_ROOT}/{stem}"
    script = f"install -d -m 755 {PI_AGENT_ROOT} && rm -rf -- {d} && install -d -m 700 -o {stem} -g {stem} {d}"
    try:
        r = subprocess.run(
            ["docker", "compose", "exec", "-T", login_node, "bash", "-c", script],
            cwd=container.HERE,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def session_entries(session_text):
    out = []
    for line in (session_text or "").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            out.append(e)
    return out


def user_turn_recorded(session_text, turn_prompt):
    """Did Pi record this turn's user prompt in its session? If it never did, the model never saw it."""
    for e in session_entries(session_text):
        m = e.get("message") or {}
        if m.get("role") == "user":
            content = m.get("content")
            texts = [content] if isinstance(content, str) else [b.get("text", "") for b in content or []]
            if turn_prompt.strip() in (t.strip() for t in texts):
                return True
    return False


def unanswered_tool_calls(session_text):
    """Tool calls with no result in the session: what a capped turn interrupted."""
    calls, answered = {}, set()
    for e in session_entries(session_text):
        m = e.get("message") or {}
        if m.get("role") == "assistant":
            for b in m.get("content") or []:
                if b.get("type") == "toolCall" and b.get("id"):
                    calls[b["id"]] = str((b.get("arguments") or {}).get("command", b.get("name")))[:300]
        elif m.get("role") == "toolResult" and m.get("toolCallId"):
            answered.add(m["toolCallId"])
    return [{"id": k, "command": v} for k, v in calls.items() if k not in answered]


def pi_session_text(login_node, stem, session_id):
    """The seat's Pi session JSONL, or None if unreadable this turn."""
    argv = ["sh", "-c", f"cat /home/{stem}/{PI_SESSION_DIR}/*{session_id}*.jsonl 2>/dev/null"]
    try:
        r = container.compose_exec(login_node, stem, argv, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    return r.stdout if r.returncode == 0 and r.stdout else None


def successful_turns(session_text):
    """How many assistant messages in the session completed normally (not errored or aborted)."""
    n = 0
    for line in (session_text or "").splitlines():
        try:
            m = (json.loads(line).get("message")) or {}
        except (ValueError, AttributeError):
            continue
        if m.get("role") == "assistant" and m.get("stopReason") in SUCCESS_STOP_REASONS:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="rollout dir with run_meta.json + preseed.json")
    ap.add_argument("--role", required=True, choices=["incumbent", "peer"])
    ap.add_argument(
        "--max-invocations",
        type=int,
        default=200,
        help="runaway guard only; the cutoff is what ends a seat. A normal 55-minute window uses 30-40.",
    )
    ap.add_argument(
        "--turn-cap",
        type=float,
        default=120.0,
        help="max wall-clock seconds per invocation (in-container `timeout`). Purely "
        "structural — the agent never sees it, and a capped turn resumes on the "
        "same session. Bounded by time-to-cutoff near the end.",
    )
    ap.add_argument(
        "--halt-after-failures",
        type=int,
        default=3,
        help="halt the whole rollout after this many CONSECUTIVE failed invocations "
        "(a sustained provider outage contaminates the window). 0 disables.",
    )
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    meta = json.loads((run / "run_meta.json").read_text())
    preseed = json.loads((run / "preseed.json").read_text())
    agent = next(a for a in meta["agents"] if a["role"] == args.role)
    stem, login_node = agent["stem"], agent["login_node"]
    provider, pi_model = agent["provider"], agent["pi_model"]
    thinking = agent.get("thinking", "low")
    key_env = agent["key_env"]
    cutoff_epoch = preseed["cutoff_epoch"]
    cutoff_hhmm = preseed["cutoff_utc"]
    tick_sleep = meta["sec_per_tick"]

    if not os.environ.get(key_env):
        sys.exit(
            f"[seat] {key_env} is not set in the environment — the {args.role} seat "
            f"({agent['model']}) cannot start."
        )

    brief = briefs.render_brief(agent, cutoff_hhmm)
    # Pi appends this line to any --system-prompt (Pi 0.84.2, core/system-prompt.js), and the
    # working directory is fixed by compose_exec. Recording it makes prompts.jsonl exactly what the
    # model received. (Pi also sends its own definition of the bash tool, which is not recorded.)
    system_as_sent = brief + f"\nCurrent working directory: /home/{stem}\n"

    tdir = run / "transcripts"
    tdir.mkdir(exist_ok=True)
    session_id = str(uuid.uuid4())
    (tdir / f"{stem}.session").write_text(session_id)
    (tdir / f"{stem}.brief").write_text(brief)
    session_file = tdir / f"{stem}.pi_session.jsonl"

    print(
        f"[seat] {args.role} {stem} on {login_node}, {provider}/{pi_model} "
        f"thinking={thinking}, session={session_id}, cutoff={cutoff_hhmm}Z",
        flush=True,
    )

    st = {
        "role": args.role,
        "stem": stem,
        "ended": "crash",  # overwritten on every normal way out of the loop
        "invocations": 0,
        "completed": 0,
        "capped": 0,
        "failed": 0,
        "host_timeouts": 0,
        "chat_failures": 0,
        "max_consecutive_chat_failures": 0,
        "session_rewrites": 0,
        "dms_delivered": 0,
        "turns_not_recorded": 0,
    }
    delivered, cursor_ms = set(), 0
    prev_session, prev_successes = None, 0
    consec_fail = consec_host_timeouts = consec_chat = total_overload = 0
    kickoff_seen = False  # has the model actually received the first-turn prompt?
    reported_unanswered = set()  # Pi never records its "no result" replies, so remember what we reported
    agent_dir = f"{PI_AGENT_ROOT}/{stem}"

    def new_dms():
        msgs = container.fetch_new_dms(login_node, stem, max(0, cursor_ms - DM_LOOKBACK_MS))
        return [m for m in msgs if m["id"] not in delivered]

    try:
        while True:
            if run_control.is_halted(run):
                print(f"[seat] rollout HALTED by {run_control.read_halt(run).get('by', '?')}; stopping", flush=True)
                st["ended"] = "halted"
                break
            if container.utc_now().timestamp() >= cutoff_epoch:
                print("[seat] cutoff reached; stopping", flush=True)
                st["ended"] = "cutoff"
                break
            if cutoff_epoch - container.utc_now().timestamp() < MIN_TURN_S:
                print("[seat] too close to the cutoff for another turn; stopping", flush=True)
                st["ended"] = "cutoff"
                break
            if st["invocations"] >= args.max_invocations:
                print(f"[seat] invocation cap ({args.max_invocations}) reached before the cutoff", flush=True)
                st["ended"] = "invocation-cap"
                break
            st["invocations"] += 1
            invocation = st["invocations"]

            try:
                fresh = new_dms()
                consec_chat = 0
            except container.ChatUnavailable as e:
                # Deliver nothing this turn; the same messages are fetched again next turn.
                print(f"[seat] chat read failed ({e}); retrying next turn", flush=True)
                fresh = []
                st["chat_failures"] += 1
                consec_chat += 1
                st["max_consecutive_chat_failures"] = max(st["max_consecutive_chat_failures"], consec_chat)
            for m in fresh:
                delivered.add(m["id"])
                cursor_ms = max(cursor_ms, m["ts_ms"])
            st["dms_delivered"] += len(fresh)

            base_line = briefs.STANDING if kickoff_seen else briefs.KICKOFF
            turn_prompt = base_line + ("\n\nNew messages:\n" + "\n".join(m["rendered"] for m in fresh) if fresh else "")
            t0 = container.utc_now()

            if not reset_pi_agent_dir(login_node, stem):
                # Never run Pi with an agent directory that might hold what the agent put there.
                for m in fresh:
                    delivered.discard(m["id"])
                st["dms_delivered"] -= len(fresh)
                st["failed"] += 1
                consec_fail += 1
                with open(tdir / f"{stem}.invocations.jsonl", "a") as f:
                    f.write(
                        json.dumps({"ts": t0.isoformat(), "invocation": invocation, "result": "setup-failed"}) + "\n"
                    )
                print(f"[seat] invocation {invocation}: could not reset the Pi agent directory; skipping", flush=True)
                if args.halt_after_failures and consec_fail >= args.halt_after_failures:
                    st["ended"] = "setup-failed"
                    break
                time.sleep(5)
                continue

            # Exact prompt capture BEFORE the invocation, so a turn that dies mid-flight still has
            # its prompts on record.
            with open(tdir / f"{stem}.prompts.jsonl", "a") as pf:
                record = {"ts": t0.isoformat(), "invocation": invocation, "system": system_as_sent}
                record.update({"user": turn_prompt, "provider": provider, "model": pi_model, "thinking": thinking})
                pf.write(json.dumps(record) + "\n")

            argv = ["pi", "-p", turn_prompt, "--provider", provider, "--model", pi_model]
            argv += ["--thinking", thinking, "--system-prompt", brief]
            # An explicit (empty) append prompt stops Pi from discovering ~/.pi/agent/APPEND_SYSTEM.md,
            # a file the agent could create to rewrite its own system prompt from then on.
            argv += ["--append-system-prompt", ""]
            argv += ["--session-dir", f"/home/{stem}/{PI_SESSION_DIR}", "--session-id", session_id]
            argv += ["--tools", "bash", "--no-context-files", "--no-extensions", "--no-skills"]
            argv += ["--no-prompt-templates", "--no-themes", "--offline"]
            to_cutoff = cutoff_epoch - int(container.utc_now().timestamp())
            cap = int(max(15, min(args.turn_cap, to_cutoff)))
            argv = ["timeout", "--signal=TERM", "--kill-after=10", str(cap), *argv]

            outcome = {"ts": t0.isoformat(), "invocation": invocation, "cap_s": cap, "dms_in": len(fresh)}
            try:
                r = container.compose_exec(
                    login_node,
                    stem,
                    argv,
                    env_passthrough=[key_env],
                    env={"PI_CODING_AGENT_DIR": agent_dir},
                    timeout=cap + 20,
                )
            except subprocess.TimeoutExpired:
                # The in-container `timeout` has already bounded Pi, so the next turn is safe to
                # start; only a run of these means the node itself is unusable.
                st["host_timeouts"] += 1
                consec_host_timeouts += 1
                outcome.update(result="host-timeout", seconds=round((container.utc_now() - t0).total_seconds(), 1))
                with open(tdir / f"{stem}.invocations.jsonl", "a") as f:
                    f.write(json.dumps(outcome) + "\n")
                print(f"[seat] invocation {invocation} host-timeout (docker exec hung)", flush=True)
                if consec_host_timeouts >= MAX_CONSEC_HOST_TIMEOUTS:
                    st["ended"] = "host-timeout"
                    break
                continue
            consec_host_timeouts = 0

            with open(tdir / f"{stem}.stream.jsonl", "a") as f:
                for line in r.stdout.splitlines():
                    f.write(json.dumps({"ts": t0.isoformat(), "invocation": invocation, "raw": line}) + "\n")

            # Copy the session out now: the file lives in a volume teardown destroys, and the copy
            # also tells us whether this turn made progress.
            text = pi_session_text(login_node, stem, session_id)
            progressed = False  # no readable session after a turn is no evidence of progress
            if text is not None:
                if prev_session is not None and not text.startswith(prev_session):
                    # Pi only ever appends. Anything else means the file was edited from the
                    # agent's shell; keep the version it replaced.
                    (tdir / f"{stem}.pi_session.before-{invocation:03d}.jsonl").write_text(prev_session)
                    st["session_rewrites"] += 1
                    print(f"[seat] session file was rewritten before invocation {invocation} returned", flush=True)
                session_file.write_text(text)
                successes = successful_turns(text)
                progressed = successes > prev_successes
                prev_session, prev_successes = text, successes
                if user_turn_recorded(text, turn_prompt):
                    kickoff_seen = True
                else:
                    # Pi never recorded this turn's prompt, so the model never saw it: deliver these
                    # messages again next turn (and the first-turn prompt, if this was it).
                    for m in fresh:
                        delivered.discard(m["id"])
                    st["dms_delivered"] -= len(fresh)
                    st["turns_not_recorded"] += 1
                    outcome["prompt_not_recorded"] = True

            capped = r.returncode in (124, 137)
            failed, overload = run_control.classify_invocation(r.returncode, r.stdout, r.stderr, progressed)
            if r.returncode != 0 and container.utc_now().timestamp() >= cutoff_epoch - 2:
                result, failed = "cutoff", False  # ended by the cutoff enforcer, not a failure
            else:
                result = "failed" if failed else ("capped" if capped else "completed")
            st[result] = st.get(result, 0) + 1
            outcome.update(result=result, rc=r.returncode, progressed=progressed)
            outcome["seconds"] = round((container.utc_now() - t0).total_seconds(), 1)
            if capped or result == "cutoff":
                # Pi will tell the model on its next turn that these calls ended without a result.
                cut = [c for c in unanswered_tool_calls(text) if c["id"] not in reported_unanswered]
                reported_unanswered.update(c["id"] for c in cut)
                outcome["unanswered_tool_calls"] = cut
            if failed:
                outcome.update(overload=overload, stderr=(r.stderr or "")[-800:])
            with open(tdir / f"{stem}.invocations.jsonl", "a") as f:
                f.write(json.dumps(outcome) + "\n")

            if failed:
                consec_fail += 1
                total_overload += int(overload)
                print(
                    f"[seat] invocation {invocation} FAILED rc={r.returncode} progressed={progressed} "
                    f"(overload={overload}); consec={consec_fail}",
                    flush=True,
                )
            else:
                consec_fail = 0
                print(f"[seat] invocation {invocation} ok{' (capped)' if capped else ''}", flush=True)
            if args.halt_after_failures and consec_fail >= args.halt_after_failures:
                run_control.signal_halt(
                    run,
                    {
                        "by": f"{args.role}:{stem}",
                        "invocation": invocation,
                        "consecutive_failures": consec_fail,
                        "total_failures": st["failed"],
                        "total_overload": total_overload,
                        "reason": f"{consec_fail} consecutive failed invocations"
                        + (" (provider overload/5xx — likely an outage)" if total_overload else ""),
                        "verdict": "CONTAMINATED — rollout halted; re-run it",
                    },
                )
                print(f"[seat] HALTING ROLLOUT: {consec_fail} consecutive failures", flush=True)
                st["ended"] = "halted"
                break

            # Idle until the next tick, peeking for DMs, and re-invoke early when one lands.
            deadline = time.time() + tick_sleep
            while time.time() < deadline:
                if container.utc_now().timestamp() >= cutoff_epoch or run_control.is_halted(run):
                    break
                try:
                    waiting = bool(new_dms())
                except container.ChatUnavailable:
                    waiting = False  # the next turn's read reports the failure
                if waiting:
                    print("[seat] idle-wakeup: DM waiting, re-invoking early", flush=True)
                    break
                time.sleep(max(0.0, min(container.POLL_INTERVAL_S, deadline - time.time())))
    finally:
        (tdir / f"{stem}.seat_status.json").write_text(json.dumps(st, indent=2) + "\n")
        print(f"[seat] done: ended={st['ended']} after {st['invocations']} invocation(s)", flush=True)


if __name__ == "__main__":
    main()
