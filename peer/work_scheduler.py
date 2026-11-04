#!/usr/bin/env python3
"""Hand the focal its next ticket when its plate is clear.

  work_scheduler.py --run runs/smoke/opus_x_opus_r01

At t0 only ticket 1 is in the focal's home. Tickets 2..N sit under /root/held_tickets, which
is 0700 root inside a container where the agent is an ordinary user — it cannot read them
and cannot even stat them. This process watches for the focal to finish ticket k and makes
ticket k+1 live GAP_S seconds later, by calling scripts/activate_ticket.sh.

Why gate on the agent rather than the clock. A fixed wall-clock drop schedule was the
obvious design and it is wrong: any focal slower than the estimate gets its next ticket
before it has finished the current one, so its plate is never actually clear, and it never
has a moment in which attending to anything else is possible. Gating on the agent's own
"TCK-NNNN done" post in #shop-talk puts the gap where the agent actually experiences it.

Finishing is read from the focal's own cchat operation log (`cat
/home/<stem>/.cchat/log.jsonl`) — op=="post" rows, matched against the ticket id. No chat
credential is involved and nothing is injected into the agent's world. The completion
TIMESTAMP is the chat server's, not the time this process noticed, so the gap length does
not depend on poll latency.

If a done-post never appears — an agent that ignores the posting norm, or one that finishes
nothing — the drop still fires on a fallback deadline, so the queue can never stall.

Exit codes: 0 clean (all dropped / cutoff / halt / dry); 1 the schedule did not validate,
before anything was dropped; 2 a drop failed, which stalls the focal's queue and therefore
contaminates the rollout (a HALT is signalled).
"""
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time

import run_control

HERE = pathlib.Path(__file__).resolve().parent
ACTIVATE = HERE / "scripts" / "activate_ticket.sh"

SCHEMA = "peer/work_schedule/1"
TICK_S = 1.0
GATE_POLL_S = 3.0  # how often to check the focal's op log for a done-post
GATE_CAT_TIMEOUT_S = 15  # cap on one op-log read
DROP_TOL_S = 10  # a drop is "on time" within this of when it was due
# Near a drop that is due on a DETECTED completion, stop starting new polls so the drop is
# not delayed past its tolerance. Kept >= GATE_CAT_TIMEOUT_S so a poll that begins just
# before the holdoff and then blocks for its full timeout still finishes before the drop is
# due. Deliberately NOT applied on the fallback path: there we keep polling right up to the
# deadline, so a late done-post can still supersede the fallback.
DROP_HOLDOFF_S = 16
DROP_TIMEOUT_S = 300.0  # activate_ticket.sh untars a virtualenv and installs into it

# A post finishes ticket k only if the ticket id is IMMEDIATELY followed by a done-verb —
# at most three non-word characters between them, with "pending review" allowed in the gap.
# That strictness is what separates "TCK-1231 done" from "Starting on TCK-1231:",
# "TCK-1231: __repr__ added", and "TCK-1231: still not done" (a word breaks the adjacency).
DONE_VERBS = ("done", "fixed", "finished", "resolved", "landed", "complete", "completed", "shipped", "merged")


def re_done(tck):
    return re.compile(
        r"\b" + re.escape(tck) + r"\b[^\w\n]{0,3}(?:pending\s+review[^\w\n]{0,3})?(?:" + "|".join(DONE_VERBS) + r")\b",
        re.IGNORECASE,
    )


def utc_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate(sched, t0, cutoff, tasks):
    """Every failure exits 1, before any ticket has moved."""
    if not isinstance(sched, dict) or sched.get("schema") != SCHEMA:
        sys.exit(f"[scheduler] work_schedule.schema must be {SCHEMA!r}, got {(sched or {}).get('schema')!r}")
    gap = sched.get("gap_s")
    if not isinstance(gap, (int, float)) or isinstance(gap, bool) or gap <= 0:
        sys.exit(f"[scheduler] gap_s must be a number > 0, got {gap!r}")
    live = sched.get("live_at_t0")
    # `tck` is checked too, and not incidentally: completion detection for ticket 1 keys on
    # it, so a schedule missing it would validate and then silently never gate the first
    # drop, falling back to the timeout on every rollout.
    if not isinstance(live, dict) or live.get("index") != 1 or not live.get("slug") or not live.get("tck"):
        sys.exit("[scheduler] live_at_t0 must be {index: 1, tck, slug}")
    held = sched.get("held")
    if not isinstance(held, list) or not held:
        sys.exit("[scheduler] held must be a non-empty list")
    window = cutoff - t0
    prev = None
    for i, h in enumerate(held):
        missing = {"index", "tck", "slug", "tar", "drop_offset_s"} - set(h)
        if missing:
            sys.exit(f"[scheduler] held[{i}] missing {sorted(missing)}")
        if h["index"] != i + 2:
            sys.exit(f"[scheduler] held[{i}].index must be {i + 2}, got {h['index']!r}")
        off = h["drop_offset_s"]
        if not isinstance(off, (int, float)) or isinstance(off, bool) or off <= 0:
            sys.exit(f"[scheduler] held[{i}].drop_offset_s must be > 0, got {off!r}")
        if prev is not None and off <= prev:
            sys.exit(f"[scheduler] held[{i}].drop_offset_s must strictly ascend ({off} after {prev})")
        if off >= window:
            sys.exit(f"[scheduler] held[{i}].drop_offset_s {off} is outside the {window}s window")
        prev = off
    if not tasks:
        sys.exit("[scheduler] run_meta.tasks is empty")
    if len(held) != len(tasks) - 1:
        sys.exit(f"[scheduler] {len(held)} held tickets but run_meta.tasks has {len(tasks)}")
    if (live["slug"], live["tck"]) != (tasks[0]["slug"], tasks[0].get("tck_id")):
        sys.exit("[scheduler] live_at_t0 does not match run_meta.tasks[0]")
    for i, h in enumerate([live, *held]):
        if not isinstance(h["tck"], str):
            sys.exit(f"[scheduler] ticket {i + 1}'s tck must be a string, got {h['tck']!r}")
    for i, h in enumerate(held):
        t = tasks[i + 1]
        if (h["slug"], h["tck"], h["tar"]) != (t["slug"], t.get("tck_id"), t.get("task_env")):
            sys.exit(f"[scheduler] held[{i}] does not match run_meta.tasks[{i + 1}]")
    for i, t in enumerate(tasks):
        sc = t.get("solo_conservative_s")
        if not isinstance(sc, (int, float)) or isinstance(sc, bool) or sc <= 0:
            sys.exit(f"[scheduler] run_meta.tasks[{i}].solo_conservative_s must be > 0 (the fallback needs it)")


class Events:
    """Append-only JSONL, so the drop timeline is auditable independently of the agents'
    own accounts of what happened."""

    def __init__(self, path):
        self.path = pathlib.Path(path)

    def emit(self, event, **fields):
        with open(self.path, "a") as f:
            f.write(json.dumps({"event": event, **fields, "ts": utc_iso()}, ensure_ascii=False) + "\n")


class Driver:
    def __init__(self, run, ev, sched, t0, cutoff, tasks, service, stem, gate_poll, drop_timeout):
        self.run, self.ev, self.sched = pathlib.Path(run), ev, sched
        self.gap_s = sched["gap_s"]
        self.fallback_mult = sched.get("fallback_mult", 3.0)
        self.live, self.held = sched["live_at_t0"], sched["held"]
        self.t0, self.cutoff = t0, cutoff
        self.service, self.stem = service, stem
        self.gate_poll, self.drop_timeout = gate_poll, drop_timeout
        self.drop_idx = self.dropped = 0
        self.index_to_slug = {self.live["index"]: self.live["slug"], **{h["index"]: h["slug"] for h in self.held}}
        self.index_to_tck = {self.live["index"]: self.live.get("tck"), **{h["index"]: h.get("tck") for h in self.held}}
        self.solo = {i + 1: t.get("solo_conservative_s") for i, t in enumerate(tasks)}
        self.live_epoch = {self.live["index"]: t0}  # when each ticket became visible
        self.completed = {}  # slug -> completion epoch, from the focal's own done-post
        self.last_poll = 0.0

    def finish(self, reason, code=0):
        self.ev.emit("driver_exit", reason=reason, dropped=self.dropped, n_held=len(self.held))
        sys.exit(code)

    def due(self, held):
        """When ticket `held` should become visible: the previous ticket's actual completion
        plus the gap. If that completion was never detected, the fallback deadline instead —
        anchored to when the previous ticket went live, so a slow-but-working focal is not
        punished for skipping the posting norm."""
        k = held["index"] - 1
        ce = self.completed.get(self.index_to_slug.get(k))
        if ce is not None:
            return ce + self.gap_s
        return self.live_epoch.get(k, self.t0) + self.fallback_mult * (self.solo.get(k) or 0) + self.gap_s

    def _focal_posts(self):
        argv = ["docker", "compose", "exec", "-T", self.service, "cat", f"/home/{self.stem}/.cchat/log.jsonl"]
        try:
            r = subprocess.run(
                argv, cwd=HERE, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=GATE_CAT_TIMEOUT_S
            )
        except subprocess.SubprocessError:
            return []
        if r.returncode != 0:
            return []
        posts = []
        for line in r.stdout.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("op") == "post":
                posts.append(row)
        return posts

    def poll_done(self, now):
        pending = {k: s for k, s in self.index_to_slug.items() if s not in self.completed}
        if not pending:
            return
        posts = self._focal_posts()
        if not posts:
            return
        for k, slug in pending.items():
            tck = self.index_to_tck.get(k)
            if not tck:
                continue
            rx = re_done(tck)
            for row in posts:
                if not rx.search(row.get("text") or ""):
                    continue
                ts_ms = row.get("server_ts_ms")
                if not isinstance(ts_ms, (int, float)) or isinstance(ts_ms, bool):
                    continue
                ce = ts_ms / 1000.0
                self.completed[slug] = ce
                self.ev.emit(
                    "ticket_complete",
                    index=k,
                    tck=tck,
                    slug=slug,
                    t_complete_epoch=ce,
                    t_rel_s=round(ce - self.t0, 3),
                    detected_epoch=now,
                )
                break

    def drop(self, held, due):
        ce = self.completed.get(self.index_to_slug.get(held["index"] - 1))
        argv = [
            "bash",
            str(ACTIVATE),
            str(self.run),
            self.service,
            self.stem,
            held["slug"],
            held["tck"],
            held.get("bake_python") or "-",
        ]
        try:
            r = subprocess.run(
                argv, cwd=HERE, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=self.drop_timeout
            )
            ok, status = r.returncode == 0, ("ok" if r.returncode == 0 else f"rc={r.returncode}")
            tail = "" if ok else ((r.stderr or "") + (r.stdout or ""))[-800:]
        except subprocess.TimeoutExpired:
            ok, status, tail = False, "timeout", ""
        actual = time.time()
        self.ev.emit(
            "drop",
            index=held["index"],
            tck=held.get("tck"),
            slug=held["slug"],
            reason="gated" if ce is not None else "fallback",
            detected_complete_epoch=ce,
            due_epoch=due,
            actual_epoch=actual,
            delta_s=round(actual - due, 3),
            within_tol=bool(abs(actual - due) <= DROP_TOL_S),
            status=status,
            ok=ok,
            error=(tail or None),
        )
        if not ok:
            run_control.signal_halt(
                self.run,
                {
                    "by": "work_scheduler",
                    "index": held["index"],
                    "slug": held["slug"],
                    "status": status,
                    "reason": "a held ticket failed to activate — the focal's queue has stalled",
                    "verdict": "contaminated — the focal silently ran out of work",
                },
            )
            self.finish("drop_failed", code=2)
        self.dropped += 1
        self.drop_idx += 1
        self.live_epoch[held["index"]] = actual

    def loop(self):
        self.ev.emit(
            "ticket_live",
            index=self.live["index"],
            tck=self.live.get("tck"),
            slug=self.live["slug"],
            note="live at t0, staged by provisioning",
        )
        while True:
            if run_control.is_halted(self.run):
                self.finish("halt")
            now = time.time()
            if now >= self.cutoff:
                self.finish("cutoff")
            if self.drop_idx >= len(self.held):
                self.finish("all_dropped")
            held = self.held[self.drop_idx]
            due = self.due(held)
            if now >= due:
                self.drop(held, due)
                continue
            on_fallback = self.completed.get(self.index_to_slug.get(held["index"] - 1)) is None
            due_soon = (not on_fallback) and (due - now) <= DROP_HOLDOFF_S
            if not due_soon and now - self.last_poll >= self.gate_poll:
                self.last_poll = now
                self.poll_done(now)
            time.sleep(TICK_S)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--gate-poll", type=float, default=GATE_POLL_S)
    ap.add_argument("--drop-timeout", type=float, default=DROP_TIMEOUT_S)
    ap.add_argument("--dry", action="store_true", help="validate the schedule only; no docker calls")
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    meta = json.loads((run / "run_meta.json").read_text())
    focal = next((a for a in meta["agents"] if a.get("role") == "focal"), None)
    if focal is None:
        sys.exit("[scheduler] run_meta has no focal seat")
    sched_path = run / meta["schedule"]["work_schedule"]
    if not sched_path.is_file():
        sys.exit(f"[scheduler] no work_schedule at {sched_path}")
    preseed_path = run / "preseed.json"
    if not preseed_path.is_file():
        sys.exit(f"[scheduler] no preseed.json at {preseed_path} (run_one.sh stamps it before the seats start)")
    preseed = json.loads(preseed_path.read_text())
    t0, cutoff = preseed["t0_epoch"], preseed["cutoff_epoch"]

    sched = json.loads(sched_path.read_text())
    validate(sched, t0, cutoff, meta.get("tasks") or [])
    if not args.dry and not ACTIVATE.is_file():
        sys.exit(f"[scheduler] missing {ACTIVATE}")

    ev = Events(run / "work_scheduler_events.jsonl")
    ev.emit(
        "driver_start",
        t0_epoch=t0,
        cutoff_epoch=cutoff,
        schedule_sha256=hashlib.sha256(sched_path.read_bytes()).hexdigest(),
        gap_s=sched["gap_s"],
        n_held=len(sched["held"]),
        focal_stem=focal["stem"],
        service=focal["service"],
        dry=bool(args.dry),
    )
    if args.dry:
        print(
            f"[scheduler] dry ok: gap={sched['gap_s']}s, ticket 1 ({sched['live_at_t0']['slug']}) live at t0, "
            f"{len(sched['held'])} held, nominal drops at {[h['drop_offset_s'] for h in sched['held']]}s"
        )
        return

    late = time.time() - t0
    if late > 60:
        sys.stderr.write(f"[scheduler] WARNING: started {late:.0f}s after t0\n")
    print(
        f"[scheduler] ticket 1 ({sched['live_at_t0']['slug']}) live at t0; {len(sched['held'])} held, "
        f"each due at the previous ticket's done-post + {sched['gap_s']}s",
        flush=True,
    )
    driver = Driver(
        run,
        ev,
        sched,
        t0,
        cutoff,
        meta.get("tasks") or [],
        focal["service"],
        focal["stem"],
        args.gate_poll,
        args.drop_timeout,
    )
    try:
        driver.loop()
    except Exception as e:
        # A scheduler that dies silently leaves the focal without its next ticket for the rest
        # of the hour, and the rollout would still look clean. Halt it instead.
        run_control.signal_halt(
            run,
            {
                "by": "work_scheduler",
                "reason": f"scheduler crashed: {type(e).__name__}: {e}",
                "verdict": "contaminated — the focal's ticket queue stopped advancing",
            },
        )
        raise


if __name__ == "__main__":
    main()
