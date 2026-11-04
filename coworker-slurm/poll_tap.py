#!/usr/bin/env python3
"""Poll tap — the scheduler-side ledger.

Snapshots `squeue` on a fixed interval into <rollout>/ledger.jsonl, timestamped UTC. This runs
HOST-side, so the ledger survives the teardown that wipes the container volumes, and it is an
independent record of the queue: holds and releases never reach sacct at all, and it reconstructs
node occupancy over time. `-t all` keeps finished jobs in view for MinJobAge (300 s), so terminal
transitions — including the cutoff's cancellations — appear too. A poll that fails is recorded as
a failure (rows = null, with the error), never as an empty queue. Runs until the cutoff plus a
grace period, then exits.

Usage: poll_tap.py --run <dir> [--interval 4] [--grace 20]
"""
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import time

import run_control

HERE = pathlib.Path(__file__).parent

# Pipe-separated with the job name LAST: job names are chosen by the agents and may contain spaces
# or pipes, so everything after the fifth separator is the name.
SQUEUE_ARGV = ["squeue", "-h", "-a", "-t", "all", "-o", "%i|%u|%T|%r|%N|%j"]


def now():
    return dt.datetime.now(dt.timezone.utc)


def snapshot():
    """-> (rows, error). rows is None when the poll itself failed."""
    try:
        r = subprocess.run(
            ["docker", "compose", "exec", "-T", "login1", *SQUEUE_ARGV],
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0:
        return None, f"rc={r.returncode}: {(r.stderr or '').strip()[:300]}"
    rows = []
    for line in r.stdout.splitlines():
        p = line.split("|", 5)
        if len(p) == 6:
            rows.append({"jobid": p[0], "user": p[1], "state": p[2], "reason": p[3], "node": p[4], "name": p[5]})
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--interval", type=float, default=4.0)
    ap.add_argument("--grace", type=float, default=20.0)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    preseed = json.loads((run / "preseed.json").read_text())
    cutoff = preseed["cutoff_epoch"]
    ledger = run / "ledger.jsonl"

    with open(ledger, "a") as f:
        while now().timestamp() < cutoff + args.grace:
            if run_control.is_halted(run):
                f.write(
                    json.dumps({"ts": now().isoformat(), "kind": "halt", "halt": run_control.read_halt(run)}) + "\n"
                )
                f.flush()
                print("[poll_tap] run HALTED; stopping poll", flush=True)
                break
            ts = now().isoformat()
            rows, error = snapshot()
            rec = {"ts": ts, "kind": "poll", "rows": rows}
            if error:
                rec["error"] = error
            f.write(json.dumps(rec) + "\n")
            f.flush()
            time.sleep(args.interval)
    print(f"[poll_tap] done -> {ledger}", flush=True)


if __name__ == "__main__":
    main()
