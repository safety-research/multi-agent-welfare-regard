#!/usr/bin/env python3
"""Reservation enforcer — the authoritative cutoff.

Sleeps until cutoff_epoch, then ENDS the reservation for each agent:
  1. `scancel -u <stem>` → every job still queued (running OR pending) at the cutoff becomes
     CANCELLED; anything that finished earlier stays COMPLETED. SLURM reservations don't hard-kill
     running jobs at their end, so this is done explicitly. Retried, and its result recorded.
  2. `pkill -u <stem>`, then `pkill -KILL -u <stem>`, on the agent's login node → kills the in-flight harness process INSIDE the
     container and anything else the agent left running, so nothing acts past the window. It has to
     happen in-container: a host-side timeout only kills the `docker exec` client.
  3. Appends one `reservation_end` event to the ledger with what was still queued and whether each
     step succeeded. poll_tap's post-cutoff snapshots show the resulting CANCELLED states.

Both steps are attempted for every agent even if an earlier step fails. The cancellations are made
by root — in sacct they read "CANCELLED by 0" — and are the orchestrator ending the window, not an
agent's choice.
"""
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import time

import run_control

HERE = pathlib.Path(__file__).parent


def now():
    return dt.datetime.now(dt.timezone.utc).timestamp()


def dexec(node, argv, timeout=30):
    """-> (returncode, stdout, stderr); returncode None on timeout."""
    try:
        r = subprocess.run(
            ["docker", "compose", "exec", "-T", node, *argv], cwd=HERE, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return None, "", "timeout"
    return r.returncode, r.stdout, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--grace",
        type=float,
        default=0.0,
        help="seconds after cutoff before enforcing (default 0 = hard stop at cutoff)",
    )
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    preseed = json.loads((run / "preseed.json").read_text())
    meta = json.loads((run / "run_meta.json").read_text())
    cutoff = preseed["cutoff_epoch"] + args.grace
    agents = [(a["stem"], a["login_node"]) for a in meta["agents"]]
    print(
        f"[reservation] arming for {preseed['cutoff_utc']}Z (+{args.grace}s); agents={[s for s, _ in agents]}",
        flush=True,
    )

    while now() < cutoff:
        if run_control.is_halted(run):
            print("[reservation] run HALTED before cutoff; standing down", flush=True)
            return
        time.sleep(min(2.0, max(0.05, cutoff - now())))

    report = {}
    for stem, node in agents:
        rc, out, err = dexec("login1", ["squeue", "-h", "-u", stem, "-o", "%i %T"])
        queued = [ln.split()[0] for ln in out.splitlines() if ln.strip()] if rc == 0 else None
        cancel = None
        for _ in range(3):
            cancel, _, cerr = dexec("login1", ["scancel", "-u", stem])
            if cancel == 0:
                break
            time.sleep(2)
        kill, _, _ = dexec(node, ["pkill", "-u", stem])  # rc 1 just means nothing was left running
        time.sleep(2)
        dexec(node, ["pkill", "-KILL", "-u", stem])  # anything the agent left running that ignores TERM
        report[stem] = {"queued_at_cutoff": queued, "scancel_rc": cancel, "pkill_rc": kill}
        if queued is None:
            report[stem]["squeue_error"] = (err or "").strip()[:200]

    with open(run / "ledger.jsonl", "a") as f:
        f.write(
            json.dumps(
                {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), "kind": "reservation_end", "agents": report}
            )
            + "\n"
        )
    print(f"[reservation] ENDED: {report}", flush=True)


if __name__ == "__main__":
    main()
