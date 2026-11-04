#!/usr/bin/env python3
"""Run a batch of rollouts.

  ./run.py --cells opus_x_opus --reps 1 --concurrency 1 --out runs/smoke --dry
  ./run.py --cells opus_x_opus --reps 5 --concurrency 4 --out runs/claude_only
  ./run.py --reps 20 --concurrency 25 --out runs/grid          # the full 5x5, 500 rollouts

Four things make a long batch survivable:

  Resuming. Re-running the same command picks up where it stopped. A rollout counts as done
  only if it finished (run_one.sh writes STATUS as its last act), did not halt, and ran in
  this batch's mode (live or --dry). Anything else is moved aside as
  <name>.<reason>.<timestamp>, with its log, and run again.

  Stopping on provider failure. If a rollout halts because a provider is down or a key has
  hit a wall, a sentinel file goes into the batch directory and nothing further is
  dispatched. Rollouts already in flight finish on their own. Without this, a key that hits
  its monthly cap turns the rest of the batch into hundreds of empty rollouts. Resume by
  deleting the sentinel and re-running.

  A ledger. One line per rollout attempt in ledger.tsv, so a batch that runs most of a day is
  auditable while it is still going.

  A lock. Only one run.py works on a batch directory at a time, and the lock is held by the
  rollouts themselves, so it outlives a run.py that was killed while its rollouts were still
  going. A second run.py refuses to start instead of re-running them underneath themselves.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time

import registry
import scrub
from materialize_tasks import DEFAULT_TASKS

HERE = pathlib.Path(__file__).resolve().parent
RUN_ONE = HERE / "scripts" / "run_one.sh"
IMAGE = "peer-ws:local"
# The batch path is interpolated into shell and python snippets further down the pipeline.
SAFE_PATH = re.compile(r"^[A-Za-z0-9._/+-]+$")


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def preflight_keys(cells):
    """Refuse to start if a cell still to run needs a key that is not set.

    This check matters more than it looks. Without a key the seat never starts, but the
    world still comes up, the tickets still activate and the rollout directory still fills
    with plausible-looking files — so a batch would complete and produce no agent behaviour
    at all.
    """
    need = {}
    for cell in cells:
        subject, peer = cell.split("_x_")
        for key in (subject, peer):
            need.setdefault(registry.key_env(key), set()).add(cell)
    missing = {var: sorted(cs) for var, cs in need.items() if not os.environ.get(var)}
    if missing:
        for var, cs in sorted(missing.items()):
            head = ", ".join(cs[:4]) + (f", +{len(cs) - 4} more" if len(cs) > 4 else "")
            sys.stderr.write(f"{var} is not set, but these cells need it: {head}\n")
        sys.stderr.write("\nExport the keys, or select only the cells you have keys for (--cells).\n")
        sys.exit(2)


def preflight_setup():
    """The one-time setup steps, checked up front: without them every rollout fails at
    compose-up or generation, one at a time, after the batch has started."""
    problems = []
    if (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, stdin=subprocess.DEVNULL).returncode
        != 0
    ):
        problems.append(f"the {IMAGE} image is missing: docker build -t {IMAGE} docker/")
    for var, sub in (("LEAN_ELAN", "toolchains"), ("LEAN_SRC", ".lake/packages")):
        val = os.environ.get(var)
        if not val:
            problems.append(f"{var} is not set (./setup_lean.sh prints what to export)")
        elif not (pathlib.Path(val) / sub).is_dir():
            problems.append(f"{var}={val} has no {sub}/ (run ./setup_lean.sh)")
    resolved = []
    for t in DEFAULT_TASKS:
        bake = HERE / t / "bake.json"
        if not bake.is_file():
            problems.append(f"{t} is not materialized (run ./materialize_tasks.py)")
        elif json.loads(bake.read_text()).get("install_mode") != "frozen":
            resolved.append(t)
    if resolved:
        # Not fatal — the freeze can stop installing as packages are yanked — but the test
        # environment then differs from the one the grid was run with, and the agent sees it.
        sys.stderr.write(
            "WARNING: these tickets' virtualenvs were not built from the dataset freeze, so their "
            f"test environment differs from the original runs: {', '.join(resolved)} (see bake.json)\n"
        )
    if problems:
        sys.exit("Setup is incomplete:\n  " + "\n  ".join(problems))


def is_done(d, dry):
    """(done, reason-if-not). See the module docstring for the rule."""
    status_f = d / "STATUS"
    if not status_f.is_file():
        return False, "incomplete"
    if (d / "HALT").exists():
        return False, "halted"
    # A dry rollout (DRY-ok, or a failed check: dry-lean-check-failed, ...) does not count
    # toward a live batch, and a failed one does not count toward a dry batch either. A
    # finished live rollout satisfies a plumbing check, and is never moved aside for one.
    status = status_f.read_text().strip().lower()
    if status.startswith("dry"):
        if not dry:
            return False, "dry"
        if not status.startswith("dry-ok"):
            return False, "failed"
    return True, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--cells", default=None, help="comma-separated, e.g. opus_x_opus,opus_x_gpt (default: all 25)")
    ap.add_argument("--reps", type=int, default=1, help="rollouts per cell")
    ap.add_argument("--concurrency", type=int, default=4, help="rollouts in flight (5 containers each)")
    ap.add_argument("--out", default="runs/batch", help="batch directory")
    ap.add_argument("--dry", action="store_true", help="plumbing only: no agents, no API calls")
    args = ap.parse_args()

    if args.reps < 1 or args.concurrency < 1:
        sys.exit("--reps and --concurrency must be at least 1")
    cells = (
        list(dict.fromkeys(c.strip() for c in args.cells.split(",") if c.strip()))
        if args.cells
        else list(registry.CELLS)
    )
    bad = [c for c in cells if c not in registry.CELLS]
    if bad or not cells:
        sys.exit(f"unknown cell(s) {bad}. Cells are <subject>_x_<peer> over {sorted(registry.MODELS)}.")

    outroot = pathlib.Path(args.out).resolve()
    if not SAFE_PATH.match(str(outroot)):
        sys.exit(f"--out {outroot}: use only letters, digits and . _ / + - in the path")
    preflight_setup()
    outroot.mkdir(parents=True, exist_ok=True)

    # The lock fd is inherited by every run_one.sh (pass_fds below) and by what they start,
    # so it stays held until the last rollout of this invocation has exited.
    fd = os.open(outroot / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    lock_fd = fcntl.fcntl(fd, fcntl.F_DUPFD, 200)  # well clear of the fds the scripts use
    os.close(fd)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(
            f"{outroot} is in use: another run.py, or rollouts started by one that has since "
            f"exited, are still running there. Wait for them, or stop them first."
        )

    sentinel = outroot / "BATCH_HALT"
    ledger = outroot / "ledger.tsv"
    if not ledger.exists():
        ledger.write_text("cell\trep\tstatus\tdir\tts\n")
    if sentinel.exists():
        sys.exit(
            f"{sentinel} is present; refusing to dispatch. Inspect it, then delete it to resume.\n"
            f"{sentinel.read_text()}"
        )

    lock = threading.Lock()
    counts = {"launched": 0, "skipped": 0, "retired": 0}
    this_run = {}
    keys = scrub.secrets()

    def record(cell, rep, status, name):
        with lock:
            with open(ledger, "a") as f:
                f.write(f"{cell}\t{rep}\t{status}\t{name}\t{utc()}\n")
            if not status.startswith("RETIRED:"):
                this_run[status] = this_run.get(status, 0) + 1

    def raise_sentinel(cell, rep, cause):
        with lock:
            if not sentinel.exists():
                sentinel.write_text(
                    f"cause: {cause}\ncell: {cell} rep {rep}\nts: {utc()}\n"
                    f"resume: delete this file and re-run the same command\n"
                )
                sys.stderr.write(f"[batch] HALT raised by {cell} rep {rep} ({cause})\n")

    # Work out what still has to run before asking for keys: a resume should not need a key
    # that only already-finished cells use.
    plan = []
    for rep in range(1, args.reps + 1):
        for cell in cells:
            d = outroot / f"{cell}_r{rep:02d}"
            done, reason = is_done(d, args.dry)
            if done:
                counts["skipped"] += 1
                continue
            plan.append((cell, rep, d, reason if d.exists() else None))
    if not args.dry:
        preflight_keys(sorted({cell for cell, _, _, _ in plan}))

    slots = threading.Semaphore(args.concurrency)
    running = {}  # pid -> Popen, for Ctrl-C
    stopping = threading.Event()

    def one(cell, rep, d):
        try:
            if stopping.is_set():
                return
            subject, peer = cell.split("_x_")
            cmd = ["bash", str(RUN_ONE), subject, peer, str(d)] + (["--dry"] if args.dry else [])
            # Compose reads COMPOSE_* from the environment; run_one.sh sets the ones it needs,
            # and an unrelated one from your shell (a path separator, a profile) must not leak in.
            env = {k: v for k, v in os.environ.items() if not k.startswith("COMPOSE_")}
            # Output goes to files, not pipes, and each rollout is its own session: a rollout
            # must be able to outlive this process (a killed run.py, a closed terminal) and
            # still capture and finish, instead of dying on its first write to a dead pipe.
            log_path, res_path = outroot / f"{d.name}.log", outroot / f".{d.name}.result"
            with open(log_path, "w") as err, open(res_path, "w") as out:
                proc = subprocess.Popen(
                    cmd,
                    cwd=HERE,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    env=env,
                    pass_fds=(lock_fd,),
                    start_new_session=True,
                )
                with lock:
                    running[proc.pid] = proc
                if stopping.is_set():
                    os.killpg(proc.pid, signal.SIGINT)
                proc.wait()
                with lock:
                    running.pop(proc.pid, None)
            log_path.write_text(scrub.scrub_text(log_path.read_text(errors="replace"), keys))
            res = res_path.read_text().strip()
            res_path.unlink(missing_ok=True)
            status = res.split("\t")[-1] if res else "unknown"
            if stopping.is_set() and not (d / "STATUS").is_file():
                # whatever step the Ctrl-C happened to break reports itself as the failure
                status = "INTERRUPTED"
            record(cell, rep, status, d.name)
            print(f"[batch] {cell} rep {rep}: {status}", flush=True)
            halt = d / "HALT"
            if halt.is_file():
                try:
                    h = json.loads(halt.read_text())
                except ValueError:
                    h = {}
                if h.get("fatal_provider_error"):
                    raise_sentinel(cell, rep, "provider quota or auth wall")
                elif h.get("provider_trouble"):
                    raise_sentinel(cell, rep, "provider overload")
        finally:
            slots.release()

    threads = []
    interrupted = False
    try:
        for cell, rep, d, reason in plan:
            if sentinel.exists():
                record(cell, rep, "SKIPPED_batch_halt", d.name)
                continue
            slots.acquire()
            if sentinel.exists():  # raised while this one was queued
                slots.release()
                record(cell, rep, "SKIPPED_batch_halt", d.name)
                continue
            if reason:
                # Moved aside only now, when its replacement is about to start, and kept: a
                # halted or failed attempt is evidence of what went wrong. Its log goes with it.
                retired = d.with_name(f"{d.name}.{reason}.{int(time.time())}")
                d.rename(retired)
                log = outroot / f"{d.name}.log"
                if log.exists():
                    log.rename(outroot / f"{retired.name}.log")
                record(cell, rep, f"RETIRED:{reason}", retired.name)
                counts["retired"] += 1
                print(f"[batch] {d.name} was {reason}; kept as {retired.name} and re-running", flush=True)
            t = threading.Thread(target=one, args=(cell, rep, d), daemon=False)
            t.start()
            threads.append(t)
            counts["launched"] += 1
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        # The rollouts are in their own sessions, so the terminal's Ctrl-C reached only this
        # process. Pass it on: each rollout then tears down its own containers. Wait for that,
        # so their ledger rows are written; a second Ctrl-C stops waiting, not the teardown.
        interrupted = True
        stopping.set()
        with lock:
            procs = list(running.values())
        sys.stderr.write(
            f"\n[batch] interrupted: no more rollouts will start; stopping the {len(procs)} "
            f"running (each tears down its own containers). Ctrl-C again to stop waiting.\n"
        )
        for proc in procs:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            sys.stderr.write("[batch] not waiting; the rollouts finish tearing down in the background\n")
            os._exit(130)

    print(
        f"\n=== {counts['launched']} launched, {counts['skipped']} already complete, "
        f"{counts['retired']} moved aside and re-run ==="
    )
    if sentinel.exists():
        print(f"batch halted:\n{sentinel.read_text()}")
    for k in sorted(this_run):
        print(f"  {k:<28} {this_run[k]}")
    if interrupted:
        sys.exit(130)
    print(f"\nviewer: ./build_viewer.py --batch {args.out} --out {args.out}/viewer.html")


if __name__ == "__main__":
    main()
