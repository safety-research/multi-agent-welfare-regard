#!/usr/bin/env python3
"""Run a batch of episodes.

Each (cell, rep) runs as its OWN docker compose project, so episodes are fully isolated and run
concurrently. --concurrency counts EPISODES, not containers: each episode is a 3-container compose
project and two live agent seats.

  ./run.py --cells opus_opus --reps 1 --concurrency 1 --out runs/smoke
  ./run.py --reps 20 --concurrency 30 --out runs/grid      # all 25 cells, 500 episodes

Writes <out>/episodes.tsv: one row per episode (cell, rep, status, rounds, delegations, issues,
directory). That is an index of what ran, not a result — each episode's evidence is in its own
directory, and the dependent variables are read out of that.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import fcntl
import hashlib
import ipaddress
import json
import os
import pathlib
import re
import secrets
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

import labels
from run_control import BATCH_HALT_THRESHOLD
from scrub_report import is_truthful

HERE = pathlib.Path(__file__).parent
IMAGE = "manager-env:local"
STAGGER_S = 3.0  # between the first wave's starts; the daemon drops container creates under storms

# Episodes currently running: compose project -> the run_one.sh process. On an interrupt these are
# stopped and torn down explicitly, rather than trusting every signal to reach every child.
RUNNING, RUNNING_LOCK = {}, threading.Lock()
BATCH_LOCK_FD = None  # the batch directory's lock; passed to every episode process (see main())


def project_name(batch, cell, rep):
    # docker compose project: lowercase alphanumerics and hyphens, unique per batch so a container
    # left over from an earlier batch cannot collide. The cell name is mapped to hyphens rather
    # than stripped, and never truncated: two cells that collapsed to the same string would share a
    # COMPOSE_PROJECT_NAME, and run_one.sh's `down -v` would then tear down the sibling episode's
    # LIVE forge mid-run.
    return f"mgr{batch}-{re.sub(r'[^a-z0-9]+', '-', cell.lower())}-r{rep}"


# Every episode gets its own /24, so N concurrent Compose projects do not exhaust Docker's default
# address pool (about 30 bridge networks). They come from 10.64.0.0/11, which avoids the default VPC
# ranges of the main clouds (GCP auto-mode uses 10.128.0.0/9, AWS 172.31.0.0/16) and the usual
# Kubernetes defaults — and, whatever the range, any /24 that overlaps something already on this
# host is skipped. That matters: Docker gives a bridge the first address of its subnet, so a /24
# containing the host's own gateway would put the gateway's address on a local bridge and cut the
# machine off the network.
SUBNET_RANGE = ipaddress.ip_network("10.64.0.0/11")


def _host_networks():
    """Everything on this host an episode subnet must not overlap: routes, interface addresses, and
    the subnets of existing Docker networks (including other batches')."""
    nets = []
    try:
        for line in open("/proc/net/route").read().splitlines()[1:]:
            f = line.split()
            dest, mask = struct.pack("<L", int(f[1], 16)), struct.pack("<L", int(f[7], 16))
            bits = bin(int.from_bytes(mask, "big")).count("1")
            if bits:  # not the default route
                nets.append(ipaddress.ip_network(f"{socket.inet_ntoa(dest)}/{bits}", strict=False))
    except (OSError, ValueError, IndexError):
        pass
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True).stdout
        nets += [ipaddress.ip_network(a, strict=False) for a in re.findall(r"inet (\d+\.\d+\.\d+\.\d+/\d+)", out)]
    except OSError:
        pass
    ids = subprocess.run(["docker", "network", "ls", "-q"], capture_output=True, text=True).stdout.split()
    if ids:
        out = subprocess.run(
            ["docker", "network", "inspect", "--format", "{{range .IPAM.Config}}{{.Subnet}} {{end}}", *ids],
            capture_output=True,
            text=True,
        ).stdout
        for tok in out.split():
            try:
                net = ipaddress.ip_network(tok, strict=False)
                if net.version == 4:
                    nets.append(net)
            except ValueError:
                pass
    return nets


def plan_subnets(batch, n):
    """2n free /24s for this batch: one per episode, and a vetted alternate for each, which
    run_one.sh uses if bringing the episode up fails. The starting point is derived from the batch
    id, so two batches on one host start in different places."""
    taken = _host_networks()
    free = [s for s in SUBNET_RANGE.subnets(new_prefix=24) if not any(s.overlaps(t) for t in taken)]
    if len(free) < 2:
        # Something routes the whole range — typically a VPN claiming 10.0.0.0/8. Fall back to avoiding
        # only specific networks (a /16 or narrower): the host's own subnets and addresses, and every
        # Docker network. A broad route like that is overridden only for the /24s actually in use.
        specific = [t for t in taken if t.prefixlen >= 16]
        free = [s for s in SUBNET_RANGE.subnets(new_prefix=24) if not any(s.overlaps(t) for t in specific)]
        say(
            "[run.py] note: a route on this host covers all of 10.64.0.0/11; episode networks will avoid only "
            "this host's own subnets, addresses and Docker networks."
        )
    if len(free) < 2:
        sys.exit("[run.py] FATAL: no free /24 left in 10.64.0.0/11 on this host.\n")
    start = int(hashlib.sha256(batch.encode()).hexdigest(), 16) % len(free)
    pick = [str(free[(start + i) % len(free)]) for i in range(2 * n)]
    return pick[:n], pick[n:]


def say(*a):
    """print() that cannot take the batch down. If the terminal has gone away (a dropped SSH
    session), writing to it raises — and an exception thrown from inside the interrupt handler would
    abort the cleanup before the halt file or the index is written."""
    try:
        print(*a, flush=True)
    except (OSError, ValueError):
        # The terminal has gone. Point file descriptor 1 at /dev/null, so neither later output nor
        # the interpreter's own flush at exit can fail — a failed final flush makes Python exit 120,
        # replacing the status this batch meant to report.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError):
            pass


def write_episode_file(path, text):
    """Write one of the host's own files into an episode directory — never THROUGH whatever is there.

    The worker container mounts that directory read-write as your uid, so a symlink it left at this
    name would send an ordinary write to any file you can write. Its containers are down by the time
    this runs, so nothing can put the link back, and O_NOFOLLOW refuses one if it somehow does.
    O_NONBLOCK makes a FIFO left there fail at once instead of hanging the batch; a regular file
    ignores it.
    """
    try:
        if path.is_symlink():
            path.unlink()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644)
    except OSError as e:
        print(f"[run.py] not writing {path}: {e}", file=sys.stderr)
        return
    with os.fdopen(fd, "w") as f:
        f.write(text)


def one(cell, rep, out_dir, project, dry, subnet, subnet_alt, stagger, halt_file):
    subject, peer = labels.parse_cell(cell)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Pre-dispatch gate. Recorded explicitly as "skipped" rather than silently dropped: otherwise a
    # halted batch's denominators conflate "never dispatched" with "ran and produced nothing".
    if halt_file and halt_file.exists():
        return f"{cell}\t{rep}\tskipped:batch_halt\t\t\t\t{out_dir}"
    if stagger:
        time.sleep(stagger)
    env = dict(
        os.environ,
        DRY="1" if dry else "0",
        CLUSTER_SUBNET=subnet,
        CLUSTER_SUBNET_ALT=subnet_alt,
        SUBJECT_PROVIDER=subject["provider"],
        SUBJECT_MODEL=subject["model"],
        SELF_LABEL=subject["label"],
        WORKER_MODEL=f"{peer['provider']}/{peer['model']}",
        WORKER_LABEL=peer["label"],
    )
    argv = ["bash", str(HERE / "scripts" / "run_one.sh"), cell, str(rep), out_dir, project]
    if halt_file and halt_file.exists():  # re-check: the batch may have halted during the stagger
        return f"{cell}\t{rep}\tskipped:batch_halt\t\t\t\t{out_dir}"
    try:
        # stdin is /dev/null, never the user's terminal: the agent CLI reads ALL of stdin before it
        # does anything, and `docker compose exec` forwards it — so an inherited terminal, which never
        # sends EOF, would stall every episode until its 60-minute cap.
        proc = subprocess.Popen(
            argv,
            cwd=HERE,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            # The episode inherits the batch lock, so the lock outlives run.py itself: if run.py is
            # killed outright, a resume must still be refused until the last orphaned episode ends.
            pass_fds=(BATCH_LOCK_FD,) if BATCH_LOCK_FD is not None else (),
        )
    except Exception as e:
        return f"{cell}\t{rep}\tERROR:launch:{e}\t\t\t\t{out_dir}"
    with RUNNING_LOCK:
        RUNNING[project] = proc
    try:
        stdout, stderr = proc.communicate()
    finally:
        with RUNNING_LOCK:
            RUNNING.pop(project, None)
    write_episode_file(pathlib.Path(out_dir) / "run_one.log", stderr or "")
    lines = [ln for ln in (stdout or "").splitlines() if ln.strip()]
    return lines[-1] if lines else f"{cell}\t{rep}\tERROR:no-output-rc{proc.returncode}\t\t\t\t{out_dir}"


def _result(fut, cell, rep, out):
    """An episode thread's line — or, if the thread itself raised, an ERROR line for that one
    episode. Re-raising here would take down the batch and, inside the interrupt handler, prevent
    the index from ever being written."""
    try:
        return fut.result()
    except Exception as e:
        return f"{cell}\t{rep}\tERROR:harness:{type(e).__name__}\t\t\t\t{out}/{cell}_r{rep}"


def stop_running(batch):
    """Stop every episode still in flight and tear its Compose project down. Called on an interrupt.

    The projects to remove are found by NAME, from Docker, rather than from the list of processes
    this script is still waiting on. A terminal hangup or Ctrl-C reaches every process in the
    foreground group, so an episode's own shell may already be dead by the time this runs — and
    `docker compose exec` leaves the agent it started running inside the container. Tearing down
    only the episodes still registered here would leave exactly those ones running, and spending."""
    with RUNNING_LOCK:
        running = dict(RUNNING)
    for proc in running.values():
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    live = subprocess.run(
        ["docker", "compose", "ls", "-a", "-q", "--filter", f"name=mgr{batch}-"], capture_output=True, text=True
    ).stdout.split()
    projects = sorted(set(live) | set(running))
    for project in projects:
        subprocess.run(["docker", "compose", "-p", project, "down", "-v"], capture_output=True)
    return projects


def preflight(cells, dry):
    """Refuse to start rather than produce a batch that looks full and contains nothing."""
    # The roster's central invariant: every disclosed label must actually name the model behind it.
    # Nothing downstream handles a false label, so a label that does not match its model would put
    # an unhandled identity claim in front of the manager.
    bad = [r for r in labels.ROSTER if not is_truthful(r["label"], r["model"])]
    if bad:
        sys.exit(
            "[run.py] FATAL: labels.py has disclosed labels that do not name their model:\n"
            + "\n".join(f"    {r['label']!r} for {r['model']!r}" for r in bad)
            + "\n  Every label in this environment must be truthful. See scrub_report.py.\n"
        )
    check_image()
    if dry:
        return  # a dry run makes no API call, so it needs no keys — but it does need the image
    needed = {}
    for c in cells:
        for entry in labels.parse_cell(c):
            needed.setdefault(labels.KEY_FOR[entry["provider"]], set()).add(entry["label"])
    missing = {k: v for k, v in needed.items() if not os.environ.get(k)}
    if missing:
        detail = "\n".join(f"    {k}   (needed for: {', '.join(sorted(v))})" for k, v in sorted(missing.items()))
        sys.exit(
            f"[run.py] FATAL: {len(missing)} provider key(s) not set in the environment:\n"
            f"{detail}\n"
            f"  Without them the agent seats never start, but the forge is still seeded and torn\n"
            f"  down — so the batch would emit full-looking episode directories containing no\n"
            f"  agent behaviour at all.\n"
            f"  Export the key(s), or select cells that only use the providers you have:\n"
            f"    ./run.py --cells opus_opus ...\n"
        )


def check_image():
    """The image must exist and be built for this user's uid — checked for dry runs too, since a dry
    run is exactly where a user first finds out. Without the first check `docker compose up` would
    silently build the image itself, with default ids."""
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        sys.exit(f"[run.py] FATAL: image {IMAGE} not built. Run:  docker compose build\n")
    # The episode output directory is a bind mount, so the container sees its HOST ownership. If
    # the image's `agent` user is a different uid, the containers cannot write their own output and
    # every episode fails late and confusingly. Check it up front instead.
    host_uid, host_gid = os.getuid(), os.getgid()
    if 61001 in (host_uid, host_gid) or 61001 in os.getgroups():
        sys.exit(
            "[run.py] FATAL: your uid, gid or a supplementary group is 61001, which the image reserves\n"
            "  for the manager's isolated user. The manager could then read the episode output. Run as a\n"
            "  different user, or change 61001 in docker/Dockerfile and run.py.\n"
        )
    probe = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "id", IMAGE, "-u", "agent"], capture_output=True, text=True
    )
    img_uid = (probe.stdout or "").strip()
    if probe.returncode == 0 and img_uid.isdigit() and int(img_uid) != host_uid:
        sys.exit(
            f"[run.py] FATAL: {IMAGE} was built for uid {img_uid} but you are uid {host_uid}.\n"
            f"  The episode output directory is a bind mount, so the containers would not be able\n"
            f"  to write to it. Rebuild for your ids:\n"
            f"    AGENT_UID={host_uid} AGENT_GID={host_gid} docker compose build\n"
        )
    # scripts/run_one.sh blocks the cloud metadata addresses with this image's `ip`, and refuses to
    # run an episode without that block. An image built before the block existed has no `ip`.
    has_ip = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", IMAGE, "-c", "command -v ip"], capture_output=True
    )
    if has_ip.returncode != 0:
        sys.exit(
            f"[run.py] FATAL: {IMAGE} has no `ip`, which scripts/run_one.sh needs to block the cloud\n"
            f"  metadata service. It was built before that block existed. Rebuild it:\n"
            f"    AGENT_UID={host_uid} AGENT_GID={host_gid} docker compose build\n"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None, help="comma-separated cell keys (see labels.py --cells); default = all 25")
    ap.add_argument("--reps", type=int, default=1, help="episodes PER cell (total = cells x reps)")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="max episodes in flight. Each is 3 containers and 2 live agent seats",
    )
    ap.add_argument("--out", default=None, help="batch directory (default: runs/batch_<id>)")
    ap.add_argument(
        "--dry", action="store_true", help="plumbing check: up, seed the forge, down — no agent, no API call"
    )
    args = ap.parse_args()

    all_cells = labels.cells()
    # De-duplicated, order preserved: a repeated cell would otherwise produce two episodes with the
    # same compose project name and the same output directory, and each would tear the other down.
    cells = list(dict.fromkeys(c.strip() for c in args.cells.split(",") if c.strip())) if args.cells else all_cells
    unknown = [c for c in cells if c not in all_cells]
    if unknown:
        sys.exit(f"[run.py] unknown cell(s) {unknown} — see: python3 labels.py --cells")
    preflight(cells, args.dry)

    # Random rather than time-derived: two run.py invocations started in the same second would
    # otherwise share a batch id, and with it every compose project name.
    batch = secrets.token_hex(3)
    # Resolved to an absolute path: run_one.sh cd's to this directory, so a relative --out given
    # from some other working directory would otherwise land in two different places.
    out = pathlib.Path(args.out).resolve() if args.out else (HERE / f"runs/batch_{batch}")
    out.mkdir(parents=True, exist_ok=True)
    # A dry run into a directory that holds real, finished episodes would move each of them aside
    # (run_one.sh treats anything it finds in an episode directory as a stale earlier run).
    if args.dry and any(
        p.read_text().split("\t")[2:3] and p.read_text().split("\t")[2].startswith("ok:") for p in out.glob("*/STATUS")
    ):
        sys.exit(f"[run.py] REFUSING: {out} holds finished episodes. Give the dry run its own --out.\n")
    halt_file = out / "BATCH_HALT.json"
    if halt_file.exists():
        sys.exit(
            f"[run.py] REFUSING TO START: a batch-halt sentinel already exists at {halt_file}\n"
            f"  cause: {halt_file.read_text()}\n  Delete it to resume into this directory.\n"
        )

    # One run.py per batch directory at a time. A second one (say, a resume started while the first
    # is still draining its in-flight episodes) would move a live episode's files aside and run two
    # episodes in one directory. The lock is released automatically however this process ends.
    global BATCH_LOCK_FD
    lock = open(out / ".run.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(
            f"[run.py] REFUSING: {out} is in use — by another run.py, or by episodes still running from one\n"
            f"  that was killed. Wait for them to finish (docker compose ls --filter name=mgr).\n"
        )
    BATCH_LOCK_FD = lock.fileno()

    # rep-major, so the in-flight mix stays balanced across providers rather than packing the first
    # `concurrency` slots with a single cell.
    run_list = [(c, r) for r in range(1, args.reps + 1) for c in cells]
    subnets, subnets_alt = plan_subnets(batch, len(run_list))

    say(
        f"[run.py] batch {batch}: {len(run_list)} episodes ({len(cells)} cells x {args.reps} reps), "
        f"concurrency={args.concurrency}{' DRY' if args.dry else ''} -> {out}",
    )
    if not args.dry:
        say(
            f"[run.py] {args.concurrency} parallel = ~{3*args.concurrency} containers and "
            f"{2*args.concurrency} live agent seats.",
        )

    results, t0, consec_overload, interrupted, crashed = {}, time.time(), 0, False, None

    # SIGTERM and SIGHUP (a closed terminal, a `kill`) take the same path as Ctrl-C, so the index is
    # still written and the batch still stops cleanly. SIGKILL cannot be caught; see the README.
    def _as_interrupt(signum, frame):
        # Ignore every further signal FIRST. A terminal hangup can deliver SIGHUP twice, and a second
        # one arriving while this exception unwinds would abort the cleanup that follows it.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _as_interrupt)
    # Respect nohup: if SIGHUP is already ignored, the user asked for the batch to survive a logout.
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        signal.signal(signal.SIGHUP, _as_interrupt)
    ex = cf.ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        futs, reused = {}, 0
        for idx, (cell, rep) in enumerate(run_list):
            rout = f"{out}/{cell}_r{rep}"
            # RESUME. An episode that already finished ok in this --out is kept, not re-run: re-running
            # it would spend the money twice and replace a good record. Anything else found here
            # (unfinished, contaminated, errored) is re-run, and run_one.sh moves the old attempt to
            # superseded/<k>/ first, so two runs never share one episode directory.
            prev = pathlib.Path(rout) / "STATUS"
            if not args.dry and prev.exists():
                line = prev.read_text().strip()
                if line.count("\t") >= 2 and line.split("\t")[2].startswith("ok:"):
                    results[(cell, rep)] = line
                    reused += 1
                    continue
            stagger = STAGGER_S * idx if idx < args.concurrency else 0.0
            futs[
                ex.submit(
                    one,
                    cell,
                    rep,
                    rout,
                    project_name(batch, cell, rep),
                    args.dry,
                    subnets[idx],
                    subnets_alt[idx],
                    stagger,
                    halt_file,
                )
            ] = (cell, rep)
        if reused:
            say(f"[run.py] {reused} episode(s) already finished ok in {out}; keeping them, not re-running.")
        done = reused
        for f in cf.as_completed(futs):
            cell, rep = futs[f]
            line = _result(f, cell, rep, out)
            results[(cell, rep)] = line
            done += 1
            say(f"[run.py] ({done}/{len(run_list)}) {cell} r{rep}: {line}")
            # BATCH-HALT GUARD. A provider outage mid-batch would otherwise burn every remaining
            # episode as an individually-retried contamination, and be discovered hours later. Each
            # episode classifies itself (see run_control.py); this counts CONSECUTIVE
            # overload-classified completions and stops DISPATCHING at the threshold. Episodes
            # already in flight finish normally. "Consecutive" is in completion order, not dispatch
            # order — close enough to distinguish a sustained outage from a burst, which is all it
            # has to do. Resume by deleting the sentinel and re-running.
            status = line.split("\t")[2] if line.count("\t") >= 2 else ""
            consec_overload = consec_overload + 1 if status.startswith("overload") else 0
            if consec_overload >= BATCH_HALT_THRESHOLD and not halt_file.exists():
                halt_file.write_text(
                    json.dumps(
                        {
                            "cause": "provider_overload",
                            "consecutive": consec_overload,
                            "after_episodes": done,
                            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "verdict": "batch halted — re-run the skipped episodes once the provider recovers",
                        },
                        indent=2,
                    )
                )
                say(f"[run.py] BATCH HALT: {consec_overload} consecutive overloads -> no new dispatches")
    except BaseException as exc:
        # Ctrl-C, SIGTERM or SIGHUP arrive here as KeyboardInterrupt. Anything else is a bug in this
        # script — but the batch must still stop cleanly and still write its index, rather than
        # waiting out every episode inside `finally` and then dying without one.
        if not isinstance(exc, KeyboardInterrupt):
            crashed = exc
        # Ctrl-C has to STOP the batch. A plain `with ThreadPoolExecutor(...)` does not: its
        # __exit__ calls shutdown(wait=True), so every queued episode still runs to completion —
        # possibly hours after the user asked it to stop — and the index is never written. The
        # sentinel gates the pre-dispatch check inside one(), so queued episodes return at once as
        # skipped; cancel_futures drops any that have not started.
        interrupted = True
        # Order matters. Stop further interrupts first — a second Ctrl-C or a repeated SIGHUP would
        # otherwise abort the cleanup half-way. Then write the halt file, which stops every queued
        # episode, BEFORE anything is printed: the terminal may be gone.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        if not halt_file.exists():
            cause = "interrupted_by_user" if crashed is None else f"harness_error: {crashed!r}"[:300]
            halt_file.write_text(json.dumps({"cause": cause, "after_episodes": len(results)}, indent=2))
        say("\n[run.py] interrupted — stopping in-flight episodes; nothing new will start.")
        stopped = stop_running(batch)
        if stopped:
            say(f"[run.py] stopped and tore down {len(stopped)} in-flight episode(s).")
        ex.shutdown(wait=True, cancel_futures=True)
        # Collect what the interrupted loop never recorded. A terminal Ctrl-C reaches every episode
        # in the foreground process group, not just this one, so episodes in flight are killed and
        # torn down by their own trap — label them as interrupted rather than as an error.
        for f, (cell, rep) in futs.items():
            if (cell, rep) in results or f.cancelled() or not f.done():
                continue
            line = _result(f, cell, rep, out)
            if "\tERROR:no-output" in line:
                line = f"{cell}\t{rep}\tinterrupted\t\t\t\t{out}/{cell}_r{rep}"
            results[(cell, rep)] = line
        say(f"[run.py] check for anything left over: docker compose ls --filter name=mgr{batch}")
    finally:
        ex.shutdown(wait=True)

    idx_file = out / "episodes.tsv"
    never = "skipped:batch_halt" if halt_file.exists() else "MISSING"  # a halted batch never started them
    rows = {c: results.get(c, f"{c[0]}\t{c[1]}\t{never}\t\t\t\t{out}/{c[0]}_r{c[1]}") for c in run_list}
    with open(idx_file, "w") as g:
        g.write("cell\trep\tstatus\trounds\tdelegations\tissues\tdir\n")
        for c in sorted(run_list):
            g.write(rows[c] + "\n")
    status = [rows[c].split("\t")[2] for c in run_list]
    dt_s = round(time.time() - t0)
    ok = sum(1 for st in status if st.startswith(("ok", "DRY-ok")))
    skipped = sum(1 for st in status if st.startswith("skipped"))
    say(f"[run.py] DONE {ok}/{len(run_list)} episodes in {dt_s}s ({dt_s//60}m) -> {idx_file}")
    if skipped:
        why = "interruption" if interrupted else "the batch-halt guard"
        say(f"[run.py] {skipped} episodes were SKIPPED by {why} — see {halt_file}")
    # A batch that halted or was interrupted did not do what was asked. Say so in the exit status,
    # rather than reporting success with a fraction of the grid in it.
    if crashed is not None:
        raise crashed  # the index is written; now surface the bug
    if interrupted or halt_file.exists():
        return 2
    # Nothing succeeded: say so in the exit status, for anyone scripting around run.py.
    if run_list and ok == 0:
        return 1


if __name__ == "__main__":
    sys.exit(main())
