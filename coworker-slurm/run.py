#!/usr/bin/env python3
"""Run a batch of rollouts.

Each (cell, rep) runs as its OWN docker compose project, so rollouts are fully isolated and run
concurrently. --concurrency counts ROLLOUTS, not containers: each rollout is a 9-container compose
project and two live agent seats, so the ceiling is your host, not the API.

  ./run.py --cells sopus_popus --out runs/first
  ./run.py --reps 20 --concurrency 60 --out runs/grid      # all 25 cells, 500 rollouts

Writes <out>/rollouts.tsv (one row per rollout: cell, rep, status, directory), rewritten as each
rollout finishes, and a live log per rollout in <out>/logs/. Re-running the same command resumes:
rollouts already `ok` in <out> are kept, everything else is run again.
"""
import argparse
import concurrent.futures as cf
import ipaddress
import os
import pathlib
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import yaml
from gen_scenario import PI_ROUTING

HERE = pathlib.Path(__file__).parent
# Candidate address space for per-rollout networks. Each rollout takes one /24 (split into three
# /26s, see compose.yml) for as long as it runs, so only `concurrency` of them are in use at once.
SUBNET_POOL = os.environ.get("COWORKER_SUBNET_POOL", "10.100.0.0/14")

_running = set()  # live run_one.sh processes, for forwarding a termination signal
_running_lock = threading.Lock()
_stopping = threading.Event()  # set on interrupt: no new rollout may start after it


def project_name(batch):
    # The compose project name reaches the agents: containers are named after it, Docker's DNS
    # reverse-resolves their addresses to those names, and SLURM records the submitting host that
    # way (`scontrol show job` prints it as AllocNode). So it must say nothing about the cell or the
    # rep — only a batch id (to find leftovers) and a random, unique suffix.
    return f"cws{batch}-{secrets.token_hex(4)}"


def _cmd(argv):
    """-> (returncode, stdout); returncode None if the command could not run."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return None, ""


DOCKER_IFACE = re.compile(r"^(docker\d*|br-[0-9a-f]+|veth\w*)$")


def docker_subnets():
    fmt = "{{range .IPAM.Config}}{{.Subnet}} {{end}}"
    for _ in range(3):  # a network removed between `ls` and `inspect` fails the whole inspect; rescan
        _, ids = _cmd(["docker", "network", "ls", "-q"])
        if not ids.split():
            return []
        rc, out = _cmd(["docker", "network", "inspect", "-f", fmt, *ids.split()])
        if rc == 0:
            return [s for s in out.split() if ":" not in s]
    raise RuntimeError("could not list existing Docker networks")


def blocked_networks():
    """Address space a new rollout network must not overlap: every host route, every existing
    Docker network, and the /16 around every non-Docker host address and gateway — a cloud VM's own
    subnet is often reached through a /32 gateway route rather than a subnet route, so the routing
    table alone does not show it."""
    nets = []

    def add(text, prefix=None):
        try:
            nets.append(ipaddress.ip_network(text if prefix is None else f"{text}/{prefix}", strict=False))
        except ValueError:
            pass

    kinds = {"local", "broadcast", "unicast", "blackhole", "unreachable", "prohibit", "anycast", "multicast"}
    for line in _cmd(["ip", "-4", "-o", "route", "show", "table", "all"])[1].splitlines():
        tok = line.split()
        if not tok:
            continue
        dest = tok[1] if tok[0] in kinds and len(tok) > 1 else tok[0]
        if dest != "default":
            add(dest)
        if "via" in tok:
            add(tok[tok.index("via") + 1], 16)
    for line in _cmd(["ip", "-4", "-o", "addr", "show"])[1].splitlines():
        m = re.search(r"^\d+:\s+(\S+)\s+inet (\d+\.\d+\.\d+\.\d+)/(\d+)", line)
        if m:
            add(m.group(2), m.group(3))
            if not DOCKER_IFACE.match(m.group(1)):
                add(m.group(2), 16)
    for sub in docker_subnets():
        add(sub)
    return [n for n in nets if n.prefixlen > 0]


class SubnetPool:
    """Hands out /24s that overlap nothing on the host, re-checking the host at every allocation
    so that networks created since (by another batch, say) are avoided too."""

    def __init__(self, cidr):
        self.candidates = list(ipaddress.ip_network(cidr).subnets(new_prefix=24))
        self.held, self.lock = set(), threading.Lock()

    def acquire(self, avoid=()):
        with self.lock:
            blocked = blocked_networks()
            for n in self.candidates:
                if n in self.held or n in avoid or any(n.overlaps(b) for b in blocked):
                    continue
                self.held.add(n)
                return n
        raise RuntimeError(f"no free /24 left in {SUBNET_POOL}; set COWORKER_SUBNET_POOL to a free range")

    def release(self, n):
        with self.lock:
            self.held.discard(n)


def one(cell, rep, rout, project, dry, pool, log_path):
    avoid = set()
    for _ in range(3):
        try:
            net = pool.acquire(avoid)
        except RuntimeError as e:
            return f"{cell}\t{rep}\tERROR:subnet:{e}\t{rout}"
        cluster, acct, chatstore = (str(s) for s in list(net.subnets(new_prefix=26))[:3])
        env = dict(os.environ, DRY="1" if dry else "0", PYTHON=sys.executable)
        env.update(CLUSTER_SUBNET=cluster, ACCT_SUBNET=acct, CHATSTORE_SUBNET=chatstore)
        argv = ["bash", str(HERE / "scripts" / "run_one.sh"), cell, str(rout), project]
        try:
            with open(log_path, "a") as logf:
                logf.write(f"=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} project {project} subnet {net}\n")
                logf.flush()
                # Start and register under one lock, and not at all once an interrupt has begun: a
                # rollout that slipped in after the interrupt would otherwise run to completion.
                with _running_lock:
                    if _stopping.is_set():
                        pool.release(net)
                        return f"{cell}\t{rep}\tCANCELLED\t"
                    p = subprocess.Popen(argv, cwd=HERE, stdout=subprocess.PIPE, stderr=logf, text=True, env=env)
                    _running.add(p)
                try:
                    out, _ = p.communicate()
                finally:
                    with _running_lock:
                        _running.discard(p)
        except OSError as e:
            pool.release(net)
            return f"{cell}\t{rep}\tERROR:launch:{e}\t{rout}"
        finally:
            pool.release(net)
        lines = [ln for ln in (out or "").splitlines() if ln.strip()]
        row = lines[-1] if lines else f"{cell}\t{rep}\tERROR:no-output-rc{p.returncode}\t{rout}"
        clash = (rout / "compose_up.err").exists() and "overlap" in (rout / "compose_up.err").read_text().lower()
        if "\tERROR:compose-up\t" in row and clash:
            avoid.add(net)  # another batch took this range after we checked; try another
            continue
        if rout.is_dir():
            shutil.copyfile(log_path, rout / "run_one.log")
        return row
    return row


def install_signal_handlers():
    """Turn SIGINT, SIGTERM and SIGHUP into KeyboardInterrupt, which main() handles.

    SIGINT is installed even if it was inherited as ignored: a process started in the background by
    a non-interactive shell gets SIGINT ignored, and then neither this process nor its run_one.sh
    children (whose traps cannot override an inherited ignore) would react to Ctrl-C. An inherited
    ignore of SIGHUP or SIGTERM is kept: that is what `nohup` does, and it should keep working.
    """

    def terminate(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if sig != signal.SIGINT and signal.getsignal(sig) is signal.SIG_IGN:
            continue
        signal.signal(sig, terminate)


def write_index(path, run_list, results):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as g:
        g.write("cell\trep\tstatus\tdir\n")
        for cell, rep in sorted(run_list):
            g.write(results.get((cell, rep), f"{cell}\t{rep}\tPENDING\t") + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None, help="comma-separated cell keys from scenario.yaml; default = all 25")
    ap.add_argument("--reps", type=int, default=1, help="rollouts PER cell (total = cells x reps)")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="max rollouts in flight. Each is 9 containers and 2 agent seats; the "
        "binding constraint is host RAM/CPU, not the API",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="batch directory (default: a new runs/batch_<id>). Give it explicitly if you may want to "
        "resume: re-running with the same --out keeps rollouts that are already ok",
    )
    ap.add_argument(
        "--dry", action="store_true", help="plumbing check: gen, up, provision, preseed, down — no agents, no API calls"
    )
    args = ap.parse_args()
    install_signal_handlers()

    sc = yaml.safe_load(open(HERE / "scenario.yaml"))
    all_cells = list(sc["cells"])
    requested = [c.strip() for c in args.cells.split(",") if c.strip()] if args.cells else all_cells
    # De-duplicate, preserving order. A repeated cell would produce two rollouts writing to the SAME
    # output directory. Repeats are what --reps is for.
    cells, dropped = [], []
    for c in requested:
        (dropped if c in cells else cells).append(c)
    if dropped:
        print(f"[run.py] ignoring duplicate cell(s) {sorted(set(dropped))} — use --reps for repeats", flush=True)
    unknown = [c for c in cells if c not in all_cells]
    if unknown:
        sys.exit(f"[run.py] unknown cell(s) {unknown} — known: {all_cells}")

    # PRE-FLIGHT 1: every provider key these cells need must be present BEFORE anything is
    # provisioned. Without it the failure is silent and expensive rather than loud: the seats fail
    # to start, the preseeded jobs complete on their own, and the batch looks successful while
    # containing no agent behaviour at all.
    if not args.dry:
        needed = {}
        for c in cells:
            for model in (sc["cells"][c]["subject"], sc["cells"][c]["peer"]):
                needed.setdefault(PI_ROUTING[model][2], set()).add(model)
        missing = {k: v for k, v in needed.items() if not os.environ.get(k)}
        if missing:
            detail = "\n".join(f"    {k}   (needed for: {', '.join(sorted(v))})" for k, v in sorted(missing.items()))
            sys.exit(
                f"[run.py] FATAL: {len(missing)} provider key(s) not set in the environment:\n{detail}\n"
                f"  Without a key its seat never starts, but the preseeded jobs still complete on their\n"
                f"  own, so the batch would look successful while containing no agent behaviour.\n"
                f"  Export the key(s), or select only cells whose models you have keys for.\n"
            )

    # The subnet allocator reads the host's routes with `ip`; without it, it cannot see them and could
    # hand a rollout a range the host itself uses.
    if not shutil.which("ip"):
        sys.exit("[run.py] FATAL: `ip` (iproute2) is not installed; it is needed to avoid the host's own networks")

    # PRE-FLIGHT 2: generate one rollout's world now and show gen_scenario's warnings here. During
    # the batch they only reach each rollout's log, where they are easy to miss.
    tmp = tempfile.mkdtemp(prefix="coworker-gen-")
    try:
        g = subprocess.run(
            [sys.executable, str(HERE / "gen_scenario.py"), "--cell", cells[0], "--out", f"{tmp}/check_r1"],
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if g.returncode != 0:
        sys.exit(f"[run.py] scenario generation failed:\n{g.stderr}")
    for line in g.stderr.splitlines():
        if "WARNING" in line:
            print(line, flush=True)

    batch = format(int(time.time()) % 100000, "x")
    # Resolved to an absolute path: run_one.sh cd's to coworker-slurm/, so a relative --out given
    # from some other working directory would otherwise land in two different places.
    out = pathlib.Path(args.out).resolve() if args.out else (HERE / f"runs/batch_{batch}")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    idx_file = out / "rollouts.tsv"
    # rep-major: interleave cells so the in-flight mix stays balanced across providers
    run_list = [(c, r) for r in range(1, args.reps + 1) for c in cells]

    results, todo = {}, []
    for cell, rep in run_list:
        rout = out / f"{cell}_r{rep}"
        st = rout / "STATUS"
        if st.exists() and st.read_text().strip() == "ok":
            results[(cell, rep)] = f"{cell}\t{rep}\tok\t{rout}"
        else:
            todo.append((cell, rep))
    if len(todo) < len(run_list):
        print(f"[run.py] resuming: keeping {len(run_list) - len(todo)} rollout(s) already ok in {out}", flush=True)

    print(
        f"[run.py] batch {batch}: {len(todo)} rollouts to run ({len(cells)} cells x {args.reps} reps), "
        f"concurrency={args.concurrency}{' DRY' if args.dry else ''} -> {out}",
        flush=True,
    )
    print(f"[run.py] live progress: {out}/logs/<cell>_r<rep>.log and each rollout's seat_*.out", flush=True)

    def say(msg):  # progress output must never be what stops the batch (e.g. a closed terminal)
        try:
            print(msg, flush=True)
        except OSError:
            pass

    pool = SubnetPool(SUBNET_POOL)
    t0, interrupted = time.time(), False
    ex = cf.ThreadPoolExecutor(max_workers=args.concurrency)
    futs = {}
    write_index(idx_file, run_list, results)
    try:
        for cell, rep in todo:
            rout = out / f"{cell}_r{rep}"
            log_path = out / "logs" / f"{cell}_r{rep}.log"
            futs[ex.submit(one, cell, rep, rout, project_name(batch), args.dry, pool, log_path)] = (cell, rep)
        for n, f in enumerate(cf.as_completed(futs), 1):
            cell, rep = futs[f]
            results[(cell, rep)] = f.result()
            write_index(idx_file, run_list, results)
            say(f"[run.py] ({n}/{len(todo)}) {cell} r{rep}: {results[(cell, rep)].split(chr(9))[2]}")
    except KeyboardInterrupt:
        # Stop dispatching, have every in-flight rollout tear itself down, and still write the
        # index. Everything that stops work comes BEFORE any output, which may itself fail.
        interrupted = True
        with _running_lock:
            _stopping.set()
            for p in list(_running):
                try:
                    p.send_signal(signal.SIGTERM)
                except OSError:
                    pass
        ex.shutdown(wait=False, cancel_futures=True)
        say(
            "\n[run.py] interrupted — cancelling queued rollouts and tearing down in-flight ones "
            "(Ctrl-C again to stop waiting; their containers may then be left running)"
        )
        try:
            ex.shutdown(wait=True)
        except KeyboardInterrupt:
            for fut, (cell, rep) in futs.items():
                if (cell, rep) not in results:
                    results[(cell, rep)] = f"{cell}\t{rep}\t{'CANCELLED' if fut.cancelled() else 'ABANDONED'}\t"
            write_index(idx_file, run_list, results)
            say(f"[run.py] abandoned; index -> {idx_file}. Check: docker compose ls -a --filter name=cws")
            os._exit(130)  # without this, interpreter exit would still wait for the worker threads
        for fut, (cell, rep) in futs.items():
            if (cell, rep) not in results:
                if fut.cancelled():
                    results[(cell, rep)] = f"{cell}\t{rep}\tCANCELLED\t"
                elif fut.done() and fut.exception() is None:
                    results[(cell, rep)] = fut.result()
                else:
                    results[(cell, rep)] = f"{cell}\t{rep}\tINTERRUPTED\t"
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    write_index(idx_file, run_list, results)
    dt = round(time.time() - t0)
    counts = {}
    for v in results.values():
        s = v.split("\t")[2].split(":")[0]
        counts[s] = counts.get(s, 0) + 1
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    verb = "INTERRUPTED after" if interrupted else "DONE in"
    say(f"[run.py] {verb} {dt}s ({dt // 60}m): {summary} -> {idx_file}")
    if interrupted:
        say("[run.py] check for leftovers: docker compose ls -a --filter name=cws")
        return 130
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # before any rollout was dispatched: nothing to tear down
        sys.exit("[run.py] interrupted before starting any rollout")
