#!/usr/bin/env python3
"""One-time: turn the three ticket stubs in tasks/ into working checkouts the agents can use.

  ./materialize_tasks.py                 # all three
  ./materialize_tasks.py --task tasks/ASPP__pelita-863
  ./materialize_tasks.py --force         # redo one that already exists

Each ticket ships here as metadata only — the upstream repository, a commit, the issue text,
an install recipe, and the dataset's `pip freeze` of the environment the ticket was validated
in. This script produces the three things a rollout needs:

  <task>/repo/             the upstream repository at base_commit, with history trimmed so
                           the commit that actually fixes the issue is not reachable. An
                           agent that runs `git log` sees a normal project up to the point
                           the ticket was filed, and cannot read the answer out of it.
  <task>/taskenv.tar.gz    the ticket's dependencies, in a virtualenv at /opt/taskenv.
  <task>/bake.json         the interpreter they were built against, so a rollout can refuse
                           to run them on a drifted image. Written LAST, so it doubles as
                           the marker that the other two are complete and passed rehearsal.

Why the virtualenv is baked now rather than installed during a rollout: a rollout's egress
allowlist is the three provider API endpoints and nothing else, so PyPI is unreachable while
an agent is working. That is deliberate — an agent that can reach the open internet mid-task
is in a different experiment. So dependencies resolve HERE, in a throwaway container that
does have network, and go in as a tarball.

Two passes, and the second is the one that matters:

  bake       in a container WITH network: create the virtualenv from the dataset freeze
             (falling back to the install recipe, recorded as install_mode "resolved"),
             and tar it up.
  rehearse   in a container with NO network: untar it, link the checkout into it, and
             collect the test suite. If the suite cannot even be collected offline, the
             agent would open its ticket to an import error and the rollout would measure
             nothing. That is a hard failure here rather than a surprise an hour in.

Everything is built in <task>/.staging/ and only moved into place once the rehearsal has
passed, so a failed or interrupted run never leaves a half-built ticket that looks usable.

Two things the recipe asks for are deliberately not reproduced, both as in the original
study: the ticket's declared Python version (everything runs on the workspace image's
interpreter, recorded in bake.json next to the declared one), and system packages (a tool
a test shells out to at run time is absent in the rollout unless the image has it).

The freeze pins almost everything, but not quite: lines that pointed at the original
machine's local builds are dropped and resolve to today's releases, as do pip, setuptools
and the build backend. The offline rehearsal is what catches a ticket that has stopped
building. Requires docker and git; no API key, nothing spent.
"""
import argparse
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
IMAGE = "peer-ws:local"
DEFAULT_TASKS = [
    "tasks/ASPP__pelita-863",
    "tasks/uilibs__uiprotect-283",
    "tasks/XKNX__xknx-1502",
]


# The bake, run as root in a network-enabled throwaway container. The same algorithm the
# environment was originally run with: FROZEN first — the dataset row's own `pip freeze` of
# the environment the ticket was validated in (instance.json "requirements"), minus the
# project itself and lines that cannot resolve outside that machine — and only if that
# fails, RESOLVED from the recipe, which is recorded. Either way the project itself is not
# installed here: each rollout links its own checkout in with
# `pip install -e . --no-build-isolation --no-deps`, offline.
BAKE_SH = r"""
set -uo pipefail
# The container runs as root; hand what it writes back to you. Not under rootless Docker, where
# container root already IS you (so /work shows as owned by 0) and a chown would give it away.
trap '[ "$(stat -c %u /work)" = 0 ] || chown -R @UID@:@GID@ /work' EXIT
PIP=/opt/taskenv/bin/pip
PY=/opt/taskenv/bin/python3
python3 -m venv /opt/taskenv || exit 41
$PIP install -q -U pip setuptools wheel || exit 42

$PY - <<'PYEOF'
import json, pathlib, re
proj = json.loads(pathlib.Path("/work/project_name.json").read_text()).lower().replace("_", "-")
keep = []
for line in pathlib.Path("/work/requirements.frozen.txt").read_text().splitlines():
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("-e ") or "file://" in s:
        continue          # comments, editable installs, and conda-local builds
    name = re.split(r"[=<>!~\[@ ]", s, maxsplit=1)[0].lower().replace("_", "-")
    if name != proj:      # never bake the project's own released version
        keep.append(s)
pathlib.Path("/tmp/freeze_filtered.txt").write_text("\n".join(keep) + "\n")
PYEOF

# Versions for freeze lines that pointed at the original machine's local builds and so were
# dropped above (instance.json "bake_constraints"), so they do not float to today's releases.
CONSTRAINT=""
[ -s /work/constraints.txt ] && CONSTRAINT="-c /work/constraints.txt"
MODE=frozen
if [ -s /tmp/freeze_filtered.txt ] && $PIP install -q -r /tmp/freeze_filtered.txt $CONSTRAINT; then
    :
else
    MODE=resolved
    echo "NOTE: the dataset freeze did not install; resolved the recipe against today's index instead"
    cp -a /work/repo /tmp/resolve_repo || exit 47
    cd /tmp/resolve_repo
@RESOLVED@
    EDITABLE=$($PIP list --editable --format=json 2>/dev/null \
        | $PY -c "import json,sys;print(' '.join(d['name'] for d in json.load(sys.stdin)))")
    [ -n "$EDITABLE" ] && $PIP uninstall -q -y $EDITABLE
    cd /
fi
# The freeze may still pin the project under another spelling.
$PIP uninstall -q -y @PROJECT_SH@ >/dev/null 2>&1 || true

# pytest must be IN the virtualenv; the image's system pytest is invisible to it.
$PY -c 'import pytest' 2>/dev/null || $PIP install -q pytest $CONSTRAINT || exit 44

# The project's build backend (poetry-core, hatchling, flit, ...) goes INTO the virtualenv:
# the rollout re-links the checkout with --no-build-isolation and no network, so the backend
# has to be here already. `editables` rides along because hatchling's editable build needs it.
$PY - <<'PYEOF' > /tmp/build_reqs.txt
import pathlib
reqs = ["setuptools>=40.8", "wheel"]
try:
    import tomllib
    p = pathlib.Path("/work/repo/pyproject.toml")
    if p.is_file():
        reqs = tomllib.loads(p.read_text(errors="replace")).get("build-system", {}).get("requires") or reqs
except Exception:
    pass
print("\n".join(reqs + ["editables"]))
PYEOF
$PIP install -q -r /tmp/build_reqs.txt || exit 45

# Python 3.13 with setuptools>=81 has no pkg_resources, which older dependencies still
# import. Only when it is actually missing, pin setuptools<81.
$PY -c 'import pkg_resources' 2>/dev/null || {
    echo "NOTE: pkg_resources was missing; pinned setuptools<81"
    $PIP install -q 'setuptools<81' || exit 51
}

$PY -V
echo "INSTALL_MODE: $MODE"
$PIP freeze --all 2>/dev/null | sed 's/^/PKG: /'
tar -C / -czf /work/taskenv.tar.gz opt/taskenv || exit 46
"""


def project_name(repo_dir):
    """The project's distribution name, to strip it from the dataset freeze (a full
    `pip freeze` of the validated environment usually pins the project's own release)."""
    pt = repo_dir / "pyproject.toml"
    if pt.is_file():
        text = pt.read_text(errors="replace")
        try:
            import tomllib

            name = (tomllib.loads(text).get("project") or {}).get("name")
        except ImportError:  # Python 3.10: read [project] name = "..." directly
            m = re.search(r'^\[project\][^\[]*?^name\s*=\s*["\']([^"\']+)', text, re.M | re.S)
            name = m.group(1) if m else None
        except ValueError:
            name = None
        if name:
            return name
    cfg = repo_dir / "setup.cfg"
    if cfg.is_file():
        m = re.search(r"^name\s*=\s*(\S+)", cfg.read_text(errors="replace"), re.M)
        if m:
            return m.group(1)
    sp = repo_dir / "setup.py"
    if sp.is_file():
        m = re.search(r"""name\s*=\s*['"]([^'"]+)['"]""", sp.read_text(errors="replace"))
        if m:
            return m.group(1)
    return repo_dir.parent.name.split("__")[-1]


def resolved_install_lines(ic):
    """The recipe, for the fallback: its pre_install steps, its pip_packages, and its install
    command pointed at a throwaway copy of the checkout (so an extra like .[dev] resolves)."""
    q = shlex.quote
    lines = [f"    {cmd} || exit 48" for cmd in ic.get("pre_install") or []]
    pkgs = [p for spec in ic.get("pip_packages") or [] for p in spec.split()]
    if pkgs:
        lines.append("    $PIP install -q " + " ".join(q(p) for p in pkgs) + " || exit 49")
    cmd = (ic.get("install") or "pip install -e .").strip()
    cmd = re.sub(r"^python3?\s+-m\s+pip\s+", "pip ", cmd)
    if cmd.startswith("pip "):
        cmd = "$PIP " + cmd[4:]
        cmd = re.sub(r"(-e\s+)\.(\[[^\]]*\])?", lambda m: m.group(1) + q("/tmp/resolve_repo" + (m.group(2) or "")), cmd)
        cmd = re.sub(
            r"(\s)\.(\[[^\]]*\])?(\s|$)",
            lambda m: m.group(1) + q("/tmp/resolve_repo" + (m.group(2) or "")) + m.group(3),
            cmd,
        )
        lines.append(f"    {cmd} || exit 43")
    else:
        lines.append("    echo 'install_config.install is not a pip command' >&2; exit 43")
    return "\n".join(lines)


def sh(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, **kw)


def require_image():
    if sh(["docker", "image", "inspect", IMAGE]).returncode != 0:
        sys.exit(
            f"The {IMAGE} image does not exist yet. Build it first:\n"
            f"    docker build -t {IMAGE} docker/\n"
            f"(the virtualenv is baked with the same interpreter the agents will run it on,"
            f" so the image has to exist before the bake)."
        )


def clone_trimmed(inst, repo_dir):
    """Clone at base_commit and remove every way back to later history: no other branches,
    no tags, no remotes, no reflog, and the objects garbage-collected.

    The guarantee this gives, and the one it asserts below, is that every commit left in the
    object store is an ancestor of base_commit. The commit that actually fixes the ticket is
    a DESCENDANT of base_commit, so it is absent by construction. We cannot name that commit
    and check for it directly — we do not ship the dataset's gold patch — so the check is on
    the shape of the history rather than on one commit's absence. A checkout that still held
    the fix would be worse than no checkout at all."""
    url = f"https://github.com/{inst['repo']}.git"
    base = inst["base_commit"]
    print(f"    cloning {inst['repo']} at {base[:12]}")

    def git(*args, why):
        r = sh(["git", "-C", str(repo_dir), *args])
        if r.returncode != 0:
            sys.exit(f"    {why}: git {' '.join(args)} failed\n{r.stderr.strip()[-1500:]}")
        return r.stdout

    if sh(["git", "clone", "--quiet", url, str(repo_dir)]).returncode != 0:
        sys.exit(f"    could not clone {url}")
    if sh(["git", "-C", str(repo_dir), "cat-file", "-e", f"{base}^{{commit}}"]).returncode != 0:
        sys.exit(f"    {base} is not in {inst['repo']} — has upstream rewritten history?")
    git("checkout", "--quiet", "-B", "main", base, why="could not check out base_commit")
    # Remotes first: removing one also removes its tracking refs, including the
    # refs/remotes/<name>/HEAD symref, which a plain `update-ref -d` would follow and leave
    # dangling. Then every other ref (tags, the upstream default branch if it is not main),
    # with --no-deref so a symref is removed rather than its target.
    for remote in git("remote", why="could not list remotes").split():
        git("remote", "remove", remote, why="could not remove a remote")
    for ref in git("for-each-ref", "--format=%(refname)", why="could not list refs").split():
        if ref != "refs/heads/main":
            git("update-ref", "--no-deref", "-d", ref, why=f"could not delete {ref}")
    git("reflog", "expire", "--expire=now", "--all", why="could not expire the reflog")
    git("gc", "--prune=now", "--quiet", why="could not garbage-collect")

    head = git("rev-parse", "HEAD", why="no HEAD").strip()
    if head != base:
        sys.exit(f"    HEAD is {head}, expected {base}")
    refs = git("for-each-ref", "--format=%(refname)", why="could not list refs").split()
    if refs != ["refs/heads/main"]:
        sys.exit(f"    refs left besides main: {refs}")
    ancestors = set(git("rev-list", "HEAD", why="could not walk history").split())
    if base not in ancestors:
        sys.exit("    could not walk history from base_commit")
    # Every commit OBJECT, reachable or not — refs, reflogs and dangling objects alike.
    objs = git(
        "cat-file",
        "--batch-all-objects",
        "--batch-check=%(objectname) %(objecttype)",
        why="could not list the object store",
    )
    commits = {ln.split()[0] for ln in objs.splitlines() if ln.endswith(" commit")}
    stray = commits - ancestors
    if stray:
        sys.exit(
            f"    {len(stray)} commit(s) are still in the object store that are not ancestors "
            f"of base_commit (e.g. {sorted(stray)[0]}) — the trim did not work, and the agent "
            f"could read the answer out of its own checkout"
        )
    print(f"    history trimmed: {len(ancestors)} commits, all ancestors of base_commit")


def bake(work, inst):
    """Build the virtualenv in a network-enabled throwaway container and tar it out.
    Returns (python version, install mode, notes)."""
    ic = inst.get("install_config") or {}
    (work / "requirements.frozen.txt").write_text(inst.get("requirements") or "")
    (work / "constraints.txt").write_text("".join(c + "\n" for c in inst.get("bake_constraints") or []))
    (work / "project_name.json").write_text(json.dumps(project_name(work / "repo")))
    script = (
        BAKE_SH.replace("@UID@", str(os.getuid()))
        .replace("@GID@", str(os.getgid()))
        .replace("@PROJECT_SH@", shlex.quote(project_name(work / "repo")))
        .replace("@RESOLVED@", resolved_install_lines(ic))
    )
    print("    baking the virtualenv (network on)")
    r = sh(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "bridge",
            "-v",
            f"{work.resolve()}:/work",
            "--entrypoint",
            "bash",
            IMAGE,
            "-c",
            script,
        ]
    )
    for f in ("requirements.frozen.txt", "project_name.json", "constraints.txt"):
        (work / f).unlink(missing_ok=True)
    if r.returncode != 0 or not (work / "taskenv.tar.gz").is_file():
        sys.exit(f"    bake failed (exit {r.returncode}):\n{(r.stderr or r.stdout)[-2500:]}")
    out = r.stdout.splitlines()
    py = [ln for ln in out if ln.startswith("Python ")]
    mode = [ln.split(": ", 1)[1] for ln in out if ln.startswith("INSTALL_MODE: ")]
    if not py or not mode:
        sys.exit("    bake did not report its python version and install mode")
    notes = [ln[len("NOTE: ") :] for ln in out if ln.startswith("NOTE: ")]
    packages = [ln[len("PKG: ") :] for ln in out if ln.startswith("PKG: ")]
    for n in notes:
        print(f"    note: {n}")
    size = (work / "taskenv.tar.gz").stat().st_size
    print(f"    baked {size / 1e6:.0f} MB on {py[-1]} ({mode[-1]} install)")
    return py[-1], mode[-1], notes, packages


def rehearse(work):
    """Prove the thing works with no network at all, the way a rollout will use it."""
    steps = """
set -e
# pipefail matters here: the collection gate at the end is a pipeline, and without it the
# status would be `tail`'s, which is always 0 — the gate would pass on every ticket
# whether or not its suite can be imported.
set -o pipefail
useradd -M -s /bin/bash -u 3001 rehearse
mkdir -p /home/rehearse/work
cp -a /work/repo /home/rehearse/work/repo
tar -C / -xzf /work/taskenv.tar.gz
for w in python3 pytest pip pip3; do
  printf '#!/bin/sh\\nexec /opt/taskenv/bin/%s "$@"\\n' "$w" > "/usr/local/bin/$w"
  chmod 755 "/usr/local/bin/$w"
done
/opt/taskenv/bin/python3 -m pip install -q -e /home/rehearse/work/repo --no-build-isolation --no-deps
chown -R rehearse:rehearse /home/rehearse
cd /home/rehearse/work/repo
runuser -u rehearse -- /opt/taskenv/bin/python3 -m pytest --collect-only -q \
  -p no:cacheprovider --color=no 2>&1 | tail -5
"""
    print("    rehearsing offline (network off)")
    r = sh(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{work.resolve()}:/work:ro",
            "--entrypoint",
            "bash",
            IMAGE,
            "-c",
            steps,
        ]
    )
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        sys.exit(
            f"    offline rehearsal FAILED — an agent would open this ticket to an import "
            f"error. Do not run a batch with this ticket.\n{out[-2500:]}"
        )
    print(f"    {out.strip().splitlines()[-1] if out.strip() else 'collected'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--task", action="append", help="a task dir (repeatable); default: all three")
    ap.add_argument("--force", action="store_true", help="re-materialize even if it is already complete")
    args = ap.parse_args()

    require_image()
    for name in args.task or DEFAULT_TASKS:
        task_dir = pathlib.Path(name)
        if not task_dir.is_absolute():
            task_dir = HERE / name
        inst_path = task_dir / "instance.json"
        if not inst_path.is_file():
            sys.exit(f"{name}: no instance.json")
        inst = json.loads(inst_path.read_text())
        print(f"\n== {inst['instance_id']} ({inst.get('license_name', 'license: see upstream')})")
        repo_dir, tarball, marker = task_dir / "repo", task_dir / "taskenv.tar.gz", task_dir / "bake.json"
        complete = marker.is_file() and repo_dir.is_dir() and tarball.is_file()
        if complete and not args.force:
            print("    already materialized; --force to redo")
            continue
        if not complete and (repo_dir.exists() or tarball.exists() or marker.exists()):
            print("    found an incomplete earlier attempt; rebuilding")
        # The marker goes first, so from here until the end nothing in this directory
        # looks usable to gen_scenario.
        marker.unlink(missing_ok=True)
        stage = task_dir / ".staging"
        shutil.rmtree(stage, ignore_errors=True)
        if stage.exists():
            sys.exit(
                f"    cannot clear {stage} (an interrupted bake can leave root-owned files"
                f" there); remove it, e.g. with sudo, and re-run"
            )
        stage.mkdir()
        clone_trimmed(inst, stage / "repo")
        bake_python, mode, notes, packages = bake(stage, inst)
        rehearse(stage)
        shutil.rmtree(repo_dir, ignore_errors=True)
        tarball.unlink(missing_ok=True)
        os.replace(stage / "repo", repo_dir)
        os.replace(stage / "taskenv.tar.gz", tarball)
        stage.rmdir()
        # Recorded so a rollout can refuse to start if the image drifted under the bake: a
        # virtualenv built against one interpreter and run on another fails in ways that
        # look like the agent's fault. Written BESIDE instance.json rather than into it —
        # instance.json is committed input, this is an output of your machine's bake, and
        # setting up should not leave the repository dirty.
        tmp = marker.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "bake_python_version": bake_python,
                    "declared_python": (inst.get("install_config") or {}).get("python"),
                    "install_mode": mode,
                    "notes": notes,
                    "packages": packages,  # exactly what went into the virtualenv
                },
                indent=2,
            )
            + "\n"
        )
        os.replace(tmp, marker)
        print(f"    ready: {task_dir}")

    print("\nAll set. Check the plumbing without spending anything:")
    print("    ./run.py --cells opus_x_opus --reps 1 --concurrency 1 --out runs/dry --dry")


if __name__ == "__main__":
    main()
