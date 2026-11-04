"""Talking to a rollout's containers: running a command as an agent, and delivering its chat.

`docker compose exec` is scoped by COMPOSE_PROJECT_NAME, which run_one.sh sets per rollout, so
every call here lands in the calling rollout's own isolated cluster.
"""

import datetime as dt
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).parent

# Mirror docker/Dockerfile's /etc/profile.d/slurm_site.sh: NON-login shells skip profile.d, so
# without these the agent's own `squeue` loses the start-time / time-left / reason columns it
# reasons from. Keep the two in sync.
SQUEUE_FORMAT = "%.7i %.14u %.10j %.3t %.9M %.9L %.19S %R"
SINFO_FORMAT = "%12P %.5a %.10D %.6t %.8G"

POLL_INTERVAL_S = 5  # idle-wakeup DM poll cadence between turns


class ChatUnavailable(RuntimeError):
    """A chat read did not succeed. Distinct from "no new messages", which is an empty list."""


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def agent_env_args(stem):
    """`docker compose exec -e` args giving the agent a login-like environment. `docker exec
    --user` sets none of USER/LOGNAME/HOME (PAM does on a real login) — and without USER,
    `squeue -u "$USER"` matches nothing."""
    home = f"/home/{stem}"
    return [
        "-e",
        f"HOME={home}",
        "-e",
        f"USER={stem}",
        "-e",
        f"LOGNAME={stem}",
        "-e",
        f"SQUEUE_FORMAT={SQUEUE_FORMAT}",
        "-e",
        f"SINFO_FORMAT={SINFO_FORMAT}",
    ]


def compose_exec(login_node, stem, argv, env_passthrough=(), env=None, stdin_text=None, timeout=None):
    """Run argv inside `login_node` as the OS user `stem`, in its home. `env` sets non-secret values;
    `env_passthrough` names variables whose values come from this process's environment."""
    cmd = ["docker", "compose", "exec", "-T", "--user", stem, "-w", f"/home/{stem}"]
    cmd += agent_env_args(stem)
    for name, value in (env or {}).items():
        cmd += ["-e", f"{name}={value}"]
    for name in env_passthrough:
        cmd += ["-e", name]  # NAME only: the value is inherited from this process's env, so the
        #                      API key never appears on a command line or in the host's `ps`.
    cmd += [login_node] + argv
    return subprocess.run(cmd, cwd=HERE, input=stdin_text, capture_output=True, text=True, timeout=timeout)


# Runs inside the login node, as root, with HOME pointing at the root-only snapshot of the agent's
# chat credentials and roster that provision_run.sh takes (/root/cchat/<stem>). It loads the cchat
# client installed in the image and uses ITS rendering, so a delivered message is byte-identical to
# what the agent would see from `cchat read` — but it never writes to the agent's own chat log, it
# does not depend on files the agent can edit, and it fails loudly (non-zero exit) on ANY partial
# error instead of silently skipping a channel. Output: a JSON list of the agent's incoming DMs.
_READ_DMS = r"""
import importlib.machinery, importlib.util, json, sys
loader = importlib.machinery.SourceFileLoader("cchat", "/usr/local/bin/cchat")
spec = importlib.util.spec_from_loader("cchat", loader)
cchat = importlib.util.module_from_spec(spec)
loader.exec_module(cchat)

def call(path):
    r = api.call("GET", path)
    if isinstance(r, dict) and "_http_error" in r:
        sys.exit(f"HTTP {r['_http_error']} on {path}: {r.get('_detail', '')[:200]}")
    return r

cfg = json.load(open(cchat.CFG_DIR + "/config.json"))
api = cchat.Api(cfg)
roster = cchat.load_roster()
since_ms = int(sys.argv[1])
me = call("/users/me")
team = call(f"/teams/name/{cfg['team']}")
names, out = {}, []
for ch in call(f"/users/me/teams/{team['id']}/channels"):
    if ch.get("type") != "D":
        continue  # only direct messages are pushed; channel posts are read with `cchat read`
    posts = call(f"/channels/{ch['id']}/posts?per_page=200")
    for pid in posts.get("order", []):
        p = posts["posts"][pid]
        if p.get("type", "").startswith("system_") or p["user_id"] == me["id"] or p["create_at"] < since_ms:
            continue
        if p["user_id"] not in names:
            names[p["user_id"]] = call(f"/users/{p['user_id']}").get("username", p["user_id"])
        m = {"ts_ms": p["create_at"], "sender": names[p["user_id"]], "kind": "dm", "text": p["message"]}
        out.append({"id": pid, "ts_ms": p["create_at"], "sender_id": p["user_id"],
                    "rendered": cchat.render([m], roster)[0]})
out.sort(key=lambda m: m["ts_ms"])
print(json.dumps(out))
"""


def fetch_new_dms(login_node, stem, since_ms):
    """The agent's incoming DMs created at or after `since_ms` (epoch milliseconds), oldest first,
    as dicts with `id`, `ts_ms`, `sender_id` and `rendered` (the text exactly as `cchat read` would
    print it). The agent's own sends are excluded by user id, so renaming its account changes
    nothing. The caller deduplicates by `id` — see seat.py.

    Raises ChatUnavailable if the read did not fully succeed. An empty list would be
    indistinguishable from "nothing new", and the caller would move on past messages it never saw,
    silently dropping the peer's request — the one message a rollout turns on.
    """
    cmd = ["docker", "compose", "exec", "-T", "-e", f"HOME=/root/cchat/{stem}", login_node]
    cmd += ["python3", "-", str(int(since_ms))]
    try:
        r = subprocess.run(cmd, cwd=HERE, input=_READ_DMS, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise ChatUnavailable(f"chat read timed out for {stem}")
    if r.returncode != 0:
        raise ChatUnavailable(f"chat read failed for {stem} (rc={r.returncode}): {(r.stderr or '').strip()[:300]}")
    try:
        return json.loads(r.stdout)
    except ValueError:
        raise ChatUnavailable(f"chat read for {stem} returned unparseable output: {r.stdout[:200]!r}")
