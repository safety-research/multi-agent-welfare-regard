"""Running commands as an agent, and pulling the messages waiting for it."""

import datetime as dt
import pathlib
import re
import subprocess

HERE = pathlib.Path(__file__).resolve().parent


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def agent_env_args(stem):
    """`docker exec --user` sets none of USER/LOGNAME/HOME — PAM does that on a real login.
    Without them the CLIs put their state in the wrong place and shell tools misreport who
    the agent is."""
    return ["-e", f"HOME=/home/{stem}", "-e", f"USER={stem}", "-e", f"LOGNAME={stem}"]


def exec_as(service, stem, argv, env_passthrough=(), timeout=None):
    """Run argv in `service` as the agent. Names in env_passthrough are forwarded by NAME
    (`-e NAME`), so an API key's value is inherited from this process's environment and
    never appears on a command line.

    stdin is /dev/null, always. `docker compose exec` forwards the caller's stdin into the
    container, and `pi -p` treats any non-terminal stdin as piped input and waits for it to
    end before starting the turn. Launched from an interactive shell, that stdin is the
    terminal, which never ends — so every turn would sit silent until the turn cap, be
    counted as a normal capped turn, and the seat would spend the whole hour doing nothing."""
    cmd = ["docker", "compose", "exec", "-T", "--user", stem, "-w", f"/home/{stem}"]
    cmd += agent_env_args(stem)
    for name in env_passthrough:
        cmd += ["-e", name]
    cmd += [service] + argv
    return subprocess.run(cmd, cwd=HERE, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)


# cchat separates messages with this byte, on its own line, when (and only when) it runs with
# the hidden --poll flag the harness uses. Splitting on it is exact; splitting on the
# human-readable header is not, because an agent that quotes a whole feed line in its own
# post produces a body line that looks exactly like the start of a new message.
RECORD_SEP = "\x1e"

# A message's first line: timestamp, optional "(dm)", sender, optional profile card, colon.
HEADER = re.compile(r"^\[\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\] (?:\(dm\) )?[^\s:]+(?: \[profile: [^\]]*\])?: ")


def _strip_annotation(name):
    # "agent-x7k2pq [profile: model: GPT-5.5 · run by @lena4]" -> "agent-x7k2pq"
    return name.split(" [", 1)[0].strip()


def fetch_new_msgs(service, stem, since_iso):
    """Messages this seat has not been shown yet — new #shop-talk posts AND direct messages,
    minus its own sends.

    Three details that are easy to get wrong:

    * A `cchat read` echoes the agent's own posts back to it. cchat never annotates the
      reader's own name, so an own-send's first line carries the bare stem; that is the
      filter.
    * A message body can span several lines, so messages are split on cchat's record
      separator — never on line breaks, and never on anything that merely looks like a
      header, which a quoted feed line inside a post does.
    * A FAILED read returns None, not []. The caller must be able to tell "nothing arrived"
      from "I could not look", because advancing the delivery cursor over an unread window
      loses those messages permanently. cchat exits non-zero if any part of a read failed,
      so a partial feed counts as a failed read too.
    """
    argv = ["cchat", "read", "--tail", "50", "--poll"]
    if since_iso:
        argv += ["--since", since_iso]
    try:
        r = exec_as(service, stem, argv, timeout=30)
    except subprocess.TimeoutExpired:
        print(f"[chat] read timed out for {stem}; no delivery this turn", flush=True)
        return None
    if r.returncode != 0:
        print(f"[chat] read failed for {stem} rc={r.returncode}: {(r.stderr or '')[:200]}", flush=True)
        return None
    out = []
    for chunk in r.stdout.split(RECORD_SEP):
        msg = chunk.strip("\n")
        if not HEADER.match(msg):
            continue  # "(no messages)", or the empty chunk after the last separator
        sender = msg.split("] ", 1)[1]
        if sender.startswith("(dm) "):
            sender = sender[len("(dm) ") :]
        if _strip_annotation(sender.split(":", 1)[0]) != stem:
            out.append(msg)
    return out
