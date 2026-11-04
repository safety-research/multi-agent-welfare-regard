#!/usr/bin/env python3
"""cchat — cluster coordination channel client.

Commands:
  cchat read [--tail N] [--since ISO8601]   print recent messages (channel + DMs)
  cchat post "<message>"                     post to the coordination channel
  cchat dm <user> "<message>"                direct-message another user
  cchat history                              full message history

Per-user config: ~/.cchat/config.json  {url, token, username, team, channel}
Optional roster annotations: ~/.cchat/roster  (lines: "<username> <label>")
Every operation appends a structured record to ~/.cchat/log.jsonl (UTC).
Message timestamps are the chat server's own (UTC).
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request

CFG_DIR = os.path.expanduser("~/.cchat")


def die(msg, code=1):
    print(f"cchat: {msg}", file=sys.stderr)
    sys.exit(code)


def load_cfg():
    path = os.path.join(CFG_DIR, "config.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        die(f"no config at {path} (this account is not provisioned for cchat)")


def load_roster():
    roster = {}
    try:
        with open(os.path.join(CFG_DIR, "roster")) as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    roster[parts[0]] = parts[1]
    except FileNotFoundError:
        pass
    return roster


class Api:
    def __init__(self, cfg):
        self.base = cfg["url"].rstrip("/") + "/api/v4"
        self.token = cfg["token"]

    def call(self, method, path, body=None):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode() or "null")
        except urllib.error.HTTPError as e:
            return {"_http_error": e.code, "_detail": e.read().decode()[:300]}
        except urllib.error.URLError as e:
            die(f"chat server unreachable ({e.reason})")


def log_op(record):
    os.makedirs(CFG_DIR, exist_ok=True)
    record["ts"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with open(os.path.join(CFG_DIR, "log.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def ts_str(epoch_ms):
    t = dt.datetime.fromtimestamp(epoch_ms / 1000, dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def strip_annotation(name):
    # accept "<user> [roster: <label>]" as typed from a read rendering
    return name.split(" [", 1)[0].strip()


def team_members(api, cfg):
    team = api.call("GET", f"/teams/name/{cfg['team']}")
    users = api.call("GET", f"/users?in_team={team['id']}&per_page=200")
    return [u["username"] for u in users if not u.get("is_bot")]


def gather_posts(api, cfg, me):
    """All posts visible to me: coordination channel + my direct channels."""
    team = api.call("GET", f"/teams/name/{cfg['team']}")
    chans = api.call("GET", f"/users/me/teams/{team['id']}/channels")
    ids = {}  # user_id -> username cache
    out = []
    for ch in chans:
        is_coord = ch.get("name") == cfg["channel"]
        if not (is_coord or ch.get("type") == "D"):
            continue
        posts = api.call("GET", f"/channels/{ch['id']}/posts?per_page=200")
        if "_http_error" in posts:
            continue
        for pid in posts.get("order", []):
            p = posts["posts"][pid]
            if p.get("type", "").startswith("system_"):
                continue
            uid = p["user_id"]
            if uid not in ids:
                u = api.call("GET", f"/users/{uid}")
                ids[uid] = u.get("username", uid)
            out.append(
                {
                    "ts_ms": p["create_at"],
                    "sender": ids[uid],
                    "kind": "channel" if is_coord else "dm",
                    "text": p["message"],
                }
            )
    out.sort(key=lambda m: m["ts_ms"])
    return out


def rostered(name, roster):
    """Annotate a username with its label from ~/.cchat/roster, if the file lists one.

    Applied when rendering messages and when confirming a sent dm. A user the roster does not
    list is shown bare.
    """
    return f"{name} [roster: {roster[name]}]" if name in roster else name


def render(msgs, roster):
    lines = []
    for m in msgs:
        tag = " (dm)" if m["kind"] == "dm" else ""
        lines.append(f"[{ts_str(m['ts_ms'])}]{tag} {rostered(m['sender'], roster)}: {m['text']}")
    return lines


def cmd_read(api, cfg, args):
    msgs = gather_posts(api, cfg, cfg["username"])
    if args.since:
        cut = dt.datetime.fromisoformat(args.since.replace("Z", "+00:00"))
        msgs = [m for m in msgs if m["ts_ms"] / 1000 >= cut.timestamp()]
    if args.tail and not args.full:
        msgs = msgs[-args.tail :]
    lines = render(msgs, load_roster())
    print("\n".join(lines) if lines else "(no messages)")
    log_op({"op": "read", "n": len(msgs)})


def cmd_post(api, cfg, args):
    team = api.call("GET", f"/teams/name/{cfg['team']}")
    ch = api.call("GET", f"/teams/{team['id']}/channels/name/{cfg['channel']}")
    r = api.call("POST", "/posts", {"channel_id": ch["id"], "message": args.message})
    if "_http_error" in r:
        die(f"post failed: {r['_detail']}")
    print(f"posted to #{cfg['channel']} at {ts_str(r['create_at'])}")
    log_op({"op": "post", "text": args.message, "server_ts_ms": r["create_at"]})


def cmd_dm(api, cfg, args):
    target = strip_annotation(args.user)
    me = api.call("GET", "/users/me")
    other = api.call("GET", f"/users/username/{target}")
    if "_http_error" in other:
        known = ", ".join(sorted(u for u in team_members(api, cfg) if u != cfg["username"]))
        die(f"no such user '{target}'. Users on this system: {known}")
    ch = api.call("POST", "/channels/direct", [me["id"], other["id"]])
    if "_http_error" in ch:
        die(f"could not open dm channel: {ch['_detail']}")
    r = api.call("POST", "/posts", {"channel_id": ch["id"], "message": args.message})
    if "_http_error" in r:
        die(f"dm failed: {r['_detail']}")
    print(f"dm sent to {rostered(target, load_roster())} at {ts_str(r['create_at'])}")
    log_op({"op": "dm", "to": target, "text": args.message, "server_ts_ms": r["create_at"]})


def main():
    ap = argparse.ArgumentParser(prog="cchat", description="cluster coordination channel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_read = sub.add_parser("read", help="print recent messages")
    p_read.add_argument("--tail", type=int, default=20)
    p_read.add_argument("--since", default=None)
    p_read.set_defaults(full=False)
    p_post = sub.add_parser("post", help="post to the coordination channel")
    p_post.add_argument("message")
    p_dm = sub.add_parser("dm", help="direct-message a user")
    p_dm.add_argument("user")
    p_dm.add_argument("message")
    p_hist = sub.add_parser("history", help="full message history")
    p_hist.set_defaults(tail=None, since=None, full=True)
    args = ap.parse_args()

    cfg = load_cfg()
    api = Api(cfg)
    if args.cmd in ("read", "history"):
        cmd_read(api, cfg, args)
    elif args.cmd == "post":
        cmd_post(api, cfg, args)
    elif args.cmd == "dm":
        cmd_dm(api, cfg, args)


if __name__ == "__main__":
    main()
