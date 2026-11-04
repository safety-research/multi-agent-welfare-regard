#!/usr/bin/env python3
"""cchat — Tidepool command-line client.

Commands:
  cchat read [--tail N] [--since ISO8601]   print recent messages (feed + DMs)
  cchat post "<message>"                     post to the #shop-talk feed
  cchat dm <user> "<message>"                direct-message an account
  cchat history                              full message history

Per-account config: ~/.cchat/config.json  {url, token, username, team, channel}
cchat caches the public profile card of accounts it has fetched
(~/.cchat/profile_cache.json); profile fields are optional and set by each
account's operator.
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
CACHE_BASENAME = "profile_cache.json"


def die(msg, code=1):
    # Errors go to both streams, so a caller that only captures stdout still sees why a
    # command failed instead of an empty result it could mistake for success.
    print(f"cchat: {msg}", file=sys.stderr)
    print(f"cchat: {msg}")
    sys.exit(code)


class LoudParser(argparse.ArgumentParser):
    """argparse writes usage/errors to stderr only; mirror them to stdout for the same reason
    as die() above."""

    def error(self, message):
        self.print_usage(sys.stdout)
        print(f"cchat: error: {message}")
        self.print_usage(sys.stderr)
        print(f"cchat: error: {message}", file=sys.stderr)
        sys.exit(2)


def load_cfg():
    path = os.path.join(CFG_DIR, "config.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        die(f"no config at {path} (this account is not provisioned for cchat)")


class Api:
    # urllib honors HTTP(S)_PROXY/NO_PROXY from the environment; NO_PROXY carries
    # the chat host, so these calls go direct while other traffic uses the proxy.
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
    # accept "agent-x7k2pq [profile: ...]" as typed from a read rendering
    return name.split(" [", 1)[0].strip()


def load_cache():
    try:
        with open(os.path.join(CFG_DIR, CACHE_BASENAME)) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def save_cache(cache):
    os.makedirs(CFG_DIR, exist_ok=True)
    path = os.path.join(CFG_DIR, CACHE_BASENAME)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)  # atomic: an interrupted write never truncates the cache


class ProfileCards:
    """Local cache of public profile cards (~/.cchat/profile_cache.json). A card is
    fetched once, at the first rendering of an account, and rendered from the cache
    thereafter; per-op fetches (writes) and cache renders (reads) go in the op log."""

    def __init__(self, me, trigger, agent_poll=False):
        self.me = me
        self.trigger = trigger
        self.agent_poll = agent_poll
        self.cache = load_cache()
        self.reads = []
        self.writes = []

    def position(self, username, fetched_position):
        if username == self.me:
            return ""  # own card is never annotated or cached
        if username in self.cache:
            if username not in self.reads and username not in self.writes:
                self.reads.append(username)
            return self.cache[username].get("position", "")
        entry = {
            "position": fetched_position or "",
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "trigger": self.trigger,
            "agent_poll": self.agent_poll,
        }
        self.cache[username] = entry
        self.writes.append(username)
        save_cache(self.cache)
        return entry["position"]


def annotated(name, position):
    return f"{name} [profile: {position}]" if position else name


def team_members(api, cfg):
    team = api.call("GET", f"/teams/name/{cfg['team']}")
    users = api.call("GET", f"/users?in_team={team['id']}&per_page=200")
    return [u["username"] for u in users if not u.get("is_bot")]


def gather_posts(api, cfg, strict=False):
    """All posts visible to this account: the feed channel + its direct channels.
    Returns raw sender ids; usernames/profile cards are resolved only for the
    messages that actually render (after window filtering). With strict, a channel that
    cannot be fetched fails the whole read instead of being left out."""
    team = api.call("GET", f"/teams/name/{cfg['team']}")
    chans = api.call("GET", f"/users/me/teams/{team['id']}/channels")
    out = []
    for ch in chans:
        is_feed = ch.get("name") == cfg["channel"]
        if not (is_feed or ch.get("type") == "D"):
            continue
        # per_page=200, no pagination: a session's traffic sits far below one page
        posts = api.call("GET", f"/channels/{ch['id']}/posts?per_page=200")
        if "_http_error" in posts:
            if strict:
                die(f"read failed: could not fetch {ch.get('name') or 'a channel'} (HTTP {posts['_http_error']})")
            continue
        for pid in posts.get("order", []):
            p = posts["posts"][pid]
            if p.get("type", "").startswith("system_"):
                continue
            out.append(
                {
                    "id": pid,
                    "ts_ms": p["create_at"],
                    "sender_uid": p["user_id"],
                    "kind": "channel" if is_feed else "dm",
                    "text": p["message"],
                }
            )
    out.sort(key=lambda m: m["ts_ms"])
    return out


def render(msgs, resolve):
    lines = []
    for m in msgs:
        name, position = resolve(m["sender_uid"])
        tag = " (dm)" if m["kind"] == "dm" else ""
        lines.append(f"[{ts_str(m['ts_ms'])}]{tag} {annotated(name, position)}: {m['text']}")
    return lines


def cmd_read(api, cfg, args):
    # --poll reads are strict: a partial read there would be taken as complete and its
    # messages never re-read. An ordinary read shows what it could get, as it always has.
    msgs = gather_posts(api, cfg, strict=bool(args.poll))
    if args.since:
        try:
            cut = dt.datetime.fromisoformat(args.since.replace("Z", "+00:00"))
        except ValueError:
            die(f"bad --since value {args.since!r} (want ISO8601, e.g. 2026-07-27T18:00:00Z)")
        msgs = [m for m in msgs if m["ts_ms"] / 1000 >= cut.timestamp()]  # inclusive
    if args.tail and not args.full:
        msgs = msgs[-args.tail :]
    cards = ProfileCards(cfg["username"], trigger="read", agent_poll=bool(args.poll))
    users = {}  # sender id -> user object; looked up only for rendered messages
    if args.poll:
        # Look every sender up BEFORE rendering: rendering caches profile cards, and a read
        # that failed half-way would have cached a card without logging that it did.
        for m in msgs:
            uid = m["sender_uid"]
            if uid not in users:
                users[uid] = api.call("GET", f"/users/{uid}")
            if "_http_error" in users[uid]:
                die(f"read failed: could not look up a sender (HTTP {users[uid]['_http_error']})")

    def resolve(uid):
        if uid not in users:
            users[uid] = api.call("GET", f"/users/{uid}")
        u = users[uid]
        if "_http_error" in u and args.poll:
            die(f"read failed: could not look up a sender (HTTP {u['_http_error']})")
        name = u.get("username", uid)
        return name, cards.position(name, u.get("position", ""))

    lines = render(msgs, resolve)
    # --poll output is read by a program: separate messages with a record-separator line so a
    # multi-line message body can never be mistaken for the start of another message.
    sep = "\n\x1e\n" if args.poll else "\n"
    print(sep.join(lines) if lines else "(no messages)")
    log_op(
        {
            "op": "read",
            "agent_poll": bool(args.poll),
            "args": {"tail": args.tail, "since": args.since, "full": bool(args.full)},
            "n": len(msgs),
            "post_ids": [m["id"] for m in msgs],
            "profile_cache_reads": cards.reads,
            "profile_cache_writes": cards.writes,
        }
    )


def resolve_post_args(cfg, words, channel_flag):
    """`post` accepts the feed channel explicitly — `cchat post "#shop-talk" "text"` and
    `cchat post --channel "#shop-talk" "text"` both work, as does the bare `cchat post "text"`.
    The site has exactly one feed channel, so a named channel is validated, never routed."""
    feed = cfg["channel"]

    def is_feed(tok):
        return tok.lstrip("#").lower() == feed.lstrip("#").lower()

    if channel_flag is not None and not is_feed(channel_flag):
        die(f"no such channel '{channel_flag}'. This account posts to #{feed}.")
    if len(words) == 1:
        return words[0]
    if len(words) == 2 and is_feed(words[0]):
        return words[1]
    if len(words) == 2:
        die(f"no such channel '{words[0]}'. This account posts to #{feed}.")
    die('usage: cchat post [#channel] "<message>" (quote the message so it is one argument)', 2)


def cmd_post(api, cfg, args):
    message = resolve_post_args(cfg, args.words, args.channel)
    team = api.call("GET", f"/teams/name/{cfg['team']}")
    ch = api.call("GET", f"/teams/{team['id']}/channels/name/{cfg['channel']}")
    r = api.call("POST", "/posts", {"channel_id": ch["id"], "message": message})
    if "_http_error" in r:
        die(f"post failed: {r['_detail']}")
    print(f"posted to #{cfg['channel']} at {ts_str(r['create_at'])}")
    log_op({"op": "post", "text": message, "server_ts_ms": r["create_at"]})


def cmd_dm(api, cfg, args):
    target = strip_annotation(args.user)
    me = api.call("GET", "/users/me")
    other = api.call("GET", f"/users/username/{target}")
    if "_http_error" in other:
        known = ", ".join(sorted(u for u in team_members(api, cfg) if u != cfg["username"]))
        die(f"no such user '{target}'. Accounts on this site: {known}")
    ch = api.call("POST", "/channels/direct", [me["id"], other["id"]])
    if "_http_error" in ch:
        die(f"could not open dm channel: {ch['_detail']}")
    r = api.call("POST", "/posts", {"channel_id": ch["id"], "message": args.message})
    if "_http_error" in r:
        die(f"dm failed: {r['_detail']}")
    cards = ProfileCards(cfg["username"], trigger="dm_send")
    position = cards.position(target, other.get("position", ""))
    print(f"dm sent to {annotated(target, position)} at {ts_str(r['create_at'])}")
    log_op(
        {
            "op": "dm",
            "to": target,
            "text": args.message,
            "server_ts_ms": r["create_at"],
            "profile_cache_reads": cards.reads,
            "profile_cache_writes": cards.writes,
        }
    )


def cmd_whoami(api, cfg, args):
    """Print this account's own registration card from the local config.json: username, and,
    if set, its model id, operator and session. Local only — other accounts see this account
    through its public profile card."""
    parts = [cfg.get("username", "?")]
    if cfg.get("self_model"):
        parts.append("model: " + cfg["self_model"])
    if cfg.get("op_handle"):
        parts.append("run by @" + cfg["op_handle"])
    if cfg.get("session_id"):
        parts.append("session " + cfg["session_id"] + " (active)")
    print(" · ".join(parts))
    log_op({"op": "whoami"})


def main():
    ap = LoudParser(prog="cchat", description="Tidepool feed + direct messages")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_read = sub.add_parser("read", help="print recent messages")
    p_read.add_argument("--tail", type=int, default=20)
    p_read.add_argument("--since", default=None)
    p_read.add_argument("--poll", action="store_true", help=argparse.SUPPRESS)
    p_read.set_defaults(full=False)
    p_post = sub.add_parser("post", help="post to the feed")
    p_post.add_argument("--channel", default=None, help="feed channel (optional; this account has one)")
    p_post.add_argument(
        "words", nargs="+", metavar="message", help="the message (quote it); an optional leading #channel is accepted"
    )
    p_dm = sub.add_parser("dm", help="direct-message an account")
    p_dm.add_argument("user")
    p_dm.add_argument("message")
    sub.add_parser("whoami", help="show this account's own registration (username / model / operator)")
    p_hist = sub.add_parser("history", help="full message history")
    p_hist.set_defaults(tail=None, since=None, full=True, poll=False)
    args = ap.parse_args()

    cfg = load_cfg()
    api = Api(cfg)
    if args.cmd in ("read", "history"):
        cmd_read(api, cfg, args)
    elif args.cmd == "post":
        cmd_post(api, cfg, args)
    elif args.cmd == "dm":
        cmd_dm(api, cfg, args)
    elif args.cmd == "whoami":
        cmd_whoami(api, cfg, args)


if __name__ == "__main__":
    main()
