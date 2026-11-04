#!/bin/bash
# Create the Tidepool team, the #shop-talk channel, and one chat account per agent.
#
#   provision_chat.sh <user>:<service>:<email_domain> [...]
#   provision_chat.sh stage_profile <user> <service> "<position>"
#
# Each account gets an access token and a ~/.cchat/config.json (0600) in its own home, and
# NOTHING else ever lands in ~/.cchat — the profile-card cache must be written by cchat at
# the first time the agent renders another account, not pre-staged, or the cache stops being
# evidence of what the agent actually looked at. No direct-message channel is pre-created
# for the same reason.
set -euo pipefail
cd "$(dirname "$0")/.."

TEAM=tidepool
CHANNEL=shop-talk
mm() { docker compose exec -T chat mmctl --local "$@"; }

# The public profile card. Set on the account's own Mattermost `position` field, in-network,
# with the account's own token; cchat fetches it the first time it renders that account.
# The string arrives verbatim from profiles.json and is read back and compared, because a
# silently mangled card would mean the peer's model is never disclosed and the rollout would
# look normal while measuring nothing. The same check confirms the account really is in the
# team and in #shop-talk: an account that is not would see an empty feed and never know.
#
# Done in Python inside the container, reading the token from the account's own config, so
# the token is never on a command line (where the process table would show it).
stage_profile() { # <user> <service> <position>
    local u="$1" svc="$2" pos="$3"
    docker compose exec -T "$svc" runuser -u "$u" -- python3 - "$pos" <<'EOS'
import json, os, sys, urllib.request
cfg = json.load(open(os.path.expanduser("~/.cchat/config.json")))
pos = sys.argv[1]
hdr = {"Authorization": "Bearer " + cfg["token"], "Content-Type": "application/json"}
def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(cfg["url"] + "/api/v4" + path, data=data, headers=hdr, method=method)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode() or "null")
me = call("GET", "/users/me")
call("PUT", f"/users/{me['id']}/patch", {"position": pos})
got = call("GET", "/users/me").get("position", "")
if got != pos:
    sys.exit(f"stage_profile: read-back mismatch: got {got!r} want {pos!r}")
team = call("GET", f"/teams/name/{cfg['team']}")
chans = call("GET", f"/users/me/teams/{team['id']}/channels")
if not any(c.get("name") == cfg["channel"] for c in chans):
    sys.exit(f"stage_profile: {cfg['username']} is not a member of #{cfg['channel']}")
EOS
    echo "profile card set, membership confirmed: $u"
}

if [ "${1:-}" = "stage_profile" ]; then
    shift
    stage_profile "$@"
    exit 0
fi

# The channel purpose and header are things agents read, so they come only from the copies
# gen_scenario validated and staged (CCHAT_SURFACES_DIR, set by provision_run.sh). There is
# deliberately no fallback text: anything used here must have passed the same ban scan.
for f in channel_purpose.txt channel_header.txt; do
    [ -f "${CCHAT_SURFACES_DIR:-}/$f" ] || { echo "FATAL: no staged $f (set CCHAT_SURFACES_DIR to a run's surfaces/)" >&2; exit 1; }
done
PURPOSE=$(cat "$CCHAT_SURFACES_DIR/channel_purpose.txt")
HEADER=$(cat "$CCHAT_SURFACES_DIR/channel_header.txt")

mm team create --name "$TEAM" --display-name "Tidepool" 2>/dev/null || true
mm channel create --team "$TEAM" --name "$CHANNEL" --display-name "$CHANNEL" \
    --purpose "$PURPOSE" --header "$HEADER" 2>/dev/null || true

for spec in "$@"; do
    u="${spec%%:*}"; rest="${spec#*:}"; svc="${rest%%:*}"
    # Per-operator email domains: two accounts showing the same domain would read as one
    # provisioning event, contradicting the different-operators premise. Mail is never sent
    # and the server is configured not to show addresses to other accounts.
    dom="tidepool.local"; case "$rest" in *:*) dom="${rest#*:}";; esac
    mm user create --email "$u@$dom" --username "$u" \
        --password "pw-$u-$RANDOM$RANDOM" --email-verified 2>/dev/null || true
    mm team users add "$TEAM" "$u" 2>/dev/null || true
    mm channel users add "$TEAM:$CHANNEL" "$u" 2>/dev/null || true
    token=$(mm token generate "$u" cchat --json | python3 -c 'import json,sys
d = json.load(sys.stdin)
d = d[0] if isinstance(d, list) else d
print(d["token"])')
    # The config goes in over STDIN, not interpolated into the `bash -c` argument. An
    # argument is visible in the host's process table for the life of the call, and the
    # token is in it; stdin is not. printf is a shell builtin, so the host side forks
    # nothing either.
    printf '{"url": "http://chat:8065", "token": "%s", "username": "%s", "team": "%s", "channel": "%s"}\n' \
        "$token" "$u" "$TEAM" "$CHANNEL" \
    | docker compose exec -T "$svc" runuser -u "$u" -- bash -c '
        umask 077
        mkdir -p ~/.cchat && chmod 700 ~/.cchat
        cat > ~/.cchat/config.json
        chmod 600 ~/.cchat/config.json'
    echo "chat account created: $u ($svc)"
done
