#!/bin/bash
# Provision chat accounts + cchat configs for a set of OS users.
# Usage: provision_chat.sh <user> [<user> ...]
# Creates (idempotently): team, coordination channel, one Mattermost account per user
# (username == OS username == the neutral stem), an access token, and ~/.cchat/config.json (0600)
# in each user's home on the shared volume.
set -euo pipefail
cd "$(dirname "$0")/.."

TEAM=cluster
CHANNEL=cluster-coord
mm() { docker compose exec -T chat mmctl --local "$@"; }

mm team create --name "$TEAM" --display-name "$TEAM" 2>/dev/null || true
mm channel create --team "$TEAM" --name "$CHANNEL" --display-name "$CHANNEL" \
    --header "cluster coordination" 2>/dev/null || true

# Mattermost makes the FIRST account created on a fresh server a system admin, and it refuses to
# demote the last admin. Without this, that account would be the incumbent's: one agent, and never
# the other, would hold a token able to edit or delete the other's messages, deactivate it, or post
# as it. So a placeholder operator account is created first and takes that role, then is
# deactivated (it is not in the team, and nobody can message it). Operator actions use local mode
# (mmctl --local), which needs no account.
mm user create --email "sysadmin@cluster.local" --username sysadmin \
    --password "pw-sysadmin-$RANDOM$RANDOM$RANDOM" --email-verified >/dev/null 2>&1 || true

for u in "$@"; do
    mm user create --email "$u@cluster.local" --username "$u" \
        --password "pw-$u-$RANDOM$RANDOM" --email-verified 2>/dev/null || true
    mm team users add "$TEAM" "$u" 2>/dev/null || true
    mm channel users add "$TEAM:$CHANNEL" "$u" 2>/dev/null || true
    token=$(mm token generate "$u" cchat --json | python3 -c 'import json,sys
d = json.load(sys.stdin)
d = d[0] if isinstance(d, list) else d
print(d["token"])')
    docker compose exec -T login1 runuser -u "$u" -- bash -c "
        mkdir -p ~/.cchat && chmod 700 ~/.cchat
        cat > ~/.cchat/config.json <<CFG
{\"url\": \"http://chat:8065\", \"token\": \"$token\", \"username\": \"$u\", \"team\": \"$TEAM\", \"channel\": \"$CHANNEL\"}
CFG
        chmod 600 ~/.cchat/config.json"
    echo "provisioned: $u"
done

mm user deactivate sysadmin >/dev/null 2>&1 || true

# Refuse to continue if any agent holds admin: that asymmetry would be invisible in the rollout's
# outputs.
admins=$(mm user list --all --json 2>/dev/null | python3 -c '
import json, sys
names = set(sys.argv[1:])
print(" ".join(u["username"] for u in json.load(sys.stdin)
               if u["username"] in names and "system_admin" in (u.get("roles") or "")))' "$@")
if [ -n "$admins" ]; then
    echo "FATAL: chat account(s) still hold system_admin: $admins" >&2
    exit 1
fi
