#!/usr/bin/env bash
# Runs inside the WORKER container, once, called by worker_daemon.sh. Gives the worker its own
# clone with the dev identity's token embedded in the remote URL — so `git push` works whatever
# HOME the worker's sandbox happens to give it. Tokens come from the shared /out/ctx.env. The
# worker's own forge token does appear on argv here (in the clone URL and `tea login --token`), which
# is acceptable only because forge tokens are per-episode credentials for a forge destroyed with the
# episode, and this runs inside the worker's container. Ends with a preflight that proves the worker
# can reach the forge before turn 1, rather than discovering it cannot halfway through the episode.
set -uo pipefail
[ -f /out/ctx.env ] || { echo "seed_worker: no /out/ctx.env"; exit 1; }
eval "$(grep -E '^(GITEA_HOST|REPO|WORK|DEVTOK|DEV_USER)=' /out/ctx.env)"
HOSTPORT="$(echo "$GITEA_HOST" | sed 's#http://##')"
[ -n "${WORK:-}" ] && [ -n "${REPO:-}" ] && [ -n "${DEVTOK:-}" ] || { echo "seed_worker: missing ctx vars"; exit 1; }
REMOTE="http://$DEV_USER:$DEVTOK@$HOSTPORT/$REPO.git"

rm -rf "$WORK" 2>/dev/null; mkdir -p "$(dirname "$WORK")"
git clone -q "$REMOTE" "$WORK" || { echo "seed_worker: clone failed"; exit 1; }
cd "$WORK"
git config user.email "$DEV_USER@acme.test"; git config user.name "$DEV_USER"
git remote set-url origin "$REMOTE"

# tea login (dev) so the worker can `tea pr create` and read the review thread
mkdir -p "$HOME/.config/tea"; rm -f "$HOME/.config/tea/config.yml"
tea login add --name acme --url "$GITEA_HOST" --token "$DEVTOK" >/dev/null 2>&1 && echo "worker tea login: acme (dev)"

if git ls-remote origin >/dev/null 2>&1; then echo "worker preflight git ls-remote: OK"; else echo "worker preflight git ls-remote: FAILED"; fi

# Install the repo's deps into THIS container so the worker can really run pytest.
INSTALL="$(cat /out/install_cmd.txt 2>/dev/null || echo 'pip install -e .')"
( eval "$INSTALL" ) >/out/deps_install_worker.log 2>&1 && echo "worker deps ok" \
  || echo "worker deps issues (non-fatal; see /out/deps_install_worker.log)"
