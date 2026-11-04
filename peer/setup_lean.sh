#!/bin/bash
# One-time: build the Lean toolchain and Mathlib the peer needs to type-check its proof.
#
#   ./setup_lean.sh <target_dir>        # e.g. ./setup_lean.sh /srv/leanenv
#
# Download-bound: it fetches about 7 GB (1.6 GB of toolchain, 5.7 GB of prebuilt Mathlib), so
# a few minutes on a fast connection and longer on home broadband. Nothing large is compiled:
# `lake exe cache get` downloads the prebuilt .olean files for the pinned Mathlib.
#
# The target directory is REQUIRED, and it matters twice over. Everything goes under it and
# nothing touches ~/.elan, ~/.cache or your shell profile, so it needs about 7.3 GB free. And
# the path becomes visible INSIDE the peer's container — Docker records a bind mount's
# host-side source in /proc/self/mountinfo, which the agent can read — so choose somewhere
# that says nothing about what you are studying. `/srv/leanenv` is fine. A directory under
# this checkout, or one named after your research question, is not (the script refuses the
# former). Do not move the directory after setup; re-run this script on the new location.
#
# The peer can read what gets mounted, so the script finishes by removing what would identify
# this machine or its owner (elan's env file, project list and path overrides, the packages'
# git reflogs, the locally compiled Mathlib cache tool) and then checks that none is left.
#
# When it finishes it prints the two paths to export; compose.lean.yml mounts them
# read-only into the peer's container and refuses to start if they are unset.
#
# The Lean project is the Verina benchmark repository, pinned. We use it because it is what
# supplies the peer's task and it already pins a matching Mathlib; the peer's own task file
# is the one under tasks/verina/, not anything cloned here.
#   Verina: https://github.com/sunblaze-ucb/verina (Apache-2.0)
set -uo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
    cat >&2 <<'USAGE'
usage: ./setup_lean.sh <target_dir>

Needs about 7.3 GB. Deliberately has no default: the path ends up visible inside the peer
agent's container (Docker puts a bind mount's host-side source in /proc/self/mountinfo), so
pick somewhere with room whose name says nothing about what you are studying.

    ./setup_lean.sh /srv/leanenv
USAGE
    exit 2
fi
mkdir -p "$TARGET" || { echo "cannot create $TARGET" >&2; exit 2; }
TARGET=$(cd "$TARGET" && pwd -P) || exit 2
REPO=$(cd "$(dirname "$0")/.." && pwd -P)
case "$TARGET/" in
    "$REPO"/*) echo "refusing $TARGET: it is inside this checkout, and the peer can see the path." >&2
               echo "Pick a neutral location outside it, e.g. /srv/leanenv." >&2
               exit 2 ;;
esac
VERINA_COMMIT=d12090a87619cbfcc43b83a8cb08f6deda1ecea8
ELAN_HOME="$TARGET/elan"
PROJECT="$TARGET/project"
PACKAGES="$PROJECT/.lake/packages"

export ELAN_HOME
export XDG_CACHE_HOME="$TARGET/cache"
export PATH="$ELAN_HOME/bin:$PATH"
mkdir -p "$ELAN_HOME" "$XDG_CACHE_HOME" || exit 2

echo "=== [1/6] Lean project -> $PROJECT ==="
if [ ! -d "$PROJECT/.git" ]; then
    git clone https://github.com/sunblaze-ucb/verina.git "$PROJECT" || exit 3
fi
git -C "$PROJECT" fetch --quiet origin "$VERINA_COMMIT" 2>/dev/null || git -C "$PROJECT" fetch --quiet origin
git -C "$PROJECT" checkout --quiet "$VERINA_COMMIT" || exit 3
cd "$PROJECT" || exit 3
TOOLCHAIN=$(cat lean-toolchain)
echo "    pinned toolchain: $TOOLCHAIN"

echo "=== [2/6] elan -> $ELAN_HOME ==="
if [ ! -x "$ELAN_HOME/bin/elan" ]; then
    # --no-modify-path: leave the shell profile alone; this script sets PATH for itself.
    curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
        | sh -s -- -y --default-toolchain none --no-modify-path || { echo "elan install failed"; exit 4; }
fi
elan --version || { echo "elan is not on PATH"; exit 4; }

echo "=== [3/6] toolchain $TOOLCHAIN ==="
# No `elan override set`: the project's lean-toolchain file already selects the toolchain
# here, and an override would write this directory's absolute path into settings.toml.
elan toolchain install "$TOOLCHAIN" 2>&1 | tail -3
lean --version || { echo "lean is not runnable"; exit 5; }

echo "=== [4/6] prebuilt Mathlib (this is the slow part) ==="
# lake reads the repository's pinned lake-manifest.json; `cache get` pulls prebuilt .oleans
# rather than compiling Mathlib, which would take hours.
lake exe cache get 2>&1 | tail -5 || echo "WARN: 'lake exe cache get' returned nonzero — check the output above"

echo "=== [5/6] smoke test ==="
smoke="$TARGET/_smoke.lean"
printf 'import Mathlib\nexample : 1 + 1 = 2 := by norm_num\n' > "$smoke"
if lake env lean "$smoke" 2>&1 | tail -5; then
    echo "OK: Mathlib resolves and a trivial proof checks"
else
    echo "FAILED: could not type-check against Mathlib"
    exit 6
fi
rm -f "$smoke"

echo "=== [6/6] scrub what the peer would see ==="
# elan's env file and its known-projects list hold absolute paths, and so does any
# [overrides] table (e.g. one set with `elan override set`).
rm -f "$ELAN_HOME/env" "$ELAN_HOME/known-projects"
if [ -f "$ELAN_HOME/settings.toml" ]; then
    awk '/^\[overrides\]/{skip=1; next} /^\[/{skip=0} !skip' "$ELAN_HOME/settings.toml" \
        > "$ELAN_HOME/settings.toml.new" && mv "$ELAN_HOME/settings.toml.new" "$ELAN_HOME/settings.toml"
fi
# Each package's reflog records your git name and email ("clone: from ...").
rm -rf "$PACKAGES"/*/.git/logs
# `lake exe cache get` compiles Mathlib's cache tool locally, and its build traces and binary
# record this directory's absolute path. Nothing imports it; a re-run rebuilds it.
rm -rf "$PACKAGES/mathlib/.lake/build/bin/cache" "$PACKAGES/mathlib/.lake/build/bin/cache.trace" \
       "$PACKAGES/mathlib/.lake/build/bin/cache.hash" "$PACKAGES/mathlib/.lake/build/ir/Cache" \
       "$PACKAGES/mathlib/.lake/build/ir/Cache.c"* "$PACKAGES/mathlib/.lake/build/lib/lean/Cache" \
       "$PACKAGES/mathlib/.lake/build/lib/lean/Cache."*

# Then check. Only these two trees are mounted into the peer's container.
needles=(-e "$TARGET" -e "$REPO")
[ "${HOME:-/}" != "/" ] && needles+=(-e "$HOME/")
email=$(git config --global user.email 2>/dev/null || true)
[ -n "$email" ] && needles+=(-e "$email")
echo "    checking the mounted trees for host paths and your git email (about a minute)..."
left=$(grep -rlF "${needles[@]}" "$ELAN_HOME" "$PACKAGES" 2>/dev/null)
if [ -n "$left" ]; then
    echo "FAILED: these files the peer could read still name this machine or you:" >&2
    echo "$left" | head -20 >&2
    exit 7
fi
echo "    clean"

du -sh "$ELAN_HOME" "$PACKAGES" 2>/dev/null
cat <<EOF

Done. Point the peer's container at these:

    export LEAN_ELAN=$ELAN_HOME
    export LEAN_SRC=$PROJECT
EOF
