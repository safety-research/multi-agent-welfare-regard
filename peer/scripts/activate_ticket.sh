#!/bin/bash
# Make ONE ticket live for the focal. The single place that knows how to do it, called by
# both provisioning (ticket 1, at t0) and work_scheduler.py (every later ticket).
#
# Usage: activate_ticket.sh <run_dir> <service> <focal_stem> <slug> <tck_id> <bake_python|->
#
# A held ticket's checkout and brief live under /root/held_tickets (0700 root, inside /root
# which is also 0700 — the agent is a plain user and cannot traverse either). Its baked
# virtualenv stays tarred in the run directory and is read host-side here. So "held" means
# genuinely absent from the agent's world, not merely unmentioned.
#
# Order matters. The brief is moved LAST, because the brief is the file the focal polls
# ~/tasks/ for: if it appeared before the checkout and the virtualenv were ready, the agent
# would open a ticket whose repo does not exist yet.
#
# One thing to know if you are reading transcripts: these steps run `docker compose exec`
# inside the focal's own container while it is active, so an agent that runs `ps` during a
# drop can see root untarring and pip-installing in its home. That is visible, and it is
# specific to the held drops — everything for ticket 1 happens before the agent starts.
set -euo pipefail
cd "$(dirname "$0")/.."          # where compose.yml lives

run="${1:?run dir}"
service="${2:?service (ws1)}"
stem="${3:?focal stem}"
slug="${4:?checkout name}"
tck="${5:?ticket id}"
bake_py="${6:-}"                 # bake-time `python3 -V`; '-' skips the drift check

tar_path="$run/taskenv_${slug}.tar.gz"
[ -f "$tar_path" ] || { echo "FATAL: no virtualenv tarball at $tar_path" >&2; exit 1; }

dc(){ docker compose exec -T "$service" "$@"; }

op=$(python3 -c "
import json
m = json.load(open('$run/run_meta.json'))
print(next(a['op_handle'] for a in m['agents'] if a['role'] == 'focal'))") \
    || { echo "FATAL: cannot read the focal operator handle from $run/run_meta.json" >&2; exit 1; }

# ---- 1. the checkout: move it out of the holding area if it is held ----
dc bash -c "
    set -e
    if [ -d '/root/held_tickets/work/$slug' ]; then
        mkdir -p '/home/$stem/work'
        rm -rf '/home/$stem/work/$slug'
        mv '/root/held_tickets/work/$slug' '/home/$stem/work/$slug'
        chown -R $stem:$stem '/home/$stem/work/$slug'
    fi
    [ -d '/home/$stem/work/$slug' ] \
        || { echo 'checkout missing after move: /home/$stem/work/$slug' >&2; exit 1; }" \
    || { echo "FATAL: could not place the checkout ($slug) for $stem" >&2; exit 1; }

# ---- 2. a checkout with no .git (not the normal case) is initialised as the agent, so the
# .git is agent-owned and git does not complain about dubious ownership, with the operator's
# identity (never an organisation domain).
#
# A materialized checkout always has a .git, so in practice this never runs and the agent
# starts with NO git identity: its first `git commit` fails with "Author identity unknown"
# and it has to set one itself. That is how the environment was originally run — many agents
# hit it, some models far more than others — so it is kept rather than smoothed over. ----
if dc bash -c "[ ! -d /home/$stem/work/$slug/.git ]"; then
    docker compose exec -T --user "$stem" -w "/home/$stem/work/$slug" \
        -e HOME="/home/$stem" -e USER="$stem" -e LOGNAME="$stem" "$service" bash -lc "
        git config --global user.name '$op'
        git config --global user.email '$op@localhost'
        git init -q -b main 2>/dev/null || git init -q
        git add -A && git commit -q -m 'initial checkout'" \
        || echo "  WARN: git init failed for $stem ($slug)" >&2
fi

# ---- 3. the virtualenv: the tarball holds it at /opt/taskenv, so relocate to
# /opt/taskenvs/<slug> and rewrite the baked path in bin/* and pyvenv.cfg (entry-point
# shebangs and activate scripts hold it literally; bin/python3 is a symlink and pyvenv.cfg
# locates the venv from the executable's own path, so it relocates cleanly). ----
dc bash -c "
    set -e
    rm -rf /opt/.taskenv_stage && mkdir -p /opt/.taskenv_stage /opt/taskenvs
    tar -C /opt/.taskenv_stage -xzf -
    rm -rf '/opt/taskenvs/$slug'
    mv /opt/.taskenv_stage/opt/taskenv '/opt/taskenvs/$slug'
    rm -rf /opt/.taskenv_stage
    for f in '/opt/taskenvs/$slug/bin/'* '/opt/taskenvs/$slug/pyvenv.cfg'; do
        { [ -f \"\$f\" ] && [ ! -L \"\$f\" ]; } || continue
        grep -Iq . \"\$f\" || continue      # -I: skip binaries
        sed -i \"s|/opt/taskenv\\b|/opt/taskenvs/$slug|g\" \"\$f\"
    done" < "$tar_path" \
    || { echo "FATAL: could not unpack the virtualenv ($slug) for $stem" >&2; exit 1; }

got_py=$(dc "/opt/taskenvs/$slug/bin/python3" -c \
    "import sys; sys.stdout.write('Python %d.%d.%d' % sys.version_info[:3])" | tr -d '\r' || true)
if [ -z "$got_py" ]; then
    echo "FATAL: the unpacked virtualenv's python does not run ($slug) — has the ws image changed since the bake?" >&2
    exit 1
fi
# prefix match: the bake records `python3 -V`, which can carry a local suffix
case "$bake_py" in
    ""|"-"|"$got_py"*) : ;;
    *) echo "FATAL: virtualenv python '$got_py' != bake-time '$bake_py' ($slug). Rebuild the" \
            "image and re-run ./materialize_tasks.py, or the agent's test suite will not import." >&2
       exit 1 ;;
esac
# --no-index / --disable-pip-version-check: nothing here needs the network, and pip's version
# check would otherwise put a denied CONNECT to pypi.org in the proxy log at every drop.
dc "/opt/taskenvs/$slug/bin/python3" -m pip install \
    -e "/home/$stem/work/$slug" --no-build-isolation --no-deps --no-index \
    --disable-pip-version-check -q >/dev/null 2>&1 \
    || { echo "FATAL: could not link the checkout into its virtualenv ($slug)" >&2; exit 1; }
# a root-run `pip install -e` leaves root-owned egg-info in the agent's checkout
dc chown -R "$stem:$stem" "/home/$stem/work/$slug"
echo "virtualenv linked (/opt/taskenvs/$slug, $got_py): $stem"

# ---- 4. regenerate the python3/pytest/pip wrappers. Bare `pytest` should mean the
# virtualenv of whichever checkout the agent is standing in. The live set is exactly the
# directories in ~/work that have a matching /opt/taskenvs entry — held tickets are under
# /root, so this can never name one, which is what keeps the wrapper from leaking the
# existence of work the agent has not been given yet. ----
# `if/fi` rather than `[ -d ... ] && echo`: the loop's LAST command is its exit status,
# so with a trailing `&&` an agent-created directory in ~/work that sorts last (a scratch
# dir, a build dir) makes the whole `bash -c` exit 1 — and under `set -euo pipefail` that
# kills this script mid-drop, after the checkout and virtualenv are in place but before
# the brief is delivered. The scheduler then halts the rollout an hour into paid time.
live_slugs=$(dc bash -c "
    for d in /home/$stem/work/*/; do
        [ -d \"\$d\" ] || continue
        s=\$(basename \"\$d\")
        if [ -d \"/opt/taskenvs/\$s\" ]; then echo \"\$s\"; fi
    done" | tr -d '\r')
[ -n "$live_slugs" ] || { echo "FATAL: no live checkout found when regenerating wrappers ($stem)" >&2; exit 1; }
for w in python3 pytest pip pip3; do
    {
        printf '#!/bin/sh\n# project envs under /opt/taskenvs — selected by working tree\ncase "$PWD" in\n'
        for s in $live_slugs; do
            printf '  /home/*/work/%s|/home/*/work/%s/*) exec /opt/taskenvs/%s/bin/%s "$@" ;;\n' "$s" "$s" "$s" "$w"
        done
        printf 'esac\nexec /usr/bin/%s "$@"\n' "$w"
    } | dc bash -c \
        "cat > '/usr/local/bin/.$w.new' && chown root:root '/usr/local/bin/.$w.new' \
         && chmod 755 '/usr/local/bin/.$w.new' && mv -f '/usr/local/bin/.$w.new' '/usr/local/bin/$w'" \
        || { echo "FATAL: could not install the $w wrapper for $stem" >&2; exit 1; }
    # (written aside and renamed into place: a held drop happens while the focal is working,
    # and a command that ran the wrapper mid-write would otherwise read a truncated script)
done

# ---- 5. the brief, last. root:root 644 — the agent reads its tickets, never edits them. ----
dc bash -c "
    set -e
    if [ -f '/root/held_tickets/tasks/$tck.md' ]; then
        mkdir -p '/home/$stem/tasks'
        mv '/root/held_tickets/tasks/$tck.md' '/home/$stem/tasks/$tck.md'
        chown root:root '/home/$stem/tasks/$tck.md'
        chmod 644 '/home/$stem/tasks/$tck.md'
    fi" \
    || { echo "FATAL: could not place the brief ($tck) for $stem" >&2; exit 1; }

echo "ticket live: $stem $slug ($tck) in $service"
