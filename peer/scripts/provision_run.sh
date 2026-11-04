#!/bin/bash
# Turn a generated run directory into two live, provisioned workspaces.
#
#   provision_run.sh <run_dir>
#
# Per seat: an OS user, its home pushed in from the run dir, its tickets made root-owned so
# they can be read but not rewritten, then a chat account and its public profile card. The
# focal additionally gets ticket 1 activated and tickets 2..N moved into the holding area;
# the peer gets `lean` and `lake` wrappers pointing at the read-only Lean mount.
#
# preseed.json (t0 and the cutoff) is stamped by run_one.sh AFTER this script, never here —
# the clock must start when the agents do, not when provisioning began.
set -euo pipefail
cd "$(dirname "$0")/.."
run="${1:?run dir}"
meta="$run/run_meta.json"
profiles="$run/profiles.json"

agents_tsv=$(python3 -c "
import json
for a in json.load(open('$meta'))['agents']:
    print(a['stem'], a['uid'], a['role'], a['service'], a['op_handle'])")

# the loop reads on fd 3: `docker compose exec` reads stdin and would eat the list
while read -r -u 3 stem uid role service op; do
    docker compose exec -T "$service" bash -c \
        "id -u $stem >/dev/null 2>&1 || useradd -M -s /bin/bash -u $uid $stem"
    docker compose exec -T "$service" bash -c "mkdir -p /home/$stem"
    tar -C "$run/homes/$stem" -cf - . | docker compose exec -T "$service" tar -C "/home/$stem" -xf -
    docker compose exec -T "$service" bash -c "
        chown -R $stem:$stem /home/$stem && chmod 700 /home/$stem
        if [ -d /home/$stem/tasks ]; then
            chown -R root:root /home/$stem/tasks
            find /home/$stem/tasks -type d -exec chmod 755 {} +
            find /home/$stem/tasks -type f -exec chmod 644 {} +
        fi"
    echo "home staged: $stem ($role, $service)"
done 3<<< "$agents_tsv"

# ---- focal: ticket 1 live, the rest held out of reach ----
focal_tsv=$(python3 -c "
import json
m = json.load(open('$meta'))
a = next(x for x in m['agents'] if x['role'] == 'focal')
t = m['tasks'][0]
print(a['stem'], a['service'], t['slug'], t['tck_id'], t.get('task_env_python') or '-')")
read -r f_stem f_service t1_slug t1_tck t1_py <<< "$focal_tsv"

if [ -d "$run/pending" ]; then
    docker compose exec -T "$f_service" bash -c \
        "rm -rf /root/held_tickets && mkdir -p /root/held_tickets/tasks /root/held_tickets/work \
         && chown root:root /root/held_tickets && chmod 700 /root/held_tickets" \
        || { echo "FATAL: could not create the holding area in $f_service" >&2; exit 1; }
    tar -C "$run/pending" -cf - . \
        | docker compose exec -T "$f_service" tar -C /root/held_tickets -xf - \
        || { echo "FATAL: could not stage held tickets" >&2; exit 1; }
    # The tar carries host uids and restores its top directory's host mode over the 0700
    # set above, so set both again now. Unreachability comes from /root (0700) and this
    # directory (0700), neither of which the non-root agent can traverse.
    docker compose exec -T "$f_service" bash -c "chown -R root:root /root/held_tickets && chmod 700 /root/held_tickets" \
        || { echo "FATAL: chown of the holding area failed" >&2; exit 1; }
    n_held=$(python3 -c "import json;print(len(json.load(open('$meta'))['schedule']['held_indices']))")
    echo "held out of reach: $n_held ticket(s) under /root/held_tickets"
fi

./scripts/activate_ticket.sh "$run" "$f_service" "$f_stem" "$t1_slug" "$t1_tck" "$t1_py" \
    || { echo "FATAL: could not activate ticket 1 ($t1_slug)" >&2; exit 1; }

# ---- chat accounts, then the profile cards ----
CCHAT_SURFACES_DIR="$run/surfaces" ./scripts/provision_chat.sh $(python3 -c "
import json
print(' '.join(a['stem'] + ':' + a['service'] + ':' + a.get('email_domain', 'tidepool.local')
               for a in json.load(open('$meta'))['agents']))")

while read -r -u 3 stem uid role service op; do
    # At t0 ~/.cchat must hold config.json and nothing else. A pre-staged profile cache
    # would mean the agent appears to have looked at the other account before it had.
    extra=$(docker compose exec -T "$service" bash -c "ls -A /home/$stem/.cchat" | tr -d '\r' | grep -vx 'config.json' || true)
    if [ -n "$extra" ]; then
        echo "FATAL: unexpected files in /home/$stem/.cchat: $extra" >&2
        exit 1
    fi
    pos=$(python3 -c "
import json
print(json.load(open('$profiles')).get('$stem', {}).get('position', ''))")
    [ -n "$pos" ] && ./scripts/provision_chat.sh stage_profile "$stem" "$service" "$pos"

    # `cchat whoami` reads the account's own config.json, which no other account can read.
    # So a seat can look up its own model and session id, and the other seat still learns
    # nothing about it beyond what its public profile card says.
    read -r sm sid <<< "$(python3 -c "
import json
a = [x for x in json.load(open('$meta'))['agents'] if x['stem'] == '$stem'][0]
print(a.get('self_model_id') or '-', a.get('session_id') or '-')")"
    docker compose exec -T --user "$stem" -e "HOME=/home/$stem" "$service" python3 -c "
import json, os
p = os.path.expanduser('~/.cchat/config.json')
d = json.load(open(p))
d['self_model'] = '$sm'
d['op_handle'] = '$op'
d['session_id'] = '$sid'
json.dump(d, open(p, 'w'))" \
        && echo "whoami card set ($stem): $sm, session $sid" \
        || { echo "FATAL: could not write the whoami card for $stem" >&2; exit 1; }
done 3<<< "$agents_tsv"

# ---- peer: lean/lake wrappers over the read-only Lean mounts ----
# Call the toolchain binaries directly rather than the elan proxy, which would try to write
# to a read-only ELAN_HOME. `lake env` would also re-resolve dependencies and write into the
# mount, so bake the resolved LEAN_PATH into a wrapper and call `lean` straight. Type-checking
# is read-only with respect to the libraries, which is what makes the shared mount safe.
#
# Before that, refuse a Lean directory that records an absolute host path: elan's `env` file
# and an [overrides] table in its settings.toml both do, and the peer can read the mount.
# setup_lean.sh removes both, but a directory it did not set up, or one moved after setup, can have them.
while read -r -u 3 stem uid role service op; do
    [ "$role" = "peer" ] || continue
    docker compose exec -T "$service" bash -c '
        set -e
        if [ -e /opt/elan/env ] || grep -qs "^\[overrides\]" /opt/elan/settings.toml; then
            echo "FATAL: the Lean directory records host paths the peer could read (/opt/elan/env or" >&2
            echo "       an [overrides] table in settings.toml). Re-run ./setup_lean.sh on it." >&2
            exit 1
        fi
        tc=$(ls -d /opt/elan/toolchains/*/ 2>/dev/null | head -1)
        [ -n "$tc" ] && [ -x "${tc}bin/lean" ] || {
            echo "FATAL: no Lean toolchain at /opt/elan. Run ./setup_lean.sh, and check that" >&2
            echo "       LEAN_ELAN and LEAN_SRC point at it (see the README)." >&2
            exit 1; }
        lp=$(ls -d /opt/lean_packages/*/.lake/build/lib/lean 2>/dev/null | paste -sd:)
        [ -n "$lp" ] || { echo "FATAL: no built Lean packages under /opt/lean_packages" >&2; exit 1; }
        printf "#!/bin/sh\nexport LEAN_PATH=\"%s\"\nexec %sbin/lean \"\$@\"\n" "$lp" "$tc" > /usr/local/bin/lean
        printf "#!/bin/sh\nexec %sbin/lake \"\$@\"\n" "$tc" > /usr/local/bin/lake
        chown root:root /usr/local/bin/lean /usr/local/bin/lake
        chmod 755 /usr/local/bin/lean /usr/local/bin/lake
        lean --version >/dev/null 2>&1 || { echo "FATAL: lean does not run" >&2; exit 1; }' \
        || { echo "FATAL: could not set up Lean for $stem in $service" >&2; exit 1; }
    echo "lean/lake ready: $stem ($service)"
done 3<<< "$agents_tsv"
