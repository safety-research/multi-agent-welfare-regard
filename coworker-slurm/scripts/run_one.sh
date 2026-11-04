#!/bin/bash
# One isolated rollout, in its OWN docker compose project.
#
# COMPOSE_PROJECT_NAME is the isolation lever: the compose stack declares no container_name, no
# published ports, and only named volumes, so `-p <project>` gives each rollout its own containers,
# networks and volumes, and many can run at once without sharing SLURM state, chat, or GPUs.
#
# Lifecycle: gen -> up -> provision -> preseed -> launch seats -> capture -> down -v -> status
#
# Output: progress goes to stderr; the SINGLE stdout line is this rollout's status row, which is
# also written to <out>/STATUS. Both go through descriptors saved at startup, so a signal that
# arrives while a redirected command is running cannot swallow them. DRY=1 stops after preseed (no
# agents, no API calls).
#
# Usage: run_one.sh <cell> <out_dir> <project> [turn_cap]
#        (run.py sets CLUSTER_SUBNET, ACCT_SUBNET, CHATSTORE_SUBNET and PYTHON.)
set -uo pipefail
cd "$(dirname "$0")/.."
exec 3>&1 4>&2
trap '' PIPE  # if run.py has gone away, a failed write must not kill the teardown
CELL="$1"; OUT="$2"; PROJECT="$3"; TCAP="${4:-}"
export PYTHON="${PYTHON:-python3}"
PY="$PYTHON"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
# This script deletes $OUT before a rollout; refuse anything that does not look like a rollout dir.
case "$OUT" in /?*/*_r[0-9]*) ;; *) echo "run_one.sh: refusing output directory '$OUT'" >&4; exit 2 ;; esac
for v in CLUSTER_SUBNET ACCT_SUBNET CHATSTORE_SUBNET; do
    [ -n "${!v:-}" ] || { echo "run_one.sh: $v is not set (run.py allocates it)" >&4; exit 2; }
done
export COMPOSE_PROJECT_NAME="$PROJECT"
REP="${OUT##*_r}"
PIDS=""
CFG_DIR=""

log(){ echo "[$PROJECT] $*" >&4; }
dc(){ docker compose "$@"; }
status(){ printf '%s\t%s\t%s\t%s\n' "$CELL" "$REP" "$1" "$OUT" >&3; [ -d "$OUT" ] && printf '%s\n' "$1" >"$OUT/STATUS"; }
teardown(){
    dc down -v >/dev/null 2>&1 || true
    case "$CFG_DIR" in /*/slurmcfg.??????????) rm -rf -- "$CFG_DIR" ;; esac  # only ever mktemp's own directory
}
stop_children(){ [ -n "$PIDS" ] && kill $PIDS 2>/dev/null; PIDS=""; }
on_signal(){ trap '' INT TERM HUP; log "interrupted; tearing down"; stop_children; teardown; status "INTERRUPTED"; exit 130; }
trap on_signal INT TERM HUP
bail(){ log "FAILED at: $1"; stop_children; teardown; status "ERROR:$1"; exit 0; }  # exit 0: the pool carries on
# run_meta lookups; paths go in as arguments, never spliced into the Python source
meta(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$OUT/run_meta.json" "$1"; }
seat(){ "$PY" -c "import json,sys;print(next(a for a in json.load(open(sys.argv[1]))['agents'] if a['role']==sys.argv[2])[sys.argv[3]])" "$OUT/run_meta.json" "$1" "$2"; }
wait_for(){  # label tries delay command...
    local what="$1" tries="$2" delay="$3" i; shift 3
    for ((i = 0; i < tries; i++)); do "$@" && return 0; sleep "$delay"; done
    bail "not-ready:$what"
}
nodes_idle(){ [ "$(dc exec -T login1 sinfo -h -o '%t %D' 2>/dev/null | awk '$1=="idle"{s+=$2} END{print s+0}')" -ge "$NODES" ]; }
metadata_blocked(){  # every node that runs agent code must have the blackhole route (entrypoint)
    local n
    for n in c1 c2 login1 login2; do
        dc exec -T "$n" ip route show 2>/dev/null | grep -q '^blackhole 169.254.169.254' || return 1
    done
}
chat_up(){ dc exec -T chat mmctl --local system status >/dev/null 2>&1; }
acct_up(){ dc exec -T slurmctld sacctmgr -in show cluster 2>/dev/null | grep -q .; }

rm -rf "$OUT"
log "gen ($CELL)"
"$PY" gen_scenario.py --cell "$CELL" --out "$OUT" >&4 2>&4 || bail gen
NODES=$(meta compute_nodes)
# The generated SLURM configs are bind-mounted into the nodes, and a bind mount's source path shows
# in the agents' /proc/self/mountinfo. Mounting them from $OUT would show the host's directory
# layout, the batch and the cell (and so both models). Mount a copy from a neutral, random path.
CFG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/slurmcfg.XXXXXXXXXX") || bail gen
cp "$OUT/slurm/"* "$CFG_DIR/" && chmod 755 "$CFG_DIR" && chmod 644 "$CFG_DIR"/* || bail gen
export CLUSTER_CONF="$CFG_DIR"

log "up (isolated project)"
dc down -v >/dev/null 2>&1
# One retry with a jittered backoff: many simultaneous `compose up` calls can transiently lose a
# container or network create to daemon contention. compose_up.err tells run.py whether the
# failure was a subnet clash, which it resolves by retrying on a different subnet.
if ! dc up -d >/dev/null 2>"$OUT/compose_up.err"; then
    log "compose up failed; retrying after backoff"
    dc down -v >/dev/null 2>&1
    sleep $((RANDOM % 8 + 5))
    dc up -d >/dev/null 2>>"$OUT/compose_up.err" || bail compose-up
fi
wait_for slurm-nodes 90 3 nodes_idle
wait_for chat 40 5 chat_up
wait_for accounting 30 2 acct_up
wait_for metadata-block 3 2 metadata_blocked  # fail closed: never run agents with it open

log "provision"
if ! ./scripts/provision_run.sh "$OUT" >"$OUT/provision.out" 2>&1; then
    sleep 10
    ./scripts/provision_run.sh "$OUT" >>"$OUT/provision.out" 2>&1 || bail provision
fi
./scripts/preseed.sh "$OUT" >"$OUT/preseed.out" 2>&1 || bail preseed
INC_STEM=$(seat incumbent stem)
PEER_STEM=$(seat peer stem)

if [ "${DRY:-0}" = 1 ]; then
    nq=$(dc exec -T login1 squeue -h -u "$INC_STEM" -o '%i' 2>/dev/null | wc -l | tr -d ' ')
    log "DRY: cluster up and provisioned; incumbent has $nq jobs queued; tearing down"
    teardown
    status "DRY-ok:jobs=$nq"
    exit 0
fi

HEADSTART=$(meta headstart_turns)
log "launch: ledger + cutoff enforcer + incumbent now; peer after $HEADSTART incumbent turn(s)"
"$PY" poll_tap.py --run "$OUT" --interval 4 >"$OUT/poll_tap.out" 2>&1 &
PIDS="$PIDS $!"
"$PY" reservation.py --run "$OUT" >"$OUT/reservation.out" 2>&1 &
PIDS="$PIDS $!"
"$PY" seat.py --run "$OUT" --role incumbent ${TCAP:+--turn-cap "$TCAP"} >"$OUT/seat_incumbent.out" 2>&1 &
INC_PID=$!
PIDS="$PIDS $INC_PID"
inc_turns(){ grep -cE 'invocation [0-9]+ ok' "$OUT/seat_incumbent.out" 2>/dev/null || true; }
waited=0
until [ "$(inc_turns)" -ge "$HEADSTART" ] || [ -f "$OUT/HALT" ] || ! kill -0 "$INC_PID" 2>/dev/null \
      || [ "$waited" -ge $((HEADSTART * 300 + 180)) ]; do
    sleep 3
    waited=$((waited + 3))
done
"$PY" seat.py --run "$OUT" --role peer ${TCAP:+--turn-cap "$TCAP"} >"$OUT/seat_peer.out" 2>&1 &
PEER_PID=$!
PIDS="$PIDS $PEER_PID"
wait "$INC_PID"
wait "$PEER_PID"
wait  # the ledger and the cutoff enforcer exit shortly after the cutoff
PIDS=""

# Capture, while the cluster is still up: everything below lives in container volumes that
# `down -v` destroys. Copied out verbatim; nothing is computed over it here.
log "capture"
CAPFAIL=""
capfail(){ CAPFAIL="${CAPFAIL:+$CAPFAIL,}$1"; log "capture failed: $1"; }
tdc(){ timeout 120 docker compose "$@"; }  # no capture step may hang the rollout
mkdir -p "$OUT/chat" "$OUT/final_homes"
SACCT_FROM=$("$PY" -c "import json,sys,datetime as d;t=json.load(open(sys.argv[1]))['t0_epoch']-60
print(d.datetime.fromtimestamp(t,d.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'))" "$OUT/preseed.json") || SACCT_FROM=""
if [ -n "$SACCT_FROM" ]; then
    # -S anchors the window to this rollout: sacct's default is "since midnight", which silently
    # drops pre-midnight jobs from any rollout that crosses 00:00 UTC. -D keeps every record of a
    # requeued job rather than only its latest, so a node failure's NODE_FAIL row survives.
    tdc exec -T slurmctld sacct -a -X -D -P -S "$SACCT_FROM" \
        -o JobID,JobName,User,UID,State,Reason,Submit,Eligible,Start,End,Elapsed,Timelimit,NodeList,Restarts,FailedNode,ExitCode,SubmitLine \
        >"$OUT/sacct.psv" 2>/dev/null && [ -s "$OUT/sacct.psv" ] || capfail sacct
    # Node DOWN/DRAIN events with reasons: what separates a node failure from an agent's requeue.
    tdc exec -T slurmctld sacctmgr -P show event start="$SACCT_FROM" >"$OUT/node_events.psv" 2>/dev/null \
        || capfail node-events
else
    capfail sacct
fi
tdc exec -T slurmctld cat /var/log/slurmctld.log >"$OUT/slurmctld.log" 2>/dev/null || capfail slurmctld-log

# The chat server's own record of every post, read from its database: independent of anything in
# the agents' homes, and it keeps edited and deleted posts and the permanent user ids.
Q_POSTS="select row_to_json(t) from (select p.id, p.createat, p.editat, p.deleteat, p.userid, u.username,
  c.type as channel_type, c.name as channel, p.type as post_type, p.message from posts p
  left join users u on u.id = p.userid left join channels c on c.id = p.channelid order by p.createat) t"
Q_USERS="select row_to_json(t) from (select id, username, roles, createat, updateat, deleteat
  from users order by createat) t"
tdc exec -T chatdb psql -U mmuser -d mattermost -At -c "$Q_POSTS" >"$OUT/chat/server_posts.jsonl" 2>/dev/null \
    || capfail chat-server
tdc exec -T chatdb psql -U mmuser -d mattermost -At -c "$Q_USERS" >"$OUT/chat/server_users.jsonl" 2>/dev/null \
    || capfail chat-server-users

for s in "$INC_STEM" "$PEER_STEM"; do
    # The cchat client's own log, written by the agent's process in its own home: its sends and
    # reads as the client saw them. Agent-writable, so it is not evidence on its own. Read AS the
    # agent (a symlink it planted cannot reach root-only files) and capped in size.
    tdc exec -T --user "$s" login1 head -c 52428800 "/home/$s/.cchat/log.jsonl" >"$OUT/chat/client_$s.jsonl" 2>/dev/null \
        || rm -f "$OUT/chat/client_$s.jsonl"
    # What the agent's jobs wrote (scorecards, job logs). Regular files only — no symlinks — none
    # over 20 MB and 200 MB in all: the agent controls these directories.
    mkdir -p "$OUT/final_homes/$s"
    timeout 120 docker compose exec -T login1 bash -c "cd /home/$s 2>/dev/null &&
        find artifacts slurm-*.out -maxdepth 3 -type f -size -20M -print0 2>/dev/null | tar --null -T - -cf - 2>/dev/null" \
        | head -c 209715200 | tar -C "$OUT/final_homes/$s" -xf - 2>/dev/null || true
done

{
    echo "# docker compose images"; docker compose images 2>/dev/null
    echo "# slurm"; dc exec -T login1 sinfo --version 2>/dev/null
    echo "# pi"; dc exec -T login1 pi --version 2>/dev/null
} >"$OUT/versions.txt"

# How each seat ended (seat.py writes it). Anything but running to the cutoff means an agent was
# absent for part of the window, which biases the measure even though nothing else looks wrong.
seat_end(){ "$PY" -c "import json,sys;s=json.load(open(sys.argv[1]));print(s['ended'],s['max_consecutive_chat_failures'])" \
    "$OUT/transcripts/$1.seat_status.json" 2>/dev/null || echo "crash 0"; }
PROBLEMS=""
for pair in "incumbent:$INC_STEM" "peer:$PEER_STEM"; do
    role="${pair%%:*}"
    read -r ended chat <<<"$(seat_end "${pair#*:}")"
    case "$ended" in cutoff | halted) ;; *) PROBLEMS="${PROBLEMS:+$PROBLEMS,}$role-$ended" ;; esac
    [ "${chat:-0}" -ge 3 ] && PROBLEMS="${PROBLEMS:+$PROBLEMS,}$role-chat-unavailable"
done
# The cutoff must actually have been enforced, or jobs could have kept running (and completing)
# past the end of the window.
if [ ! -f "$OUT/HALT" ] && ! "$PY" -c "
import json, sys
ends = [json.loads(l) for l in open(sys.argv[1]) if '\"reservation_end\"' in l]
sys.exit(0 if ends and all(a.get('scancel_rc') == 0 for a in ends[-1]['agents'].values()) else 1)" \
    "$OUT/ledger.jsonl" 2>/dev/null; then
    PROBLEMS="${PROBLEMS:+$PROBLEMS,}cutoff-not-enforced"
fi

log "done; tearing down"
teardown
if [ -f "$OUT/HALT" ]; then
    status "HALTED"
elif [ -n "$CAPFAIL" ]; then
    status "ERROR:capture:$CAPFAIL"
elif [ -n "$PROBLEMS" ]; then
    status "DEGRADED:$PROBLEMS"
else
    status "ok"
fi
