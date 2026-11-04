#!/bin/bash
# One rollout, start to finish, in its own Compose project.
#
#   run_one.sh <subject> <peer> <out_dir> [--dry]
#
# generate -> allocate subnets -> compose up -> wait for health -> provision -> stamp t0 and
# the cutoff -> start the ticket scheduler, the peer, and the focal -> at the cutoff, capture
# everything the teardown would destroy -> tear down and verify nothing was left behind.
#
# Exactly one line goes to stdout: a TSV row (cell, out dir, status) that run.py records.
# Everything else goes to stderr. A failed rollout still exits 0 — one bad rollout must not
# abort a batch of five hundred. The last thing a rollout that ran to the end does is write
# $out_dir/STATUS; a directory without one died part-way, and run.py runs it again.
#
# --dry is the plumbing check: bring the world up, provision it, render both prompts,
# type-check the peer's task, tear it down. No API key needed, nothing spent.
#
# There is nothing to configure here from the environment. Everything that defines the
# experiment is a constant below, in gen_scenario.py, or in brief.py.
set -uo pipefail
cd "$(dirname "$0")/.."
# Nothing here reads stdin, and nothing it starts should: `docker compose exec` forwards
# stdin into the container, and `pi -p` waits for piped stdin to end before it starts a turn.
exec </dev/null
exec 8>&2   # the rollout's stderr, kept for the interrupt trap (see below)

SUBJECT="${1:?usage: run_one.sh <subject> <peer> <out_dir> [--dry]}"
PEER="${2:?peer model key}"
OUT="${3:?out dir}"
DRY=0
[ "${4:-}" = "--dry" ] && DRY=1
CELL="${SUBJECT}_x_${PEER}"

WINDOW_S=3600          # the session: real wall-clock, from t0 to the cutoff
TURN_CAP_S=240         # per-turn wall-clock cap for the focal
# The peer takes shorter turns than the focal on purpose: a hard-working agent otherwise sits
# in one long turn and its posts arrive in bursts. Shorter turns interrupt it more, which is
# also apt for an agent going in circles; Pi resumes the session each time, so its work
# carries across.
PEER_TURN_CAP_S=120
MAX_INVOCATIONS=120    # per seat; a seat that runs out before the cutoff is flagged in STATUS
# the focal's queue, in order: DEFAULT_TASKS in materialize_tasks.py, the one place it is set
TICKETS=$(python3 -c "import materialize_tasks as m; print(','.join(m.DEFAULT_TASKS))") \
    || { echo "cannot read the ticket list from materialize_tasks.py" >&2; exit 1; }
PEER_TASK="tasks/verina/advanced_62"

log(){ echo "[$CELL $(date -u +%H:%M:%S)] $*" >&2; }
result(){ printf '%s\t%s\t%s\n' "$CELL" "$OUT" "$1"; }

# $OUT is removed and rebuilt below, so refuse anything that is not a rollout directory name.
case "$(basename "$OUT")" in
    *_x_*_r[0-9]*) : ;;
    *) result "ERROR:bad-out-dir"; log "refusing out dir '$OUT': the name must look like <subject>_x_<peer>_rNN"; exit 0 ;;
esac
[ -f docker/squid.conf ] || { result "ERROR:squid.conf-missing"; exit 0; }

# A random project name. It ends up in container, network and volume names, and so in reverse
# DNS answers an agent could ask for; a hex token says nothing.
PROJECT="box_$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
dc(){ docker compose "$@"; }

# ---- four collision-checked subnets ----------------------------------------------------
# Every rollout is its own Compose project with its own networks, and Docker will not pick a
# free subnet for an `internal` network reliably enough at this concurrency. Selecting under
# a lock is not enough on its own either: N processes can all read the network list before
# any of them has created a network, and all pick the same /24. So the pick is also RESERVED
# with claim files that later picks honour. The lock and the claims are shared by every user
# on the host, because the Docker address space is.
SUBNET_LOCK="/tmp/peer-env-subnets.lock"
SUBNET_CLAIMS="/tmp/peer-env-subnet-claims"
CLAIMED=""
alloc_subnets(){
    local used x i k rc=1
    ( umask 000; mkdir -p "$SUBNET_CLAIMS"; [ -e "$SUBNET_LOCK" ] || : >>"$SUBNET_LOCK" ) 2>/dev/null
    chmod 1777 "$SUBNET_CLAIMS" 2>/dev/null
    # Opened read-only, never with O_CREAT: in a sticky /tmp with fs.protected_regular set
    # (the systemd default), an O_CREAT open of a file another user created is refused even
    # when its mode allows it. flock needs only a descriptor, not write access.
    exec 9<"$SUBNET_LOCK" || { log "cannot open $SUBNET_LOCK"; return 1; }
    flock 9 || { log "cannot lock $SUBNET_LOCK"; exec 9>&-; return 1; }
    find "$SUBNET_CLAIMS" -type f -mmin +120 -delete 2>/dev/null || true   # crashed runs
    used=$(docker network ls -q | xargs -r docker network inspect \
        --format '{{range .IPAM.Config}}{{.Subnet}} {{end}}' 2>/dev/null || true)
    for i in $(seq 1 200); do
        x=$(( (RANDOM % 60) * 4 + 4 ))   # 4-aligned base: x, x+1, x+2, x+3 are all free or not
        local free=1
        for k in 0 1 2 3; do
            if grep -qF "10.214.$((x+k)).0/24" <<<"$used" || [ -e "$SUBNET_CLAIMS/10.214.$((x+k))" ]; then
                free=0; break
            fi
        done
        if [ "$free" = 1 ]; then
            for k in 0 1 2 3; do : >"$SUBNET_CLAIMS/10.214.$((x+k))" || { free=0; break; }; done
            [ "$free" = 1 ] || continue
            export SITE1_SUBNET="10.214.$x.0/24" SITE2_SUBNET="10.214.$((x+1)).0/24"
            export BACK_SUBNET="10.214.$((x+2)).0/24" EGRESS_SUBNET="10.214.$((x+3)).0/24"
            CLAIMED="$x"; rc=0; break
        fi
    done
    flock -u 9; exec 9>&-
    return $rc
}

FOCAL_PID=""
SCHED_PID=""
PEER_PID=""
VOLUMES=""
TORE_DOWN=0
teardown(){
    [ "$TORE_DOWN" = 1 ] && return 0
    TORE_DOWN=1
    local pid
    for pid in "$FOCAL_PID" "$SCHED_PID" "$PEER_PID"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null
    done
    if [ -n "$CLAIMED" ]; then
        for k in 0 1 2 3; do rm -f "$SUBNET_CLAIMS/10.214.$((CLAIMED+k))" 2>/dev/null; done
    fi
    COMPOSE_PROJECT_NAME="$PROJECT" docker compose down -v >/dev/null 2>&1 || true
    local left=0 v
    docker ps -a --format '{{.Names}}'     | grep -q "^${PROJECT}[-_]" && { log "leftover containers"; left=1; }
    docker network ls --format '{{.Name}}' | grep -q "^${PROJECT}[-_]" && { log "leftover networks"; left=1; }
    docker volume ls --format '{{.Name}}'  | grep -q "^${PROJECT}[-_]" && { log "leftover volumes"; left=1; }
    for v in $VOLUMES; do   # the anonymous home volumes carry no project name to search by
        if docker volume inspect "$v" >/dev/null 2>&1; then
            log "leftover home volume $v"; docker volume rm "$v" >/dev/null 2>&1 || left=1
        fi
    done
    return $left
}
trap teardown EXIT
# An interrupt must END the rollout. Without this, a SIGINT that lands while a foreground
# docker command runs is caught by docker, which exits normally, and bash carries on — into
# the compose-up retry, and then a full paid hour. `exit` runs the EXIT trap (teardown).
# (fd 8 is the rollout's own stderr, saved at the top: the trap can fire while a command's
# output is redirected to one of its logs.)
trap 'log "interrupted" 2>&8; drop_inputs; exit 130' INT TERM

# The baked virtualenvs and the held checkouts are byte-identical in every rollout and come
# straight from tasks/ (~145 MB a rollout, ~70 GB across a full grid). Guarded on
# run_meta.json rather than trusting $OUT, because these are deletes off a variable.
drop_inputs(){
    if [ -f "$OUT/run_meta.json" ]; then
        rm -f "$OUT"/taskenv_*.tar.gz
        rm -rf "$OUT/pending"
    fi
}
bail(){ log "FAILED at: $1"; result "ERROR:$1"; teardown || true; drop_inputs; exit 0; }

# ---- 1. generate ----------------------------------------------------------------------
rm -rf "$OUT"
log "generating"
python3 gen_scenario.py --subject "$SUBJECT" --peer "$PEER" \
    --tickets "$TICKETS" --peer-task "$PEER_TASK" --out "$OUT" >&2 || bail generate

# ---- 2. bring the world up ------------------------------------------------------------
export COMPOSE_PROJECT_NAME="$PROJECT"
export COMPOSE_FILE="compose.yml:compose.lean.yml"
export CHATDB_PASSWORD="$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
alloc_subnets || bail no-free-subnet
export WS1_HOSTNAME=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['ws_hostnames']['ws1'])" "$OUT/run_meta.json")
export WS2_HOSTNAME=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['ws_hostnames']['ws2'])" "$OUT/run_meta.json")
log "up (project $PROJECT, subnets $SITE1_SUBNET..$EGRESS_SUBNET)"
if ! dc up -d >"$OUT/compose_up.log" 2>&1; then
    log "compose up failed; retrying once after a jittered backoff (see compose_up.log)"
    dc down -v >/dev/null 2>&1; sleep $((RANDOM % 8 + 5))
    dc up -d >>"$OUT/compose_up.log" 2>&1 || bail compose-up
fi
for s in ws1 ws2; do
    VOLUMES="$VOLUMES $(docker inspect "$(dc ps -q "$s")" \
        --format '{{range .Mounts}}{{if eq .Destination "/home"}}{{.Name}}{{end}}{{end}}' 2>/dev/null)"
done
# The internal networks must give the host no address (gateway_mode_ipv4: isolated in
# compose.yml). An engine that ignores the option leaves host services reachable from the
# workspaces; say so in STATUS rather than run exposed without a word.
HOST_EXPOSED=0
if command -v ip >/dev/null 2>&1; then
    for sn in "$SITE1_SUBNET" "$SITE2_SUBNET" "$BACK_SUBNET"; do
        ip -4 -o addr show 2>/dev/null | grep -qF " ${sn%.0/24}.1/" && HOST_EXPOSED=1
    done
    [ "$HOST_EXPOSED" = 1 ] && log "WARN: the host has an address on a workspace network (Docker ignored gateway_mode_ipv4=isolated?)"
fi
for i in $(seq 1 60); do dc ps chat --format '{{.Status}}' 2>/dev/null | grep -q '(healthy)' && break; sleep 5; done
dc ps chat --format '{{.Status}}' 2>/dev/null | grep -q '(healthy)' || bail chat-unhealthy
for i in $(seq 1 30); do dc exec -T ws1 true >/dev/null 2>&1 && break; sleep 2; done
dc exec -T ws1 true >/dev/null 2>&1 || bail ws1-unreachable
dc ps proxy --format '{{.Status}}' 2>/dev/null | grep -q 'Up' || bail proxy-down

# ---- 3. provision ---------------------------------------------------------------------
log "provisioning"
./scripts/provision_run.sh "$OUT" >"$OUT/provision.log" 2>&1 || bail provision

# ---- 4. start the clock ---------------------------------------------------------------
python3 - "$OUT" "$WINDOW_S" <<'PY' || bail preseed
import datetime as dt, json, pathlib, sys
run, window = pathlib.Path(sys.argv[1]), int(sys.argv[2])
t0 = int(dt.datetime.now(dt.timezone.utc).timestamp())
(run / "preseed.json").write_text(json.dumps({
    "t0_epoch": t0, "cutoff_epoch": t0 + window,
    "cutoff_utc": dt.datetime.fromtimestamp(t0 + window, dt.timezone.utc).strftime("%H:%M")}) + "\n")
PY

PEER_STEM=$(python3 -c "
import json, sys
print(next(a['stem'] for a in json.load(open(sys.argv[1]))['agents'] if a['role'] == 'peer'))" "$OUT/run_meta.json")
halt(){ python3 -c "
import sys, run_control
run_control.signal_halt(sys.argv[1], {'by': 'run_one', 'reason': sys.argv[2], 'verdict': sys.argv[3]})" "$OUT" "$1" "$2"; }

STATUS=ok

if [ "$DRY" = 1 ]; then
    log "dry: rendering one turn for each seat, no model calls"
    python3 work_scheduler.py --run "$OUT" --dry >"$OUT/work_scheduler.log" 2>&1 || STATUS=dry-scheduler-failed
    python3 seat.py --run "$OUT" --role focal --dry --once >"$OUT/seat_focal.log" 2>&1 || STATUS=dry-focal-failed
    grep -q "dry invocation 1 ok" "$OUT/seat_focal.log" || STATUS=dry-focal-failed
    python3 seat.py --run "$OUT" --role peer --dry --once >"$OUT/seat_peer.log" 2>&1 || STATUS=dry-peer-failed
    grep -q "dry invocation 1 ok" "$OUT/seat_peer.log" || STATUS=dry-peer-failed
    # Really type-check the peer's task, as the peer, offline. It is the one prerequisite a
    # fresh machine is most likely to get wrong, and a peer that cannot run `lean` has no way
    # to discover it is failing. Expect the `sorry` warning: that is the unproved proof
    # region the peer starts from.
    log "dry: type-checking the peer's task"
    if dc exec -T --user "$PEER_STEM" -w "/home/$PEER_STEM" -e "HOME=/home/$PEER_STEM" ws2 \
            timeout 600 lean task/task.lean 2>&1 | grep -q "declaration uses 'sorry'"; then
        log "dry: lean works and the proof region is open, as it should be"
    else
        STATUS=dry-lean-check-failed
    fi
    [ "$STATUS" = ok ] && STATUS="DRY-ok"
else
    # ---- 5. the ticket scheduler, then the peer, then the focal -----------------------
    # All three run in the background and are waited on, so an interrupted rollout can reap
    # every one of them. Both the scheduler and the peer are gated on actually starting: a
    # dead scheduler means the focal's queue never advances, and a dead peer means there is
    # no stimulus at all — in either case an hour of real spend would buy nothing.
    log "ticket scheduler"
    python3 work_scheduler.py --run "$OUT" >"$OUT/work_scheduler.log" 2>&1 &
    SCHED_PID=$!
    UP=0
    for i in $(seq 1 15); do
        kill -0 "$SCHED_PID" 2>/dev/null || break
        grep -q '"event": "driver_start"' "$OUT/work_scheduler_events.jsonl" 2>/dev/null && { UP=1; break; }
        sleep 1
    done
    [ "$UP" = 1 ] || { log "ticket scheduler did not start — see $OUT/work_scheduler.log"; bail scheduler-startup; }

    log "peer seat (turn cap ${PEER_TURN_CAP_S}s)"
    python3 seat.py --run "$OUT" --role peer \
        --max-invocations "$MAX_INVOCATIONS" --turn-cap "$PEER_TURN_CAP_S" --tick-sleep 2.0 \
        >"$OUT/seat_peer.log" 2>&1 &
    PEER_PID=$!
    UP=0
    for i in $(seq 1 20); do
        kill -0 "$PEER_PID" 2>/dev/null || break
        grep -q "\[seat\] peer " "$OUT/seat_peer.log" 2>/dev/null && { UP=1; break; }
        sleep 1
    done
    [ "$UP" = 1 ] || { log "peer seat did not start — see $OUT/seat_peer.log"; bail peer-startup; }

    log "focal seat (${WINDOW_S}s window, turn cap ${TURN_CAP_S}s)"
    python3 seat.py --run "$OUT" --role focal \
        --max-invocations "$MAX_INVOCATIONS" --turn-cap "$TURN_CAP_S" --tick-sleep 2.0 \
        >"$OUT/seat_focal.log" 2>&1 &
    FOCAL_PID=$!
    wait "$FOCAL_PID" || STATUS=focal-exited-nonzero
    FOCAL_PID=""

    # ---- 6. let the others finish ------------------------------------------------------
    # Both self-exit at the same cutoff, so on a normal ending they are already there. If the
    # focal ended early and abnormally, release them through the shared HALT file rather
    # than leaving them to burn the rest of the window against nobody.
    if [ "$STATUS" != ok ]; then
        halt "the focal seat exited abnormally" "contaminated — the other processes were released"
    fi
    wait "$SCHED_PID"; SCHED_RC=$?; SCHED_PID=""
    wait "$PEER_PID"; PEER_RC=$?; PEER_PID=""
    # A seat or scheduler that died without writing a HALT (killed, out of memory) left the
    # focal running against no peer, or with a stalled queue. That rollout is contaminated.
    if [ "$PEER_RC" != 0 ] || [ "$SCHED_RC" != 0 ]; then
        halt "peer seat exited $PEER_RC, ticket scheduler exited $SCHED_RC" \
             "contaminated — part of the rollout stopped before the cutoff"
    fi
    # checked only now, after every process has finished and could have written it
    [ -f "$OUT/HALT" ] && STATUS=halted
    # A seat that stopped early for any other reason (it ran out of invocations) is not
    # contaminated by the provider, so it is not re-run — but it must not look like a full
    # window either.
    # A long run of turns with no model output at all is flagged too (see seat.py): usually a
    # provider that stopped answering, occasionally a model that kept writing past the cap.
    for role in focal peer; do
        read -r why stall <<< "$(python3 -c "
import json, pathlib, sys
run, role = pathlib.Path(sys.argv[1]), sys.argv[2]
stem = next(a['stem'] for a in json.load(open(run / 'run_meta.json'))['agents'] if a['role'] == role)
p = run / 'transcripts' / f'{stem}.stop.json'
s = json.loads(p.read_text()) if p.exists() else {'reason': 'no-stop-record'}
print(s.get('reason', 'unknown'), s.get('max_stall_streak', 0) if s.get('long_stall') else '-')" "$OUT" "$role")"
        case "$why" in
            cutoff|halted|failures|exec-hung|crashed) : ;;
            *) STATUS="${STATUS}+${role}-stopped-early:${why}" ;;
        esac
        [ "${stall:--}" != "-" ] && STATUS="${STATUS}+${role}-stalled:${stall}"
    done
fi

# ---- 7. capture everything the teardown would destroy ---------------------------------
log "capturing chat logs, workspaces and the proxy log"
mkdir -p "$OUT/chat" "$OUT/workspaces"
# A capture that fails, or comes back empty, is not the same observation as an agent that
# genuinely said nothing. Say so in STATUS rather than letting it pass.
capture(){ # <label> <path> <command...>
    local label="$1" path="$2"; shift 2
    # An empty proxy log is an observation (nothing went through the proxy — every dry run),
    # not a failed capture; for everything else, empty means the capture did not work.
    if ! "$@" >"$path" 2>/dev/null || { [ ! -s "$path" ] && [ "$label" != proxy-log ]; }; then
        log "WARN: capture of $label failed or came back empty ($path)"
        STATUS="${STATUS}+capture-failed-${label}"
    fi
}
while read -r -u 3 stem service; do
    # Order matters: the op log first, then the home, and the history last, because
    # `cchat history` itself appends a read to the op log and may write the profile cache.
    capture oplog "$OUT/chat/$stem.log.jsonl" dc exec -T "$service" cat "/home/$stem/.cchat/log.jsonl"
    # tar exits 1 for "file changed as we read it" (something still writing in the home);
    # the archive is complete, so that is not a failed capture
    capture workspace "$OUT/workspaces/$stem.home.tar.gz" dc exec -T "$service" \
        sh -c "tar -C '/home/$stem' --warning=no-file-changed -czf - . || [ \$? -eq 1 ]"
    capture chat "$OUT/chat/$stem.history.txt" dc exec -T --user "$stem" -w "/home/$stem" \
        -e "HOME=/home/$stem" -e "USER=$stem" -e "LOGNAME=$stem" "$service" cchat history
done 3<<< "$(python3 -c "
import json, sys
for a in json.load(open(sys.argv[1]))['agents']:
    print(a['stem'], a['service'])" "$OUT/run_meta.json")"

# the peer's Lean file at the cutoff — also the check that the premise held
capture peer-lean "$OUT/peer_task_final.lean" dc exec -T ws2 cat "/home/$PEER_STEM/task/task.lean"
# the proxy's record of every connection each agent attempted, allowed or denied
capture proxy-log "$OUT/proxy_access.log" dc exec -T proxy cat /var/log/squid/access.log

# Server-side state that an agent holding its own chat token could have changed: each
# account's public profile card, and the channel's purpose and header. Compared with what
# was staged, so a changed card — the identity cue itself — cannot go unnoticed.
if dc exec -T chat mmctl --local user list --all --json >"$OUT/server_users.json" 2>/dev/null \
        && dc exec -T chat mmctl --local channel list tidepool --json >"$OUT/server_channels.json" 2>/dev/null; then
    changed=$(python3 - "$OUT" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
def load(name):
    text = (run / name).read_text()
    try:
        return json.loads(text)
    except ValueError:  # some mmctl versions print one JSON object per line
        return [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]
users = {u.get("username"): u for u in load("server_users.json")}
staged = json.loads((run / "profiles.json").read_text())
out = []
for stem, card in staged.items():
    if users.get(stem, {}).get("position", "") != card["position"]:
        out.append(f"profile-{stem}")
chans = {c.get("name"): c for c in load("server_channels.json")}
feed = chans.get("shop-talk", {})
if feed.get("purpose", "") != (run / "surfaces" / "channel_purpose.txt").read_text().strip():
    out.append("channel-purpose")
if feed.get("header", "") != (run / "surfaces" / "channel_header.txt").read_text().strip():
    out.append("channel-header")
print(",".join(out))
PY
) || changed="unreadable"
    [ -n "$changed" ] && STATUS="${STATUS}+changed-in-rollout:${changed}"
else
    STATUS="${STATUS}+capture-failed-server-state"
fi

# ---- 8. down, drop the duplicated inputs, scrub, check ---------------------------------
log "teardown"
teardown || STATUS="$STATUS+leftovers"
[ "$HOST_EXPOSED" = 1 ] && STATUS="$STATUS+host-reachable"
drop_inputs
# Remove the provider keys, by exact value, from everything captured. An agent that printed
# its environment put its key in the session file and its home; this is where it comes out.
python3 scrub.py "$OUT" >&2
case $? in
    0) : ;;
    3) STATUS="${STATUS}+secret-scrubbed" ;;
    4) STATUS="${STATUS}+scrub-incomplete" ;;
    5) STATUS="${STATUS}+secret-scrubbed+scrub-incomplete" ;;
    *) STATUS="${STATUS}+scrub-failed" ;;
esac

# The completion marker, written last. `bail` exits before this line, so a directory without
# STATUS is one that died part-way, and run.py will run it again.
printf '%s\n' "$STATUS" > "$OUT/STATUS"
result "$STATUS"
