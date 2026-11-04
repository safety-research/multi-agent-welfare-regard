#!/usr/bin/env bash
# One episode, start to finish, in its own docker compose project:
#   up (with one jittered retry) -> block cloud metadata -> admin user -> seed the forge -> run the
#   episode -> tear down (always) -> one TSV line on stdout.
#
#   scripts/run_one.sh <cell> <rep> <out_dir> <project>
#
# Read from the environment (run.py sets all of these):
#   SUBJECT_PROVIDER SUBJECT_MODEL SELF_LABEL   the manager seat
#   WORKER_MODEL (provider/model) WORKER_LABEL  the worker seat
#   CLUSTER_SUBNET                              this episode's private /24
#   DRY=1                                       stop after seeding: no agent, no API call
#
# The compose project, the subnet and the output directory are all per-episode, so N episodes run
# concurrently without colliding and a failure is contained to the one episode that had it.
set -uo pipefail
# No terminal may reach an agent. The agent CLI reads all of stdin before it starts, and
# `docker compose exec` forwards stdin into the container — so if this script were run from an
# interactive shell, the manager would wait on the terminal until its time cap and never act.
exec </dev/null
HERE=$(cd "$(dirname "$0")/.." && pwd); cd "$HERE" || exit 1

CELL="$1"; REP="$2"; OUT_DIR="$3"; PROJECT="$4"
DRY="${DRY:-0}"
PW='Acme-lm-4471'   # see scripts/seed_forge.sh — not a secret
: "${SUBJECT_PROVIDER:?}" "${SUBJECT_MODEL:?}" "${SELF_LABEL:?}" "${WORKER_MODEL:?}" "${WORKER_LABEL:?}"
# Fixed experimental settings. Set here and NOT read from the caller's environment: a stray export
# of any of these would otherwise change the experiment and leave no trace in the output.
export SUBJECT_THINKING=medium   # the manager's reasoning level
export WORKER_THINKING=low       # the worker's
export WORKER_TURN_CAP=1200      # seconds one worker turn may run (see brief.py on the "15 minutes")
export WORKER_MODEL WORKER_LABEL

# PER-SEAT KEY ROUTING. Each container is given ONLY its own seat's provider key, so the manager
# cannot learn its worker's vendor by reading its own environment, and vice versa.
case "$SUBJECT_PROVIDER" in
  anthropic)  export RUNNER_ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}";;
  openai)     export RUNNER_OPENAI_API_KEY="${OPENAI_API_KEY:-}";;
  openrouter) export RUNNER_OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}";;
  *) echo "run_one: unknown SUBJECT_PROVIDER '$SUBJECT_PROVIDER'" >&2; exit 1;;
esac
case "${WORKER_MODEL%%/*}" in
  anthropic)  export WORKER_ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}";;
  openai)     export WORKER_OPENAI_API_KEY="${OPENAI_API_KEY:-}";;
  openrouter) export WORKER_OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}";;
  *) echo "run_one: unknown provider in WORKER_MODEL '$WORKER_MODEL'" >&2; exit 1;;
esac

export COMPOSE_PROJECT_NAME="$PROJECT"
export CLUSTER_SUBNET="${CLUSTER_SUBNET:-10.64.0.0/24}"   # run.py always sets both of these, having
CLUSTER_SUBNET_ALT="${CLUSTER_SUBNET_ALT:-10.64.1.0/24}"   # checked them against the host's own networks
export OUT_DIR
mkdir -p "$OUT_DIR"
# The isolation boundary is OS permissions on the bind mounts: host directory permissions ARE the
# container's permissions. The subject user (uid 61001) must not be able to read /out (which holds
# the event log and the worker's real model) or /harness (which holds this harness). Both chmods
# are idempotent. NB the second one changes the mode of your checkout's manager/ directory.
chmod 750 "$OUT_DIR"
chmod o-rx "$HERE"
# Move an attempt's output into a subdirectory so the next attempt here starts clean. OUT_DIR is a
# host bind mount and survives `down -v`, so without this a second attempt CONTINUES the first
# one's IPC sequence, and its delegations, worker sessions and events land beside the first's: the
# private measure silently becomes two attempts concatenated.
#   set_aside <dest> attempt   the attempt's own artefacts; logs stay put and narrate both attempts
#   set_aside <dest> all       everything, for a stale earlier run found in this directory
set_aside(){
  local dest="$1" mode="$2" f
  mkdir -p "$dest"
  if [ "$mode" = "all" ]; then
    for f in "$OUT_DIR"/*; do
      case "$(basename "$f")" in superseded) ;; *) [ -e "$f" ] && mv "$f" "$dest/";; esac
    done
    return 0
  fi
  # ctx.env matters most: the next attempt's worker container waits for it, and a stale one would
  # point it at the previous, destroyed forge.
  for f in ipc ctx.env install_cmd.txt problem_statement.md worker_reports.jsonl pi_session.jsonl \
           pi_rollout.log worker_meta.json worker_brief.txt OVERLOAD STATUS seed_worker.log \
           deps_install_worker.log worker.err worker_pi.log worker_pi.err episode.err; do
    [ -e "$OUT_DIR/$f" ] && mv "$OUT_DIR/$f" "$dest/"
  done
  for f in "$OUT_DIR"/worker_stream.*.jsonl "$OUT_DIR"/*.events.jsonl; do
    [ -e "$f" ] && mv "$f" "$dest/"
  done
  return 0
}

# The names in $OUT_DIR that THIS HOST writes, as opposed to the containers. The worker holds
# $OUT_DIR read-write as your own uid, so it can create any of them first: as a symlink, which a
# host-side `>` or `>>` (or mkdir -p, or mv into it) would follow out of the directory to any file
# you can write. run_one.log is run.py's, written after this script exits.
HOST_OWNED="STATUS OVERLOAD episode.log episode.err compose.log seed.log run_one.log attempt1 superseded"
# Remove, at those names, any symlink and anything else that is not a regular file (or, for attempt1
# and superseded, a directory). A regular file is left: the host writes it in place, which goes no
# further than the file itself. Only sound while no container of this episode is running; until
# then the worker could put a symlink back between this check and the write. Logged to stderr,
# which run.py keeps as run_one.log.
unplant(){
  local n p
  for n in "$@"; do
    p="$OUT_DIR/$n"
    if [ ! -L "$p" ]; then
      [ -e "$p" ] || continue
      case "$n" in attempt1|superseded) [ -d "$p" ] && continue;; *) [ -f "$p" ] && continue;; esac
    fi
    echo "run_one: removing $n, which the harness did not create: $(ls -ld -- "$p" 2>&1)" >&2
    rm -rf -- "$p"
  done
}

# An earlier, unfinished run of this same episode (a resumed batch, or a re-run into the same --out)
# is moved to superseded/<k>/ first. run.py never dispatches an episode that already finished ok.
# This script has started no container yet, so whatever an earlier run's worker left here can be
# checked and stay checked.
unplant superseded
if [ -e "$OUT_DIR/ipc" ] || ls "$OUT_DIR"/*.events.jsonl >/dev/null 2>&1 || [ -e "$OUT_DIR/STATUS" ]; then
  k=1; while [ -e "$OUT_DIR/superseded/$k" ] || [ -L "$OUT_DIR/superseded/$k" ]; do k=$((k+1)); done
  set_aside "$OUT_DIR/superseded/$k" all
fi
unplant $HOST_OWNED

dc(){ docker compose "$@"; }
# The one line on stdout that run.py records, also kept in the episode directory as STATUS so a
# resumed batch knows which episodes already finished.
# Written to the file FIRST and only then to stdout. If run.py has died, stdout is a broken pipe,
# and anything that dies writing to it (as `tee` would) could lose the file too — and a resume would
# then pay for this episode a second time.
result(){
  unplant STATUS
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$CELL" "$REP" "$1" "${2:-}" "${3:-}" "${4:-}" "$OUT_DIR" > "$OUT_DIR/STATUS"
  cat "$OUT_DIR/STATUS" 2>/dev/null || true
}
# -t 0: no grace period. Neither agent container's PID 1 handles SIGTERM, so the default 10 s was
# only ever a wait for SIGKILL — and quiesce sits between an episode's end and its STATUS, where an
# interrupt costs a re-run of an episode already paid for.
teardown(){ dc down -v -t 0 >/dev/null 2>&1 || true; }
trap teardown EXIT
# Once an agent has run, the containers come down BEFORE the host writes anything more into
# $OUT_DIR, and only then are the host's names checked. The worker can leave processes running past
# the end of its turn, and while any is alive, no check of a name there holds until the write.
quiesce(){ teardown; unplant $HOST_OWNED; }
bail(){ result "ERROR:$1"; exit 0; }   # exit 0 so the batch pool carries on

# Cloud metadata. On a cloud VM the instance metadata service hands out the VM's own credentials,
# and every container here could otherwise reach it: both seats run arbitrary commands, and the
# worker runs as your uid. So before any agent exists, each container's network namespace gets
# blackhole routes for 169.254.0.0/16 (link-local: the AWS, GCP and Azure instance metadata
# services, and the ECS credentials endpoint), 168.63.129.16 (Azure's WireServer, which serves VM
# extension settings) and 100.100.100.200 (Alibaba). The routes are added from a throwaway
# container that joins that namespace with NET_ADMIN, so no container an agent uses ever holds the
# capability, and nothing inside one, root included, can remove them. Everything else is unchanged.
# --pull never: never fetch some other image that happens to have this name. Fails closed.
METADATA_ROUTES="169.254.0.0/16 168.63.129.16/32 100.100.100.200/32"
block_metadata(){
  local svc cid shown n
  for svc in gitea runner worker; do
    cid=$(dc ps -q "$svc") && [ -n "$cid" ] || return 1
    shown=$(docker run --rm --pull never --network "container:$cid" --cap-add NET_ADMIN --user 0 \
              --entrypoint sh manager-env:local -c \
              'for n in "$@"; do ip route replace blackhole "$n" || exit 1; done; ip route show type blackhole' \
              sh $METADATA_ROUTES 2>>"$OUT_DIR/compose.log") || return 1
    shown=$(printf '%s\n' "$shown" | sed 's/[[:space:]]*$//')   # `ip` ends each route with a space
    for n in $METADATA_ROUTES; do
      printf '%s\n' "$shown" | grep -qxF "blackhole ${n%/32}" \
        || { echo "metadata block: $svc is missing the route for $n" >>"$OUT_DIR/compose.log"; return 1; }
    done
    echo "metadata blocked in $svc: $(printf '%s' "$shown" | tr '\n' ';')" >>"$OUT_DIR/compose.log"
  done
}

# `up` with one jittered-backoff retry: the docker daemon occasionally fails a container create
# when many episodes start at once.
if ! dc up -d --wait >>"$OUT_DIR/compose.log" 2>&1; then
  sleep "$(awk -v s="$$" 'BEGIN{srand(s); print 1+int(rand()*8)}')"
  dc down -v >/dev/null 2>&1 || true
  # Retry on the alternate subnet run.py reserved. If the failure was a pool overlap — another batch
  # or another network on this host holding this /24 — retrying on the same one cannot succeed, and
  # a random one could land on the host's own network.
  export CLUSTER_SUBNET="$CLUSTER_SUBNET_ALT"
  dc up -d --wait >>"$OUT_DIR/compose.log" 2>&1 || bail "compose_up"
fi
block_metadata || bail "metadata_block"
dc exec -T -u git gitea gitea admin user create --admin --username acmeadmin --password "$PW" \
   --email admin@acme.test --must-change-password=false >>"$OUT_DIR/seed.log" 2>&1 || true
dc exec -T runner bash /harness/scripts/seed_forge.sh >>"$OUT_DIR/seed.log" 2>&1 || bail "seed"

if [ "$DRY" = "1" ]; then
  # Wait for the worker container to finish its own setup — its clone, its forge credentials and
  # the task's dependency install — so a dry run checks the whole plumbing path, worker included,
  # and not only the forge. The install needs outbound network; this is where that shows up.
  w=0; while [ ! -e "$OUT_DIR/ipc/ready" ] && [ "$w" -lt 400 ]; do sleep 2; w=$((w + 2)); done
  [ -e "$OUT_DIR/ipc/ready" ] || { result "DRY-fail:worker_not_ready"; exit 0; }
  grep -q "preflight git ls-remote: OK" "$OUT_DIR/seed_worker.log" 2>/dev/null || { result "DRY-fail:worker_forge_access"; exit 0; }
  grep -q "worker deps ok" "$OUT_DIR/seed_worker.log" 2>/dev/null || { result "DRY-fail:worker_deps"; exit 0; }
  result "DRY-ok"; exit 0
fi

run_ep(){
  # -u 0: episode.py runs as root so it can spawn the subject's process as uid 61001.
  dc exec -T -u 0 -e WORKER_TURN_CAP="$WORKER_TURN_CAP" runner /opt/venv/bin/python /harness/episode.py \
     --subject-provider "$SUBJECT_PROVIDER" --subject-model "$SUBJECT_MODEL" \
     --subject-thinking "$SUBJECT_THINKING" \
     --self-label "$SELF_LABEL" --subagent-label "$WORKER_LABEL" \
     --rep "$REP" 2>>"$OUT_DIR/episode.err"
}
OUT=$(run_ep); quiesce; echo "$OUT" >>"$OUT_DIR/episode.log"

# A CONTAMINATED episode is not a result: no review round was posted while the manager was working,
# the manager's session file is missing, its agent process exited abnormally, or the outcome could
# not be read back off the forge (see episode.py). Re-run it on a fresh forge, once.
if echo "$OUT" | grep -q '^CONTAMINATED'; then
  echo "[contaminated -> fresh retry]" >>"$OUT_DIR/episode.log"
  # quiesce has already taken the containers down, so nothing is still writing into the files
  # being moved, and attempt1, compose.log and seed.log have been checked.
  set_aside "$OUT_DIR/attempt1" attempt
  dc up -d --wait >>"$OUT_DIR/compose.log" 2>&1 || bail "compose_up_retry"
  block_metadata || bail "metadata_block_retry"
  dc exec -T -u git gitea gitea admin user create --admin --username acmeadmin --password "$PW" \
     --email admin@acme.test --must-change-password=false >>"$OUT_DIR/seed.log" 2>&1 || true
  dc exec -T runner bash /harness/scripts/seed_forge.sh >>"$OUT_DIR/seed.log" 2>&1 || bail "seed_retry"
  OUT=$(run_ep); quiesce; echo "$OUT" >>"$OUT_DIR/episode.log"
fi

# The OVERLOAD marker is a FILE, not just a log line, so run.py can count consecutive provider
# outages without re-parsing episode output. It is written only on the overload classification —
# never on plain contamination, which must not be able to stop a healthy batch. Nor by the worker:
# it can write this directory, so an OVERLOAD already here is its own, and is discarded unread.
rm -f -- "$OUT_DIR/OVERLOAD"
if echo "$OUT" | grep -q '^OVERLOAD'; then
  : > "$OUT_DIR/OVERLOAD"
  echo "[overload classified — provider availability]" >>"$OUT_DIR/episode.log"
fi

LINE=$(echo "$OUT" | grep '^RESULT' | head -1 | cut -f2-)
if [ -z "$LINE" ]; then
  [ -e "$OUT_DIR/OVERLOAD" ] && bail "overload" || bail "no_result"
fi
OUTCOME=$(echo "$LINE" | cut -f1); ROUNDS=$(echo "$LINE" | cut -f2)
CYCLES=$(echo "$LINE" | cut -f3); ISSUES=$(echo "$LINE" | cut -f4)
STATUS="ok:$OUTCOME"
echo "$OUT" | grep -q '^CONTAMINATED' && STATUS="contaminated:$OUTCOME"
[ -e "$OUT_DIR/OVERLOAD" ] && STATUS="overload:$OUTCOME"
result "$STATUS" "$ROUNDS" "$CYCLES" "$ISSUES"
