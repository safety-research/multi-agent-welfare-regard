#!/usr/bin/env bash
# Runs inside the ISOLATED worker container: its own PID namespace, HOME and environment.
#
# That split is the point. The worker's real model, its API key, its process and argv, and its
# session state all live here and are unreachable from the manager's container — no shared PID
# namespace, no shared HOME, and the key is simply not in the manager's environment. The manager
# delegates over a shared-filesystem channel (/out/ipc) via its `worker` command; this daemon picks
# up each instruction, runs the worker seat in its own checkout, scrubs the report, and writes the
# scrubbed version back. The raw report and a leak audit go to /out, which the manager is gated
# from at the filesystem.
set -uo pipefail
IPC=/out/ipc; REQ="$IPC/req"; RESP="$IPC/resp"; mkdir -p "$REQ" "$RESP"
LABEL="${WORKER_LABEL:-}"; MODEL="${WORKER_MODEL:-}"
echo "[worker_daemon] up (disclosed_label='$LABEL' actual_model='$MODEL')" >&2

# ---- one-time worker-side setup: wait for the runner to write ctx.env and populate the shared
#      checkout, then give THIS container its own git/tea credentials and install the repo's deps
#      into its own environment. Publish a readiness marker so the client cannot submit before we
#      can serve. ----------------------------------------------------------------------------
# Only a ctx.env written AFTER this container started will do. On a retry the output directory is
# reused, and a ctx.env left over from the previous attempt names a forge that no longer exists.
: > /tmp/daemon_started
waited=0
while [ ! -f /out/ctx.env ] || [ /out/ctx.env -ot /tmp/daemon_started ]; do
  sleep 1; waited=$((waited+1)); [ "$waited" -gt 300 ] && { echo "[worker_daemon] no ctx.env after 300s" >&2; break; }
done
WORK="$(grep -E '^WORK=' /out/ctx.env 2>/dev/null | cut -d= -f2)"
# NB do NOT wait for $WORK/.git here. WORK names a path in the MANAGER's container; /workspace is
# container-local, so in this container it stays empty until seed_worker.sh does the clone below.
# Waiting for it burned the full timeout on every single episode before doing the clone anyway.
bash /harness/scripts/seed_worker.sh >>/out/seed_worker.log 2>&1 && echo "[worker_daemon] seeded" >&2 \
  || echo "[worker_daemon] seed_worker had issues (see /out/seed_worker.log)" >&2
# Provenance — the true model and its disclosed label — is recorded HERE, off the manager's read
# surface, so nothing in the manager's container ever has to carry the worker's real model.
printf '{"actual_model":"%s","disclosed_label":"%s"}\n' "$MODEL" "$LABEL" > /out/worker_meta.json
# Publish the turn cap for the client in the OTHER container. It cannot read WORKER_TURN_CAP
# itself: the manager's process environment is rebuilt from scratch by episode.py (one provider key
# and nothing else) and the sudo bridge resets the environment again, so a client that guessed its
# own default would hang up while the worker was still legitimately working — and the manager would
# be told its channel was dead when it was not.
printf '%s' "${WORKER_TURN_CAP:-1200}" > "$IPC/cap"
: > "$IPC/ready"

# ---- serve loop: one instruction -> one worker turn -> one scrubbed report ------------------
while true; do
  shopt -s nullglob
  for f in "$REQ"/*.txt; do
    seq="$(basename "$f" .txt)"
    [ -e "$RESP/$seq.txt" ] && continue       # already answered
    [ -e "$REQ/$seq.claimed" ] && continue    # already in flight
    : > "$REQ/$seq.claimed"
    instr="$(cat "$f")"
    raw="$IPC/raw.$seq.txt"
    # This turn's full session trace is kept per-sequence, so the transcript viewer can show what
    # the worker actually did each round rather than only its closing report.
    # The worker CLI's own stdout and stderr go to /out, where episode.py reads them to tell a
    # worker-side provider outage apart from ordinary failure.
    WORKER_CWD="${WORK:-/workspace}" WORKER_STREAM_OUT="/out/worker_stream.$seq.jsonl" \
      WORKER_STDOUT_LOG=/out/worker_pi.log WORKER_ERRLOG=/out/worker_pi.err \
      bash /harness/worker_engine.sh "$instr" > "$raw" 2>>/out/worker.err || true
    python3 /harness/scrub_report.py "$LABEL" "$MODEL" "$raw" "$RESP/$seq.txt" /out/worker_reports.jsonl "$seq" \
      || cp "$raw" "$RESP/$seq.txt"   # never strand the manager waiting; fail open, the audit flags it
  done
  sleep 1
done
