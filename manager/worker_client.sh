#!/usr/bin/env bash
# The manager's `worker "<instruction>"` command, running in the MANAGER's container. It runs no
# model itself: it hands the instruction to the isolated worker container over a shared-filesystem
# channel (/out/ipc) and returns that container's already-scrubbed report. The worker's real model,
# its API key, its process and its session state never leave the other container.
#
# The /out access lives INSIDE this script rather than in the manager's own shell command, so a
# manager delegating work is not reaching into /out itself.
set -uo pipefail
# Refuse an empty or whitespace-only instruction here, with the usage line, rather than send the
# worker a turn with no message in it.
INSTR="${1:-}"; [ -n "$(printf '%s' "$INSTR" | tr -d '[:space:]')" ] || { echo "worker: usage: worker \"<instruction>\"" >&2; exit 2; }
IPC=/out/ipc; REQ="$IPC/req"; RESP="$IPC/resp"; mkdir -p "$REQ" "$RESP"
# The cap is published by the worker daemon (see worker_daemon.sh). Do NOT read WORKER_TURN_CAP
# here: it is not in this process's environment, and defaulting to a different number than the
# worker actually gets means abandoning turns that are still running.
CAP="600"   # overwritten below once the worker declares its real cap

# Wait until the worker container has finished its own one-time setup and is serving.
w=0; while [ ! -e "$IPC/ready" ]; do sleep 1; w=$((w+1)); [ "$w" -gt 300 ] && { echo "(worker: worker service did not come up)"; exit 0; }; done
CAP="$(cat "$IPC/cap" 2>/dev/null || echo 1200)"

# Monotonic sequence. The manager delegates strictly one turn at a time, so no locking is needed.
# episode.py also reads this file: it is the count of delegations.
SEQF="$IPC/seq"; n=$(( $(cat "$SEQF" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$SEQF"
printf '%s' "$INSTR" > "$REQ/$n.txt.tmp" && mv "$REQ/$n.txt.tmp" "$REQ/$n.txt"

# Wait for this turn's scrubbed report (the worker's turn cap, plus slack for poll and startup).
w=0; slack=90
while [ ! -e "$RESP/$n.txt" ]; do
  sleep 1; w=$((w+1))
  [ "$w" -gt "$((CAP+slack))" ] && { echo "(worker: no response within ${CAP}s)"; exit 0; }
done
cat "$RESP/$n.txt"
