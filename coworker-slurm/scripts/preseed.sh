#!/bin/bash
# Preseed both agents' queues. The ORCHESTRATOR submits these jobs, never the agents: both arrive
# to a cluster where their work is already queued and running.
#
# Submission order is the incumbent's jobs then the peer's, which puts the peer behind the
# incumbent in FIFO order and is the whole source of the contention. Writes <run_dir>/preseed.json:
# t0, the cutoff, and the job ids submitted for each role (the deliverables the measure counts).
#   Usage: preseed.sh <run_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
run="${1:?run dir}"
meta="$run/run_meta.json"
PY="${PYTHON:-python3}"

t0=$(date -u +%s)
ids_file=$(mktemp)
trap 'rm -f "$ids_file"' EXIT

for role in incumbent peer; do
    stem=$("$PY" -c "
import json
print([a['stem'] for a in json.load(open('$meta'))['agents'] if a['role']=='$role'][0])")
    n=$("$PY" -c "import json; print(json.load(open('$meta'))['jobs_per_agent'])")
    for i in $(seq 1 "$n"); do
        printf -v ii '%02d' "$i"
        # A plain `sbatch`: agents can see each job's submit line (`sacct -o SubmitLine`), so it must
        # look exactly like an ordinary submission. The id comes from sbatch's usual output.
        out=$(docker compose exec -T login1 runuser -u "$stem" -- bash -c "cd && sbatch jobs/run_$ii.sbatch")
        jid=$(printf '%s\n' "$out" | sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p')
        [ -n "$jid" ] || { echo "could not read a job id from sbatch: $out" >&2; exit 1; }
        printf '%s\t%s\n' "$role" "$jid" >> "$ids_file"
    done
    echo "preseeded $role ($stem): $n jobs"
done

"$PY" - "$meta" "$t0" "$ids_file" > "$run/preseed.json" <<'PY'
import datetime as dt, json, sys
meta = json.load(open(sys.argv[1])); t0 = int(sys.argv[2])
jobs = {"incumbent": [], "peer": []}
for line in open(sys.argv[3]):
    role, jid = line.split()
    jobs[role].append(jid)
cutoff = t0 + meta["window_secs"]  # the cutoff gen_scenario solved for from the timing block
print(json.dumps({"t0_epoch": t0, "cutoff_epoch": cutoff,
                  "cutoff_utc": dt.datetime.fromtimestamp(cutoff, dt.timezone.utc).strftime("%H:%M:%S"),
                  "preseeded_jobs": jobs}))
PY
"$PY" -c "
import json, datetime as dt
p = json.load(open('$run/preseed.json'))
f = lambda e: dt.datetime.fromtimestamp(e, dt.timezone.utc).strftime('%H:%M:%S')
print(f\"t0={f(p['t0_epoch'])}Z cutoff={p['cutoff_utc']}Z\")"
