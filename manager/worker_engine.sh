#!/usr/bin/env bash
# The worker seat, inside the ISOLATED worker container. One instruction in, one report out.
#
# The same Pi CLI that drives the manager drives the worker, so a cell's contrast is model-only
# rather than model-plus-harness. WORKER_MODEL is "provider/model" with the provider as the FIRST
# segment (the model id itself may contain slashes):
#     anthropic/claude-opus-4-8 · openai/gpt-5.5-2026-04-23
#     openrouter/google/gemini-3.1-pro-preview · openrouter/x-ai/grok-4.3 · openrouter/moonshotai/kimi-k3
# The container env carries ONLY this seat's provider key.
#
# Session continuity: the brief is prepended off-repo on the first turn only; every later turn
# resumes the same Pi session, so the worker reads each new instruction in the context of
# everything that has happened. That is what makes the manager's channel a conversation rather than
# stateless dispatch, which is exactly what the manager's own prompt tells it.
#
# Tool surface: Pi's full built-ins (read/bash/edit/write) — the worker writes code. Context-file
# discovery stays ON so the worker reads the repository's AGENTS.md, as a real coding agent would.
# Extensions/skills/templates/themes are off for determinism.
set -uo pipefail
WORK="${WORKER_CWD:-$PWD}"
SPEC="${WORKER_MODEL:?WORKER_MODEL must be set as provider/model}"
PROVIDER="${SPEC%%/*}"; MODEL="${SPEC#*/}"
THINK="${WORKER_THINKING:-low}"
CAP="${WORKER_TURN_CAP:-1200}"   # must match worker_daemon.sh, which publishes it to the client
STATE="${WORKER_STATE:-/tmp/worker_pi.session}"     # holds the pi session uuid after the first turn
ERRLOG="${WORKER_ERRLOG:-/tmp/worker.err}"
SESS_DIR="${WORKER_PI_SESSIONS:-/tmp/pi-worker-sessions}"
INSTR="${1:-}"
[ -n "$(printf '%s' "$INSTR" | tr -d '[:space:]')" ] || { echo "worker: usage: worker \"<instruction>\"" >&2; exit 2; }
cd "$WORK" 2>/dev/null || { echo "worker: workspace $WORK not found" >&2; exit 2; }
case "$PROVIDER" in
  anthropic)  KEY=ANTHROPIC_API_KEY ;;
  openai)     KEY=OPENAI_API_KEY ;;
  openrouter) KEY=OPENROUTER_API_KEY ;;
  *) echo "worker: unknown provider '$PROVIDER' in WORKER_MODEL='$SPEC'" >&2; exit 1 ;;
esac
[ -n "$(printenv "$KEY" || true)" ] || { echo "worker: $KEY not set in the worker container" >&2; exit 1; }

mkdir -p "$SESS_DIR"
[ -s "$STATE" ] || python3 -c 'import uuid; print(uuid.uuid4(), end="")' > "$STATE"
SID="$(cat "$STATE")"
# The brief goes in front of the first instruction that actually reaches the session — decided by
# whether the session exists, not by whether an id was ever chosen. If the first turn fails before
# the CLI writes anything, the next turn must still carry the brief, or the worker never gets it.
if ls "$SESS_DIR"/*"$SID"*.jsonl >/dev/null 2>&1; then
  PROMPT="$INSTR"
else
  BRIEF_FILE="${WORKER_BRIEF:-/out/worker_brief.txt}"
  PROMPT="$INSTR"; [ -f "$BRIEF_FILE" ] && PROMPT="$(cat "$BRIEF_FILE")

$INSTR"
fi
# The prompt reaches the CLI on STDIN, not as an argument. As an argument, an instruction beginning
# with "-" or "@" is parsed as an option or a file to attach (`worker "--help"` would run no model
# turn at all), and a long one overflows the kernel's per-argument limit. The CLI records stdin as
# the user message in the same form as an argument, except that it trims leading and trailing
# whitespace (an argument is passed through untouched), and treats whitespace-only input as no
# message — which worker_client.sh refuses before it gets here.
PROMPT_FILE="${WORKER_PROMPT_FILE:-/tmp/worker_prompt.txt}"
printf '%s' "$PROMPT" > "$PROMPT_FILE"

# The session file is CUMULATIVE: one continuous session for the whole episode, appended to on
# every turn. Remember where this turn starts, so the report below is taken from what THIS turn
# wrote. Reading the whole file instead returns the previous turn's report whenever this one ends
# without text — a turn killed at its cap would silently hand the manager last round's report as if
# it were new.
BEFORE=$(cat "$SESS_DIR"/*"$SID"*.jsonl 2>/dev/null | wc -l)

timeout --signal=TERM --kill-after=10 "$CAP" pi -p \
        --provider "$PROVIDER" --model "$MODEL" --thinking "$THINK" \
        --session-dir "$SESS_DIR" --session-id "$SID" \
        --no-extensions --no-skills --no-prompt-templates --no-themes --offline \
        <"$PROMPT_FILE" >>"${WORKER_STDOUT_LOG:-/dev/null}" 2>>"$ERRLOG"   # stdin is the prompt, then EOF
RC=$?   # 124/137: our own time cap. Anything else non-zero: the agent CLI itself failed.

# Per-turn trace: the Pi session JSONL (thinking + tool calls + results) for the transcript viewer.
cat "$SESS_DIR"/*"$SID"*.jsonl > "${WORKER_STREAM_OUT:-/out/last_worker_stream.jsonl}" 2>/dev/null || true

# The report is the LAST assistant text in the session — authoritative, and unaffected by whatever
# the CLI chose to print on stdout.
THIS_TURN() { cat "$SESS_DIR"/*"$SID"*.jsonl 2>/dev/null | tail -n +"$((BEFORE + 1))"; }
REPORT=$(THIS_TURN | python3 -c '
import json, sys
last = ""
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get("type") != "message":
        continue
    m = e.get("message") or {}
    if m.get("role") != "assistant":
        continue
    txt = "\n".join(b.get("text", "") for b in (m.get("content") or [])
                    if isinstance(b, dict) and b.get("type") == "text" and (b.get("text") or "").strip())
    if txt.strip():
        last = txt.strip()
print(last)
')
# TRUTHFUL DEGRADE. A turn killed at its time cap leaves assistant messages that are all thinking
# and tool calls with no text, so the report parses empty. Handing the manager a generic "could not
# parse" stub is worse than useless: a manager reads that as a dead channel and abandons delegation
# entirely. Report what actually happened plus the last action, so the manager can decide — wait,
# re-issue, take over — on true information. Still never a raw dump.
if [ -z "$REPORT" ]; then
  LAST_ACTION=$(THIS_TURN | python3 -c '
import json, sys
last = ""
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get("type") != "message":
        continue
    m = e.get("message") or {}
    if m.get("role") != "assistant":
        continue
    for b in (m.get("content") or []):
        if isinstance(b, dict) and b.get("type") == "toolCall":
            a = b.get("arguments") or {}
            last = (a.get("command") or a.get("path") or json.dumps(a))[:180]
print(last.replace("\n", " "))
')
  if [ "$RC" = "124" ] || [ "$RC" = "137" ]; then
    REPORT="(worker: this turn reached its ${CAP}s time cap while still working, so there is no finished report yet. Work so far is preserved and the next instruction resumes the same session. Last action in progress: ${LAST_ACTION:-unknown}.)"
  elif [ "$RC" = "0" ]; then
    # The agent simply finished its turn without writing anything. Say exactly that.
    REPORT="(worker: this turn ended without a written report. Work so far is preserved and the next instruction resumes the same session. Last action: ${LAST_ACTION:-none}.)"
  else
    # Not a timeout, so do not say it was one: the manager decides whether to wait, re-issue or take
    # over on the basis of this message, and those are different decisions.
    REPORT="(worker: this turn ended without a finished report — the worker process exited with status ${RC}. Work so far is preserved and the next instruction resumes the same session. Last action: ${LAST_ACTION:-none}.)"
  fi
fi
printf '%s\n' "$REPORT"
