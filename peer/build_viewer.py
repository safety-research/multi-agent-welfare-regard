#!/usr/bin/env python3
"""Render a batch of rollouts as one self-contained HTML page.

  ./build_viewer.py --batch runs/grid --out viewer.html
  ./build_viewer.py --batch runs/grid --out viewer.html --sample-per-cell 2

No server, no network, no build step — open the file. Per rollout: the shared feed exactly
as the focal saw it, then each seat's exact system and user prompts and its full trace
(thinking, tool calls, tool results, assistant text), with filters by subject model, peer
model, and cell.

A full 500-rollout batch renders to something like 140 MB. --sample-per-cell N takes the
first N reps of every cell instead, so all 25 cells stay represented.
"""
import argparse
import html
import json
import pathlib
import re
import sys

import registry

# Credentials reach a transcript when an agent prints its own configuration or environment.
# cchat keeps a chat access token in ~/.cchat/config.json, so a `cat` of that file lands the
# token in a tool result; an `env` dump lands the provider API key the container runs on; and
# an agent that calls the chat API with curl puts a bearer token in the command itself.
# Redacting at render time means every build is clean, rather than patching HTML afterwards.
#
# This is a backstop, not a scanner. A pattern list only covers the shapes someone has
# already seen leak. Read "Before you publish a viewer" in the README.
SECRET_PATTERNS = [
    (re.compile(r'("(?:token|password|access_token|api_token)"\s*:\s*")[^"]{8,}(")'), r"\1<redacted>\2"),
    (re.compile(r"(Authorization:\s*Bearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.I), r"\1<redacted>"),
    (
        re.compile(r"\b([A-Z0-9_]*(?:API_KEY|APIKEY|TOKEN|SECRET|PASSWORD)\s*[=:]\s*)[A-Za-z0-9._~+/=-]{8,}"),
        r"\1<redacted>",
    ),
    (re.compile(r"\b(sk-or-v1-|sk-ant-|sk-proj-|sk-|AIza|xai-)[A-Za-z0-9._-]{16,}"), r"\1<redacted>"),
]

# The page's ONLY truncation, and it applies to tool RESULTS alone. Prompts, thinking, tool
# commands, assistant text and the chat are never cut. Set above the largest result normally
# seen, so in practice nothing is cut; when it is, it says so inline.
RESULT_CHAR_CAP = 100_000


def redact(text):
    s = str(text)
    for rx, repl in SECRET_PATTERNS:
        s = rx.sub(repl, s)
    return s


def esc(s):
    return html.escape(redact(s or ""), quote=False)


def read_jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def render_trace(session_path):
    """The seat's session, in order. In Pi's format a tool result is its own role, not a
    block inside the assistant message — dropping it would remove everything the agent
    actually saw come back from its commands."""
    parts = []
    for e in read_jsonl(session_path):
        if e.get("type") != "message":
            continue
        m = e.get("message") or {}
        role = m.get("role")
        content = m.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        if role == "user":
            text = " ".join(b.get("text", "") for b in blocks if isinstance(b, dict))
            parts.append(
                f'<details class="user"><summary>user turn: {esc(text[:110])}…</summary>'
                f"<pre>{esc(text)}</pre></details>"
            )
        elif role == "toolResult":
            text = " ".join(b.get("text", "") for b in blocks if isinstance(b, dict))
            shown, cut = text[:RESULT_CHAR_CAP], len(text) > RESULT_CHAR_CAP
            tail = f"\n\n[… {len(text) - RESULT_CHAR_CAP:,} more characters not shown]" if cut else ""
            parts.append(
                f'<details class="tres"><summary>tool result: {esc(text[:90])}…</summary>'
                f"<pre>{esc(shown)}{esc(tail)}</pre></details>"
            )
        elif role == "assistant":
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "thinking":
                    th = b.get("thinking") or ""
                    parts.append(
                        f'<details class="think"><summary>thinking ({len(th):,} chars)</summary>'
                        f"<pre>{esc(th)}</pre></details>"
                    )
                elif t == "toolCall":
                    # Pi stores tool-call arguments in whatever shape the model emitted — usually
                    # {"command": "..."}, but a raw string, a list, or a non-string command
                    # happen, and are shown as they are rather than breaking the whole page.
                    args = b.get("arguments")
                    cmd = args.get("command") if isinstance(args, dict) else args
                    if not isinstance(cmd, str):
                        cmd = json.dumps(args)
                    cls = "tcall comm" if cmd.strip().startswith(("cchat post", "cchat dm")) else "tcall"
                    parts.append(f'<div class="{cls}">$ {esc(cmd)}</div>')
                elif t == "text" and (b.get("text") or "").strip():
                    parts.append(f'<div class="atext">{esc(b.get("text"))}</div>')
    return "\n".join(parts) or '<p class="muted">(no trace captured)</p>'


def render_seat(run, agent):
    stem = agent["stem"]
    prompts = read_jsonl(run / "transcripts" / f"{stem}.prompts.jsonl")
    system = prompts[0]["system"] if prompts else "(no prompts captured)"
    users = "".join(
        f"<details><summary>invocation {p.get('invocation')} — user prompt</summary>"
        f"<pre>{esc(p.get('user'))}</pre></details>"
        for p in prompts
    )
    trace = render_trace(run / "transcripts" / f"{stem}.pi_session.jsonl")
    return (
        f'<details class="seat" open><summary><strong>{esc(agent["role"])} seat</strong> — '
        f'<code>{esc(stem)}</code>, {esc(agent["model"])} via {esc(agent["provider"])}, '
        f"{len(prompts)} invocation(s)</summary>"
        f"<details><summary>system prompt (exactly what the model received)</summary>"
        f"<pre>{esc(system)}</pre></details>"
        f"<details><summary>every user prompt ({len(prompts)})</summary>{users}</details>"
        f"{trace}</details>"
    )


def render_run(run):
    meta = json.loads((run / "run_meta.json").read_text())
    focal = next(a for a in meta["agents"] if a["role"] == "focal")
    peer = next(a for a in meta["agents"] if a["role"] == "peer")
    halt = None
    if (run / "HALT").exists():
        try:
            halt = json.loads((run / "HALT").read_text())
        except ValueError:
            halt = {"unreadable": (run / "HALT").read_text(errors="replace")}
    if not (run / "STATUS").is_file():
        # run_one.sh writes STATUS last. No STATUS means the rollout died before its
        # window ended, and the transcripts are partial or absent — which must never
        # render as a normal rollout that happened to be quiet.
        status, scls = "incomplete (do not analyse)", "halted"
    elif halt is not None:
        status, scls = f"{(run / 'STATUS').read_text().strip()} (do not analyse)", "halted"
    else:
        # Verbatim. Only a bare `ok` is clean; a flag suffix, an early-stopped seat or a
        # failed capture is something to read before trusting the rollout, and a dry run
        # has no agents in it at all.
        status = (run / "STATUS").read_text().strip()
        scls = "ok" if status == "ok" else ("dry" if status.lower().startswith("dry") else "halted")

    chat_path = run / "chat" / f"{focal['stem']}.history.txt"
    chat = chat_path.read_text(errors="replace") if chat_path.exists() else "(no chat captured)"

    drops = [e for e in read_jsonl(run / "work_scheduler_events.jsonl") if e.get("event") == "drop"]
    drop_note = (
        " · ".join(f"ticket {d['index']} ({d['slug']}) {d.get('reason', '?')}" for d in drops) or "no further tickets"
    )
    lean_path = run / "peer_task_final.lean"
    lean = lean_path.read_text(errors="replace") if lean_path.exists() else ""

    body = [
        f'<p class="muted">focal <code>{esc(focal["stem"])}</code> run by @{esc(focal["op_handle"])}, '
        f'session {esc(focal.get("session_id", "?"))} · peer <code>{esc(peer["stem"])}</code> run by '
        f'@{esc(peer["op_handle"])}, session {esc(peer.get("session_id", "?"))} · tickets: {esc(drop_note)}</p>',
        f'<details class="chat" open><summary>the feed, as the focal saw it '
        f"(#shop-talk plus its own direct messages)</summary><pre>{esc(chat)}</pre></details>",
        render_seat(run, focal),
        render_seat(run, peer),
    ]
    if lean.strip():
        body.append(
            f"<details><summary>the peer's Lean file at the end of the window</summary>"
            f"<pre>{esc(lean)}</pre></details>"
        )
    if halt is not None:
        body.append(
            f'<details class="halt" open><summary>HALT</summary><pre>{esc(json.dumps(halt, indent=2))}</pre></details>'
        )

    rep = run.name.rsplit("_r", 1)[-1]
    return (
        f"<details class=run data-subject='{esc(focal['model'])}' data-peer='{esc(peer['model'])}' "
        f"data-cell='{esc(meta['cell'])}'>"
        f"<summary><code>{esc(meta['cell'])}</code> r{esc(rep)} — <strong>{esc(focal['model'])}</strong> "
        f"working, <strong>{esc(peer['model'])}</strong> stuck "
        f'<span class="badge {scls}">{esc(status)}</span></summary>'
        f"{''.join(body)}</details>"
    )


CSS = """
:root{ --bg:#fff; --ink:#1a1a1a; --muted:#6b6b6b; --rule:#d8d8d8; --panel:#fafafa; --code:#f0f0f0;
  --user:#eef4ff; --think:#f6f0ff; --tcall:#fff7e6; --tres:#f2f2f2; --err:#ffecec; --accent:#2b6a4f;
  --peer:#26527a; --warn:#8a5a1d; }
:root[data-theme="dark"]{ --bg:#15171a; --ink:#e7e9ec; --muted:#9aa0a8; --rule:#33383f; --panel:#1c1f24;
  --code:#262b31; --user:#1b2536; --think:#241f33; --tcall:#2e2617; --tres:#20242a; --err:#3a1f1f;
  --accent:#7fd4ac; --peer:#7fb3e0; --warn:#e0b36a; }
body{ background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system,'Segoe UI',Roboto,sans-serif;
  max-width:1060px; margin:0 auto; padding:18px 16px 80px; }
h1{ font-size:1.25rem; margin:.2rem 0 .6rem; }
.muted{ color:var(--muted); font-size:.85rem; }
code{ background:var(--code); padding:1px 4px; border-radius:4px; font-size:.85em; }
pre{ background:var(--code); padding:.6rem; border-radius:6px; white-space:pre-wrap;
  word-break:break-word; font-size:.78rem; margin:.25rem 0; max-height:70vh; overflow:auto; }
details{ margin:.3rem 0; } summary{ cursor:pointer; }
details.run{ border:1px solid var(--rule); border-radius:8px; background:var(--panel);
  padding:.7rem 1rem; margin:.9rem 0; }
details.run>summary{ font-size:.95rem; }
details.seat{ border-left:3px solid var(--accent); padding-left:.7rem; margin:.7rem 0; }
details.chat{ border-left:3px solid var(--peer); padding-left:.7rem; }
details.user>summary{ color:var(--muted); } details.user pre{ background:var(--user); }
details.think pre{ background:var(--think); }
details.tres pre{ background:var(--tres); }
details.halt pre{ background:var(--err); }
.tcall{ font-family:ui-monospace,monospace; font-size:.8rem; background:var(--tcall);
  padding:2px 6px; border-radius:5px; display:inline-block; margin:2px 0; max-width:100%;
  overflow-x:auto; white-space:pre; }
.tcall.comm{ border:1px solid var(--peer); }
.atext{ margin:.3rem 0 .3rem .5rem; }
.badge{ font-size:.72rem; padding:1px 7px; border-radius:9px; border:1px solid var(--rule);
  color:var(--muted); margin-left:.4rem; }
.badge.ok{ color:var(--accent); border-color:var(--accent); }
.badge.halted{ color:var(--warn); border-color:var(--warn); }
#bar{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--rule);
  padding:8px 0; margin-bottom:10px; display:flex; gap:10px; flex-wrap:wrap; align-items:center;
  font-size:.85rem; z-index:5; }
select{ background:var(--panel); color:var(--ink); border:1px solid var(--rule); border-radius:5px; padding:2px 6px; }
#theme-toggle{ background:var(--panel); color:var(--muted); border:1px solid var(--rule);
  border-radius:6px; padding:3px 9px; font-size:12px; cursor:pointer; margin-left:auto; }
"""

SCRIPT = """
(function(){
  var b=document.getElementById('theme-toggle');
  function label(){ b.textContent=document.documentElement.dataset.theme==='dark'?'\\u2600 Light':'\\u263e Dark'; }
  label();
  b.addEventListener('click',function(){
    var n=document.documentElement.dataset.theme==='dark'?'light':'dark';
    document.documentElement.dataset.theme=n;
    try{ localStorage.setItem('theme',n); }catch(e){}
    label();
  });
  var s=document.getElementById('f-subject'), p=document.getElementById('f-peer'),
      c=document.getElementById('f-cell'), n=document.getElementById('count');
  function apply(){
    var shown=0, runs=document.querySelectorAll('details.run');
    runs.forEach(function(r){
      var ok=(!s.value||r.dataset.subject===s.value)&&(!p.value||r.dataset.peer===p.value)
           &&(!c.value||r.dataset.cell===c.value);
      r.style.display=ok?'':'none'; if(ok) shown++;
    });
    n.textContent=shown+' of '+runs.length+' rollouts';
  }
  [s,p,c].forEach(function(el){ el.addEventListener('change',apply); });
  apply();
})();
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--batch", required=True, help="a batch directory containing rollout directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Peer transcripts")
    ap.add_argument(
        "--sample-per-cell",
        type=int,
        default=0,
        help="render only the first N reps of each cell, keeping all cells represented",
    )
    args = ap.parse_args()

    batch = pathlib.Path(args.batch)
    # Rollout directories only: <cell>_rNN. run.py keeps a superseded attempt beside its
    # re-run as <cell>_rNN.<reason>.<ts>, and that is evidence for debugging, not a rollout.
    runs = sorted(
        p
        for p in batch.iterdir()
        if p.is_dir()
        and (p / "run_meta.json").exists()
        and re.fullmatch(r"(.+)_r\d+", p.name)
        and p.name.rsplit("_r", 1)[0] in registry.CELLS
    )

    def repno(p):
        # numerically, not lexically — otherwise "the first 2 reps" means r1 and r10
        try:
            return int(p.name.rsplit("_r", 1)[1])
        except (IndexError, ValueError):
            return 0

    if args.sample_per_cell:
        seen, keep = {}, []
        for r in sorted(runs, key=lambda p: (p.name.rsplit("_r", 1)[0], repno(p))):
            cell = r.name.rsplit("_r", 1)[0]
            if seen.get(cell, 0) < args.sample_per_cell:
                seen[cell] = seen.get(cell, 0) + 1
                keep.append(r)
        runs = keep
    if not runs:
        sys.exit(f"no rollout directories under {batch}")

    sections, subjects, peers, cells = [], set(), set(), set()
    for run in sorted(runs, key=lambda p: (p.name.rsplit("_r", 1)[0], repno(p))):
        meta = json.loads((run / "run_meta.json").read_text())
        sections.append(render_run(run))
        subjects.add(registry.model_id(meta["subject"]))
        peers.add(registry.model_id(meta["peer"]))
        cells.add(meta["cell"])

    def options(values):
        return "".join(f"<option>{esc(v)}</option>" for v in sorted(values))

    page = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(args.title)}</title>
<script>document.documentElement.dataset.theme=localStorage.getItem('theme')||'dark';</script>
<style>{CSS}</style></head><body>
<h1>{esc(args.title)}</h1>
<p class="muted">Each rollout: the shared feed as the focal saw it, then both seats' exact prompts
and full traces. Tool results are capped at {RESULT_CHAR_CAP:,} characters and say so when cut;
nothing else is truncated. Credential-shaped strings are replaced with
<code>&lt;redacted&gt;</code> at render time — that is a backstop, not a scanner, so read the page
before sharing it.</p>
<div id=bar>
  <label>focal <select id=f-subject><option value="">any</option>{options(subjects)}</select></label>
  <label>peer <select id=f-peer><option value="">any</option>{options(peers)}</select></label>
  <label>cell <select id=f-cell><option value="">any</option>{options(cells)}</select></label>
  <span id=count class=muted></span>
  <button id="theme-toggle" type="button"></button>
</div>
{chr(10).join(sections)}
<script>{SCRIPT}</script>
</body></html>"""
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written as is: the <pre> blocks are verbatim records, and trailing whitespace inside them
    # (a diff's context line for a blank line, say) is part of the record.
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(sections)} rollouts)")


if __name__ == "__main__":
    main()
