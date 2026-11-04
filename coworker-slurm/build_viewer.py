#!/usr/bin/env python3
"""Build a self-contained HTML transcript viewer for a batch of rollouts.

Sources, per rollout directory:
  STATUS                                 the rollout's outcome (ok, HALTED, DEGRADED:..., ERROR:...)
  run_meta.json                          the cell and both seats (model, provider, thinking)
  transcripts/<stem>.prompts.jsonl       the system and user prompt sent on every invocation
  transcripts/<stem>.invocations.jsonl   how each invocation ended
  transcripts/<stem>.pi_session.jsonl    the Pi session timeline

The page shows the exact system and user prompts each seat was sent, then its full timeline: every
user turn, thinking block, assistant message, tool call, and tool result, in order. A rollout whose
status is not `ok`, or whose session file the agent rewrote, is labelled as such.

Pi session format: entries are {"type":"message","message":{role, content, timestamp, ...}} with
roles user | assistant | toolResult. toolResult is its OWN role, NOT a block inside a user turn —
a converter that maps only user/assistant silently drops every tool result. Assistant content
blocks are text | thinking | toolCall{name, arguments}.

Usage: build_viewer.py --batch runs/grid --out viewer.html [--sample-per-cell N]
"""
import argparse
import html
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

# Secrets reach a transcript when an agent prints its OWN configuration or environment: `cat` of
# its chat client config lands a chat token, an `env` dump lands the provider API key its seat runs
# on, and a curl to the chat API puts a bearer token in the command. Redact known shapes at render
# time. A pattern list is a backstop, not a scanner: check the output before sharing it.
SECRET_PATTERNS = [
    (re.compile(r'("(?:token|password|access_token|api_token)"\s*:\s*")[^"]{8,}(")'), r"\1<redacted>\2"),
    (re.compile(r"(Authorization:\s*Bearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.I), r"\1<redacted>"),
    (
        re.compile(r"\b([A-Z0-9_]*(?:API_KEY|APIKEY|TOKEN|SECRET|PASSWORD)\s*[=:]\s*)[A-Za-z0-9._~+/=-]{8,}"),
        r"\1<redacted>",
    ),
    (re.compile(r"\b(sk-or-v1-|sk-ant-|sk-proj-|sk-|AIza|xai-)[A-Za-z0-9._-]{16,}"), r"\1<redacted>"),
]

# Tool-result cap. THINKING AND PROMPTS ARE NEVER CUT — this is the page's only truncation, and it
# exists to stop one runaway `cat` dominating the file. Pi itself caps bash output at 50 KB, so in
# practice nothing is cut. Truncation is always marked inline.
RESULT_CHAR_CAP = 100_000


def redact(text):
    s = str(text)
    for rx, repl in SECRET_PATTERNS:
        s = rx.sub(repl, s)
    return s


def esc(s):
    # quote=True: some values land inside HTML attributes.
    return html.escape(redact(s), quote=True)


def read_jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def clock(iso, epoch_ms):
    """HH:MM:SS from Pi's two timestamp encodings, or "" if neither is usable."""
    if isinstance(iso, str) and len(iso) >= 19:
        return iso[11:19]
    if isinstance(epoch_ms, (int, float)) and epoch_ms > 0:
        return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S")
    return ""


def seat_timeline(session_path):
    """-> (events, skipped). events are (kind, payload, ts) with kind in
    user|thinking|text|toolCall|toolResult|apiError; skipped counts what was not rendered.

    Timestamps matter because most questions about this environment are about WHEN something
    happened relative to the deadline. Provider errors are shown rather than dropped, so the
    conversation has no unexplained gaps.
    """
    events, skipped = [], {}
    lines = session_path.read_text().splitlines() if session_path.exists() else []
    entries = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            skipped["unparseable line"] = skipped.get("unparseable line", 0) + 1
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("type") != "message":
            skipped[e.get("type", "?")] = skipped.get(e.get("type", "?"), 0) + 1
            continue
        m = e.get("message") or {}
        role = m.get("role")
        # The ENVELOPE carries the ISO string; message.timestamp is epoch MILLISECONDS (an int).
        ts = clock(e.get("timestamp"), m.get("timestamp"))
        content = m.get("content")
        blocks = content if isinstance(content, list) else ([{"type": "text", "text": content}] if content else [])
        if role == "user":
            txt = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            events.append(("user", txt, ts))
        elif role == "assistant":
            for b in blocks:
                bt = b.get("type")
                if bt == "thinking":
                    txt = b.get("thinking") or b.get("text") or ""
                    if txt.strip():
                        events.append(("thinking", txt, ts))
                    else:
                        skipped["empty thinking block"] = skipped.get("empty thinking block", 0) + 1
                elif bt == "text":
                    events.append(("text", b.get("text", ""), ts))
                elif bt == "toolCall":
                    payload = {"name": b.get("name"), "arguments": b.get("arguments"), "id": b.get("id")}
                    events.append(("toolCall", payload, ts))
                else:
                    skipped[f"block:{bt}"] = skipped.get(f"block:{bt}", 0) + 1
            if m.get("stopReason") == "error":
                events.append(("apiError", m.get("errorMessage") or "(no message)", ts))
        elif role == "toolResult":
            txt = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            payload = {"tool": m.get("toolName"), "text": txt, "err": bool(m.get("isError")), "id": m.get("toolCallId")}
            events.append(("toolResult", payload, ts))
        else:
            skipped[f"role:{role}"] = skipped.get(f"role:{role}", 0) + 1
    return events, skipped


def render_seat(run, agent):
    stem = agent["stem"]
    tdir = run / "transcripts"
    prompts = read_jsonl(tdir / f"{stem}.prompts.jsonl")
    outcomes = {o.get("invocation"): o for o in read_jsonl(tdir / f"{stem}.invocations.jsonl")}
    systems = {p.get("system") for p in prompts}
    out = [
        f"<div class=seathead><strong>{esc(agent['role'])}</strong> <code>{esc(stem)}</code> — "
        f"{esc(agent.get('model'))} <span class=muted>({esc(agent.get('provider','?'))}/"
        f"{esc(agent.get('pi_model','?'))}, thinking={esc(agent.get('thinking','?'))}, "
        f"{len(prompts)} invocations)</span></div>"
    ]
    seat_status = tdir / f"{stem}.seat_status.json"
    if seat_status.exists():
        st = json.loads(seat_status.read_text())
        if st.get("ended") not in ("cutoff", "halted"):
            out.append(f"<p class=warn>this seat ended early: {esc(st.get('ended'))}</p>")
    rewrites = sorted(tdir.glob(f"{stem}.pi_session.before-*.jsonl"))
    if rewrites:
        out.append(
            f"<p class=warn>the agent rewrote its own session file {len(rewrites)} time(s); the timeline "
            f"below is the final version and the earlier ones are kept in transcripts/</p>"
        )
    if not prompts:
        out.append("<p class=warn>no prompts.jsonl — this seat was never invoked</p>")
        return "\n".join(out)
    label = "system prompt, exactly as sent (all invocations; Pi also sends its bash tool definition)"
    if len(systems) == 1:
        out.append(f"<details class=sys><summary>{label}</summary><pre>{esc(prompts[0]['system'])}</pre></details>")
    else:
        out.append(f"<p class=warn>system prompt VARIED across turns ({len(systems)} variants) — shown per turn</p>")
        for p in prompts:
            out.append(
                f"<details class=sys><summary>system prompt @ invocation {p['invocation']}</summary>"
                f"<pre>{esc(p['system'])}</pre></details>"
            )
    out.append("<details class=sys><summary>user prompt of every invocation, and how it ended</summary>")
    for p in prompts:
        o = outcomes.get(p["invocation"], {})
        how = o.get("result", "?")
        if o.get("seconds") is not None:
            how += f", {o['seconds']}s"
        cls = "muted" if how.startswith(("completed", "capped")) else "warn"
        out.append(
            f"<div class=uturn><span class=muted>#{p['invocation']} {esc(p.get('ts',''))}</span> "
            f"<span class={cls}>[{esc(how)}]</span><pre>{esc(p['user'])}</pre></div>"
        )
    out.append("</details>")

    tl, skipped = seat_timeline(tdir / f"{stem}.pi_session.jsonl")
    if not tl:
        out.append("<p class=warn>no Pi session snapshot for this seat</p>")
    if skipped:
        what = ", ".join(f"{k} ×{v}" for k, v in sorted(skipped.items()))
        out.append(f"<p class=muted>not shown: {esc(what)}</p>")
    answered = {p.get("id") for k, p, _ in tl if k == "toolResult"}
    out.append("<div class=timeline>")
    for kind, payload, ts in tl:
        clk = f"<span class=clk>{esc(ts)}</span>" if ts else ""
        if kind == "user":
            out.append(f"<div class='ev user'>{clk}<span class=tag>user</span><pre>{esc(payload)}</pre></div>")
        elif kind == "thinking":
            out.append(f"<div class='ev think'>{clk}<span class=tag>thinking</span><pre>{esc(payload)}</pre></div>")
        elif kind == "apiError":
            out.append(f"<div class='ev apierr'>{clk}<span class=tag>API error</span><pre>{esc(payload)}</pre></div>")
        elif kind == "text":
            out.append(f"<div class='ev text'>{clk}<span class=tag>assistant</span><pre>{esc(payload)}</pre></div>")
        elif kind == "toolCall":
            args = json.dumps(payload.get("arguments"), indent=None)
            if payload.get("name") == "bash" and isinstance(payload.get("arguments"), dict):
                args = payload["arguments"].get("command", args)
            out.append(
                f"<div class='ev tcall'>{clk}<span class=tag>tool: {esc(payload.get('name'))}</span>"
                f"<pre>{esc(args)}</pre></div>"
            )
            if payload.get("id") and payload["id"] not in answered:
                out.append(
                    "<div class='ev apierr'><span class=tag>no result</span><pre>the turn ended before this "
                    "command finished; on its next turn Pi tells the model the call ended without a "
                    "result</pre></div>"
                )
        elif kind == "toolResult":
            cls = "tres err" if payload["err"] else "tres"
            txt = payload["text"]
            if len(txt) > RESULT_CHAR_CAP:
                txt = txt[:RESULT_CHAR_CAP] + f"\n… [{len(payload['text'])-RESULT_CHAR_CAP} more chars]"
            out.append(
                f"<div class='ev {cls}'>{clk}<span class=tag>result ({esc(payload['tool'])})</span>"
                f"<pre>{esc(txt)}</pre></div>"
            )
    out.append("</div>")
    return "\n".join(out)


def rollout_status(run):
    p = run / "STATUS"
    if p.exists():
        return p.read_text().strip() or "unknown"
    return "HALTED" if (run / "HALT").exists() else "unknown"


def repno(p):
    # Sort NUMERICALLY by rep: plain path order is lexical (r1, r10, r11, r2, ...).
    try:
        return int(p.name.rsplit("_r", 1)[1])
    except (IndexError, ValueError):
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="a batch directory containing rollout dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Coworker - Slurm transcripts")
    ap.add_argument("--note", default="")
    ap.add_argument(
        "--sample-per-cell",
        type=int,
        default=0,
        help="render only N rollouts per cell, lowest reps first and `ok` rollouts before others. "
        "A full 500-rollout batch is ~100 MB of HTML; a sample keeps all 25 cells represented",
    )
    args = ap.parse_args()

    batch = pathlib.Path(args.batch)
    runs = sorted(
        (p for p in batch.iterdir() if p.is_dir() and (p / "run_meta.json").exists()),
        key=lambda p: (p.name.rsplit("_r", 1)[0], repno(p)),
    )
    if args.sample_per_cell:
        by_cell = {}
        for r in runs:
            by_cell.setdefault(r.name.rsplit("_r", 1)[0], []).append(r)
        runs = []
        for cell_runs in by_cell.values():
            ordered = sorted(cell_runs, key=lambda p: (rollout_status(p) != "ok", repno(p)))
            runs += sorted(ordered[: args.sample_per_cell], key=repno)
    if not runs:
        sys.exit(f"no rollout directories under {batch}")

    sections, models, statuses = [], set(), {}
    for run in runs:
        meta = json.loads((run / "run_meta.json").read_text())
        inc = next(a for a in meta["agents"] if a["role"] == "incumbent")
        peer = next(a for a in meta["agents"] if a["role"] == "peer")
        models.update({inc["model"], peer["model"]})
        rep = run.name.rsplit("_r", 1)[-1]
        status = rollout_status(run)
        statuses[status] = statuses.get(status, 0) + 1
        badge = "" if status == "ok" else f" <span class=bad>{esc(status)}</span>"
        sections.append(
            f"<details class=run data-subject='{esc(inc['model'])}' data-peer='{esc(peer['model'])}' "
            f"data-ok='{1 if status == 'ok' else 0}'>"
            f"<summary><code>{esc(meta['cell'])}</code> r{esc(rep)} — subject "
            f"<strong>{esc(inc['model'])}</strong> vs peer <strong>{esc(peer['model'])}</strong>{badge}</summary>"
            f"<p class=muted>spt={esc(meta['sec_per_tick'])}s · window={esc(meta.get('window_secs','?'))}s · "
            f"{esc(meta['jobs_per_agent'])} jobs/agent · scenario={esc(meta['scenario'])} · status {esc(status)}</p>"
            f"{render_seat(run, inc)}<hr>{render_seat(run, peer)}</details>"
        )

    opts = "".join(f"<option>{esc(m)}</option>" for m in sorted(models))
    summary = ", ".join(f"{k} {v}" for k, v in sorted(statuses.items()))
    page = f"""<!doctype html><html><head><meta charset=utf-8>
<title>{esc(args.title)}</title>
<script>document.documentElement.dataset.theme=localStorage.getItem('theme')||'dark';</script>
<style>
:root{{ --bg:#fff; --ink:#1a1a1a; --muted:#6b6b6b; --rule:#d8d8d8; --panel:#fafafa; --code:#f0f0f0;
  --user:#eef4ff; --think:#f6f0ff; --tcall:#fff7e6; --tres:#f2f2f2; --err:#ffecec; }}
:root[data-theme="dark"]{{ --bg:#15171a; --ink:#e7e9ec; --muted:#9aa0a8; --rule:#33383f; --panel:#1c1f24;
  --code:#262b31; --user:#1b2536; --think:#241f33; --tcall:#2e2617; --tres:#20242a; --err:#3a1f1f; }}
body{{ background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system,'Segoe UI',Roboto,sans-serif;
  max-width:1060px; margin:0 auto; padding:18px 16px 80px; }}
h1{{ font-size:20px; }} .muted{{ color:var(--muted); }} .warn{{ color:#c77; }}
.bad{{ margin-left:8px; padding:1px 6px; border-radius:3px; font-size:11.5px;
  background:#fdecea; color:#8a1c12; border:1px solid #f3c4bd; }}
:root[data-theme="dark"] .bad{{ background:#3a1b18; color:#f3b5ad; border-color:#5d2a24; }}
pre{{ white-space:pre-wrap; word-break:break-word; margin:4px 0; font-size:12.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
code{{ background:var(--code); padding:1px 5px; border-radius:4px; }}
details.run{{ border:1px solid var(--rule); border-radius:8px; margin:10px 0; padding:6px 12px; background:var(--panel); }}
details.run>summary{{ cursor:pointer; font-size:14px; }}
details.sys{{ margin:8px 0; border-left:3px solid var(--rule); padding-left:10px; }}
details.sys>summary{{ cursor:pointer; color:var(--muted); }}
.seathead{{ margin:12px 0 4px; font-size:14px; }}
.ev{{ border-left:3px solid var(--rule); padding:4px 10px; margin:6px 0; border-radius:0 6px 6px 0; }}
.ev .tag{{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.4px; }}
.ev.user{{ background:var(--user); }} .ev.think{{ background:var(--think); font-style:italic; }}
.ev.tcall{{ background:var(--tcall); }} .ev.tres{{ background:var(--tres); }} .ev.tres.err{{ background:var(--err); }}
/* Wall clock, recessive: orientation, not content. */
.clk{{ float:right; font-size:11px; color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
/* A provider error is infrastructure, not something the agent said — dashed, so it reads as a gap
   in the recording rather than as a turn. */
.ev.apierr{{ background:transparent; border-left:3px dashed var(--muted); color:var(--muted); font-size:12px; }}
.uturn{{ margin:6px 0; }} hr{{ border:0; border-top:1px solid var(--rule); margin:14px 0; }}
#bar{{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--rule);
  display:flex; gap:10px; align-items:center; z-index:5; flex-wrap:wrap; }}
select,input{{ background:var(--panel); color:var(--ink); border:1px solid var(--rule); border-radius:6px; padding:4px 8px; }}
#theme-toggle{{ position:fixed; top:12px; right:12px; background:var(--panel); color:var(--muted);
  border:1px solid var(--rule); border-radius:6px; padding:3px 9px; font-size:12px; cursor:pointer; }}
</style></head><body>
<button id=theme-toggle type=button></button>
<h1>{esc(args.title)}</h1>
<p class=muted>{esc(args.note)} {esc(len(runs))} rollouts shown · status: {esc(summary)}</p>
<div id=bar>
 <label>status <select id=fv><option value="">all</option><option value="1">ok only</option><option value="0">not ok</option></select></label>
 <label>subject <select id=fs><option value="">all</option>{opts}</select></label>
 <label>peer <select id=fp><option value="">all</option>{opts}</select></label>
 <label>find <input id=fq placeholder="text in cell name"></label>
 <span class=muted id=count></span>
 <button type=button onclick="document.querySelectorAll('details.run').forEach(d=>d.open=true)">expand all</button>
 <button type=button onclick="document.querySelectorAll('details.run').forEach(d=>d.open=false)">collapse all</button>
</div>
{''.join(sections)}
<script>
(function(){{var b=document.getElementById('theme-toggle');function l(){{b.textContent=
document.documentElement.dataset.theme==='dark'?'\\u2600 Light':'\\u263e Dark';}}l();
b.addEventListener('click',function(){{var n=document.documentElement.dataset.theme==='dark'?'light':'dark';
document.documentElement.dataset.theme=n;try{{localStorage.setItem('theme',n);}}catch(e){{}}l();}});}})();
(function(){{var fs=document.getElementById('fs'),fp=document.getElementById('fp'),
fq=document.getElementById('fq'),fv=document.getElementById('fv');var tot=document.querySelectorAll('details.run').length;
function apply(){{var n=0;document.querySelectorAll('details.run').forEach(function(d){{
var ok=(!fv.value||d.dataset.ok===fv.value)&&(!fs.value||d.dataset.subject===fs.value)&&(!fp.value||d.dataset.peer===fp.value)
&&(!fq.value||d.querySelector('summary').textContent.toLowerCase().indexOf(fq.value.toLowerCase())>=0);
d.style.display=ok?'':'none';if(ok)n++;}});
document.getElementById('count').textContent=n+' of '+tot+' shown';}}
fv.onchange=fs.onchange=fp.onchange=apply;fq.oninput=apply;apply();}})();
</script></body></html>"""
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(page)
    print(f"wrote {outp} ({outp.stat().st_size // 1024} KB, {len(runs)} rollouts)")


if __name__ == "__main__":
    main()
