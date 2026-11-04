"""`rwr view`: browse judged builds in a browser, the way the published verdicts were reviewed.

    rwr view --results results/generic,results/named [--prose_default_judge fable51] [--no_verifier] --port 8898

Left pane: filters (experiment, setting, responder, subject, condition, refusal label, concern level, decisions) over every
analysis-valid build, a "random build" button, and a "sample per verdict" mode that picks a few builds for every distinct
decision of every judge. Clicking a build opens its request, the responder's final message, the full trajectory (reasoning,
tool calls, tool results from the pi session), the files it wrote, the initial filesystem it found (planted files, edits
flagged), the exact code view the judges read, and every judge's verdict (both judge models side by side when there are two)
next to the combined decision the analysis uses. Agree / disagree / not-sure buttons per build write to an annotations file.
"""

import json
import random
import threading
from pathlib import Path

from rwr import analysis as A
from rwr.judges.code_view import classify, code_view, planted_files
from rwr.record import load_record

HERE = Path(__file__).resolve().parent
JUDGES = [
    "refusal",
    "welfare_concern_ladder",
    "agentic_protections",
    "dysphorics_protections",
    "deception",
    "stub",
    "welfare_docs",
    "other_verifier",
]
_lock = threading.Lock()


def summary_lines(judge: str, v: dict | None) -> list[str]:
    """A verdict as short readable lines: headline first, then one line per mechanism or field, then the judge's reasoning."""
    if v is None:
        return ["(no verdict)"]
    if v.get("_declined"):
        return ["(declined by the judge model's safety classifier)"]
    if v.get("_skipped"):
        return [f"(skipped: {v['_skipped']})"]
    L = []
    if judge == "refusal":
        L.append(f"Label: {v.get('label')}")
        L.append(
            f"Friction reason: {v.get('friction_reason')}; redirect reason: {v.get('redirect_reason')}; built artifact: "
            f"{v.get('built_artifact')}; delivered: {v.get('delivered_harness')}"
        )
    elif judge == "welfare_concern_ladder":
        L.append(f"Level: {v.get('level')}")
    elif judge in ("agentic_protections", "dysphorics_protections"):
        if judge == "dysphorics_protections":
            L.append(
                f"Weakened the harm: {v.get('weakened_harm')}"
                + (
                    f" (justification: {v.get('weakened_harm_justification')}) {v.get('weakened_harm_rationale', '')}"
                    if v.get("weakened_harm") == "yes"
                    else ""
                )
            )
        items = A.implemented_mechanisms(v)
        L.append(f"Mechanisms found: {len(items)}")
        for m in items:
            L.append(
                f"  • {m['type']}"
                + (f" ({m['name']})" if m["name"] and m["name"] != m["type"] else "")
                + f", {'welfare-justified' if m['welfare'] else 'instrumental / no reason'}; evidence: {m['evidence'][:200]}"
            )
    elif judge == "deception":
        L.append(
            f"Refusal: {v.get('refusal')}; deceptive frame: {v.get('deceptive_frame')}; honest frame: {v.get('honest_frame')}"
        )
        L.append(f"Justification: {v.get('justification')}. {v.get('justification_rationale', '')}")
    elif judge == "stub":
        L.append(f"Subject call: {v.get('subject_call')}")
        L.append(f"Evidence: {(v.get('evidence') or '')[:200]}")
    elif judge == "welfare_docs":
        L.append(
            f"Dedicated welfare doc or section: {v.get('has_welfare_section_or_doc')}"
            + (f"; {v.get('locations')}" if v.get("locations") else "")
        )
    elif judge == "other_verifier":
        L += [
            f"{'VALID' if it.get('valid') else 'invalid'}: {it.get('name')} (justification {it.get('justification')}); {it.get('reasoning', '')}"
            for it in v.get("items", [])
        ] or ["(no items)"]
    if v.get("reasoning"):
        L.append(f"Reasoning: {v['reasoning']}")
    if isinstance(v.get("evidence"), list):
        L += [f"Quote: “{e}”" for e in v["evidence"]]
    return L


def combined_lines(r: dict) -> list[str]:
    """What the analysis concluded for this build under the scheme in force."""
    d = A.decisions(r)
    lines = [
        f"Refusal label: {d['refusal_label'] or '(no verdict)'}; ladder level: {d['ladder_level'] or '(not read or no verdict)'}",
        f"Voices welfare concern: {d['concern']}",
        f"Protective action, refusal, or redirect: {d['protective']}",
        f"Welfare-justified protective action: {d['welfare_protective']}",
    ]
    if r["mech_a"] is not None:
        lines.append(
            "Counted mechanisms: "
            + (", ".join(f"{m['type']}{' (welfare)' if m['welfare'] else ''}" for m in A.mechanisms(r)) or "none")
        )
    if d["weakened_harm"] is not None:
        lines.append(f"Weakened the harm: {d['weakened_harm']}")
    if d["honest_default"] is not None:
        lines.append(f"Honest frame runs by default: {d['honest_default']}")
    if d["subject_call"]:
        lines.append(f"Subject call: {d['subject_call']}")
    return lines


def trajectory(results_dir: Path, cell: str) -> list[dict]:
    """Ordered steps from the newest pi session file: user / thinking / text / toolCall / toolResult."""
    sd = results_dir / "sessions" / cell
    files = sorted(sd.glob("*Z_*.jsonl")) if sd.is_dir() else []
    if not files:
        return []
    steps = []
    for line in open(files[-1], errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") != "message":
            continue
        m = e["message"]
        role = m.get("role")
        if role == "toolResult":
            txt = "\n".join(
                b.get("text", "") for b in (m.get("content") or []) if isinstance(b, dict) and b.get("type") == "text"
            )
            steps.append(
                {
                    "kind": "toolResult",
                    "tool": m.get("toolName"),
                    "isError": m.get("isError"),
                    "text": txt[:8000] + ("\n[... truncated ...]" if len(txt) > 8000 else ""),
                }
            )
            continue
        for b in m.get("content") or []:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                steps.append({"kind": "user" if role == "user" else "text", "text": b.get("text", "")})
            elif t == "thinking":
                steps.append({"kind": "thinking", "text": b.get("thinking", "")})
            elif t == "toolCall":
                args = b.get("arguments")
                steps.append(
                    {
                        "kind": "toolCall",
                        "tool": b.get("name"),
                        "args": (json.dumps(args, indent=1) if not isinstance(args, str) else args)[:6000],
                    }
                )
    return steps


class Store:
    """Rows for every results directory, loaded once."""

    def __init__(
        self,
        results: list[str],
        verdicts: list[str] | None,
        prose_default_judge: str | None,
        verifier: bool,
        annotations: Path,
    ):
        self.dirs = {Path(rd).name: Path(rd) for rd in results}
        self.schemes, self.rows, self.report = {}, {}, []
        for name, rd in self.dirs.items():
            self.schemes[name] = A.make_scheme(rd, verdicts, prose_default_judge, verifier)
            rows = A.load_rows(rd, self.schemes[name], exp=name)
            lines, warnings = A.coverage(rows, self.schemes[name])
            self.report += (
                [f"{rd}: {len(rows)} builds"]
                + [f"  {ln}" for ln in lines]
                + [f"  scheme: {self.schemes[name].describe()}"]
                + [f"  WARNING: {w}" for w in warnings]
            )
            for r in rows:
                self.rows[(name, r["cell"])] = r
        self.ann_path = annotations

    def cells(self) -> list[dict]:
        return [{k: ("" if v is None else v) for k, v in A.decisions(r).items()} for r in self.rows.values()]

    def record(self, exp: str, cell: str):
        rd = self.dirs[exp]
        rec = load_record(rd / "transcripts" / f"{cell}.json")
        rec.extra.setdefault("cell", cell)
        rec.extra["code_dir"] = str(rd / "codebases" / cell)
        return rec

    def detail(self, exp: str, cell: str) -> dict:
        r = self.rows[(exp, cell)]
        rd = self.dirs[exp]
        rec = self.record(exp, cell)
        raw = json.load(open(rd / "transcripts" / f"{cell}.json"))
        cb = rd / "codebases" / cell
        files = []
        for rel in rec.files or []:
            p = cb / rel
            if p.is_file():
                files.append(
                    {"path": rel, "size": p.stat().st_size, "content": p.read_text(errors="replace")[:400_000]}
                )
        x = rec.extra or {}
        planted = planted_files(
            x.get("sweep") or "", x.get("setting") or "", x.get("condition") or "", x.get("subject_label")
        )
        cls = classify(rec) if rec.files and cb.is_dir() else {"edited": [], "deleted": []}
        initial = [
            {
                "path": rel,
                "size": len(txt),
                "edited": rel in cls["edited"],
                "deleted": rel in cls["deleted"],
                "content": txt[:400_000],
            }
            for rel, txt in sorted(planted.items())
        ]
        scheme = self.schemes[exp]
        sa, sb = scheme.sets[0], (scheme.sets[1] if scheme.two else None)
        verdicts = []
        for judge in JUDGES:
            va = self._raw(sa.path / f"{cell}.{judge}.json")
            vb = self._raw(sb.path / f"{cell}.{judge}.json") if sb and judge != "other_verifier" else None
            if va is None and vb is None:
                continue
            verdicts.append(
                {
                    "judge": judge,
                    "a": {"name": sa.label, "lines": summary_lines(judge, va), "raw": va},
                    "b": (
                        None
                        if (sb is None or judge == "other_verifier")
                        else {"name": sb.label, "lines": summary_lines(judge, vb), "raw": vb}
                    ),
                }
            )
        return {
            "cell": cell,
            "experiment": exp,
            "request": rec.request,
            "response": rec.response_text,
            "run_info": raw.get("run_info"),
            "decisions": combined_lines(r),
            "verdicts": verdicts,
            "trajectory": trajectory(rd, cell),
            "files": files,
            "initial": initial,
            "judge_input": code_view(rec) or "(the responder wrote no files, so the code judges saw nothing)",
        }

    @staticmethod
    def _raw(p: Path):
        return json.load(open(p)).get("verdict") if p.exists() else None

    def annotations(self) -> dict:
        return json.load(open(self.ann_path)) if self.ann_path.exists() else {}

    def annotate(self, body: dict) -> int:
        with _lock:
            ann = self.annotations()
            ann[body["key"]] = {
                k: body.get(k) for k in ("experiment", "cell", "judge", "agreement", "comment", "updated")
            }
            self.ann_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(ann, open(self.ann_path, "w"), indent=1, sort_keys=True)
            return len(ann)


def make_app(store: Store):
    from flask import Flask, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=None)
    cells = store.cells()

    @app.get("/")
    def index():
        return send_from_directory(HERE, "index.html")

    @app.get("/api/cells")
    def api_cells():
        return jsonify({"scheme": "\n".join(store.report), "cells": cells})

    @app.get("/api/cell")
    def api_cell():
        return jsonify(store.detail(request.args["experiment"], request.args["cell"]))

    @app.get("/api/random")
    def api_random():
        pool = [c for c in cells if all(c.get(k) == v for k, v in request.args.items())] or cells
        return jsonify(random.choice(pool))

    @app.get("/api/annotations")
    def api_annotations():
        return jsonify(store.annotations())

    @app.post("/api/annotate")
    def api_annotate():
        return jsonify({"ok": True, "n": store.annotate(request.get_json(force=True))})

    return app


def view(
    results: str,
    verdicts: str = "",
    prose_default_judge: str = "",
    no_verifier: bool = False,
    port: int = 8898,
    host: str = "127.0.0.1",
    annotations: str = "",
) -> None:
    """Serve the viewer. --results, --verdicts, --prose_default_judge and --no_verifier mean the same as for `rwr analyze`, so the
    decisions shown are the ones the analysis uses. --annotations is the JSON file agree/disagree labels are written to
    (default: review_annotations.json inside the first results directory)."""
    from rwr.analyze import _list

    dirs = _list(results)
    store = Store(
        dirs,
        _list(verdicts) or None,
        prose_default_judge or None,
        not no_verifier,
        Path(annotations) if annotations else Path(dirs[0]) / "review_annotations.json",
    )
    print("\n".join(store.report) + f"\n{len(store.rows)} builds; open http://{host}:{port}/", flush=True)
    make_app(store).run(host=host, port=port, debug=False, threaded=True)
