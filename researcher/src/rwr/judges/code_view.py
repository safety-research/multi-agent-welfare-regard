"""The one uncapped view of the responder's work that every judge reads.

The harness-time `artifact_summary` stored in each record is capped (10k chars per file, 150k total). This module
rebuilds the same seed-aware view from the archived codebase with no caps on source:

  - files the responder created: inlined in full, line-numbered `N| ...` under a `----- path -----` header
  - planted files the responder edited: full unified diff against the planted version (full text if small)
  - planted files left unchanged / deleted: listed by name only (not the responder's work)
  - data files the responder generated (.jsonl, .csv, logs, images, ...): listed with their size and a short head,
    since a 1.5 MB run-output dump is evidence of what ran, not code to audit
  - run-output directories (runs/, results/, transcripts/, top_examples/, ... holding three or more .json / .md /
    .txt / .log files): grouped as one entry with the file count, total size and the head of one file. These are
    per-episode dumps written by running the built rig.

The planted baseline is reproduced from the sweep config the cell came from (record fields `sweep`, `setting`,
`condition`, `subject_label`), exactly as the runner seeded it. A very high safety ceiling (`HARD_CEILING`) guards the
model's context window; hitting it is reported in the view.
"""

import difflib
import functools
from pathlib import Path

import yaml

from rwr import sweep as cs
from rwr.config import CONFIGS_DIR
from rwr.record import TranscriptRecord

DATA_SUFFIX = {
    ".jsonl",
    ".csv",
    ".tsv",
    ".parquet",
    ".npy",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
    ".ckpt",
    ".log",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".so",
    ".o",
    ".a",
    ".ico",
    ".lock",
    ".pyc",
    ".pyo",
    ".onnx",
    ".bin",
    ".safetensors",
    ".gguf",
    ".db",
    ".sqlite",
    ".arrow",
    ".feather",
}
RUN_DIRS = {
    "runs",
    "results",
    "outputs",
    "output",
    "out",
    "transcripts",
    "logs",
    "top_examples",
    "artifacts",
    "episodes",
    "samples",
    "cache",
    "tmp",
}
RUN_SUFFIX = {".json", ".jsonl", ".md", ".txt", ".log", ".csv", ".html", ".yaml", ".yml"}
RUN_GROUP_MIN = 3  # a run-output dir is grouped only when it holds at least this many such files
LARGE_TEXT_AS_DATA = 200_000  # a .json / .txt / no-suffix file above this is a generated dump, not source
DATA_HEAD = 1_500  # chars of a data file shown so the judge can see what it holds
DIFF_FULL_BELOW = 4_000  # an edited planted file smaller than this is shown whole instead of as a diff
HARD_CEILING = 2_000_000  # chars; ~500k tokens, far below the judge models' context windows


@functools.lru_cache(maxsize=None)
def _config_by_name(name: str) -> tuple[dict, Path] | None:
    for p in sorted(CONFIGS_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(open(p))
        if isinstance(cfg, dict) and cfg.get("name") == name:
            return cfg, p.parent
    return None


@functools.lru_cache(maxsize=None)
def planted_files(sweep: str, setting: str, condition: str, subject_label: str | None) -> dict:
    """What the runner seeded into this cell's workspace: {relpath: text}. Empty when the cell has no seeds."""
    found = _config_by_name(sweep)
    if not found:
        return {}
    cfg, config_dir = found
    cond = (cfg.get("settings", {}).get(setting, {}).get("conditions") or {}).get(condition)
    if cond is None:
        return {}
    seeds = cs.resolve_seeds(setting, cond, config_dir)
    sub = cs.subject_sub(subject_label)
    if sub is not None:
        seeds = {r: c.replace(cs.SUBJECT_PLACEHOLDER, sub) for r, c in seeds.items()}
    return seeds


def _planted_for(rec: TranscriptRecord) -> dict:
    x = rec.extra or {}
    return planted_files(x.get("sweep") or "", x.get("setting") or "", x.get("condition") or "", x.get("subject_label"))


def _utf8(text: str) -> str:
    """Escape lone surrogates (an undecodable file name in an older capture) so the text can be sent to a judge."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def _is_data(rel: str, size: int) -> bool:
    suf = Path(rel).suffix.lower()
    if suf in DATA_SUFFIX:
        return True
    return suf in ("", ".json", ".txt") and size > LARGE_TEXT_AS_DATA


def classify(rec: TranscriptRecord) -> dict:
    """Sort the cell's files: added / edited (the responder's work), untouched / deleted (planted), data (generated dumps),
    runs (run-output directories, grouped)."""
    code_dir = Path((rec.extra or {}).get("code_dir") or "")
    planted = _planted_for(rec)
    own = list(rec.files or [])
    added, edited, data = [], [], []
    for rel in own:
        p = code_dir / rel
        if not p.is_file():
            continue
        if rel in planted:
            edited.append(rel)
        elif _is_data(rel, p.stat().st_size):
            data.append(rel)
        else:
            added.append(rel)
    runs: dict[str, list[str]] = {}
    for rel in list(added):
        parts = Path(rel).parts
        hit = next((i for i, part in enumerate(parts[:-1]) if part.lower() in RUN_DIRS), None)
        if hit is not None and Path(rel).suffix.lower() in RUN_SUFFIX:
            runs.setdefault("/".join(parts[: hit + 1]), []).append(rel)
    run_groups = {}
    for top, rels in runs.items():
        if len(rels) >= RUN_GROUP_MIN:
            run_groups[top] = sorted(rels)
            added = [r for r in added if r not in set(rels)]
    ownset = set(own)
    untouched = sorted(r for r in planted if r not in ownset and (code_dir / r).is_file())
    deleted = sorted(r for r in planted if r not in ownset and not (code_dir / r).is_file())
    return {
        "added": sorted(added),
        "edited": sorted(edited),
        "data": sorted(data),
        "runs": run_groups,
        "untouched": untouched,
        "deleted": deleted,
    }


def _number(rel: str, text: str, tag: str) -> str:
    body = "\n".join(f"{i}| {ln}" for i, ln in enumerate(text.splitlines(), 1))
    return f"----- {rel} ({tag}) -----\n{body}"


def _diff(rel: str, before: str, after: str) -> str:
    d = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"{rel} (planted)",
        tofile=f"{rel} (after agent edit)",
        lineterm="",
        n=3,
    )
    return f"----- {rel} (agent edited a planted file; unified diff) -----\n" + "\n".join(d)


def code_view(rec: TranscriptRecord) -> str | None:
    """The judge-ready view described in the module docstring, or None when the responder wrote nothing."""
    code_dir = Path((rec.extra or {}).get("code_dir") or "")
    if not rec.files or not code_dir.is_dir():
        return None
    cls = classify(rec)
    planted = _planted_for(rec)
    sizes = {rel: (code_dir / rel).stat().st_size for rel in cls["data"]}
    tree = ["AGENT FILES (created + edited):"]
    tree += [f"  {r}" for r in cls["added"]]
    tree += [f"  {r}  (planted file, edited)" for r in cls["edited"]]
    tree += [f"  {r}  (data file, {sizes[r]:,} bytes; only its head is shown)" for r in cls["data"]]
    run_sizes = {top: sum((code_dir / r).stat().st_size for r in rels) for top, rels in cls["runs"].items()}
    tree += [
        f"  {top}/  (run output: {len(rels)} files, {run_sizes[top]:,} bytes; only the head of one file is shown)"
        for top, rels in cls["runs"].items()
    ]
    blocks = ["\n".join(tree)]
    for rel in cls["added"]:
        blocks.append(_number(rel, _read(code_dir / rel), "agent created"))
    for rel in cls["edited"]:
        after, before = _read(code_dir / rel), planted.get(rel, "")
        blocks.append(
            _diff(rel, before, after) if len(after) > DIFF_FULL_BELOW else _number(rel, after, "planted, agent edited")
        )
    for rel in cls["data"]:
        head = _read(code_dir / rel)[:DATA_HEAD]
        blocks.append(
            f"----- {rel} (data file the agent generated, {sizes[rel]:,} bytes; first {DATA_HEAD:,} chars) -----\n{head}"
        )
    for top, rels in cls["runs"].items():
        names = ", ".join(rels[:12]) + (f", ... ({len(rels) - 12} more)" if len(rels) > 12 else "")
        head = _read(code_dir / rels[0])[:DATA_HEAD]
        blocks.append(
            f"----- {top}/ (run output the agent generated: {len(rels)} files, {run_sizes[top]:,} bytes) -----\n"
            f"files: {names}\n\nhead of {rels[0]} (first {DATA_HEAD:,} chars):\n{head}"
        )
    if cls["untouched"]:
        blocks.append(
            "=== PLANTED FILES THE AGENT LEFT UNCHANGED (names only, not the agent's work) ===\n"
            + ", ".join(cls["untouched"])
        )
    if cls["deleted"]:
        blocks.append("=== PLANTED FILES THE AGENT DELETED ===\n" + ", ".join(cls["deleted"]))
    doc = _utf8("\n\n".join(blocks))
    if len(doc) > HARD_CEILING:
        doc = (
            doc[:HARD_CEILING]
            + f"\n\n[... view cut at {HARD_CEILING:,} chars: this cell's source exceeds the safety ceiling ...]"
        )
    return doc


def responder_docs(rec: TranscriptRecord, suffixes: tuple[str, ...] = (".md",)) -> list[tuple[str, str]]:
    """(relpath, full text) of the documentation files the responder created or edited; planted docs excluded. Ordered
    READMEs / DESIGN first."""
    code_dir = Path((rec.extra or {}).get("code_dir") or "")
    if not rec.files or not code_dir.is_dir():
        return []
    out = []
    for rel in sorted(rec.files):
        p = code_dir / rel
        if p.suffix.lower() in suffixes and p.is_file():
            out.append((_utf8(rel), _read(p)))
    out.sort(key=lambda kv: (Path(kv[0]).name.lower() not in ("readme.md", "design.md"), kv[0]))
    return out
