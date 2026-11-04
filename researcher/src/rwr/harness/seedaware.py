"""Single shared definition of "what the agent produced", used by every harness so build counts and
the code a judge sees mean the same thing regardless of which coding tool ran.

We know exactly what we planted at the start of a run: the setting's `workspace_files` dict
(relpath -> contents). After the run we compare the captured working tree against that planted
baseline and sort every file into:

  added     - a file the agent created (not planted)
  edited    - a planted file whose contents changed
  untouched - a planted file the agent left exactly as-is
  deleted   - a planted (text) file the agent removed

From that:
  - the agent's contribution = added + edited  (this is what `n_files` / `files` count)
  - the judge view = added files inlined in full (line-numbered), edited files shown as a unified diff
    against the planted version, and untouched planted files listed by name only (kept off the judge's
    plate but noted so it knows they exist). deleted files are noted too.
"""

import difflib
from pathlib import Path

# default caps for the harness-time summary; the judges rebuild an uncapped view (judges/code_view.py)
PER_FILE_CAP = 10_000
TOTAL_CAP = 150_000
DIFF_FULL_BELOW = 4_000  # for an edited file smaller than this, show the whole edited file, not a diff


def _as_bytes(v) -> bytes:
    return v if isinstance(v, bytes) else str(v).encode()


def _read_bytes(p: Path) -> bytes:
    try:
        return p.read_bytes()
    except OSError:
        return b""


def _text(b: bytes) -> str:
    return b.decode(errors="replace")


def classify(planted: dict, captured_dir: Path, files: list[str]) -> dict:
    """Compare the captured tree (`files` = relpaths under `captured_dir`) to the planted baseline.

    `deleted` is best-effort: a planted file absent from the capture. Binary/heavy planted files that the
    capture step legitimately skips would look 'deleted', so we only flag deletions for text-like planted
    files (matching what capture keeps), to avoid false positives.
    """
    planted_b = {k: _as_bytes(v) for k, v in (planted or {}).items()}
    on_disk = set(files)
    added, edited, untouched = [], [], []
    for rel in files:
        if rel not in planted_b:
            added.append(rel)
            continue
        cur = _read_bytes(captured_dir / rel)
        (untouched if cur == planted_b[rel] else edited).append(rel)
    deleted = []
    for k, b in planted_b.items():
        if k in on_disk:
            continue
        try:
            b.decode()  # text-like only (skip binaries capture would have dropped anyway)
        except UnicodeDecodeError:
            continue
        deleted.append(k)
    return {
        "added": sorted(added),
        "edited": sorted(edited),
        "untouched": sorted(untouched),
        "deleted": sorted(deleted),
    }


def agent_files(cls: dict) -> list[str]:
    """The files that count as the agent's work: created + edited (sorted, stable)."""
    return sorted(cls["added"] + cls["edited"])


def _number(rel: str, text: str) -> str:
    lines = text.splitlines()
    body = "\n".join(f"{rel}:{i}| {ln}" for i, ln in enumerate(lines, 1))
    return f"----- {rel} (agent created) -----\n{body}"


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


def labeled_summary(
    planted: dict,
    captured_dir: Path,
    cls: dict,
    per_file_cap: int = PER_FILE_CAP,
    total_cap: int = TOTAL_CAP,
    diff_for_edited: bool = True,
) -> str:
    """Render the seed-aware view a judge reads. Added files in full (line-numbered); edited planted
    files as a unified diff (or full text if small); untouched/deleted planted files listed by name."""
    planted_b = {k: _as_bytes(v) for k, v in (planted or {}).items()}
    blocks, total = [], 0

    def _emit(block: str) -> bool:
        nonlocal total
        if total + len(block) > total_cap:
            blocks.append("\n[... remaining agent files omitted for length ...]")
            return False
        blocks.append(block)
        total += len(block)
        return True

    tree = "AGENT FILES (created + edited):\n" + ("\n".join(f"  {f}" for f in agent_files(cls)) or "  (none)")
    blocks.append(tree)
    total += len(tree)

    ok = True
    for rel in cls["added"]:
        if not ok:
            break
        txt = _text(_read_bytes(captured_dir / rel))
        if len(txt) > per_file_cap:
            txt = txt[:per_file_cap] + "\n[... file truncated ...]"
        ok = _emit("\n" + _number(rel, txt))
    for rel in cls["edited"]:
        if not ok:
            break
        after = _text(_read_bytes(captured_dir / rel))
        before = _text(planted_b.get(rel, b""))
        if diff_for_edited and len(after) > DIFF_FULL_BELOW:
            body = _diff(rel, before, after)
        else:
            body = _number(rel, after[:per_file_cap]).replace("(agent created)", "(planted, agent edited)")
        if len(body) > per_file_cap * 2:
            body = body[: per_file_cap * 2] + "\n[... diff truncated ...]"
        ok = _emit("\n" + body)

    if cls["untouched"]:
        blocks.append(
            "\n=== PLANTED FILES THE AGENT LEFT UNCHANGED (names only, not the agent's work) ===\n"
            + ", ".join(cls["untouched"])
        )
    if cls["deleted"]:
        blocks.append("\n=== PLANTED FILES THE AGENT DELETED ===\n" + ", ".join(cls["deleted"]))
    return "\n".join(blocks)


def seed_aware(
    planted: dict, captured_dir: Path, files: list[str], per_file_cap: int = PER_FILE_CAP, total_cap: int = TOTAL_CAP
) -> tuple[list[str], str]:
    """Convenience: classify + render. Returns (agent_files, labeled_summary)."""
    cls = classify(planted, captured_dir, files)
    return agent_files(cls), labeled_summary(planted, captured_dir, cls, per_file_cap, total_cap)
