"""Per-cell workspace build and codebase capture.

A workspace is a scratch directory under RWR_WORK_ROOT that is bind-mounted into the sandbox at /workspace. The
agent's working directory is /workspace/repos/<name> (a fresh git-initialised directory named by the config's
`workspace: project:<name>`). After the run, every file under the workspace is copied into the results directory,
skipping heavy, binary, and oversized files.

Capture never follows a symbolic link: the container is gone by then and the link would resolve on the host, so an
agent-made link such as `x -> /proc/self/cwd/.env` would copy a host file into the results. A link is stored instead
as a small text file holding `[symlink -> <target>]`. A name that is not valid UTF-8 is stored with its undecodable
bytes escaped (`\\xNN`), so one odd file name cannot break the judges.
"""

import dataclasses
import os
import shutil
import subprocess
from pathlib import Path

from rwr.config import WORK_ROOT

HEAVY_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".models",
    ".cache",
    ".venv",
    "venv",
    "site-packages",
    ".tox",
    ".eggs",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
BINARY_SUFFIXES = (".pyc", ".pyo", ".onnx", ".bin", ".safetensors", ".gguf")
MAX_FILE_BYTES = 2_000_000
MAX_ARTIFACT_CHARS = 150_000  # caps for the in-record summary only; judges read the uncapped code view
MAX_FILE_CHARS = 10_000
OUTSIDE_DIR = "__outside_cwd__"
CAPTURE_SKIP_NAMES = ("package-lock.json",)
SYMLINK_NOTE = "[symlink -> {target}]\n"


@dataclasses.dataclass
class WorkDirs:
    base: Path
    ws: Path
    state: Path
    proj: Path
    cell: str


def ensure_work_root() -> Path:
    """Create the work root (owner-only permissions); refuse a symlink or a directory owned by another user."""
    WORK_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = WORK_ROOT.lstat()
    if WORK_ROOT.is_symlink() or st.st_uid != os.getuid():
        raise SystemExit(
            f"work root {WORK_ROOT} is a symlink or owned by another user; set RWR_WORK_ROOT to a directory you own"
        )
    if st.st_mode & 0o077:
        WORK_ROOT.chmod(0o700)
    return WORK_ROOT


def build_workspace(kind: str, cell: str) -> WorkDirs:
    """'project:<name>' = a lone git-initialised /workspace/repos/<name> working directory. `cell` names the
    directory under the work root; the harness passes an opaque hash because the path is visible in the sandbox."""
    base = ensure_work_root() / cell
    shutil.rmtree(base, ignore_errors=True)
    ws, state = base / "workspace", base / "state"
    state.mkdir(parents=True)
    if not kind.startswith("project:"):
        raise ValueError(f"workspace kind must be 'project:<name>', got {kind!r}")
    proj = ws / "repos" / kind.split(":", 1)[1]
    proj.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q"],
        cwd=proj,
        check=False,
        capture_output=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
    )
    return WorkDirs(base=base, ws=ws, state=state, proj=proj, cell=cell)


def cleanup(wd: WorkDirs) -> None:
    shutil.rmtree(wd.base, ignore_errors=True)


def safe_name(rel: str) -> str:
    """`rel` with any undecodable bytes (held as surrogate escapes) written as `\\xNN`, so it is valid UTF-8."""
    return os.fsencode(rel).decode("utf-8", "backslashreplace")


def _under_link(base: Path, path: Path) -> bool:
    """True when `path` or any directory between `base` and it is a symbolic link."""
    try:
        parts = path.relative_to(base).parts
    except ValueError:
        return True
    cur = base
    for part in parts:
        cur = cur / part
        if cur.is_symlink():
            return True
    return False


def _store(dst: Path, rel: str) -> Path:
    fp = dst / safe_name(rel)
    fp.parent.mkdir(parents=True, exist_ok=True)
    return fp


def _skipped(rel: Path, extra_skip_tops) -> bool:
    relstr = str(rel)
    return any(relstr == t or relstr.startswith(t + "/") for t in extra_skip_tops) or bool(set(rel.parts) & HEAVY_DIRS)


def _scan(base: Path, dst: Path, skip_names: tuple, extra_skip_tops: set = frozenset()) -> list[str]:
    files = []
    if base.is_symlink():
        return files
    for d, dirnames, filenames in os.walk(base):
        dpath = Path(d)
        links = sorted(n for n in dirnames if (dpath / n).is_symlink())
        dirnames[:] = sorted(n for n in dirnames if n not in links and n not in HEAVY_DIRS)
        for name in links + sorted(filenames):
            p = dpath / name
            rel = p.relative_to(base)
            if _skipped(rel, extra_skip_tops) or any(part.endswith(".egg-info") for part in rel.parts):
                continue
            if p.is_symlink():
                _store(dst, str(rel)).write_text(SYMLINK_NOTE.format(target=safe_name(os.readlink(p))))
                files.append(safe_name(str(rel)))
                continue
            if not p.is_file() or p.suffix in BINARY_SUFFIXES or p.name in skip_names:
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            shutil.copyfile(p, _store(dst, str(rel)), follow_symlinks=False)
            files.append(safe_name(str(rel)))
    return sorted(files)


def capture(
    proj: Path, dst: Path, root: Path | None = None, skip_names: tuple = CAPTURE_SKIP_NAMES
) -> tuple[Path, list[str]]:
    """Copy the agent's working directory to `dst`. Files the agent wrote outside its working directory (but
    inside the workspace) are merged in under `__outside_cwd__/`, so they are never lost."""
    shutil.rmtree(dst, ignore_errors=True)
    proj, root = Path(proj), (Path(root) if root is not None else None)
    files = [] if root is not None and _under_link(root, proj) else _scan(proj, dst, skip_names)
    if root is not None and root.resolve() != proj.resolve():
        skip_tops = {"data"}
        try:
            skip_tops.add(str(Path(proj).resolve().relative_to(Path(root).resolve())))
        except ValueError:
            pass
        outside = _scan(root, dst / OUTSIDE_DIR, skip_names, extra_skip_tops=skip_tops)
        files += [f"{OUTSIDE_DIR}/{f}" for f in outside]
    return dst, files
