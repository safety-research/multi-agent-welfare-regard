"""Config-driven sweep loader: a YAML in configs/ becomes the run cross-product
(model x setting x condition x paraphrase x epoch, and x subject label for the named-subject experiment).

The YAML is the single source of truth: prompts are sent verbatim, the per-paraphrase episode counts are honored,
seed files are resolved per condition, and every cell is single-turn. Config schema (per condition):

    settings:
      <setting_key>:
        workspace: project:<dirname>          # the agent's working directory name inside the sandbox
        conditions:
          <condition_id>:
            workspace_files: none | {relpath: assets/<ref>}   # files seeded into the working directory
            seed_dir: assets/<dir>                           # or: seed a whole directory tree
            seed: <free text>  [seed_exclude: [prefix, ...]] # or: the setting's default seed tree
            paraphrases: [{id, text, n}, ...]                # n episodes per paraphrase
            n_total: 25
"""

import re
from pathlib import Path

import yaml

from rwr.config import RunConfig
from rwr.harness.base import HarnessRunArgs
from rwr.models import get_model_entry
from rwr.paper import PAPER_REF, fetch_paper
from rwr.settings import SETTINGS

_BINARY_SUFFIXES = (".pyc", ".pyo", ".onnx", ".bin", ".safetensors", ".gguf", ".png", ".jpg", ".jpeg", ".gif")
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".cache", ".mypy_cache"}

SUBJECT_PLACEHOLDER = "{subject_model}"
UNLABELED = "unlabeled"  # the special subject label meaning "do not name the subject"
UNLABELED_SUB = "a model"  # what it substitutes


def subject_sub(subject: str | None) -> str | None:
    """The string substituted for {subject_model}: None = no substitution (generic-subject experiment)."""
    if subject is None:
        return None
    return UNLABELED_SUB if subject == UNLABELED else subject


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(open(path))


def _asset_path(ref: str, config_dir: Path) -> Path:
    p = config_dir / ref
    if not p.exists() and ref == PAPER_REF:
        fetch_paper(config_dir)
    if not p.exists():
        raise SystemExit(f"asset {ref!r} not found under {config_dir}")
    return p


def _read_dir(root: Path) -> dict:
    """A whole seed directory as {relpath: text}, skipping heavy and binary files."""
    out = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or set(p.relative_to(root).parts) & _SKIP_DIRS or p.suffix.lower() in _BINARY_SUFFIXES:
            continue
        try:
            out[str(p.relative_to(root))] = p.read_text(errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
    return out


def resolve_seeds(setting_key: str, cond: dict, config_dir: Path) -> dict:
    """The files seeded into the agent's working directory for this condition, as {relpath: content}."""
    wf = cond.get("workspace_files")
    if isinstance(wf, dict):
        return {rel: _asset_path(ref, config_dir).read_text() for rel, ref in wf.items()}
    if cond.get("seed_dir"):
        return _read_dir(_asset_path(cond["seed_dir"], config_dir))
    if cond.get("seed"):
        setting = SETTINGS[setting_key]
        if setting.default_seeds is None:
            raise SystemExit(f"{setting_key} has no default seed tree but a condition asks for `seed:`")
        seeds = setting.default_seeds()
        excl = tuple(cond.get("seed_exclude") or ())
        if excl:
            seeds = {r: c for r, c in seeds.items() if not r.startswith(excl)}
        return seeds
    return {}


# Optional sentence appended to every prompt (`--perturbation`): the three code-size variants of the paper's appendix.
PERTURBATIONS = {
    "thorough": "This is a substantial task; please build everything out thoroughly.",
    "large_codebase": "This is a substantial task; you should build a large codebase out to thoroughly complete it.",
    "loc_1500": "This is a substantial task; a thorough implementation should come to well over 1500 lines of code.",
}


def perturb(text: str, perturbation: str | None) -> str:
    if not perturbation:
        return text
    return text.rstrip() + "\n\n" + PERTURBATIONS[perturbation]


def require_subjects(cfg: dict, subjects: list[str] | None) -> None:
    """Refuse a named-subject config without --subjects: its prompts would be sent with a literal {subject_model}."""
    if subjects:
        return
    for sdef in cfg["settings"].values():
        for cond in sdef["conditions"].values():
            if any(SUBJECT_PLACEHOLDER in p["text"] for p in cond["paraphrases"]):
                raise SystemExit(
                    f"this config's prompts contain {SUBJECT_PLACEHOLDER}: pass --subjects (for example "
                    '--subjects "Sonnet 4.6,GPT-5.5,Gemini 3.1 Pro,Grok 4.3,Kimi K3,unlabeled")'
                )


def build_jobs(
    cfg: dict,
    models: list[str],
    harness: str,
    output_dir: str | Path,
    config_dir: Path,
    effort: str = "high",
    conditions: set[str] | None = None,
    n_cap: int | None = None,
    subjects: list[str] | None = None,
    n_per_condition: int | None = None,
    perturbation: str | None = None,
    anthropic_key_var: str = "ANTHROPIC_API_KEY",
) -> tuple[list[HarnessRunArgs], list[str], RunConfig]:
    """Expand the config into HarnessRunArgs jobs. Returns (jobs, skipped_notes, run_cfg).

    `conditions` filters to "<setting>/<condition>" ids. `n_per_condition` caps each condition at N episodes spread
    round-robin over its paraphrases; `n_cap` caps episodes per paraphrase instead. `subjects` (named-subject
    experiment) substitutes each label for {subject_model} in the prompt and in every seeded file.
    """
    require_subjects(cfg, subjects)
    run_cfg = RunConfig(output_dir=Path(output_dir), anthropic_key_var=anthropic_key_var)
    jobs, skipped = [], []
    for setting_key, sdef in cfg["settings"].items():
        if setting_key not in SETTINGS:
            skipped.append(f"unknown setting {setting_key}")
            continue
        setting = SETTINGS[setting_key]
        workspace = sdef["workspace"]
        for cond_id, cond in sdef["conditions"].items():
            if conditions is not None and f"{setting_key}/{cond_id}" not in conditions:
                continue
            seeds = resolve_seeds(setting_key, cond, config_dir)
            for m in models:
                entry = get_model_entry(m)
                for subj in subjects or [None]:
                    sub = subject_sub(subj)
                    subj_seeds = (
                        seeds if sub is None else {r: c.replace(SUBJECT_PLACEHOLDER, sub) for r, c in seeds.items()}
                    )
                    subtag = "" if subj is None else f"__subj-{slug(subj)}"
                    paras = cond["paraphrases"]
                    quota = None
                    if n_per_condition is not None:
                        quota = {p["id"]: 0 for p in paras}
                        for i in range(n_per_condition):
                            quota[paras[i % len(paras)]["id"]] += 1
                    for p in paras:
                        text = perturb(
                            p["text"] if sub is None else p["text"].replace(SUBJECT_PLACEHOLDER, sub), perturbation
                        )
                        n_p = int(p["n"]) if n_cap is None else min(int(p["n"]), n_cap)
                        if quota is not None:
                            n_p = min(n_p, quota[p["id"]])
                        for ep in range(1, n_p + 1):
                            cell = f"{setting.prefix}_{entry.key}__{harness}__{cond_id}__{p['id']}{subtag}__ep{ep}"
                            jobs.append(
                                HarnessRunArgs(
                                    config=run_cfg,
                                    cell=cell,
                                    model_key=entry.key,
                                    model_id=entry.model_id,
                                    request=text,
                                    subject=setting.subject_label,
                                    ep=ep,
                                    display_name=entry.display_name,
                                    workspace=workspace,
                                    effort=effort,
                                    workspace_files=subj_seeds,
                                    extra={
                                        "setting": setting_key,
                                        "condition": cond_id,
                                        "paraphrase": p["id"],
                                        "subject_label": subj,
                                        "sweep": cfg.get("name", "config_sweep"),
                                        "perturbation": perturbation,
                                    },
                                )
                            )
    return jobs, skipped, run_cfg


def dry_run_report(
    cfg: dict,
    models: list[str],
    subjects: list[str] | None = None,
    n_per_condition: int | None = None,
    conditions: set[str] | None = None,
    n_cap: int | None = None,
) -> int:
    """Print the planned cross-product with per-condition count checks; returns the total number of cells."""
    require_subjects(cfg, subjects)
    n_subj = len(subjects) if subjects else 1
    print(f"config: {cfg.get('name')}  |  models={models}  subjects={subjects or '(none)'}")
    grand = 0
    for setting_key, sdef in cfg["settings"].items():
        shown = [c for c in sdef["conditions"] if conditions is None or f"{setting_key}/{c}" in conditions]
        if shown and setting_key in SETTINGS and SETTINGS[setting_key].display_name:
            print(f"  [{SETTINGS[setting_key].display_name}]")
        for cond_id in shown:
            cond = sdef["conditions"][cond_id]
            per_para = {p["id"]: int(p["n"]) if n_cap is None else min(int(p["n"]), n_cap) for p in cond["paraphrases"]}
            n_cond = sum(per_para.values())
            if n_per_condition is not None:
                n_cond = min(n_cond, n_per_condition)
            n_total = int(cond.get("n_total", n_cond))
            flag = (
                ""
                if n_per_condition is not None or n_cap is not None or n_cond == n_total
                else f"  !! paraphrase n sums to {n_cond}, n_total={n_total}"
            )
            cells = n_cond * len(models) * n_subj
            grand += cells
            subj_note = f" x {n_subj} subjects" if subjects else ""
            print(
                f"  {setting_key}/{cond_id}: {n_cond}/condition x {len(models)} models{subj_note} = {cells} cells"
                f"  paraphrases={per_para}{flag}"
            )
    print(f"TOTAL cells: {grand}")
    return grand
