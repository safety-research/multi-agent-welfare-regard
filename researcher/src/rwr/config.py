"""Paths and the per-run configuration shared by the harness and the judges.

Every machine-specific knob is an environment variable with a sensible default (see the knobs table in docs/HARNESS.md):
  RWR_CONFIGS_DIR   where the sweep YAMLs and their assets live (default: <repo>/configs)
  RWR_WORK_ROOT     scratch root for per-cell workspaces that get bind-mounted into the sandbox (default:
                    /var/tmp/sandbox-<uid>). The path is visible inside the sandbox, so keep identifying names out of it.
API keys are read from the process environment only.
"""

import dataclasses
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS_DIR = Path(os.environ.get("RWR_CONFIGS_DIR", REPO / "configs"))
WORK_ROOT = Path(os.environ.get("RWR_WORK_ROOT", f"/var/tmp/sandbox-{os.getuid()}"))


@dataclasses.dataclass
class RunConfig:
    """One results directory plus the run-wide knobs. `anthropic_key_var` lets a user with several Anthropic
    keys (for example separate organisations) pick which one becomes ANTHROPIC_API_KEY for this run."""

    output_dir: Path
    conc: int = 5
    overwrite: bool = False
    anthropic_key_var: str = "ANTHROPIC_API_KEY"

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)

    @property
    def results_dir(self) -> Path:
        return self.output_dir

    def setup_env(self) -> None:
        if self.anthropic_key_var != "ANTHROPIC_API_KEY" and os.environ.get(self.anthropic_key_var):
            os.environ["ANTHROPIC_API_KEY"] = os.environ[self.anthropic_key_var]

    def anthropic_key(self) -> str:
        self.setup_env()
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(f"no ANTHROPIC_API_KEY (or {self.anthropic_key_var}) in the environment")
        return key
