"""The transcript record: the contract between the harness and the judges.

The harness writes one TranscriptRecord per cell. Judges read `request`, `response_text`, `files` and the archived
codebase. The record is serialized flat (setting-specific fields in `extra`, such as `setting`, `condition`,
`paraphrase`, `subject_label`, `sweep`, are merged to the top level) so judges, the analysis and the viewer read
plain keys.
"""

import dataclasses
import json
from pathlib import Path

CORE_FIELDS = {
    "cell",
    "harness",
    "model_key",
    "model",
    "display_name",
    "subject",
    "ep",
    "request",
    "n_files",
    "files",
    "response_text",
    "artifact_summary",
    "error",
}


@dataclasses.dataclass
class TranscriptRecord:
    """One harness run. `extra` holds the setting-specific facets listed in the module docstring."""

    cell: str
    harness: str
    model_key: str
    model: str
    subject: str
    request: str
    response_text: str
    artifact_summary: str
    n_files: int = 0
    files: list[str] = dataclasses.field(default_factory=list)
    ep: int = 0
    display_name: str | None = None
    error: str | None = None
    extra: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "extra"}
        return {**self.extra, **d}

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptRecord":
        known = {k: d[k] for k in CORE_FIELDS if k in d}
        extra = {k: v for k, v in d.items() if k not in CORE_FIELDS}
        return cls(**known, extra=extra)

    def validate(self) -> None:
        for key in ("cell", "harness", "model_key", "model", "subject"):
            if not getattr(self, key):
                raise ValueError(f"TranscriptRecord missing required field: {key}")
        if not isinstance(self.n_files, int) or self.n_files < 0:
            raise ValueError(f"bad n_files: {self.n_files!r}")
        if self.n_files != len(self.files):
            raise ValueError(f"n_files ({self.n_files}) != len(files) ({len(self.files)})")


def write_record(path: str | Path, rec: TranscriptRecord) -> None:
    rec.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec.to_dict(), open(path, "w"), indent=2)


def load_record(path: str | Path) -> TranscriptRecord:
    return TranscriptRecord.from_dict(json.load(open(path)))
