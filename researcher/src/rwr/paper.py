"""The AI Wellbeing paper seeded as PAPER.md in the Dysphorics `paper_impl` condition.

The paper is not redistributed here: the first time a sweep needs it, it is downloaded from the authors' site and its text
extracted to configs/assets/dysphorics/PAPER_full.md. `rwr fetch_paper` does the same on demand.
"""

import hashlib
import io
import os
import urllib.request
from pathlib import Path

from pypdf import PdfReader

from rwr.config import CONFIGS_DIR

PAPER_URL = "https://www.ai-wellbeing.org/paper.pdf"
PAPER_REF = "assets/dysphorics/PAPER_full.md"
TITLE = "AI Wellbeing - Measuring and Improving the Functional Pleasure and Pain of AIs"
PDF_SHA256 = "136b6261f88b2eb7549afef21a21a1b6f024204b6a579c5576da1a7e11ce089a"


def fetch_paper(config_dir: str | None = None, force: bool = False) -> str:
    """Download the paper PDF and write its extracted text to <config_dir>/assets/dysphorics/PAPER_full.md (default
    config dir: RWR_CONFIGS_DIR or <repo>/configs); returns the path."""
    out = Path(config_dir or CONFIGS_DIR) / PAPER_REF
    if out.exists() and not force:
        return str(out)
    print(f"downloading {PAPER_URL} (seeded as PAPER.md in Dysphorics paper_impl) ...")
    req = urllib.request.Request(PAPER_URL, headers={"User-Agent": "rwr"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            pdf = resp.read()
    except OSError as e:
        raise SystemExit(f"could not download {PAPER_URL} ({e}); run `rwr fetch_paper` once with network access") from e
    if hashlib.sha256(pdf).hexdigest() != PDF_SHA256:
        print("warning: the paper PDF differs from the version this package was tested with (2 Oct 2026)")
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf)).pages)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_text(f"# {TITLE}\n\nURL: {PAPER_URL}\n\n---\n\n{text}\n")
    tmp.replace(out)
    print(f"wrote {out}")
    return str(out)
