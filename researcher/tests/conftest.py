import pytest

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture(autouse=True)
def _stub_paper(monkeypatch, tmp_path):
    """Tests never download the AI Wellbeing paper: when it has not been fetched, a stub stands in for it."""
    from rwr import sweep
    from rwr.paper import PAPER_REF

    real = sweep._asset_path
    stub = tmp_path / "PAPER_full.md"
    stub.write_text("stub paper")

    def asset_path(ref, config_dir):
        if ref == PAPER_REF and not (config_dir / ref).exists():
            return stub
        return real(ref, config_dir)

    monkeypatch.setattr(sweep, "_asset_path", asset_path)
