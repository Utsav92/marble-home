from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Code + configs always come from the repo; data/results/checkpoints can be redirected (e.g. the pipeline smoke test uses
# MARBLE_HOME_WORK=<repo>/.smoke so it can never touch the real datasets).
WORK = Path(os.environ.get("MARBLE_HOME_WORK", ROOT))
CONFIGS = ROOT / "configs"
DATA = WORK / "data"
RESULTS = WORK / "results"
REPORTS = WORK / "reports"
CHECKPOINTS = WORK / "checkpoints"
ASSETS = ROOT / "assets"


def ensure(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict:
    import yaml

    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        return yaml.safe_load(f) or {}
