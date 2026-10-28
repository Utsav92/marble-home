from __future__ import annotations

import ast
from pathlib import Path

from marble_home.utils.paths import load_yaml


def parse_sets(sets: list[str] | None) -> dict:
    out = {}
    for s in sets or []:
        k, _, v = s.partition("=")
        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            out[k] = v
    return out


def load_cfg(path: str | Path, sets: list[str] | None = None) -> dict:
    cfg = load_yaml(path)
    cfg.update(parse_sets(sets))
    return cfg
