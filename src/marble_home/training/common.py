from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from marble_home.utils import provenance
from marble_home.utils.paths import CHECKPOINTS, DATA, RESULTS, ensure


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Run:
    """Run directory: results/runs/<name>/{config.json, manifest.json, metrics.csv, tb/} + checkpoints/<name>.pt"""

    def __init__(self, name: str, cfg: dict, dataset_path: Path | None = None):
        self.name = name
        self.dir = RESULTS / "runs" / name
        ensure(self.dir, CHECKPOINTS)
        self.ckpt = CHECKPOINTS / f"{name}.pt"
        self.t0 = time.time()
        self.cfg = cfg
        self.rows: list[dict] = []
        self.dataset_path = dataset_path
        provenance.write_json(self.dir / "config.json", cfg)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(self.dir / "tb"))
        except Exception:
            self.tb = None

    def log(self, step: int, **kv):
        row = {"step": step, "time_s": round(time.time() - self.t0, 1), **kv}
        self.rows.append(row)
        if self.tb:
            for k, v in kv.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)
        keys = sorted({k for r in self.rows for k in r}, key=lambda k: (k not in ("step", "time_s"), k))
        with open(self.dir / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)

    def finish(self, ckpt_sha: str | None, **extra):
        man = provenance.run_manifest(
            run=self.name, seed=self.cfg.get("seed"), duration_s=round(time.time() - self.t0, 1), checkpoint=str(self.ckpt),
            checkpoint_sha256=ckpt_sha,
            dataset=str(self.dataset_path) if self.dataset_path else None,
            dataset_sha256=provenance.file_sha256(self.dataset_path) if self.dataset_path and Path(self.dataset_path).exists() else None,
            **extra)
        provenance.write_json(self.dir / "manifest.json", man)
        if self.tb:
            self.tb.close()
        return man


def dataset_action_mode(ds: str) -> str:
    """Action parametrisation the demonstrations were recorded/relabelled in (ee_abs | ee_delta)."""
    import json

    from marble_home.datasets.hdf5 import DemoFile

    with DemoFile(dataset_path(ds)) as df:
        return json.loads(df.attrs["env_args"]).get("action_mode", "ee_abs")


def dataset_path(task: str) -> Path:
    return DATA / "demonstrations" / task / "dataset.hdf5"


def episode_stats(episodes: list[dict], key: str) -> dict:
    arr = np.concatenate([e["obs"][key] for e in episodes], 0)
    return {"mean": arr.mean(0), "std": np.maximum(arr.std(0), 1e-3)}
