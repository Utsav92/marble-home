"""In-memory torch datasets over the HDF5 demos (state or image observations)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from marble_home.datasets.hdf5 import DemoFile
from marble_home.utils.paths import DATA


def load_stats(task: str) -> dict:
    with open(DATA / "splits" / f"{task}_stats.json") as f:
        return json.load(f)


def load_split(task: str) -> dict:
    with open(DATA / "splits" / f"{task}.json") as f:
        return json.load(f)


def load_episodes(task: str, split: str, obs_keys=("state",), path: str | Path | None = None,
                  max_episodes: int | None = None, split_override: dict | None = None) -> list[dict]:
    path = path or DATA / "demonstrations" / task / "dataset.hdf5"
    names = (split_override or load_split(task))[split]
    if max_episodes is not None:
        names = names[:max_episodes]
    eps = []
    with DemoFile(path) as df:
        for n in names:
            d = df.f["data"][n]
            eps.append({"obs": {k: d["obs"][k][()] for k in obs_keys}, "act": d["actions"][()].astype(np.float32),
                        "name": n})
    return eps


class StepDataset(Dataset):
    """Single-step (obs_t, a_t) pairs for behaviour cloning."""

    def __init__(self, episodes: list[dict]):
        keys = episodes[0]["obs"].keys()
        self.obs = {k: np.concatenate([e["obs"][k] for e in episodes], 0) for k in keys}
        self.act = np.concatenate([e["act"] for e in episodes], 0)

    def __len__(self):
        return len(self.act)

    def __getitem__(self, i):
        return {k: torch.from_numpy(v[i]) for k, v in self.obs.items()}, torch.from_numpy(self.act[i])


class SeqDataset(Dataset):
    """Windows for diffusion policy: obs history (To) and an action chunk (H), edge-padded.

    Window starts at s = t-(To-1); obs = obs[s:s+To]; actions = act[s:s+H] (same convention as Chi et al.).
    """

    def __init__(self, episodes: list[dict], n_obs: int, horizon: int):
        self.eps, self.To, self.H = episodes, n_obs, horizon
        self.index = [(ei, t) for ei, e in enumerate(episodes) for t in range(len(e["act"]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ei, t = self.index[i]
        e = self.eps[ei]
        T = len(e["act"])
        s = t - (self.To - 1)
        oi = np.clip(np.arange(s, s + self.To), 0, T - 1)
        ai = np.clip(np.arange(s, s + self.H), 0, T - 1)
        obs = {k: torch.from_numpy(v[oi]) for k, v in e["obs"].items()}
        return obs, torch.from_numpy(e["act"][ai])
