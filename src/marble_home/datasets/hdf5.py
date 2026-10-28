"""Versioned HDF5 demonstration format (RoboMimic-like layout, our own schema).

file attrs : format_version, task, scene (json), env_args (json), versions (json), robot_config (json)
data/demo_N: actions (T,4) reduced policy actions | actions_full (T,24) executed ik actions
             rewards (T,) | dones (T,) | task_phase (T,) | eef_error (T,) | place_error (T,)
             obs/proprio (T,58) | obs/state (T,69) | obs/<rgb_head|rgb_wrist_*|depth_head> (T,...) optional
             attrs: seed, scene_id, task, success, num_samples, init_state (flattened MuJoCo state), expert_noise, speed
Observation at index t is the observation BEFORE ``actions[t]`` is applied.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

FORMAT_VERSION = "marble_home.demo.v1"


class DemoWriter:
    def __init__(self, path: str | Path, task: str, scene: dict, env_args: dict, versions: dict, robot_config: dict):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(path, "w")
        self.f.attrs.update(format_version=FORMAT_VERSION, task=task, scene=json.dumps(scene),
                            env_args=json.dumps(env_args), versions=json.dumps(versions),
                            robot_config=json.dumps(robot_config))
        self.g = self.f.create_group("data")
        self.n = 0

    def add(self, ep: dict, **attrs) -> str:
        name = f"demo_{self.n}"
        d = self.g.create_group(name)
        for k in ("actions", "actions_full", "rewards", "dones", "task_phase", "eef_error", "place_error"):
            d.create_dataset(k, data=np.asarray(ep[k]), compression="gzip", compression_opts=4)
        og = d.create_group("obs")
        for k, v in ep["obs"].items():
            og.create_dataset(k, data=np.asarray(v), compression="gzip", compression_opts=4)
        d.attrs.update(num_samples=len(ep["actions"]), init_state=np.asarray(ep["init_state"]), **attrs)
        self.n += 1
        return name

    def close(self):
        self.g.attrs["total"] = self.n
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class DemoFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.f = h5py.File(path, "r")
        self.attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in self.f.attrs.items()}

    @property
    def demos(self) -> list[str]:
        return sorted(self.f["data"].keys(), key=lambda s: int(s.split("_")[1]))

    def demo(self, name: str) -> dict:
        d = self.f["data"][name]
        out = {k: d[k][()] for k in ("actions", "actions_full", "rewards", "dones", "task_phase", "eef_error", "place_error")}
        out["obs"] = {k: d["obs"][k][()] for k in d["obs"]}
        out["attrs"] = {k: d.attrs[k] for k in d.attrs}
        return out

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def merge(shards: list[str | Path], out: str | Path) -> dict:
    """Concatenate shard files into one dataset; returns a summary."""
    first = DemoFile(shards[0])
    with DemoWriter(out, first.attrs["task"], json.loads(first.attrs["scene"]), json.loads(first.attrs["env_args"]),
                    json.loads(first.attrs["versions"]), json.loads(first.attrs["robot_config"])) as w:
        first.close()
        steps = 0
        for s in shards:
            with DemoFile(s) as df:
                for n in df.demos:
                    ep = df.demo(n)
                    attrs = dict(ep["attrs"])
                    init = attrs.pop("init_state")
                    attrs.pop("num_samples", None)
                    ep["init_state"] = init
                    w.add(ep, **attrs)
                    steps += len(ep["actions"])
    return {"episodes": w.n, "steps": steps, "path": str(out)}
