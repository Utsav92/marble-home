"""Render camera observations for an existing state dataset by REPLAYING its demonstrations.

Cheaper than re-collecting (the expert runs at ~17 steps/s, rendering at ~6). The replay is near-deterministic (see
docs/datasets.md), so images align with the recorded actions to within the replay tolerance. A domain-randomisation
preset can be applied: each demo is replayed under its own seed-determined appearance draw.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from marble_home.datasets.hdf5 import DemoFile, DemoWriter, merge
from marble_home.simulation.gym_env import HouseholdEnv


def render_episode(env: HouseholdEnv, ep: dict, attrs: dict) -> dict:
    env.reset(seed=int(attrs["seed"]))  # applies this demo's DR draw (if any)
    sim = env.env.sim
    sim.set_state_from_flattened(np.asarray(attrs["init_state"]))
    sim.forward()
    raw = env.env._get_observations(force_update=True)
    env._raw = {**raw, **env.env._task_obs()}
    obs = {}
    for t in range(len(ep["actions_full"])):
        o = env._policy_obs(env._raw)
        if env.randomizer is not None:
            o = env.randomizer.perturb_obs(o)
        for k, v in o.items():
            obs.setdefault(k, []).append(v)
        raw_next, *_ = env.env.step(ep["actions_full"][t])
        env._raw = raw_next
    return {k: np.stack(v) for k, v in obs.items()}


def _chunk(args):
    src, names, dst, task, obs_mode, hw, dr, scene_override = args
    envs = {}
    with DemoFile(src) as df:
        first = df.attrs
        with DemoWriter(dst, task, json.loads(first["scene"]), {**json.loads(first["env_args"]), "obs_mode": obs_mode, "image_hw": hw, "dr": dr},
                        json.loads(first["versions"]), json.loads(first["robot_config"])) as w:
            for n in names:
                ep = df.demo(n)
                at = dict(ep["attrs"])
                sj = at.get("scene_json")
                scene = json.loads(sj.decode() if isinstance(sj, bytes) else sj) if sj is not None else None
                key = scene["scene_id"] if scene else "default"
                if key not in envs:
                    envs[key] = HouseholdEnv(task=task, scene=scene, obs_mode=obs_mode, image_hw=hw, dr=dr, seed=0)
                ep["obs"] = render_episode(envs[key], ep, at)
                init = at.pop("init_state")
                at.pop("num_samples", None)
                ep["init_state"] = init
                at["name_src"] = n
                w.add(ep, **at)
    for e in envs.values():
        e.close()
    return dst


def render_dataset(src: str | Path, dst: str | Path, task: str, names: list[str], obs_mode: str = "rgb", hw: int = 64,
                   dr: str | None = None, workers: int = 3) -> dict:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    chunks = [names[i::workers] for i in range(workers) if names[i::workers]]
    shards = [str(dst.with_suffix(f".part{i}.hdf5")) for i in range(len(chunks))]
    jobs = [(str(src), c, s, task, obs_mode, hw, dr, None) for c, s in zip(chunks, shards)]
    if len(jobs) == 1:
        _chunk(jobs[0])
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as p:
            p.map(_chunk, jobs)
    summ = merge(shards, dst)
    for s in shards:
        Path(s).unlink()
    return summ
