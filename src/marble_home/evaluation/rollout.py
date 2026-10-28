"""Run an agent on seeded task instances and record per-episode metrics.

Every agent sees the SAME instances for the same seed (HouseholdEnv.reset(seed=...)), which makes
paired comparisons (McNemar) valid.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from marble_home.policies.agents import make_agent
from marble_home.simulation.gym_env import HouseholdEnv


def run_episode(env: HouseholdEnv, agent, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    agent.reset(env)
    ret, done, t = 0.0, False, 0
    lat, a_prev, dA, q_hist = [], None, [], []
    grasp_attempt = False
    sim_time = 0.0
    last = {}
    while not done:
        t0 = time.perf_counter()
        a = agent.act(obs)
        lat.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        obs, r, term, trunc, info = env.step(a)
        sim_time += time.perf_counter() - t1
        ret += r
        done = term or trunc
        last = info
        if a_prev is not None:
            dA.append(float(np.linalg.norm(a - a_prev)))
        a_prev = a
        q_hist.append(np.asarray(env._raw["robot0_joint_pos"], dtype=float))
        grasp_attempt = grasp_attempt or (a[3] > 0 and info["eef_error"] < 0.06)
        t += 1
    q = np.array(q_hist)
    dt = 1.0 / env.env.control_freq
    jerk = float(np.abs(np.diff(q, n=3, axis=0)).mean() / dt**3) if len(q) > 4 else None
    e = env.env
    return {
        "seed": seed, "success": bool(info["ever_success"]), "steps": t, "return": ret,
        "items_done": int(last.get("items_done", 0)),
        "grasp_attempted": bool(grasp_attempt), "grasp_success": bool(e._was_lifted),
        "place_error": float(last["place_error"]), "eef_error": float(last["eef_error"]),
        "invalid_steps": int(last["invalid_steps"]), "collision_free": bool(last["invalid_steps"] <= e.task_cfg.max_invalid_steps),
        "peak_force": float(last["peak_force"]), "action_smoothness": float(np.mean(dA)) if dA else 0.0,
        "joint_jerk": jerk, "latency_ms": float(np.mean(lat)), "sim_steps_per_s": t / max(sim_time, 1e-9),
    }


def _worker(args):
    spec, task, seeds, env_kwargs = args
    agent = make_agent(spec)
    env = HouseholdEnv(task=task, **{**env_kwargs, **getattr(agent, "env_kwargs", {})})
    out = [run_episode(env, agent, s) for s in seeds]
    env.close()
    return out


def evaluate(spec: dict, task: str, seeds: list[int], workers: int = 1, **env_kwargs) -> list[dict]:
    """Evaluate one agent spec on the given instance seeds. env_kwargs -> HouseholdEnv (obs_mode, scene, ...)."""
    if workers <= 1:
        return _worker((spec, task, seeds, env_kwargs))
    chunks = [seeds[i::workers] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as p:
        res = p.map(_worker, [(spec, task, c, env_kwargs) for c in chunks if c])
    flat = [e for r in res for e in r]
    return sorted(flat, key=lambda e: e["seed"])


def _scene_worker(args):
    spec, task, seeds, scene, env_kwargs = args
    agent = make_agent(spec)
    env = HouseholdEnv(task=task, scene=scene, **{**env_kwargs, **getattr(agent, "env_kwargs", {})})
    out = [run_episode(env, agent, s) for s in seeds]
    env.close()
    for e in out:
        e["scene_id"] = scene["scene_id"]
    return out


def evaluate_scenes(spec: dict, task: str, scenes: list[dict], seeds: list[int], workers: int = 1, **env_kwargs) -> dict:
    """Evaluate one agent on every scene (same instance seeds per scene). Returns {scene_id: [episode records]}."""
    jobs = [(spec, task, seeds, sc, env_kwargs) for sc in scenes]
    if workers <= 1:
        res = [_scene_worker(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as p:
            res = p.map(_scene_worker, jobs)
    return {sc["scene_id"]: r for sc, r in zip(scenes, res)}
