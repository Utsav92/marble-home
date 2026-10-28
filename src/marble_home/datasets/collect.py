"""Scripted-expert demonstration collection + replay verification."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from marble_home.datasets.hdf5 import DemoFile, DemoWriter
from marble_home.policies.scripted import make_expert
from marble_home.robots import action_space as AS
from marble_home.simulation.gym_env import HouseholdEnv
from marble_home.utils import provenance

PROPRIO_AND_GOAL_DIM = 58  # 56 proprio + 2 goal; the privileged cup_pos follows in the 'state' vector


def make_env(task, scene=None, obs_mode="state", image_hw=64, seed=0, task_cfg=None, dr=None):
    return HouseholdEnv(task=task, obs_mode=obs_mode, image_hw=image_hw, scene=scene, task_cfg=task_cfg, seed=seed, dr=dr)


def run_expert_episode(env: HouseholdEnv, seed: int, noise: float, speed: float, record_obs=True) -> dict:
    obs, info = env.reset(seed=seed)
    raw = env._raw
    init_state = env.env.sim.get_state().flatten().copy()
    ex = make_expert(env.env, noise=noise, speed=speed, rng=np.random.default_rng(seed + 10_000))
    ex.reset(raw)
    rec = {k: [] for k in ("actions", "actions_full", "rewards", "dones", "task_phase", "eef_error", "place_error")}
    obs_rec = {"proprio": [], "state": []}
    done, ever = False, False
    state_env_obs = obs
    while not done:
        full = ex.act(raw)
        a = AS.full_to_reduced(full)
        # record the observation BEFORE acting
        obs_rec["state"].append(np.asarray(state_env_obs, np.float32))
        obs_rec["proprio"].append(env._proprio(raw))
        obs_next, r, term, trunc, info = env.step(a)
        raw = env._raw
        state_env_obs = obs_next
        done = term or trunc
        ever = ever or bool(info["success"])
        rec["actions"].append(a)
        rec["actions_full"].append(info["full_action"])
        rec["rewards"].append(r)
        rec["dones"].append(float(done))
        rec["task_phase"].append(info["phase"])
        rec["eef_error"].append(info["eef_error"])
        rec["place_error"].append(info["place_error"])
    out = {k: np.asarray(v, np.float32) for k, v in rec.items()}
    out["obs"] = {k: np.asarray(v, np.float32) for k, v in obs_rec.items()}
    out["init_state"] = init_state
    out["success"] = ever
    return out


def collect(task: str, n_episodes: int, out_path: str | Path, seed0: int = 0, noise: float = 0.004,
            speed_range=(0.85, 1.2), scene: dict | None = None, keep_failures: bool = False, dr: str | None = None, log=print) -> dict:
    env = make_env(task, scene=scene, seed=seed0, dr=dr)
    rng = np.random.default_rng(seed0)
    t0, kept, tried, steps = time.time(), 0, 0, 0
    with DemoWriter(out_path, task, env.env.scene, {"task": task, "obs_mode": "state", "action_mode": "ee_abs", "dr": dr},
                    provenance.versions(), {"robot": "GR1FixedLowerBody", "control": "WHOLE_BODY_MINK_IK",
                                            "action_space": "ee_abs-4d"}) as w:
        while kept < n_episodes:
            seed = seed0 + tried
            speed = float(rng.uniform(*speed_range))
            ep = run_expert_episode(env, seed, noise, speed)
            tried += 1
            if ep["success"] or keep_failures:
                w.add(ep, seed=seed, scene_id=env.env.scene["scene_id"], task=task, success=bool(ep["success"]),
                      expert_noise=noise, speed=speed, scene_json=json.dumps(env.env.scene), dr=dr or "none")
                kept += 1
                steps += len(ep["actions"])
            if tried % 10 == 0:
                log(f"[{task}] kept {kept}/{n_episodes} tried {tried} steps {steps} ({time.time() - t0:.0f}s)")
    env.close()
    return {"task": task, "kept": kept, "tried": tried, "steps": steps, "seconds": time.time() - t0, "path": str(out_path)}


# ----------------------------------------------------------------------------- replay verification
def replay_check(env: HouseholdEnv, ep: dict, attrs: dict, tol_proprio=0.02, tol_cup=5e-3) -> dict:
    """Restore the recorded initial state, replay executed full actions, compare to recorded observations.

    Not bit-exact: the mink IK solver's internal state is not part of the saved MuJoCo state, so
    trajectories drift by <1 mrad early and up to ~7 mrad during contact-rich finger closing
    (measured). Acceptance = proprio error < ``tol_proprio``, cup error < ``tol_cup`` and the same
    success outcome.
    """
    env.reset(seed=int(attrs["seed"]))
    sim = env.env.sim
    sim.set_state_from_flattened(np.asarray(attrs["init_state"]))
    sim.forward()
    raw = env.env._get_observations(force_update=True)
    env._raw = {**raw, **env.env._task_obs()}
    env._target = np.asarray(raw["robot0_base_to_right_eef_pos"], float).copy()
    errs, cup_err, success = [], 0.0, False
    for t in range(len(ep["actions_full"])):
        rec_proprio = ep["obs"]["proprio"][t]
        errs.append(float(np.abs(env._proprio(env._raw) - rec_proprio).max()))
        raw_next, r, d, info = env.env.step(ep["actions_full"][t])
        env._raw = raw_next
        success = success or bool(info["success"])
        if t + 1 < len(ep["actions_full"]):  # state obs layout: proprio(58) then cup_pos(3)
            rec_cup = ep["obs"]["state"][t + 1][PROPRIO_AND_GOAL_DIM:PROPRIO_AND_GOAL_DIM + 3]
            cup_err = max(cup_err, float(np.abs(np.asarray(raw_next["cup_pos"]) - rec_cup).max()))
    return {"max_proprio_err": max(errs), "mean_proprio_err": float(np.mean(errs)), "max_cup_err": cup_err,
            "replay_success": success,
            "ok": max(errs) < tol_proprio and cup_err < tol_cup and success == bool(attrs["success"])}
