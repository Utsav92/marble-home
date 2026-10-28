"""Milestone-1 simulator smoke test (writes results/m1_smoke.json, a PNG and an MP4).

    python scripts/test_robot.py --headless
Acceptance: 100 consecutive steps with no model error and finite states.
"""
from __future__ import annotations

import argparse
import time

import imageio
import mujoco
import numpy as np

from marble_home.policies.scripted import ScriptedExpert
from marble_home.robots import action_space as AS
from marble_home.simulation.gym_env import HouseholdEnv
from marble_home.utils import provenance
from marble_home.utils.paths import RESULTS, ensure
from marble_home.visualization.video import rollout_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="(always offscreen; flag kept for the documented CLI)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--task", default="pick_up_cup")
    args = ap.parse_args()
    out = RESULTS / "m1"
    ensure(out)
    rep = {}

    env = HouseholdEnv(task=args.task, obs_mode="rgb", image_hw=128)
    obs, info = env.reset(seed=0)
    m = env.env.sim.model._model
    rep["model"] = {"nbody": m.nbody, "njnt": m.njnt, "ngeom": m.ngeom, "nu": m.nu, "timestep": m.opt.timestep,
                    "gravity": m.opt.gravity.tolist(), "solver": int(m.opt.solver), "control_freq": env.env.control_freq}

    # joint limits valid and the reset pose inside them
    bad = []
    for j in range(m.njnt):
        if m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
            q = env.env.sim.data.qpos[m.jnt_qposadr[j]]
            if not (lo < hi) or not (lo - 1e-6 <= q <= hi + 1e-6):
                bad.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
    rep["joint_limits_ok"] = not bad
    rep["joint_limit_violations"] = bad

    # 100 consecutive steps, finite states
    t0, finite = time.time(), True
    for i in range(args.steps):
        obs, r, term, trunc, info = env.step(np.zeros(AS.ACTION_DIM, dtype=np.float32))
        finite &= bool(np.isfinite(obs["proprio"]).all() and np.isfinite(env.env.sim.data.qpos).all())
    rep["steps_ok"] = args.steps if finite else None
    rep["steps_per_s_with_camera"] = round(args.steps / (time.time() - t0), 1)
    imageio.imwrite(out / "head_rgb.png", obs["rgb_head"])
    rep["rgb_std"] = float(obs["rgb_head"].std())

    # basic physics: the cup dropped from 8 cm must come to rest on the counter surface
    env.reset(seed=1)
    e = env.env
    qa = e.sim.model.get_joint_qpos_addr(e.cup.joints[0])
    q = e.sim.data.qpos[qa[0]:qa[1]].copy()
    q[2] += 0.08
    e.sim.data.qpos[qa[0]:qa[1]] = q
    e.sim.forward()
    for _ in range(40):
        env.step(np.zeros(AS.ACTION_DIM, dtype=np.float32))
    z = float(e._last_state.cup_pos[2])
    rep["physics_drop_rest_z_err_mm"] = round(abs(z - e.cup_rest_z) * 1000, 2)
    rep["physics_ok"] = abs(z - e.cup_rest_z) < 0.004

    # rollout video: scripted expert (reduced actions through the real wrapper)
    holder = {}

    def reset_hook(o, i):
        ex = ScriptedExpert(env.env)
        ex.reset(env._raw)
        holder["ex"] = ex

    stats = rollout_video(env, lambda o: AS.full_to_reduced(holder["ex"].act(env._raw)), out / f"{args.task}_expert.mp4",
                          "scripted expert", seed=2, on_reset=reset_hook)
    rep["expert_rollout"] = stats
    rep["accepted"] = bool(rep["steps_ok"] and rep["joint_limits_ok"] and rep["physics_ok"] and rep["rgb_std"] > 1)
    rep["manifest"] = provenance.run_manifest()
    provenance.write_json(out / "m1_smoke.json", rep)
    print({k: v for k, v in rep.items() if k != "manifest"})
    print("MILESTONE 1 smoke:", "PASS" if rep["accepted"] else "FAIL")
    raise SystemExit(0 if rep["accepted"] else 1)


if __name__ == "__main__":
    main()
