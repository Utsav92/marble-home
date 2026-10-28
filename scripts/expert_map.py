"""Map scripted-expert success over a grid of cup positions (diagnostic for expert reachability).

    python scripts/expert_map.py --task pick_up_cup --xs -0.05 0 0.05 --ys -0.22 -0.18 -0.14 -0.10 -0.08
Writes results/experiments/expert_map_<task>[_tag].json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from marble_home.evaluation.rollout import run_episode
from marble_home.policies.agents import ScriptedAgent
from marble_home.simulation.gym_env import HouseholdEnv
from marble_home.utils.paths import RESULTS, ensure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_up_cup")
    ap.add_argument("--xs", type=float, nargs="+", default=[-0.05, 0.0, 0.05])
    ap.add_argument("--ys", type=float, nargs="+", default=[-0.22, -0.20, -0.18, -0.15, -0.12, -0.09, -0.08])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--waist", type=float, nargs=2, default=None, metavar=("YAW", "PITCH"), help="IK posture weights for waist yaw/pitch (default 10 10; lower = freer)")
    a = ap.parse_args()
    ov = {"ik_posture_weights": {"robot0_torso_waist_yaw": a.waist[0], "robot0_torso_waist_pitch": a.waist[1]}} if a.waist else None
    env = HouseholdEnv(task=a.task, controller_overrides=ov)
    sam = env.env.placement_initializer
    rows = []
    for x in a.xs:
        for y in a.ys:
            sam.x_range, sam.y_range = [x, x], [y, y]
            res = [run_episode(env, ScriptedAgent(), seed=1234 + r) for r in range(a.reps)]
            ok = float(np.mean([r["success"] for r in res]))
            rows.append({"x": x, "y": y, "success": ok, "eef_error": float(np.mean([r["eef_error"] for r in res])),
                         "place_error": float(np.mean([r["place_error"] for r in res])), "steps": float(np.mean([r["steps"] for r in res]))})
            print(f"x={x:+.2f} y={y:+.3f}: success {ok:.0%} eef_err {rows[-1]['eef_error']:.3f} place_err {rows[-1]['place_error']:.3f}", flush=True)
    ensure(RESULTS / "experiments")
    (RESULTS / "experiments" / f"expert_map_{a.task}{a.tag}.json").write_text(json.dumps({"task": a.task, "rows": rows}, indent=1))
    print(f"overall {np.mean([r['success'] for r in rows]):.0%} over {len(rows)} grid points")


if __name__ == "__main__":
    main()
