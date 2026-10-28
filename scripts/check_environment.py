"""Validate the environment and print an integration-status table (VERIFIED / BLOCKED).

    python scripts/check_environment.py [--json results/environment.json]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback

from marble_home.utils import provenance
from marble_home.utils.paths import ROOT


def check(name, fn, rows):
    t = time.time()
    try:
        detail = fn()
        rows.append((name, "VERIFIED", str(detail), time.time() - t))
    except Exception as e:  # noqa: BLE001
        rows.append((name, "BLOCKED", f"{type(e).__name__}: {str(e)[:160]}", time.time() - t))
        if os.environ.get("CHECK_TRACE"):
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "results" / "environment.json"))
    args = ap.parse_args()
    rows: list = []

    def imports():
        import gymnasium, h5py, mujoco, numpy, robosuite, scipy, torch, trimesh  # noqa: F401
        return f"mujoco {mujoco.__version__}, robosuite {robosuite.__version__}, torch {torch.__version__}"

    def mujoco_pin():
        import mujoco
        d = mujoco.MjData(mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><joint type='free'/><geom size='.1'/></body></worldbody></mujoco>"))
        assert hasattr(d, "qM"), "robosuite 1.5.2 needs MjData.qM (mujoco<3.4)"
        return f"mujoco {mujoco.__version__} exposes MjData.qM"

    def mink():
        from marble_home.robots.gr1 import register_mink_ik
        return "vendored WholeBodyMinkIK registered" if register_mink_ik() else "no"

    def env_step():
        import numpy as np
        from marble_home.simulation.gym_env import HouseholdEnv
        env = HouseholdEnv(task="pick_up_cup", obs_mode="rgb", image_hw=64)
        obs, _ = env.reset(seed=0)
        for _ in range(5):
            obs, *_ = env.step(env.action_space.sample() * 0)
        assert np.isfinite(obs["proprio"]).all() and obs["rgb_head"].std() > 1
        env.close()
        return "GR1FixedLowerBody + kitchen steps, RGB offscreen render OK"

    def worldlabs():
        return "WORLDLABS_API_KEY set" if os.environ.get("WORLDLABS_API_KEY") else \
            "no WORLDLABS_API_KEY -> offline importer + synthetic fixtures only"

    def disk():
        free = shutil.disk_usage(ROOT).free / 2**30
        return f"{free:.0f} GB free on {ROOT.drive or '/'}"

    for n, f in [("imports", imports), ("mujoco pin", mujoco_pin), ("mink IK controller", mink),
                 ("GR1 kitchen env + camera", env_step), ("World Labs credentials", worldlabs), ("disk", disk)]:
        check(n, f, rows)

    print(f"{'check':28s} {'status':9s} detail")
    for n, s, d, dt in rows:
        print(f"{n:28s} {s:9s} {d}  [{dt:.1f}s]")
    hw = provenance.hardware()
    print("\nhardware:", hw)
    print("python  :", sys.version.split()[0], "| profile:", "B/C (CUDA)" if hw.get("cuda") else "A (CPU only)")
    provenance.write_json(args.json, {"checks": [dict(name=n, status=s, detail=d) for n, s, d, _ in rows],
                                      "manifest": provenance.run_manifest()})
    print("wrote", args.json)
    sys.exit(0 if all(s == "VERIFIED" for _, s, _, _ in rows if _ != "World Labs credentials") else 1)


if __name__ == "__main__":
    main()
