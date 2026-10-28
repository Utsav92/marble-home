"""Relabel an ee_abs demonstration dataset into ee_delta actions (same demos, same observations).

    python scripts/relabel_delta.py --src pick_up_cup --dst pick_up_cup_delta
delta_t = (target_t - target_{t-1}) / DELTA_SCALE, with target_{-1} = the measured right-hand position at reset (== the env's
integrator start). The policy then outputs increments to the COMMANDED target rather than a target relative to the measured pose,
so tracking lag of the IK controller does not enter the policy's regression problem.
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from marble_home.datasets.hdf5 import DemoFile, DemoWriter
from marble_home.robots import action_space as AS
from marble_home.utils.paths import DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src = DATA / "demonstrations" / a.src / "dataset.hdf5"
    dst = DATA / "demonstrations" / a.dst / "dataset.hdf5"
    dst.parent.mkdir(parents=True, exist_ok=True)
    clipped = total = 0
    with DemoFile(src) as df, DemoWriter(dst, df.attrs["task"], json.loads(df.attrs["scene"]), {**json.loads(df.attrs["env_args"]), "action_mode": "ee_delta"},
                                         json.loads(df.attrs["versions"]), json.loads(df.attrs["robot_config"])) as w:
        for n in df.demos:
            ep = df.demo(n)
            pos = AS.WS_CENTER + ep["actions"][:, :3] * AS.WS_HALF
            start = ep["obs"]["proprio"][0][42:45]  # robot0_base_to_right_eef_pos at reset
            prev = np.vstack([start, pos[:-1]])
            d = (pos - prev) / AS.DELTA_SCALE
            clipped += int((np.abs(d) > 1).sum())
            total += d.size
            ep["actions"] = np.concatenate([np.clip(d, -1, 1), ep["actions"][:, 3:]], 1).astype(np.float32)
            at = dict(ep["attrs"])
            init = at.pop("init_state")
            at.pop("num_samples", None)
            ep["init_state"] = init
            w.add(ep, **at)
    shutil.copy(DATA / "splits" / f"{a.src}.json", DATA / "splits" / f"{a.dst}.json")
    shutil.copy(DATA / "splits" / f"{a.src}_stats.json", DATA / "splits" / f"{a.dst}_stats.json")
    print(f"{a.src} -> {a.dst}: {clipped}/{total} delta components clipped at +-1 ({clipped / total:.3%})")


if __name__ == "__main__":
    main()
