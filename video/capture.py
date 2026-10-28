"""Capture synchronized multi-camera footage + per-step telemetry of one policy rollout.

    python video/capture.py --task clear_table --policy scripted --seed 900001 --name hero_expert
    python video/capture.py --task clear_table --policy checkpoints/bc_mlp_clear_table_n100_s2.pt --seed 900001 --name hero_bc
Writes video/footage/<name>_<camera>.mp4 and video/footage/<name>.json (real values from the simulator, nothing synthesised).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from marble_home.policies.agents import make_agent  # noqa: E402
from marble_home.simulation.gym_env import HouseholdEnv  # noqa: E402

OUT = Path(__file__).resolve().parent / "footage"


def capture(task, policy, seed, name, cams=("eval_side", "eval_top"), size=(960, 540), scene=None):
    spec = {"type": policy} if policy in ("scripted", "random") else {"type": "checkpoint", "path": policy, "name": Path(policy).stem}
    agent = make_agent(spec)
    env = HouseholdEnv(task=task, scene=scene, **getattr(agent, "env_kwargs", {}))
    obs, info = env.reset(seed=seed)
    agent.reset(env)
    OUT.mkdir(parents=True, exist_ok=True)
    writers = {c: imageio.get_writer(OUT / f"{name}_{c}.mp4", fps=20, macro_block_size=1, quality=8) for c in cams}
    tel = []
    done, t = False, 0
    k = env.env
    while True:
        for c in cams:
            writers[c].append_data(env.render(size[0], size[1], camera=c))
        if done:
            break
        a = agent.act(obs)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        t += 1
        s = k._last_state
        tel.append({"t": t, "phase": int(info["phase"]), "items_done": int(info["items_done"]), "active": int(k.active),
                    "eef_error": float(info["eef_error"]), "place_error": float(info["place_error"]), "grip": float(a[3]),
                    "invalid_steps": int(info["invalid_steps"]), "peak_force": float(info["peak_force"]), "success": bool(info["success"]),
                    "eef": [float(v) for v in k.eef_pos_world()], "item": [float(v) for v in s.cup_pos]})
    for w in writers.values():
        w.close()
    meta = {"task": task, "policy": policy, "seed": seed, "steps": t, "success": bool(info["ever_success"]), "n_items": int(k.n_items),
            "items_done": int(info["items_done"]), "cams": list(cams), "size": list(size), "telemetry": tel}
    (OUT / f"{name}.json").write_text(json.dumps(meta))
    env.close()
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=900001)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cams", nargs="+", default=["eval_side", "eval_top"])
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=540)
    ap.add_argument("--counter-height", type=float, default=None, help="default kitchen with this counter height (m)")
    a = ap.parse_args()
    sc = None
    if a.counter_height is not None:
        import copy

        from marble_home.simulation.kitchen_arena import DEFAULT_SCENE

        sc = copy.deepcopy(DEFAULT_SCENE)
        sc.update(scene_id=f"h{int(round(a.counter_height * 1000)):04d}", counter_height=a.counter_height)
    m = capture(a.task, a.policy, a.seed, a.name, tuple(a.cams), (a.w, a.h), scene=sc)
    print(f"{a.name}: steps={m['steps']} success={m['success']} items={m['items_done']}/{m['n_items']}")
