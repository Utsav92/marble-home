"""Render an annotated MP4 rollout for a policy.

    python scripts/render_episode.py --policy best --task pick_up_cup
    python scripts/render_episode.py --policy checkpoints/bc_mlp_pick_up_cup_n240_s0.pt --task pick_up_cup --seed 900003
    python scripts/render_episode.py --policy scripted --task place_cup --camera eval_front3q
'best' = the checkpoint with the highest success rate in results/benchmark/<task>.json (run scripts/benchmark.py first).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from marble_home.policies.agents import make_agent
from marble_home.simulation.gym_env import HouseholdEnv
from marble_home.utils.paths import RESULTS
from marble_home.visualization.video import rollout_video


def resolve(policy: str, task: str) -> dict:
    if policy in ("random", "scripted"):
        return {"type": policy, "name": policy}
    if policy == "best":
        f = RESULTS / "benchmark" / f"{task}.json"
        if not f.exists():
            raise SystemExit(f"{f} not found: run scripts/benchmark.py --task {task} --auto first")
        rows = [r for r in json.loads(f.read_text())["rows"] if r["spec"]["type"] in ("checkpoint", "ppo")]
        if not rows:
            raise SystemExit("no trained policies in the benchmark")
        best = max(rows, key=lambda r: (r["summary"]["success_rate"], -r["summary"]["latency_ms"]))
        return {**best["spec"], "name": best["name"] + " (best)"}
    return {"type": "ppo" if policy.endswith(".zip") else "checkpoint", "path": policy, "name": Path(policy).stem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="best")
    ap.add_argument("--task", default="pick_up_cup")
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--camera", default="eval_side")
    ap.add_argument("--scene-spec", default=None, help="path to a scene_spec.json (e.g. a Marble-derived scene)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    spec = resolve(args.policy, args.task)
    agent = make_agent(spec)
    scene = json.loads(Path(args.scene_spec).read_text()) if args.scene_spec else None
    env = HouseholdEnv(task=args.task, scene=scene, **getattr(agent, "env_kwargs", {}))
    out = args.out or RESULTS / "videos" / f"{args.task}_{spec['name'].replace(' ', '_')}_{args.seed}.mp4"

    def hook(obs, info):
        agent.reset(env)

    stats = rollout_video(env, agent.act, out, spec["name"], seed=args.seed, camera=args.camera, on_reset=hook)
    print(stats)


if __name__ == "__main__":
    main()
