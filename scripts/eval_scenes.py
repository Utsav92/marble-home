"""Evaluate a checkpoint on the procedural scene splits (train / val / held-out test scenes).

    python scripts/eval_scenes.py --policy checkpoints/bc_mlp_scenes_pick_v2delta_multiscene_nogoal.pt --name nogoal
Same scenes and instance seeds as `run_experiments.py scenes_eval` (930000+), so results are directly comparable.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from marble_home.evaluation.metrics import summarize
from marble_home.evaluation.rollout import evaluate_scenes
from marble_home.simulation.scene_gen import split_scenes
from marble_home.utils import provenance
from marble_home.utils.paths import RESULTS, ensure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", default="pick_up_cup")
    ap.add_argument("--per-scene", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=5)
    a = ap.parse_args()
    sp = split_scenes(a.n_train, a.n_val, a.n_test, seed=0)
    seeds = list(range(930000, 930000 + a.per_scene))
    spec = {"type": "checkpoint", "path": a.policy, "name": a.name}
    out = {"task": a.task, "policy": a.policy, "per_split": {}}
    for split in ("train", "val", "test"):
        r = evaluate_scenes(spec, a.task, sp[split], seeds, workers=a.workers)
        flat = [e for v in r.values() for e in v]
        sm = summarize(flat)
        out["per_split"][split] = {"success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"],
                                   "per_scene": {sid: float(np.mean([e["success"] for e in v])) for sid, v in r.items()}}
        print(f"{a.name:14s} {split:5s}: {sm['success_rate']:.1%} [{sm['success_ci95'][0]:.1%},{sm['success_ci95'][1]:.1%}] n={sm['n']}", flush=True)
    out["gap_train_minus_test"] = out["per_split"]["train"]["success"] - out["per_split"]["test"]["success"]
    ensure(RESULTS / "experiments")
    provenance.write_json(RESULTS / "experiments" / f"scene_eval_{a.name}.json", {**out, "manifest": provenance.run_manifest()})


if __name__ == "__main__":
    main()
