"""Evaluate one policy on held-out seeded instances and write per-episode records + summary.

    python scripts/evaluate.py --policy scripted --task pick_up_cup --episodes 30
    python scripts/evaluate.py --policy checkpoints/bc_mlp_pick_up_cup_n240_s0.pt --task pick_up_cup --episodes 30
    python scripts/evaluate.py --policy random --task reach_cup --episodes 20 --ddim-steps 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from marble_home.evaluation.metrics import summarize
from marble_home.evaluation.rollout import evaluate
from marble_home.utils import provenance
from marble_home.utils.paths import RESULTS, ensure


def spec_from_arg(p: str, ddim_steps=None, name=None) -> dict:
    if p in ("random", "scripted"):
        return {"type": p, "name": name or p}
    s = {"type": "checkpoint", "path": p, "name": name or Path(p).stem}
    if ddim_steps:
        s["ddim_steps"] = ddim_steps
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="random | scripted | path/to/checkpoint.pt")
    ap.add_argument("--task", default="pick_up_cup")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=900000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ddim-steps", type=int, default=None)
    ap.add_argument("--obs-mode", default="state")
    ap.add_argument("--action-mode", default=None, help="ee_abs | ee_delta (default: whatever the checkpoint/agent implies)")
    ap.add_argument("--dr", default=None, help="domain randomisation preset for the TEST environment")
    ap.add_argument("--tag", default="")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    spec = spec_from_arg(args.policy, args.ddim_steps, args.name)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    t0 = time.time()
    kw = {"obs_mode": args.obs_mode}
    if args.action_mode:
        kw["action_mode"] = args.action_mode
    if args.dr:
        kw["dr"] = args.dr
    eps = evaluate(spec, args.task, seeds, workers=args.workers, **kw)
    summ = summarize(eps)
    out = RESULTS / "eval"
    ensure(out)
    tag = f"_{args.tag}" if args.tag else ""
    path = out / f"{spec['name']}{tag}__{args.task}.json"
    provenance.write_json(path, {"spec": spec, "task": args.task, "seeds": seeds, "summary": summ, "episodes": eps,
                                 "wall_s": time.time() - t0, "manifest": provenance.run_manifest()})
    lo, hi = summ["success_ci95"]
    print(f"{spec['name']} on {args.task}: success {summ['success_rate']:.2%} (95% CI {lo:.2%}-{hi:.2%}, n={summ['n']}) "
          f"grasp {summ['grasp_success']:.2%} place_err {summ['place_error']:.3f} m latency {summ['latency_ms']:.2f} ms -> {path}")


if __name__ == "__main__":
    main()
