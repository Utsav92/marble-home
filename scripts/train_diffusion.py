"""Train a diffusion policy and evaluate it on held-out instance seeds.

    python scripts/train_diffusion.py --config configs/diffusion.yaml [--set epochs=20]
"""
from __future__ import annotations

import argparse

from marble_home.evaluation.metrics import summarize
from marble_home.evaluation.rollout import evaluate
from marble_home.training.train_diffusion import train_diffusion
from marble_home.utils import provenance
from marble_home.utils.cli import load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/diffusion.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.set)
    res = train_diffusion(cfg)
    print("trained:", res)
    if not args.no_eval and cfg.get("eval_episodes"):
        seeds = list(range(900000, 900000 + cfg["eval_episodes"]))
        eps = evaluate({"type": "checkpoint", "path": res["ckpt"], "name": res["name"], "ddim_steps": cfg["ddim_steps"]},
                       cfg["task"], seeds, workers=args.workers)
        summ = summarize(eps)
        provenance.write_json(f"{res['run_dir']}/eval.json", {"summary": summ, "episodes": eps, "seeds": seeds})
        lo, hi = summ["success_ci95"]
        print(f"held-out success {summ['success_rate']:.2%} (95% CI {lo:.2%}-{hi:.2%}, n={summ['n']}) "
              f"latency {summ['latency_ms']:.1f} ms")


if __name__ == "__main__":
    main()
