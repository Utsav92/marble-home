"""One CNN (train_dr, test_dr) condition per process, saved immediately (survives an OOM kill of the next one).

    python scripts/reeval_cnn_one.py --train none --test visual
Patches action_mode=ee_delta into the checkpoint if missing (trainer bug, now fixed) and writes
results/experiments/images_v2_<train>_<test>.json.
"""
from __future__ import annotations

import argparse

import torch

from marble_home.evaluation.metrics import summarize
from marble_home.evaluation.rollout import evaluate
from marble_home.utils import provenance
from marble_home.utils.paths import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=["none", "visual"], required=True)
    ap.add_argument("--test", choices=["none", "visual"], required=True)
    a = ap.parse_args()
    name = f"bc_cnn_v2_{a.train}_n150_s0"
    p = f"checkpoints/{name}.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if ck["meta"].get("action_mode") != "ee_delta":
        ck["meta"]["action_mode"] = "ee_delta"
        torch.save(ck, p)
    eps = evaluate({"type": "checkpoint", "path": p, "name": name}, "pick_up_cup", list(range(900000, 900030)), workers=1, dr=a.test)
    sm = summarize(eps)
    print(f"CNN(fixed) train={a.train} test={a.test}: {sm['success_rate']:.1%}", flush=True)
    provenance.write_json(RESULTS / "experiments" / f"images_v2_{a.train}_{a.test}.json",
                          {"task": "pick_up_cup", "train_dr": a.train, "test_dr": a.test, "success": sm["success_rate"], "ci95": sm["success_ci95"],
                           "n": sm["n"], "ckpt": p, "manifest": provenance.run_manifest()})


if __name__ == "__main__":
    main()
