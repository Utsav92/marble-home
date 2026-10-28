"""Patch action_mode into the two v2 CNN checkpoints (trainer bug: it was not recorded, so eval ran them with absolute actions)
and re-evaluate. Training is unaffected: the datasets were already ee_delta. Writes results/experiments/images_v2.json.
"""
from __future__ import annotations

import json

import torch

from marble_home.evaluation.metrics import summarize
from marble_home.evaluation.rollout import evaluate
from marble_home.utils import provenance
from marble_home.utils.paths import RESULTS


def main():
    old = json.load(open(RESULTS / "experiments" / "images_v2.json"))
    rows = []
    for tag, name in (("none", "bc_cnn_v2_none_n150_s0"), ("visual", "bc_cnn_v2_visual_n150_s0")):
        p = f"checkpoints/{name}.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        ck["meta"]["action_mode"] = "ee_delta"
        torch.save(ck, p)
        prev = [r for r in old["rows"] if r["train_dr"] == tag]
        for test in ("none", "visual"):
            eps = evaluate({"type": "checkpoint", "path": p, "name": name}, "pick_up_cup", list(range(900000, 900030)), workers=1, dr=test)
            sm = summarize(eps)
            pv = next(r for r in prev if r["test_dr"] == test)
            rows.append({"train_dr": tag, "test_dr": test, "success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"],
                         "val_mse": pv["val_mse"], "train_s": pv["train_s"], "epochs": pv["epochs"]})
            print(f"CNN(fixed) train={tag} test={test}: {sm['success_rate']:.1%}", flush=True)
    provenance.write_json(RESULTS / "experiments" / "images_v2.json",
                          {"task": "pick_up_cup", "demos": old["demos"], "rows": rows,
                           "note": "first evaluation (0% everywhere) ran the delta-trained CNNs with absolute actions because train_bc_images did not record "
                                   "action_mode; checkpoints patched and re-evaluated, trainer fixed", "manifest": provenance.run_manifest()})


if __name__ == "__main__":
    main()
