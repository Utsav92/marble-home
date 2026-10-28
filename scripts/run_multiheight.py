"""Counter-height generalisation for a multi-item task: does a policy trained at one height (0.85 m) transfer, and does training across heights fix it?

    python scripts/run_multiheight.py probe  --task load_dishwasher            # scripted expert at each height
    python scripts/run_multiheight.py collect --task load_dishwasher            # demos at the training heights
    python scripts/run_multiheight.py eval    --task load_dishwasher            # single-height vs multi-height BC, train + held-out heights

Scenes are the default kitchen with only ``counter_height`` changed (the task layout follows the height, see simulation/layouts.py).
Results -> results/experiments/multiheight_<task>.json
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp

import numpy as np

from run_experiments import save, train_eval_bc

from marble_home.datasets.collect import collect
from marble_home.evaluation.rollout import evaluate_scenes
from marble_home.simulation.kitchen_arena import DEFAULT_SCENE
from marble_home.utils.paths import DATA, RESULTS

TRAIN_H = [0.82, 0.85, 0.88, 0.91]
TEST_H = [0.835, 0.865, 0.90]  # interleaved + one beyond the middle of the training span
PROBE_H = sorted(TRAIN_H + TEST_H + [0.79, 0.94])


def scene_at(h):
    sc = copy.deepcopy(DEFAULT_SCENE)
    sc.update(scene_id=f"h{int(round(h * 1000)):04d}", counter_height=float(h))
    return sc


def _collect_one(args):
    task, sc, n, out, seed0 = args
    return collect(task, n, out, seed0=seed0, noise=0.004, scene=sc, log=lambda m: None)


def probe(a):
    seeds = list(range(940000, 940000 + a.n))
    res = evaluate_scenes({"type": "scripted"}, a.task, [scene_at(h) for h in PROBE_H], seeds, workers=a.workers)
    rows = [{"counter_height": h, "expert_success": float(np.mean([e["success"] for e in res[scene_at(h)["scene_id"]]])), "n": a.n} for h in PROBE_H]
    for r in rows:
        print(f"h={r['counter_height']:.3f} expert {r['expert_success']:.0%}", flush=True)
    save(f"multiheight_probe_{a.task}", {"task": a.task, "rows": rows})


def collect_all(a):
    jobs = [(a.task, scene_at(h), a.per_height, str(DATA / "demonstrations" / f"{a.task}_mh" / scene_at(h)["scene_id"] / "shard_0.hdf5"), 60_000 + i * 1000)
            for i, h in enumerate(TRAIN_H) if a.only is None or abs(h - a.only) < 1e-9]
    with mp.get_context("spawn").Pool(a.workers) as p:
        res = p.map(_collect_one, jobs)
    print("collected", sum(r["kept"] for r in res), "demos at", TRAIN_H, flush=True)


def evaluate(a):
    """Single-height BC (the household_tasks checkpoint, trained at 0.85 m) vs multi-height BC, on train and held-out heights."""
    ds = f"{a.task}_mh_v2delta"
    if not (DATA / "splits" / f"{ds}_stats.json").exists():
        raise SystemExit(f"run collect, prepare_dataset and relabel_delta first (missing {ds})")
    from marble_home.evaluation.metrics import summarize

    rows = json.loads((RESULTS / "experiments" / "household_tasks.json").read_text())["rows"]
    single = next(r["ckpt"] for r in rows if r["task"] == a.task and r["policy"] == "bc_mlp" and r.get("seed") == 0)
    res, _, _, train_s = train_eval_bc(a.task, None, 0, a.workers, 1, dataset=ds, tag="_multiheight", extra={"run_name": f"bc_mlp_{a.task}_multiheight"})
    specs = {"bc_single_height": {"type": "checkpoint", "path": single, "name": "bc_single_height"},
             "bc_multi_height": {"type": "checkpoint", "path": res["ckpt"], "name": "bc_multi_height"}}
    seeds = list(range(950000, 950000 + a.n))
    out = {"task": a.task, "train_heights": TRAIN_H, "test_heights": TEST_H, "n_per_height": a.n, "train_s_multi": train_s,
           "demos_per_height": a.per_height, "per_policy": {}}
    for pname, spec in specs.items():
        out["per_policy"][pname] = {}
        for split, hs in (("train_heights", TRAIN_H), ("test_heights", TEST_H)):
            r = evaluate_scenes(spec, a.task, [scene_at(h) for h in hs], seeds, workers=a.workers)
            flat = [e for v in r.values() for e in v]
            sm = summarize(flat)
            out["per_policy"][pname][split] = {"success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"],
                                               "per_height": {f"{h:.3f}": float(np.mean([e["success"] for e in r[scene_at(h)["scene_id"]]])) for h in hs}}
            print(f"{pname:18s} {split:13s}: {sm['success_rate']:.1%} [{sm['success_ci95'][0]:.1%},{sm['success_ci95'][1]:.1%}] n={sm['n']}", flush=True)
            save(f"multiheight_{a.task}", out)
    save(f"multiheight_{a.task}", out)


def extrap(a):
    """Evaluate the multi-height checkpoint at heights outside the trained 0.82-0.91 m span (no new training)."""
    from marble_home.evaluation.metrics import summarize

    ckpt = next((RESULTS.parent / "checkpoints").rglob(f"bc_mlp_{a.task}_multiheight.pt"))
    hs = [0.79, 0.94]
    r = evaluate_scenes({"type": "checkpoint", "path": str(ckpt), "name": "bc_multi_height"}, a.task, [scene_at(h) for h in hs],
                        list(range(960000, 960000 + a.n)), workers=a.workers)
    rows = []
    for h in hs:
        sm = summarize(r[scene_at(h)["scene_id"]])
        rows.append({"counter_height": h, "success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"]})
        print(f"extrap h={h:.2f}: {sm['success_rate']:.1%} [{sm['success_ci95'][0]:.1%},{sm['success_ci95'][1]:.1%}] n={sm['n']}", flush=True)
    save(f"multiheight_extrap_{a.task}", {"task": a.task, "trained_span": [min(TRAIN_H), max(TRAIN_H)], "rows": rows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["probe", "collect", "eval", "extrap"])
    ap.add_argument("--task", default="load_dishwasher")
    ap.add_argument("--n", type=int, default=15, help="episodes per height")
    ap.add_argument("--per-height", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", type=float, default=None, help="collect just this training height (resume after an interrupted run)")
    a = ap.parse_args()
    {"probe": probe, "collect": collect_all, "eval": evaluate, "extrap": extrap}[a.stage](a)
