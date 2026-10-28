"""Multi-item household tasks: scripted-expert demos -> behaviour cloning (delta actions) -> held-out evaluation.

    python scripts/run_household_tasks.py clear_table load_dishwasher laundry_basket --demos 100 --workers 2

Per task (resumable; every stage skips if its output exists):
  1. collect scripted-expert demos (privileged state; successful episodes only)   data/demonstrations/<task>/
  2. prepare (merge, split, stats) and relabel absolute -> ee_delta               data/demonstrations/<task>_v2delta/
  3. train BC-MLP on the delta dataset, evaluate it AND the scripted expert on the SAME held-out seeds (900000+)
Results -> results/experiments/household_tasks.json (rows appended per task). Items are rigid-body proxies; see simulation/layouts.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from run_experiments import save, train_eval_bc

from marble_home.evaluation.metrics import summarize
from marble_home.evaluation.rollout import evaluate
from marble_home.utils import provenance
from marble_home.utils.paths import DATA, RESULTS, ROOT

PY = sys.executable


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.check_call([PY, *map(str, args)])


def prep(task, demos, workers):
    ds = f"{task}_v2delta"
    if (DATA / "splits" / f"{ds}_stats.json").exists():
        return ds
    if not (DATA / "demonstrations" / task / "dataset.hdf5").exists():
        run(ROOT / "scripts" / "collect_demos.py", "--task", task, "--episodes", demos, "--workers", workers, "--seed", 1000, "--dataset", task)
        run(ROOT / "scripts" / "prepare_dataset.py", "--task", task, "--dataset", task, "--verify", 2)
    run(ROOT / "scripts" / "relabel_delta.py", "--src", task, "--dst", ds)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--demos", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expert-eval", action="store_true", help="also evaluate the scripted expert on the held-out seeds")
    a = ap.parse_args()
    out = RESULTS / "experiments" / "household_tasks.json"
    import json
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    seeds = list(range(900000, 900000 + a.episodes))
    for task in a.tasks:
        ds = prep(task, a.demos, a.workers)
        if a.expert_eval:
            eps = evaluate({"type": "scripted"}, task, seeds, workers=a.workers)
            sm = summarize(eps)
            rows = [r for r in rows if not (r["task"] == task and r["policy"] == "scripted_expert")]
            rows.append({"task": task, "policy": "scripted_expert", "success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"],
                         "items_done_mean": sum(e["items_done"] for e in eps) / len(eps), "steps_mean": sm["steps"]})
            print(f"{task} expert: {sm['success_rate']:.1%}", flush=True)
            save("household_tasks", {"rows": rows})
        res, sm, eps, t = train_eval_bc(task, a.demos, a.seed, a.workers, a.episodes, dataset=ds, tag=f"_hh_s{a.seed}",
                                        extra={"run_name": f"bc_mlp_{task}_n{a.demos}_s{a.seed}"})
        rows = [r for r in rows if not (r["task"] == task and r["policy"] == "bc_mlp" and r.get("seed") == a.seed)]
        rows.append({"task": task, "policy": "bc_mlp", "seed": a.seed, "n_demos": a.demos, "success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"],
                     "items_done_mean": sum(e["items_done"] for e in eps) / len(eps), "steps_mean": sm["steps"], "val_mse": res["best_val_mse"],
                     "train_s": t, "ckpt": res["ckpt"]})
        print(f"{task} BC: {sm['success_rate']:.1%} items {rows[-1]['items_done_mean']:.2f}", flush=True)
        save("household_tasks", {"rows": rows})


if __name__ == "__main__":
    main()
