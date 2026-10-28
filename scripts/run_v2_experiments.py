"""Re-run the remaining experiments on controller v2 with delta actions (all results -> results/experiments/*_v2.json).

    python scripts/run_v2_experiments.py data_scaling | ddim | proprio | dr | images | curriculum
Protocol is identical to run_experiments.py (held-out instance seeds 900000+, Wilson CIs); only the controller / action space differ.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from run_experiments import OUT, load_cfg, save, train_eval_bc

from marble_home.evaluation.metrics import summarize
from marble_home.evaluation.rollout import evaluate
from marble_home.utils.paths import DATA, RESULTS, ROOT

DS = "pick_up_cup_v2delta"
TASK = "pick_up_cup"


def seeds_of(n):
    return list(range(900000, 900000 + n))


def data_scaling(a):
    rows = []
    for n in a.sizes:
        for s in a.seeds:
            res, sm, _, t = train_eval_bc(TASK, n, s, a.workers, a.episodes, dataset=DS, tag="_v2ds")
            rows.append({"n_demos": n, "seed": s, "success": sm["success_rate"], "ci95": sm["success_ci95"], "val_mse": res["best_val_mse"],
                         "train_s": t, "episodes": sm["n"], "run": res["name"]})
            print(f"n={n:4d} seed={s}: {sm['success_rate']:.1%} val_mse {res['best_val_mse']:.5f}", flush=True)
            save("data_scaling_v2", {"task": TASK, "dataset": DS, "rows": rows})


def ddim(a):
    rows = []
    for k in a.steps:
        eps = evaluate({"type": "checkpoint", "path": a.ckpt, "name": f"ddim{k}", "ddim_steps": k}, TASK, seeds_of(a.episodes), workers=a.workers)
        s = summarize(eps)
        rows.append({"ddim_steps": k, "success": s["success_rate"], "ci95": s["success_ci95"], "steps_mean": s["steps"], "joint_jerk": s["joint_jerk"]})
        print(f"ddim={k}: {s['success_rate']:.1%}", flush=True)
        save("ddim_steps_v2", {"task": TASK, "ckpt": a.ckpt, "rows": rows, "note": "latency deliberately omitted (CPU shared); see latency_v2.json"})


def proprio(a):
    rows = []
    for sd in a.seeds:
        res, sm, _, _ = train_eval_bc(TASK, None, sd, a.workers, a.episodes, dataset=DS, tag="_v2prop",
                                      extra={"obs_key": "proprio", "run_name": f"bc_mlp_v2_proprio_only_s{sd}"})
        rows.append({"seed": sd, "success": sm["success_rate"], "ci95": sm["success_ci95"], "val_mse": res["best_val_mse"], "n": sm["n"]})
        print(f"proprio-only seed={sd}: {sm['success_rate']:.1%}", flush=True)
        save("proprio_only_v2", {"task": TASK, "rows": rows, "note": "56 proprio + 2 goal dims, NO cup pose; v2 controller, delta actions"})


def _prep(dr, n, workers):
    raw, ds = f"pick_dr_{dr}_v2raw", f"pick_dr_{dr}_v2"
    if (DATA / "splits" / f"{ds}_stats.json").exists():
        return ds
    if not (DATA / "demonstrations" / raw / "dataset.hdf5").exists():
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "collect_demos.py"), "--task", TASK, "--episodes", str(n), "--workers", str(workers),
                               "--seed", "70000", "--dr", dr, "--dataset", raw])
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "prepare_dataset.py"), "--task", TASK, "--dataset", raw, "--verify", "0"])
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "relabel_delta.py"), "--src", raw, "--dst", ds])
    return ds


def dr(a):
    conds = ["none", "physics", "sensor", "combined"]
    dsets = {"none": DS}
    for c in conds[1:]:
        dsets[c] = _prep(c, a.n_demos, a.workers)
    rows = []
    for tc in conds:
        for sd in a.seeds:
            res, _, _, _ = train_eval_bc(TASK, None, sd, a.workers, 1, dataset=dsets[tc], tag="_v2dr", extra={"run_name": f"bc_mlp_v2dr_{tc}_s{sd}"})
            for te in conds:
                eps = evaluate({"type": "checkpoint", "path": res["ckpt"], "name": res["name"]}, TASK, seeds_of(a.episodes), workers=a.workers, dr=te)
                sm = summarize(eps)
                rows.append({"train": tc, "test": te, "seed": sd, "success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"]})
                print(f"train={tc:9s} seed={sd} test={te:9s}: {sm['success_rate']:.1%}", flush=True)
                save("domain_randomization_v2", {"task": TASK, "rows": rows, "n_demos": a.n_demos,
                                                 "note": "visual randomisation only affects pixels -> see image experiment"})


def images(a):
    from marble_home.training.train_bc import train_bc

    for name, d in (("pick_v2_rgb64", None), ("pick_v2_rgb64_visdr", "visual")):
        if (DATA / "splits" / f"{name}_stats.json").exists():
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "render_dataset.py"), "--task", TASK, "--src-dataset", DS, "--obs-mode", "rgb", "--hw", "64",
               "--n-demos", str(a.n_demos), "--workers", str(a.workers), "--out", name]
        if d:
            cmd += ["--dr", d]
        subprocess.check_call(cmd)
    rows = []
    for name, tag in (("pick_v2_rgb64", "none"), ("pick_v2_rgb64_visdr", "visual")):
        cfg = load_cfg("configs/bc_cnn.yaml")
        cfg.update(task=TASK, dataset=name, seed=0, run_name=f"bc_cnn_v2_{tag}_n{a.n_demos}_s0", epochs=a.epochs)
        t0 = time.time()
        res = train_bc(cfg)
        for test in ("none", "visual"):
            eps = evaluate({"type": "checkpoint", "path": res["ckpt"], "name": res["name"]}, TASK, seeds_of(a.episodes), workers=a.workers, dr=test)
            sm = summarize(eps)
            rows.append({"train_dr": tag, "test_dr": test, "success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"],
                         "val_mse": res["best_val_mse"], "train_s": time.time() - t0, "epochs": res["epochs"]})
            print(f"CNN train={tag} test={test}: {sm['success_rate']:.1%}", flush=True)
            save("images_v2", {"task": TASK, "demos": a.n_demos, "rows": rows})


def curriculum(a):
    from marble_home.training.train_rl import train_ppo

    rows = []
    for mode in ("fixed", "manual", "adaptive"):
        for sd in a.seeds:
            cfg = load_cfg("configs/rl.yaml")
            cfg.update(task="reach_cup", action_mode="ee_delta", total_timesteps=a.steps, n_envs=3, seed=sd,
                       curriculum=None if mode == "fixed" else mode, run_name=f"ppo_v2_reach_cur_{mode}_s{sd}")
            t0 = time.time()
            res = train_ppo(cfg)
            eps = evaluate({"type": "ppo", "path": res["ckpt"], "name": res["name"]}, "reach_cup", seeds_of(a.episodes), workers=a.workers)
            sm = summarize(eps)
            f = RESULTS / "curriculum" / f"{res['name']}.json"
            tl = json.load(open(f))["transitions"] if f.exists() else None
            rows.append({"mode": mode, "seed": sd, "success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"],
                         "collision_free": sm["collision_free"], "wall_s": time.time() - t0, "transitions": tl})
            print(f"curriculum={mode} seed={sd}: {sm['success_rate']:.1%}", flush=True)
            save("curriculum_v2", {"rows": rows, "total_steps_each": a.steps})


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="exp", required=True)
    for n in ("data_scaling", "ddim", "proprio", "dr", "images", "curriculum"):
        s = sp.add_parser(n)
        s.add_argument("--episodes", type=int, default=30)
        s.add_argument("--workers", type=int, default=3)
        s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        s.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50, 100, 240])
        s.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 10, 20])
        s.add_argument("--ckpt", default="checkpoints/diffusion_v2d_pick_up_cup_s0.pt")
        s.add_argument("--n-demos", type=int, default=150)
        s.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    if a.exp == "curriculum":
        a.steps = 30000 if a.steps == [1, 2, 5, 10, 20] else a.steps[0]
    globals()[a.exp.replace("-", "_")](a)


if __name__ == "__main__":
    main()
