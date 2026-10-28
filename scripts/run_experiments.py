"""Performance-optimisation experiments (spec §14). Each writes results/experiments/<name>.json (machine-readable).

    python scripts/run_experiments.py data_scaling --sizes 10 25 50 100 240 --seeds 0 1 2
    python scripts/run_experiments.py ddim --ckpt checkpoints/diffusion_pick_up_cup_n240_s0.pt --steps 1 2 5 10 20
    python scripts/run_experiments.py throughput
Protocol per condition: change ONE variable, fixed config otherwise, identical held-out instance seeds (900000+),
mean +- Wilson CI, wall-clock cost recorded.
"""
from __future__ import annotations

import argparse
import time

from marble_home.policies.agents import load_checkpoint

import numpy as np

from marble_home.evaluation.metrics import summarize
from marble_home.evaluation.rollout import evaluate
from marble_home.utils import provenance
from marble_home.utils.cli import load_cfg
from marble_home.utils.paths import RESULTS, ensure

OUT = RESULTS / "experiments"


def save(name, obj):
    ensure(OUT)
    obj["manifest"] = provenance.run_manifest()
    provenance.write_json(OUT / f"{name}.json", obj)
    print("wrote", OUT / f"{name}.json")


def train_eval_bc(task, n_demos, seed, workers, episodes, dataset=None, extra=None, tag="", eval_kwargs=None):
    from marble_home.training.train_bc import train_bc

    cfg = load_cfg("configs/bc.yaml")
    over = dict(task=task, n_demos=n_demos, seed=seed, dataset=dataset, run_name=f"bc_mlp_{dataset or task}_n{n_demos}_s{seed}{tag}")
    over.update(extra or {})  # explicit overrides win (avoids a duplicate run_name keyword)
    cfg.update(over)
    t0 = time.time()
    res = train_bc(cfg)
    train_s = time.time() - t0
    seeds = list(range(900000, 900000 + episodes))
    eps = evaluate({"type": "checkpoint", "path": res["ckpt"], "name": res["name"]}, task, seeds, workers=workers, **(eval_kwargs or {}))
    return res, summarize(eps), eps, train_s


def exp_data_scaling(a):
    rows = []
    for n in a.sizes:
        for s in a.seeds:
            res, summ, eps, train_s = train_eval_bc(a.task, n, s, a.workers, a.episodes)
            rows.append({"n_demos": n, "seed": s, "success": summ["success_rate"], "ci95": summ["success_ci95"], "val_mse": res["best_val_mse"],
                         "train_s": train_s, "env_steps_of_demos": None, "episodes": summ["n"], "run": res["name"]})
            print(f"n={n:4d} seed={s}: success {summ['success_rate']:.1%} val_mse {res['best_val_mse']:.5f} train {train_s:.0f}s", flush=True)
            save("data_scaling", {"task": a.task, "rows": rows})
    save("data_scaling", {"task": a.task, "rows": rows})


def exp_ddim(a):
    seeds = list(range(900000, 900000 + a.episodes))
    rows = []
    for k in a.steps:
        eps = evaluate({"type": "checkpoint", "path": a.ckpt, "name": f"ddim{k}", "ddim_steps": k}, a.task, seeds, workers=a.workers)
        s = summarize(eps)
        rows.append({"ddim_steps": k, "success": s["success_rate"], "ci95": s["success_ci95"], "latency_ms_per_env_step": s["latency_ms"],
                     "steps_mean": s["steps"], "joint_jerk": s["joint_jerk"]})
        print(f"ddim={k:3d}: success {s['success_rate']:.1%} latency {s['latency_ms']:.1f} ms/step", flush=True)
    save("ddim_steps", {"task": a.task, "ckpt": a.ckpt, "rows": rows})


def exp_throughput(a):
    """Where does time go? physics-only vs +controller vs +camera vs +policy, per env step (single process)."""
    import torch
    from marble_home.simulation.gym_env import HouseholdEnv

    out = {"rows": []}
    z = np.zeros(4, np.float32)
    for name, kw in [("state obs (no camera)", dict(obs_mode="state")), ("rgb 64x64", dict(obs_mode="rgb", image_hw=64)),
                     ("rgb 96x96", dict(obs_mode="rgb", image_hw=96)), ("rgbd 64x64", dict(obs_mode="rgbd", image_hw=64)),
                     ("multicam 64x64 (3 cams + depth)", dict(obs_mode="multicam", image_hw=64))]:
        env = HouseholdEnv(task="pick_up_cup", **kw)
        env.reset(seed=0)
        for _ in range(10):
            env.step(z)
        t = time.perf_counter()
        n = 60
        for _ in range(n):
            env.step(z)
        dt = (time.perf_counter() - t) / n
        out["rows"].append({"config": name, "ms_per_env_step": dt * 1000, "env_steps_per_s": 1 / dt})
        print(f"{name:34s} {dt * 1000:7.1f} ms/step  {1 / dt:6.1f} steps/s", flush=True)
        env.close()
    env = HouseholdEnv(task="pick_up_cup", obs_mode="state")
    env.reset(seed=0)
    e = env.env
    t = time.perf_counter()
    for _ in range(200):
        e.sim.step()
    raw = (time.perf_counter() - t) / 200
    out["mujoco_substep_ms"] = raw * 1000
    out["substeps_per_control_step"] = int(round(e.control_timestep / e.model_timestep))
    print(f"raw mj_step: {raw * 1000:.3f} ms x {out['substeps_per_control_step']} substeps/control step", flush=True)
    m, _ = load_checkpoint(a.ckpt)
    x = torch.randn(1, 69)
    for _ in range(5):
        m.act(x)
    t = time.perf_counter()
    for _ in range(100):
        m.act(x)
    out["bc_inference_ms"] = (time.perf_counter() - t) / 100 * 1000
    print(f"BC-MLP inference {out['bc_inference_ms']:.2f} ms", flush=True)
    try:
        dm, _ = load_checkpoint(a.diffusion_ckpt)
        xo = torch.randn(1, dm.n_obs, 69)
        out["diffusion_latency_ms_per_call"] = {}
        for k in (1, 2, 5, 10, 20):
            for _ in range(3):
                dm.sample(xo, ddim_steps=k)
            t = time.perf_counter()
            for _ in range(20):
                dm.sample(xo, ddim_steps=k)
            ms = (time.perf_counter() - t) / 20 * 1000
            out["diffusion_latency_ms_per_call"][k] = {"per_plan_ms": ms, "amortised_per_env_step_ms": ms / dm.n_act}
            print(f"diffusion DDIM-{k}: {ms:.1f} ms per plan, {ms / dm.n_act:.2f} ms per env step (n_act={dm.n_act})", flush=True)
    except FileNotFoundError:
        pass
    save("throughput", out)


# ------------------------------------------------------------------------------------------- scene generalisation
def _scene_splits(a):
    from marble_home.simulation.scene_gen import split_scenes

    return split_scenes(a.n_train, a.n_val, a.n_test, seed=0)


def exp_scenes_probe(a):
    """Task compatibility: can the scripted expert solve each scene? (drives which scenes are usable)"""
    from marble_home.evaluation.rollout import evaluate_scenes

    sp = _scene_splits(a)
    allsc = sp["train"] + sp["val"] + sp["test"]
    seeds = list(range(920000, 920000 + a.probe))
    res = evaluate_scenes({"type": "scripted"}, a.task, allsc, seeds, workers=a.workers)
    rows = []
    for split, scs in sp.items():
        for sc in scs:
            r = res[sc["scene_id"]]
            rows.append({"split": split, "scene_id": sc["scene_id"], "kind": sc["kind"], "counter_height": sc["counter_height"],
                         "expert_success": float(np.mean([e["success"] for e in r])), "n": len(r)})
            print(f"{split:5s} {sc['scene_id']:30s} h={sc['counter_height']:.3f} expert {rows[-1]['expert_success']:.0%}", flush=True)
    save("scene_probe", {"task": a.task, "rows": rows, "scenes": {k: [s["scene_id"] for s in v] for k, v in sp.items()}})


def _collect_scene(args):
    task, scene, n, out, seed0 = args
    from marble_home.datasets.collect import collect

    return collect(task, n, out, seed0=seed0, noise=0.004, scene=scene, log=lambda m: None)


def exp_scenes_collect(a):
    import multiprocessing as mp
    import subprocess
    import sys

    from marble_home.utils.paths import DATA, ROOT

    sp = _scene_splits(a)
    jobs = []
    for i, sc in enumerate(sp["train"] + sp["val"]):
        out = DATA / "demonstrations" / a.dataset / sc["scene_id"] / "shard_0.hdf5"
        jobs.append((a.task, sc, a.per_scene, str(out), 50_000 + i * 1000))
    with mp.get_context("spawn").Pool(a.workers) as p:
        res = p.map(_collect_scene, jobs)
    print("collected", sum(r["kept"] for r in res), "demos in", len(jobs), "scenes;", sum(r["steps"] for r in res), "steps")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "prepare_dataset.py"), "--task", a.task, "--dataset", a.dataset, "--verify", "4"])


def exp_scenes_eval(a):
    """Scene-level generalisation: train-scene vs held-out-scene success for multi-scene BC and a single-scene BC baseline."""
    from marble_home.evaluation.rollout import evaluate_scenes

    sp = _scene_splits(a)
    probe = {r["scene_id"]: r["expert_success"] for r in (__import__("json").load(open(OUT / "scene_probe.json"))["rows"])} if (OUT / "scene_probe.json").exists() else {}
    # multi-scene BC
    extra = {"obs_drop": a.obs_drop, "run_name": f"bc_mlp_{a.dataset}_multiscene{a.suffix}"} if a.obs_drop else {"run_name": f"bc_mlp_{a.dataset}_multiscene{a.suffix}"}
    res, summ_val, _, train_s = train_eval_bc(a.task, None, a.seed, a.workers, 6, dataset=a.dataset, tag="_multiscene", extra=extra)
    specs = {"bc_multiscene": {"type": "checkpoint", "path": res["ckpt"], "name": "bc_multiscene"}}
    if not a.only_multi:
        specs.update({"bc_singlescene": {"type": "checkpoint", "path": a.single_ckpt, "name": "bc_singlescene"},
                      "scripted_expert": {"type": "scripted"}})
    seeds = list(range(930000, 930000 + a.per_scene_eval))
    out = {"task": a.task, "train_s_multiscene": train_s, "per_policy": {}}
    for pname, spec in specs.items():
        out["per_policy"][pname] = {}
        for split in ("train", "val", "test"):
            r = evaluate_scenes(spec, a.task, sp[split], seeds, workers=a.workers)
            flat = [e for v in r.values() for e in v]
            sm = summarize(flat)
            out["per_policy"][pname][split] = {"success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"],
                                               "per_scene": {sid: float(np.mean([e["success"] for e in v])) for sid, v in r.items()}}
            print(f"{pname:16s} {split:5s}: {sm['success_rate']:.1%} [{sm['success_ci95'][0]:.1%},{sm['success_ci95'][1]:.1%}] n={sm['n']}", flush=True)
        pp = out["per_policy"][pname]
        pp["generalization_gap_train_minus_test"] = pp["train"]["success"] - pp["test"]["success"]
    out["expert_probe"] = probe
    save(f"scene_generalization{a.suffix}", out)


# ------------------------------------------------------------------------------------------- action representation
def exp_action_repr(a):
    """BC with absolute-EE-target actions (ee_abs, baseline) vs delta-EE actions (ee_delta, relabelled from the SAME demos)."""
    import json as _json
    import shutil

    from marble_home.datasets.hdf5 import DemoFile, DemoWriter
    from marble_home.robots import action_space as AS
    from marble_home.utils.paths import DATA

    src = DATA / "demonstrations" / a.task / "dataset.hdf5"
    name = f"{a.task}_delta"
    dst = DATA / "demonstrations" / name / "dataset.hdf5"
    dst.parent.mkdir(parents=True, exist_ok=True)
    clipped = total = 0
    with DemoFile(src) as df, DemoWriter(dst, a.task, _json.loads(df.attrs["scene"]), {**_json.loads(df.attrs["env_args"]), "action_mode": "ee_delta"},
                                         _json.loads(df.attrs["versions"]), _json.loads(df.attrs["robot_config"])) as w:
        for n in df.demos:
            ep = df.demo(n)
            pos = AS.WS_CENTER + ep["actions"][:, :3] * AS.WS_HALF            # absolute targets (m, base frame)
            start = ep["obs"]["proprio"][0][42:45]                               # base_to_right_eef_pos at reset (== env._target init)
            prev = np.vstack([start, pos[:-1]])
            d = (pos - prev) / AS.DELTA_SCALE
            clipped += int((np.abs(d) > 1).sum())
            total += d.size
            ep["actions"] = np.concatenate([np.clip(d, -1, 1), ep["actions"][:, 3:]], 1).astype(np.float32)
            at = dict(ep["attrs"])
            init = at.pop("init_state")
            at.pop("num_samples", None)
            ep["init_state"] = init
            w.add(ep, **at)
    shutil.copy(DATA / "splits" / f"{a.task}.json", DATA / "splits" / f"{name}.json")
    shutil.copy(DATA / "splits" / f"{a.task}_stats.json", DATA / "splits" / f"{name}_stats.json")
    print(f"relabelled to ee_delta; {clipped}/{total} delta components clipped at +-1", flush=True)
    rows = []
    for sd in a.seeds:
        res, summ, eps, train_s = train_eval_bc(a.task, None, sd, a.workers, a.episodes, dataset=name, tag="_delta",
                                                eval_kwargs={"action_mode": "ee_delta"})
        rows.append({"action_mode": "ee_delta", "seed": sd, "success": summ["success_rate"], "ci95": summ["success_ci95"],
                     "val_mse": res["best_val_mse"], "joint_jerk": summ["joint_jerk"], "action_smoothness": summ["action_smoothness"]})
        print(f"ee_delta seed={sd}: success {summ['success_rate']:.1%} jerk {summ['joint_jerk']:.2f}", flush=True)
        save("action_repr", {"task": a.task, "clipped_fraction": clipped / total, "rows": rows})
    base = [r for r in _json.load(open(OUT / "data_scaling.json"))["rows"] if r["n_demos"] == 240]
    save("action_repr", {"task": a.task, "clipped_fraction": clipped / total, "rows": rows,
                         "ee_abs_baseline_rows": [{"action_mode": "ee_abs", "seed": r["seed"], "success": r["success"], "ci95": r["ci95"], "val_mse": r["val_mse"]} for r in base],
                         "not_run": "joint-position targets and OSC operational-space commands: demos are IK-based; no joint-space expert/relabelling implemented"})


# ------------------------------------------------------------------------------------------- domain randomisation (state policies)
def exp_dr(a):
    """Train BC on demos collected under {none, physics, sensor, combined} randomisation; test every policy under every condition."""
    import subprocess
    import sys

    from marble_home.utils.paths import ROOT

    conds = ["none", "physics", "sensor", "combined"]
    from marble_home.utils.paths import DATA

    for c in conds[1:]:
        ds = f"pick_dr_{c}"
        if (DATA / "demonstrations" / ds / "dataset.hdf5").exists() and (DATA / "splits" / f"{ds}_stats.json").exists():
            print(f"dataset {ds} already prepared: skipping collection", flush=True)
            continue
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "collect_demos.py"), "--task", a.task, "--episodes", str(a.n_demos),
                               "--workers", str(a.workers), "--seed", "70000", "--dr", c, "--dataset", ds])
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "prepare_dataset.py"), "--task", a.task, "--dataset", ds, "--verify", "0"])
    seeds = list(range(900000, 900000 + a.episodes))
    rows = []
    for train_c in conds:
        for sd in a.seeds:
            ds = None if train_c == "none" else f"pick_dr_{train_c}"
            res, _, _, _ = train_eval_bc(a.task, None, sd, a.workers, 1, dataset=ds, tag=f"_dr{train_c}", extra={"run_name": f"bc_mlp_dr_{train_c}_s{sd}"})
            for test_c in conds:
                eps = evaluate({"type": "checkpoint", "path": res["ckpt"], "name": res["name"]}, a.task, seeds, workers=a.workers, dr=test_c)
                sm = summarize(eps)
                rows.append({"train": train_c, "test": test_c, "seed": sd, "success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"]})
                print(f"train={train_c:9s} seed={sd} test={test_c:9s}: {sm['success_rate']:.1%}", flush=True)
                save("domain_randomization", {"task": a.task, "rows": rows,
                                              "note": "visual randomisation only affects pixels -> not applicable to state policies (see image experiment)"})


# ------------------------------------------------------------------------------------------- image policies + visual randomisation
def exp_images(a):
    """CNN-BC (RGB head camera + proprio) trained on demos rendered without / with VISUAL randomisation; tested under both."""
    import subprocess
    import sys

    from marble_home.training.train_bc import train_bc
    from marble_home.utils.paths import ROOT

    for name, dr in (("pick_up_cup_rgb64", None), ("pick_up_cup_rgb64_visdr", "visual")):
        cmd = [sys.executable, str(ROOT / "scripts" / "render_dataset.py"), "--task", a.task, "--obs-mode", "rgb", "--hw", "64",
               "--n-demos", str(a.n_demos), "--workers", str(a.workers), "--out", name]
        if dr:
            cmd += ["--dr", dr]
        subprocess.check_call(cmd)
    seeds = list(range(900000, 900000 + a.episodes))
    rows = []
    for name, tag in (("pick_up_cup_rgb64", "none"), ("pick_up_cup_rgb64_visdr", "visual")):
        cfg = load_cfg("configs/bc_cnn.yaml")
        cfg.update(task=a.task, dataset=name, seed=0, run_name=f"bc_cnn_rgb_{tag}_n{a.n_demos}_s0", epochs=a.epochs)
        t0 = time.time()
        res = train_bc(cfg)
        train_s = time.time() - t0
        for test in ("none", "visual"):
            eps = evaluate({"type": "checkpoint", "path": res["ckpt"], "name": res["name"]}, a.task, seeds, workers=a.workers, dr=test)
            sm = summarize(eps)
            rows.append({"train_dr": tag, "test_dr": test, "success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"],
                         "val_mse": res["best_val_mse"], "train_s": train_s, "epochs": res["epochs"]})
            print(f"CNN train_dr={tag:6s} test_dr={test:6s}: {sm['success_rate']:.1%} [{sm['success_ci95'][0]:.1%},{sm['success_ci95'][1]:.1%}]", flush=True)
            save("images", {"task": a.task, "demos": a.n_demos, "rows": rows,
                            "state_bc_same_demos_reference": "bc_mlp_pick_up_cup_n100_s{0,1,2} in data_scaling.json (state obs, privileged cup pose)"})


# ------------------------------------------------------------------------------------------- curriculum
def exp_curriculum(a):
    """PPO reach_cup (ee_delta): fixed difficulty vs manually staged vs adaptive CI-based curriculum. Same total budget."""
    from marble_home.training.train_rl import train_ppo

    seeds_eval = list(range(900000, 900000 + a.episodes))
    rows = []
    for mode in ("fixed", "manual", "adaptive"):
        for sd in a.seeds:
            cfg = load_cfg("configs/rl.yaml")
            cfg.update(task="reach_cup", action_mode="ee_delta", total_timesteps=a.steps, n_envs=3, seed=sd,
                       curriculum=None if mode == "fixed" else mode, run_name=f"ppo_reach_cur_{mode}_s{sd}")
            t0 = time.time()
            res = train_ppo(cfg)
            eps = evaluate({"type": "ppo", "path": res["ckpt"], "name": res["name"]}, "reach_cup", seeds_eval, workers=a.workers)
            sm = summarize(eps)
            from marble_home.utils.paths import RESULTS as R
            tl = None
            f = R / "curriculum" / f"{res['name']}.json"
            if f.exists():
                tl = __import__("json").load(open(f))["transitions"]
            rows.append({"mode": mode, "seed": sd, "success": sm["success_rate"], "ci95": sm["success_ci95"], "n": sm["n"],
                         "collision_free": sm["collision_free"], "wall_s": time.time() - t0, "transitions": tl})
            print(f"curriculum={mode:8s} seed={sd}: held-out {sm['success_rate']:.1%} transitions={[(t['step'], t['event']) for t in (tl or [])]}", flush=True)
            save("curriculum", {"rows": rows, "total_steps_each": a.steps,
                                "note": "stage 1 = fixed cup position, stage 2 = full randomised spawn; evaluation always on the stage-2 distribution. "
                                        "Stages 3-7 of the spec are not exercised here (see curriculum.STAGES for why 5 and 7 are unavailable)."})


def exp_proprio(a):
    """Control for the image policies: BC with proprioception only (no cup information of any kind)."""
    rows = []
    for sd in a.seeds:
        res, summ, eps, train_s = train_eval_bc(a.task, None, sd, a.workers, a.episodes, extra={"obs_key": "proprio", "run_name": f"bc_mlp_proprio_only_s{sd}"})
        rows.append({"seed": sd, "success": summ["success_rate"], "ci95": summ["success_ci95"], "val_mse": res["best_val_mse"], "n": summ["n"]})
        print(f"proprio-only seed={sd}: {summ['success_rate']:.1%} [{summ['success_ci95'][0]:.1%},{summ['success_ci95'][1]:.1%}] val_mse {res['best_val_mse']:.5f}", flush=True)
        save("proprio_only", {"task": a.task, "rows": rows, "note": "obs = 56 proprio + 2 goal dims; NO cup pose. Blind baseline for the image policies."})


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="exp", required=True)
    s = sp.add_parser("data_scaling")
    s.add_argument("--task", default="pick_up_cup")
    s.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50, 100, 240])
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--episodes", type=int, default=30)
    s.add_argument("--workers", type=int, default=3)
    s.set_defaults(fn=exp_data_scaling)
    d = sp.add_parser("ddim")
    d.add_argument("--task", default="pick_up_cup")
    d.add_argument("--ckpt", default="checkpoints/diffusion_pick_up_cup_n240_s0.pt")
    d.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    d.add_argument("--episodes", type=int, default=30)
    d.add_argument("--workers", type=int, default=3)
    d.set_defaults(fn=exp_ddim)
    t = sp.add_parser("throughput")
    t.add_argument("--ckpt", default="checkpoints/bc_mlp_pick_up_cup_n240_s0.pt")
    t.add_argument("--diffusion-ckpt", default="checkpoints/diffusion_pick_up_cup_n240_s0.pt")
    t.set_defaults(fn=exp_throughput)

    g = sp.add_parser("scenes_probe")
    g.add_argument("--task", default="pick_up_cup")
    g.add_argument("--probe", type=int, default=6)
    g.add_argument("--n-train", type=int, default=10)
    g.add_argument("--n-val", type=int, default=3)
    g.add_argument("--n-test", type=int, default=5)
    g.add_argument("--workers", type=int, default=3)
    g.set_defaults(fn=exp_scenes_probe)
    c = sp.add_parser("scenes_collect")
    c.add_argument("--task", default="pick_up_cup")
    c.add_argument("--dataset", default="scenes_pick")
    c.add_argument("--per-scene", type=int, default=25)
    c.add_argument("--n-train", type=int, default=10)
    c.add_argument("--n-val", type=int, default=3)
    c.add_argument("--n-test", type=int, default=5)
    c.add_argument("--workers", type=int, default=3)
    c.set_defaults(fn=exp_scenes_collect)
    e = sp.add_parser("scenes_eval")
    e.add_argument("--task", default="pick_up_cup")
    e.add_argument("--dataset", default="scenes_pick")
    e.add_argument("--single-ckpt", default="checkpoints/bc_mlp_pick_up_cup_n240_s0.pt")
    e.add_argument("--per-scene-eval", type=int, default=10)
    e.add_argument("--obs-drop", type=int, nargs="*", default=None, help="state-vector indices to remove from the policy input (56 57 = placement goal xy)")
    e.add_argument("--suffix", default="")
    e.add_argument("--only-multi", action="store_true")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--n-train", type=int, default=10)
    e.add_argument("--n-val", type=int, default=3)
    e.add_argument("--n-test", type=int, default=5)
    e.add_argument("--workers", type=int, default=3)
    e.set_defaults(fn=exp_scenes_eval)

    r = sp.add_parser("action_repr")
    r.add_argument("--task", default="pick_up_cup")
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--episodes", type=int, default=30)
    r.add_argument("--workers", type=int, default=3)
    r.set_defaults(fn=exp_action_repr)
    q = sp.add_parser("dr")
    q.add_argument("--task", default="pick_up_cup")
    q.add_argument("--n-demos", type=int, default=300)
    q.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    q.add_argument("--episodes", type=int, default=30)
    q.add_argument("--workers", type=int, default=3)
    q.set_defaults(fn=exp_dr)

    im = sp.add_parser("images")
    im.add_argument("--task", default="pick_up_cup")
    im.add_argument("--n-demos", type=int, default=100)
    im.add_argument("--epochs", type=int, default=40)
    im.add_argument("--episodes", type=int, default=30)
    im.add_argument("--workers", type=int, default=3)
    im.set_defaults(fn=exp_images)
    cu = sp.add_parser("curriculum")
    cu.add_argument("--steps", type=int, default=24576)
    cu.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    cu.add_argument("--episodes", type=int, default=30)
    cu.add_argument("--workers", type=int, default=3)
    cu.set_defaults(fn=exp_curriculum)
    pr = sp.add_parser("proprio")
    pr.add_argument("--task", default="pick_up_cup")
    pr.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    pr.add_argument("--episodes", type=int, default=30)
    pr.add_argument("--workers", type=int, default=3)
    pr.set_defaults(fn=exp_proprio)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
