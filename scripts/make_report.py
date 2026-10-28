"""Compile reports/ (tables + plots) from logged results. Nothing is hand-entered.

    python scripts/make_report.py            # writes reports/results_tables.md and reports/figures/*.png
The narrative (reports/final_report.md) is hand-written prose that INCLUDES results_tables.md.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marble_home.evaluation.metrics import wilson
from marble_home.utils.paths import DATA, REPORTS, RESULTS, ensure

FIG = REPORTS / "figures"


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def fmt_ci(sr, ci):
    return f"{sr:.1%} [{max(0.0, ci[0]):.1%}, {min(1.0, ci[1]):.1%}]"


def curves():
    ensure(FIG)
    out = []
    for run in sorted((RESULTS / "runs").glob("*")):
        m = run / "metrics.csv"
        if not m.exists():
            continue
        df = pd.read_csv(m)
        x = "timesteps" if "timesteps" in df else "step"
        cols = [c for c in df.columns if c in ("train_mse", "val_mse", "train_loss", "val_action_mse", "val_noise_loss", "ep_return", "success_rate",
                                               "rew_reach", "rew_total")]
        if not cols:
            continue
        fig, axes = plt.subplots(1, 2 if x == "timesteps" else 1, figsize=(10 if x == "timesteps" else 6, 3.4))
        axes = np.atleast_1d(axes)
        if x == "timesteps":
            for c in ("ep_return",):
                if c in df: axes[0].plot(df[x], df[c], label=c)
            axes[0].set_xlabel("env steps"); axes[0].set_ylabel("episode return (shaped)"); axes[0].legend()
            for c in ("success_rate", "rew_reach"):
                if c in df: axes[1].plot(df[x], df[c], label=c)
            axes[1].set_xlabel("env steps"); axes[1].legend(); axes[1].set_title("training-time (stochastic policy)")
        else:
            for c in cols:
                axes[0].plot(df[x], df[c], label=c)
            axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].legend()
        fig.suptitle(run.name, fontsize=9)
        fig.tight_layout()
        p = FIG / f"curve_{run.name}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        out.append(p)
    return out


def section_datasets():
    L = ["## Datasets", "", "| dataset | episodes | steps | mean len | splits (train/val/test) | SHA-256 (first 12) | replay-verified |", "|---|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(str(DATA / "demonstrations" / "*" / "dataset_meta.json"))):
        m = jl(f)
        if m is None:
            continue
        name = Path(f).parent.name
        if "episodes" in m:
            ver = m.get("replay_verification", [])
            sp = m.get("splits", {})
            L.append(f"| {name} | {m['episodes']} | {m['steps']} | {m['mean_len']:.0f} | {sp.get('train','?')}/{sp.get('val','?')}/{sp.get('test','?')} | "
                     f"{m['checksum_sha256'][:12]} | {sum(v['ok'] for v in ver)}/{len(ver)} |")
        else:
            s = m.get("summary", {})
            L.append(f"| {name} (rendered: {m.get('obs_mode')} {m.get('hw')}px, dr={m.get('dr')}) | {s.get('episodes','?')} | {s.get('steps','?')} | - | - | {m['checksum_sha256'][:12]} | - |")
    return L + [""]


def section_geometry():
    L = ["## Geometry validation", ""]
    for f in sorted(glob.glob(str(RESULTS / "geometry" / "*" / "report.json"))):
        r = jl(f)
        L.append(f"**{r['scene']}** ({r.get('source')}): {'PASS' if r['passed'] else 'FAIL'}")
        L.append("")
        L += ["| check | result | detail |", "|---|---|---|"]
        for k, v in r["checks"].items():
            L.append(f"| {k} | {'PASS' if v['pass'] else 'FAIL'}{'' if v['required'] else ' (advisory)'} | {v['detail'][:200].replace('|', '/')} |")
        L.append("")
    return L


def section_runs():
    L = ["## Training runs (all logged)", "", "| run | dataset demos | epochs / steps | duration (s) | best val | held-out success [Wilson 95%] | latency ms | checkpoint SHA-256 |", "|---|---|---|---|---|---|---|---|"]
    for run in sorted((RESULTS / "runs").glob("*")):
        man, ev = jl(run / "manifest.json"), jl(run / "eval.json")
        if ev is None:  # evaluated standalone via scripts/evaluate.py
            alt = sorted(glob.glob(str(RESULTS / "eval" / f"{run.name}__*.json")))
            ev = jl(alt[-1]) if alt else None
        if man is None:
            continue
        best = man.get("best_val_mse", man.get("best_val_action_mse"))
        steps = man.get("epochs_run", man.get("total_timesteps", "-"))
        succ = fmt_ci(ev["summary"]["success_rate"], ev["summary"]["success_ci95"]) + f" (n={ev['summary']['n']})" if ev else "-"
        lat = f"{ev['summary']['latency_ms']:.1f}" if ev else "-"
        L.append(f"| {run.name} | {man.get('n_train_episodes', '-')} | {steps} | {man.get('duration_s', '-')} | {('%.5f' % best) if best else '-'} | {succ} | {lat} | {str(man.get('checkpoint_sha256'))[:12]} |")
    return L + [""]


def section_data_scaling():
    d = jl(RESULTS / "experiments" / "data_scaling.json")
    if not d:
        return []
    df = pd.DataFrame(d["rows"])
    L = ["## Experiment 1: data scaling (BC-MLP, state obs) - CONTROLLER v1 (not re-run on v2)", "",
         "Pooled success = all episodes over seeds; Wilson interval on the pooled count. Per-seed values show seed variance.", "",
         "| demos | seeds | per-seed success | mean val-MSE | pooled success [Wilson 95%] |", "|---|---|---|---|---|"]
    xs, ys, lo, hi = [], [], [], []
    for n, g in df.groupby("n_demos"):
        k = int(round(sum(r.success * r.episodes for r in g.itertuples())))
        nn = int(g.episodes.sum())
        l, h = wilson(k, nn)
        L.append(f"| {n} | {len(g)} | {', '.join(f'{v:.0%}' for v in g.success)} | {g.val_mse.mean():.5f} | {k / nn:.1%} [{l:.1%}, {h:.1%}] (n={nn}) |")
        xs.append(n); ys.append(k / nn); lo.append(l); hi.append(h)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.errorbar(xs, ys, yerr=[np.maximum(np.array(ys) - lo, 0), np.maximum(np.array(hi) - ys, 0)], marker="o", capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("training demonstrations"); ax.set_ylabel("held-out success"); ax.set_ylim(0, 1.05)
    fig.tight_layout(); ensure(FIG); fig.savefig(FIG / "data_scaling.png", dpi=120); plt.close(fig)
    return L + ["", "![data scaling](figures/data_scaling.png)", ""]


def section_ddim():
    d = jl(RESULTS / "experiments" / "ddim_steps.json")
    if not d:
        return []
    L = ["## Experiment 7: diffusion inference steps (same checkpoint) - CONTROLLER v1; latency column is CPU-contended and unreliable", "", "| DDIM steps | success [Wilson 95%] | latency (ms / env step, amortised over the 8-step chunk) | mean episode steps | joint jerk |", "|---|---|---|---|---|"]
    for r in d["rows"]:
        L.append(f"| {r['ddim_steps']} | {fmt_ci(r['success'], r['ci95'])} | {r['latency_ms_per_env_step']:.2f} | {r['steps_mean']:.0f} | {r['joint_jerk']:.1f} |")
    return L + [""]


def section_throughput():
    d = jl(RESULTS / "experiments" / "throughput.json")
    if not d:
        return []
    L = ["## Experiment 8: simulation throughput - measured on a SHARED CPU (another job was running): contended lower bounds, invalid for absolute claims", "", "| configuration | ms / env step | env steps / s |", "|---|---|---|"]
    for r in d["rows"]:
        L.append(f"| {r['config']} | {r['ms_per_env_step']:.1f} | {r['env_steps_per_s']:.1f} |")
    L += ["", f"Raw `mj_step`: {d['mujoco_substep_ms']:.3f} ms x {d['substeps_per_control_step']} substeps per control step; BC-MLP inference {d['bc_inference_ms']:.2f} ms.", ""]
    return L


def section_generic(name, title):
    d = jl(RESULTS / "experiments" / f"{name}.json")
    if not d:
        return []
    return [f"## {title}", "", "```json", json.dumps({k: v for k, v in d.items() if k != "manifest"}, indent=1)[:6000], "```", ""]


def section_scene_gen():
    d = jl(RESULTS / "experiments" / "scene_generalization.json")
    if not d:
        return []
    L = ["## Scene-level generalisation (train / val / held-out test SCENES)", "",
         "| policy | train scenes | val scenes | held-out test scenes | gap (train - test) |", "|---|---|---|---|---|"]
    for p, v in d["per_policy"].items():
        L.append(f"| {p} | {fmt_ci(v['train']['success'], v['train']['ci95'])} n={v['train']['n']} | {fmt_ci(v['val']['success'], v['val']['ci95'])} n={v['val']['n']} | "
                 f"{fmt_ci(v['test']['success'], v['test']['ci95'])} n={v['test']['n']} | {v['generalization_gap_train_minus_test']:+.1%} |")
    pr = jl(RESULTS / "experiments" / "scene_probe.json")
    if pr:
        L += ["", "Scripted-expert viability per scene (task compatibility):", "", "| split | scene | kind | counter height (m) | expert success |", "|---|---|---|---|---|"]
        for r in pr["rows"]:
            L.append(f"| {r['split']} | {r['scene_id']} | {r['kind']} | {r['counter_height']:.3f} | {r['expert_success']:.0%} (n={r['n']}) |")
    return L + [""]


def _bench(path):
    d = jl(path)
    return {r["name"]: r for r in d["rows"]} if d else {}


def _row(r):
    sm = r["summary"]
    e = r["episodes"]
    return (sm["success_rate"], sm["success_ci95"], 100 * float(np.mean([x["eef_error"] for x in e])), 100 * float(np.mean([x["place_error"] for x in e])),
            100 * float(max(x["place_error"] for x in e)))


def section_v1_vs_v2():
    L = ["## Accuracy: controller v1 (baseline) vs v2 (waist-assisted IK + delta actions)", "",
         "v1 = RoboSuite default IK posture weights, absolute-target actions. v2 = waist posture weight 1 + delta actions. "
         "**Not like-for-like for reach and place**: v2 also uses stricter success rules (reach 3 cm + 5-step hold vs 5 cm; place radius 3 cm vs 4.5 cm). "
         "Errors are final-pose means over the same 30 held-out instance seeds.", "",
         "| task | policy | v1 success [Wilson 95%] | v2 success [Wilson 95%] | v1 hand err (cm) | v2 hand err (cm) | v1 place err (cm) | v2 place err (cm, max) |", "|---|---|---|---|---|---|---|---|"]
    pairs = {"pick_up_cup": ("bc_mlp_pick_up_cup_n240_s0", "bc_mlp_v2delta_pick_up_cup_s0"),
             "place_cup": ("bc_mlp_place_cup_n160_s0", "bc_mlp_v2d_place_cup_s0"),
             "reach_cup": ("bc_mlp_reach_cup_n80_s0", "bc_mlp_v2d_reach_cup_s0")}
    for task, (a, b) in pairs.items():
        v1, v2 = _bench(RESULTS / "benchmark" / "v1" / f"{task}.json"), _bench(RESULTS / "benchmark" / f"{task}.json")
        for label, n1, n2 in (("scripted expert", "scripted_expert", "scripted_expert"), ("BC-MLP", a, b)):
            if n1 in v1 and n2 in v2:
                r1, r2 = _row(v1[n1]), _row(v2[n2])
                pl = task == "place_cup"
                L.append(f"| {task} | {label} | {fmt_ci(r1[0], r1[1])} | {fmt_ci(r2[0], r2[1])} | {r1[2]:.2f} | {r2[2]:.2f} | "
                         f"{(f'{r1[3]:.2f}' if pl else '-')} | {(f'{r2[3]:.2f} ({r2[4]:.2f})' if pl else '-')} |")
    v2p = _bench(RESULTS / "benchmark" / "pick_up_cup.json")
    for n in ("bc_mlp_v2_pick_up_cup_s0", "diffusion_v2d_pick_up_cup_s0"):
        if n in v2p:
            r = _row(v2p[n])
            L.append(f"| pick_up_cup | {n} (v2 only) | - | {fmt_ci(r[0], r[1])} | - | {r[2]:.2f} | - | - |")
    em = {}
    for task in ("pick_up_cup", "place_cup", "reach_cup"):
        for tag, label in (("", "v1 expert grid"), ("_v2", "v2 expert grid")):
            d = jl(RESULTS / "experiments" / f"expert_map_{task}{tag}.json")
            if d:
                em.setdefault(task, {})[label] = (np.mean([r["success"] for r in d["rows"]]), len(d["rows"]))
    L += ["", "Scripted expert over a grid of cup positions (deterministic positions across the spawn region):", "", "| task | v1 | v2 |", "|---|---|---|"]
    for task, dd in em.items():
        a_, b_ = dd.get("v1 expert grid"), dd.get("v2 expert grid")
        L.append(f"| {task} | {('%.0f%% over %d pts' % (100 * a_[0], a_[1])) if a_ else '-'} | {('%.0f%% over %d pts' % (100 * b_[0], b_[1])) if b_ else '-'} |")
    return L + [""]


def section_waist():
    d = jl(RESULTS / "experiments" / "waist_weight_scan.json")
    if not d:
        return []
    L = ["## IK waist posture weight scan (scripted pick at 4 previously failing corners + 2 easy points)", "",
         "| waist weight (yaw=pitch) | expert success | mean hand error (m) | max waist yaw (rad) | max waist pitch (rad) |", "|---|---|---|---|---|"]
    for w, v in sorted(d.items(), key=lambda kv: -float(kv[0])):
        L.append(f"| {float(w):g}{' (RoboSuite default, v1)' if float(w) == 10 else (' (v2 default)' if float(w) == 1 else '')} | {v['success']:.0%} | {v['eef_err']:.3f} | {v['max_waist_yaw']:.2f} | {v['max_waist_pitch']:.2f} |")
    return L + [""]


def section_scene_gen_v2():
    L = ["## Scene-level generalisation (10 train / 3 val / 5 held-out test scenes; 10 episodes per scene)", "",
         "| controller / policy | train scenes | val scenes | held-out test scenes | gap (train - test) |", "|---|---|---|---|---|"]
    def add(label, v):
        L.append(f"| {label} | {fmt_ci(v['train']['success'], v['train']['ci95'])} n={v['train']['n']} | {fmt_ci(v['val']['success'], v['val']['ci95'])} n={v['val']['n']} | "
                 f"{fmt_ci(v['test']['success'], v['test']['ci95'])} n={v['test']['n']} | {v['train']['success'] - v['test']['success']:+.1%} |")
    for tag, d in (("v1", jl(RESULTS / "experiments" / "scene_generalization_v1.json")), ("v2", jl(RESULTS / "experiments" / "scene_generalization.json"))):
        if d:
            for p, v in d["per_policy"].items():
                add(f"{tag}: {p}", v)
    ng = jl(RESULTS / "experiments" / "scene_eval_nogoal.json")
    if ng:
        add("v2: multi-scene BC, placement-goal inputs removed", ng["per_split"])
    return L + [""]

def section_household():
    d = jl(RESULTS / "experiments" / "household_tasks.json")
    if not d:
        return []
    rows = d["rows"]
    L = ["## Multi-item household tasks (clear_table, load_dishwasher, laundry_basket)", "",
         "Rigid-body proxies on the same GR1 right-hand pick-and-place stack: items are cylinders (cups / laundry rolls), receptacles are static "
         "(tray, counter-top dishwasher rack without door or sliding, tote with end rims); cloth is not simulated. Success = every item placed on its slot "
         "(within 3 cm, upright, settled, hand released) and still there at the end, with at most 5 invalid contact steps. State observations (privileged item pose), "
         "delta actions, 100 expert demos, held-out seeds 900000+ (30 episodes per row).", "",
         "| task | policy | seed | success [Wilson 95%] | mean items placed | mean steps |", "|---|---|---|---|---|---|"]
    for t in ("clear_table", "laundry_basket", "load_dishwasher"):
        for r in sorted((r for r in rows if r["task"] == t), key=lambda r: (r["policy"] != "scripted_expert", r.get("seed", -1))):
            L.append(f"| {t} | {r['policy']} | {r.get('seed', '-')} | {fmt_ci(r['success'], r['ci95'])} | {r['items_done_mean']:.2f} | {r['steps_mean']:.0f} |")
    L += ["", "Pooled over BC seeds:", "", "| task | BC seeds | pooled success [Wilson 95%] |", "|---|---|---|"]
    for t in ("clear_table", "laundry_basket", "load_dishwasher"):
        g = [r for r in rows if r["task"] == t and r["policy"] == "bc_mlp"]
        if g:
            k = int(round(sum(r["success"] * r["n"] for r in g)))
            n = sum(r["n"] for r in g)
            lo, hi = wilson(k, n)
            L.append(f"| {t} | {len(g)} | {k / n:.1%} [{lo:.1%}, {hi:.1%}] (n={n}) |")
    return L + [""]


def section_v2_experiments():
    E = RESULTS / "experiments"
    L = ["## Experiments re-run on controller v2 (delta actions, 30 held-out episodes per run)", ""]
    d = jl(E / "data_scaling_v2.json")
    if d:
        df = pd.DataFrame(d["rows"])
        L += ["### Data scaling (BC-MLP, state obs, 3 seeds)", "", "| demos | per-seed success | mean val-MSE | pooled success [Wilson 95%] |", "|---|---|---|---|"]
        xs, ys, lo, hi = [], [], [], []
        for n, g in df.groupby("n_demos"):
            k = int(round(sum(r.success * r.episodes for r in g.itertuples())))
            nn = int(g.episodes.sum())
            l, h = wilson(k, nn)
            L.append(f"| {n} | {', '.join(f'{v:.0%}' for v in g.success)} | {g.val_mse.mean():.5f} | {k / nn:.1%} [{l:.1%}, {h:.1%}] (n={nn}) |")
            xs.append(n); ys.append(k / nn); lo.append(l); hi.append(h)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.errorbar(xs, ys, yerr=[np.maximum(np.array(ys) - lo, 0), np.maximum(np.array(hi) - ys, 0)], marker="o", capsize=3)
        ax.set_xscale("log"); ax.set_xlabel("training demonstrations"); ax.set_ylabel("held-out success"); ax.set_ylim(0, 1.05)
        fig.tight_layout(); ensure(FIG); fig.savefig(FIG / "data_scaling_v2.png", dpi=120); plt.close(fig)
        L += ["", "![data scaling v2](figures/data_scaling_v2.png)", ""]
    d = jl(E / "ddim_steps_v2.json")
    if d:
        L += ["### Diffusion inference steps (same checkpoint; success only, latency in its own table)", "", "| DDIM steps | success [Wilson 95%] |", "|---|---|"]
        L += [f"| {r['ddim_steps']} | {fmt_ci(r['success'], r['ci95'])} |" for r in d["rows"]] + [""]
    d = jl(E / "latency_v2.json")
    if d:
        L += ["### Policy inference latency (policy only, 1 torch thread, other jobs running; min / median of repeats)", "",
              "| policy | DDIM steps | min ms per call | median ms | min ms per env step |", "|---|---|---|---|---|"]
        L += [f"| {r['policy']} | {r['ddim_steps'] if r['ddim_steps'] else '-'} | {r['min_ms']:.2f} | {r['median_ms']:.2f} | {r['per_env_step_min_ms']:.2f} |" for r in d["rows"]]
        L += ["", "A diffusion plan covers 8 control steps, so the per-env-step column divides by 8. CPU only; no GPU claim.", ""]
    d = jl(E / "proprio_only_v2.json")
    if d:
        L += ["### Proprio-only control (no cup information)", "", "| seed | success [Wilson 95%] |", "|---|---|"]
        L += [f"| {r['seed']} | {fmt_ci(r['success'], r['ci95'])} |" for r in d["rows"]] + [""]
    rows = []
    for tr in ("none", "visual"):
        for te in ("none", "visual"):
            f = jl(E / f"images_v2_{tr}_{te}.json")
            if f:
                rows.append((tr, te, f))
    if rows:
        L += ["### Image policy (CNN-BC, 64x64 RGB + proprio, 150 demos, 40 epochs)", "",
              "A trainer bug (action_mode not saved in CNN checkpoints) made the first evaluation read 0% everywhere; the numbers below are the corrected re-evaluation.", "",
              "| trained on | tested on | success [Wilson 95%] |", "|---|---|---|"]
        L += [f"| {tr} | {te} | {fmt_ci(f['success'], f['ci95'])} (n={f['n']}) |" for tr, te, f in rows] + [""]
    d = jl(E / "curriculum_v2.json")
    if d:
        L += ["### Curriculum (PPO, reach_cup, ee_delta, 30k steps each)", "", "| curriculum | seed | held-out success [Wilson 95%] |", "|---|---|---|"]
        L += [f"| {r['mode']} | {r['seed']} | {fmt_ci(r['success'], r['ci95'])} |" for r in d["rows"]] + [""]
    return L


def main():
    ensure(REPORTS, FIG)
    curves()
    L = ["<!-- generated by scripts/make_report.py from logged results; do not edit -->", ""]
    for sec in (section_v1_vs_v2, section_waist, section_scene_gen_v2, section_household, section_v2_experiments, section_datasets, section_runs, section_data_scaling, section_ddim, section_throughput, section_geometry):
        L += sec()
    for name, title in (("domain_randomization", "Experiment 4: domain randomisation - CONTROLLER v1, incomplete (v2 re-run not finished)"), ("modalities", "Experiment 2: observation modalities"),
                        ("action_repr", "Experiment 6: action representation"), ("curriculum", "Experiment 5: curriculum - CONTROLLER v1 (v2 version above)")):
        L += section_generic(name, title)
    (REPORTS / "results_tables.md").write_text("\n".join(L) + "\n")
    print("wrote", REPORTS / "results_tables.md", "and", len(list(FIG.glob('*.png'))), "figures")


if __name__ == "__main__":
    main()
