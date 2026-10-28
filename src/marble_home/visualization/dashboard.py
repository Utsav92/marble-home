"""Research dashboard (Streamlit).   streamlit run src/marble_home/visualization/dashboard.py

Reads ONLY logged artefacts: results/runs/*, results/benchmark/*, results/geometry/*, data/marble_raw/registry.json,
data/processed_worlds/*. Nothing on these pages is hand-entered.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
RUNS, BENCH, GEOM = ROOT / "results" / "runs", ROOT / "results" / "benchmark", ROOT / "results" / "geometry"
VIDEOS = ROOT / "results" / "videos"

st.set_page_config(page_title="marble-home", layout="wide")
view = st.sidebar.radio("View", ["Simulation Viewer", "Training Dashboard", "Benchmark Dashboard", "World Browser", "Experiment Explorer"])
st.sidebar.caption("Profile A (CPU). All numbers come from files under results/.")


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def runs() -> list[Path]:
    return sorted([p for p in RUNS.glob("*") if (p / "metrics.csv").exists()]) if RUNS.exists() else []


# ----------------------------------------------------------------------------------------------- simulation
if view == "Simulation Viewer":
    st.title("Simulation viewer")
    c1, c2, c3 = st.columns(3)
    task = c1.selectbox("Task", ["reach_cup", "pick_up_cup", "place_cup"], index=1)
    seed = c2.number_input("Instance seed", 0, 10**7, 900000)
    policy = c3.selectbox("Policy", ["scripted", "random"] + [p.name for p in sorted((ROOT / "checkpoints").glob("*")) if p.suffix in (".pt", ".zip")])
    if st.button("Run episode (live MuJoCo, ~10-30 s on CPU)"):
        from marble_home.policies.agents import make_agent
        from marble_home.simulation.gym_env import HouseholdEnv
        from marble_home.tasks.definitions import PHASES

        spec = {"type": policy} if policy in ("scripted", "random") else \
            {"type": "ppo" if policy.endswith(".zip") else "checkpoint", "path": str(ROOT / "checkpoints" / policy), "name": policy}
        agent = make_agent(spec)
        env = HouseholdEnv(task=task, **getattr(agent, "env_kwargs", {}))
        obs, info = env.reset(seed=int(seed))
        agent.reset(env)
        frames, rows, done = [], [], False
        while not done:
            obs, r, term, trunc, info = env.step(agent.act(obs))
            done = term or trunc
            if len(rows) % 2 == 0:
                frames.append(env.render(480, 360))
            e = info["privileged"]["eef_pos_world"]
            rows.append({"t": len(rows), "ee_x": e[0], "ee_y": e[1], "ee_z": e[2], "reward": r, "phase": PHASES[min(info["phase"], 5)],
                         "grasp_contacts": info["n_grasp_contacts"], "ee_error": info["eef_error"], "success": info["success"]})
        env.close()
        st.session_state["sim"] = (frames, pd.DataFrame(rows))
    if "sim" in st.session_state:
        frames, df = st.session_state["sim"]
        i = st.slider("Frame", 0, len(frames) - 1, 0)
        a, b = st.columns([3, 2])
        a.image(frames[i], caption=f"t={2 * i}  phase={df.phase.iloc[min(2 * i, len(df) - 1)]}")
        b.metric("Episode success", str(bool(df.success.any())))
        b.metric("Steps", len(df))
        b.metric("Max hand<->cup contacts", int(df.grasp_contacts.max()))
        st.subheader("End-effector trajectory (world frame, m)")
        st.line_chart(df.set_index("t")[["ee_x", "ee_y", "ee_z"]])
        st.subheader("Task progress")
        st.line_chart(df.set_index("t")[["ee_error", "reward"]])
    vids = sorted(VIDEOS.glob("*.mp4")) if VIDEOS.exists() else []
    if vids:
        st.subheader("Saved rollout videos")
        v = st.selectbox("Video", vids, format_func=lambda p: p.name)
        st.video(str(v))

# ----------------------------------------------------------------------------------------------- training
elif view == "Training Dashboard":
    st.title("Training")
    rs = runs()
    if not rs:
        st.info("No runs yet: python scripts/train_bc.py --config configs/bc.yaml")
    else:
        sel = st.multiselect("Runs", [r.name for r in rs], default=[rs[-1].name])
        for name in sel:
            df = pd.read_csv(RUNS / name / "metrics.csv")
            st.subheader(name)
            x = "timesteps" if "timesteps" in df else "step"
            cols = [c for c in df.columns if c not in ("step", "time_s", "timesteps", "lr")]
            st.line_chart(df.set_index(x)[cols])
            man = jload(RUNS / name / "manifest.json") or {}
            ev = jload(RUNS / name / "eval.json")
            c1, c2, c3 = st.columns(3)
            c1.metric("Duration (s)", man.get("duration_s", "?"))
            c2.metric("Epochs run", man.get("epochs_run", "-"))
            if ev:
                s = ev["summary"]
                c3.metric("Held-out success", f"{s['success_rate']:.1%}", f"CI {s['success_ci95'][0]:.0%}-{s['success_ci95'][1]:.0%}")
        cur = sorted(ROOT.glob("results/curriculum/*.json"))
        if cur:
            st.subheader("Curriculum transitions")
            st.json(jload(cur[-1]))

# ----------------------------------------------------------------------------------------------- benchmark
elif view == "Benchmark Dashboard":
    st.title("Benchmark")
    files = sorted(BENCH.glob("*.json")) if BENCH.exists() else []
    if not files:
        st.info("Run: python scripts/benchmark.py --auto")
    else:
        f = st.selectbox("Benchmark", files, format_func=lambda p: p.stem)
        d = jload(f)
        rows = []
        for r in d["rows"]:
            s = r["summary"]
            rows.append({"policy": r["name"], "success": s["success_rate"], "ci_lo": s["success_ci95"][0], "ci_hi": s["success_ci95"][1],
                         "grasp": s["grasp_success"], "collision_free": s["collision_free"], "place_err_m": s["place_error"],
                         "latency_ms": s["latency_ms"], "sim_steps_per_s": r["sim_steps_per_s"], "n": s["n"]})
        df = pd.DataFrame(rows).set_index("policy")
        st.dataframe(df.style.format("{:.3f}"))
        st.bar_chart(df[["success"]])
        st.caption("Intervals are Wilson 95% CIs over held-out instance seeds; paired McNemar tests are in reports/benchmark_*.md")
        png = ROOT / "reports" / f"benchmark_{d['task']}.png"
        if png.exists():
            st.image(str(png))

# ----------------------------------------------------------------------------------------------- worlds
elif view == "World Browser":
    st.title("Worlds")
    reg = jload(ROOT / "data" / "marble_raw" / "registry.json") or {"scenes": {}}
    st.subheader("Marble registry (versioned)")
    if not reg["scenes"]:
        st.info("No Marble scenes registered. Dry-run / fixtures: python scripts/generate_worlds.py ; python scripts/import_marble.py --make-fixture kitchen_fixture_001")
    for sid, versions in reg["scenes"].items():
        v = versions[-1]
        with st.expander(f"{sid}  [{v['api_status']}]  v{len(versions)}"):
            st.write({k: v[k] for k in ("category", "world_id", "model", "seed", "scale", "scale_source", "cost_credits", "generation_time_s")})
            st.caption("Prompt: " + (v["prompt"] or "")[:400])
            st.json({"checksums": v["checksums"], "complexity": v["complexity"], "notes": v["notes"]})
    st.subheader("Processed / procedural scenes and geometry validation")
    geo = sorted(GEOM.glob("*/report.json")) if GEOM.exists() else []
    for g in geo:
        rep = jload(g)
        st.markdown(f"**{rep['scene']}** ({rep.get('source')}): {'PASS' if rep['passed'] else 'FAIL'}")
        cols = st.columns(4)
        for col, (cam, path) in zip(cols, rep.get("screenshots", {}).items()):
            if Path(path).exists():
                col.image(path, caption=cam)
        st.table(pd.DataFrame([{"check": k, "pass": v["pass"], "detail": v["detail"][:140]} for k, v in rep["checks"].items()]))
    procs = sorted((ROOT / "data" / "processed_worlds").glob("proc_*/v1/scene_spec.json"))
    if procs:
        st.caption(f"{len(procs)} procedural fallback scenes (task-compatible: reach_cup, pick_up_cup, place_cup)")
        st.dataframe(pd.DataFrame([{"scene": (s := jload(p))["scene_id"], "kind": s["kind"], "counter_h": round(s["counter_height"], 3)} for p in procs]))

# ----------------------------------------------------------------------------------------------- experiments
else:
    st.title("Experiment explorer")
    rows = []
    for r in runs():
        man, cfg, ev = jload(r / "manifest.json") or {}, jload(r / "config.json") or {}, jload(r / "eval.json")
        rows.append({"experiment": r.name, "model": r.name.split("_")[0], "task": cfg.get("task"), "seed": cfg.get("seed"),
                     "demos": man.get("n_train_episodes"), "duration_s": man.get("duration_s"),
                     "heldout_success": ev["summary"]["success_rate"] if ev else None,
                     "checkpoint": Path(man.get("checkpoint", "")).name, "git": (man.get("git_commit") or "")[:8],
                     "dataset_sha": (man.get("dataset_sha256") or "")[:10]})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df)
        pick = st.selectbox("Details", df.experiment)
        st.json({"config": jload(RUNS / pick / "config.json"), "manifest": jload(RUNS / pick / "manifest.json")})
    else:
        st.info("No runs yet.")
