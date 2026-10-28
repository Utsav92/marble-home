"""Run the whole pipeline end to end with tiny budgets (an executable check that the README commands work).

    python scripts/run_pipeline.py --profile smoke            # ~10 min on a laptop CPU, isolated in ./.smoke
    python scripts/run_pipeline.py --profile smoke --keep-going
All outputs go to MARBLE_HOME_WORK (default <repo>/.smoke) so real datasets/checkpoints are never touched. The smoke
profile validates plumbing, NOT performance: its models are trained for seconds and its numbers mean nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def stages(profile: str):
    t = "reach_cup"
    s = [
        ("environment check", ["scripts/check_environment.py"]),
        ("robot smoke test (M1)", ["scripts/test_robot.py", "--headless", "--steps", "100"]),
        ("world plan (dry run, no credits)", ["scripts/generate_worlds.py", "--config", "configs/marble/default.yaml"]),
        ("procedural fallback scenes", ["scripts/generate_worlds.py", "--procedural", "6"]),
        ("synthetic Marble-like import (M3)", ["scripts/import_marble.py", "--make-fixture", "kitchen_fixture_001"]),
        ("geometry validation", ["scripts/validate_geometry.py", "--scene", "kitchen_fixture_001", "--probe-expert", "1"]),
        ("collect demos (M2)", ["scripts/collect_demos.py", "--task", t, "--episodes", "10", "--workers", "2"]),
        ("prepare dataset + replay verify", ["scripts/prepare_dataset.py", "--task", t, "--verify", "2"]),
        ("train BC + eval (M4)", ["scripts/train_bc.py", "--config", "configs/bc.yaml", "--set", f"task={t}", "--set", "epochs=5",
                                  "--set", "eval_episodes=4", "--workers", "2"]),
        ("train diffusion + eval (M5)", ["scripts/train_diffusion.py", "--config", "configs/diffusion.yaml", "--set", f"task={t}",
                                         "--set", "epochs=1", "--set", "eval_episodes=2", "--workers", "2"]),
        ("train PPO + eval (M6)", ["scripts/train_rl.py", "--config", "configs/rl.yaml", "--set", f"task={t}", "--set", "total_timesteps=1024",
                                   "--set", "n_steps=256", "--set", "eval_episodes=2", "--workers", "2"]),
        ("benchmark", ["scripts/benchmark.py", "--config", "configs/evaluation.yaml", "--task", t, "--auto", "--episodes", "4", "--workers", "2"]),
        ("render episode", ["scripts/render_episode.py", "--policy", "scripted", "--task", t]),
    ]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke", choices=["smoke"])
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--work", default=str(ROOT / ".smoke"))
    args = ap.parse_args()
    env = {**os.environ, "MARBLE_HOME_WORK": args.work, "PYTHONWARNINGS": "ignore"}
    Path(args.work).mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cmd in stages(args.profile):
        t0 = time.time()
        r = subprocess.run([PY, *cmd], cwd=ROOT, env=env, capture_output=True, text=True)
        ok = r.returncode == 0
        tail = "\n".join([l for l in (r.stdout + r.stderr).splitlines() if "WARNING" not in l and "INFO" not in l][-3:])
        rows.append({"stage": name, "ok": ok, "seconds": round(time.time() - t0, 1), "cmd": " ".join(cmd), "tail": tail})
        print(f"[{'OK ' if ok else 'FAIL'}] {name:38s} {time.time() - t0:6.1f}s", flush=True)
        if not ok:
            print(tail)
            if not args.keep_going:
                break
    out = Path(args.work) / "results" / "pipeline_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"profile": args.profile, "work_dir": args.work, "stages": rows, "all_ok": all(r["ok"] for r in rows)}, indent=2))
    print("ALL OK" if all(r["ok"] for r in rows) else "FAILURES", "->", out)
    sys.exit(0 if all(r["ok"] for r in rows) else 1)


if __name__ == "__main__":
    main()
