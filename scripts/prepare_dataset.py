"""Merge demo shards, verify by replay, create train/val/test splits and normalisation stats.

    python scripts/prepare_dataset.py --task pick_up_cup [--verify 5]
Outputs (task = T):
    data/demonstrations/T/dataset.hdf5       merged data
    data/demonstrations/T/dataset_meta.json  compatibility record (sec. 9) + checksum
    data/splits/T.json                       episode-level train/val/test split
    data/splits/T_stats.json                 mean/std for state, proprio, actions
"""
from __future__ import annotations

import argparse
import glob
import json
import time

import numpy as np

from marble_home.datasets.collect import make_env, replay_check
from marble_home.datasets.hdf5 import DemoFile, merge
from marble_home.utils import provenance
from marble_home.utils.paths import DATA, ensure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_up_cup")
    ap.add_argument("--dataset", default=None, help="dataset name (default: the task)")
    ap.add_argument("--verify", type=int, default=3, help="replay-verify this many random episodes")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    name = args.dataset or args.task
    d = DATA / "demonstrations" / name
    shards = sorted(glob.glob(str(d / "**" / "shard_*.hdf5"), recursive=True))
    if not shards:
        raise SystemExit(f"no shards under {d}; run scripts/collect_demos.py first")
    out = d / "dataset.hdf5"
    summ = merge(shards, out)
    print("merged", summ)

    with DemoFile(out) as df:
        names = df.demos
        eps = {n: df.demo(n) for n in names}
        info = {"scene": json.loads(df.attrs["scene"]), "versions": json.loads(df.attrs["versions"])}

    # ---- split by episode (single procedural scene -> scene-level splits are exercised in the generalisation experiments)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(names))
    n_val, n_test = max(1, int(len(names) * args.val_frac)), max(1, int(len(names) * args.test_frac))
    split = {"val": [names[i] for i in order[:n_val]], "test": [names[i] for i in order[n_val:n_val + n_test]],
             "train": [names[i] for i in order[n_val + n_test:]], "seed": args.seed}
    ensure(DATA / "splits")
    provenance.write_json(DATA / "splits" / f"{name}.json", split)

    # ---- normalisation stats from TRAIN episodes only
    def cat(key, sub=None):
        return np.concatenate([(eps[n]["obs"][key] if sub else eps[n][key]) for n in split["train"]], 0)

    stats = {}
    for k, arr in (("state", cat("state", 1)), ("proprio", cat("proprio", 1)), ("actions", cat("actions"))):
        std = arr.std(0)
        stats[k] = {"mean": arr.mean(0).tolist(), "std": np.maximum(std, 1e-3).tolist(), "n": int(len(arr))}
    provenance.write_json(DATA / "splits" / f"{name}_stats.json", stats)

    # ---- replay verification (action-observation alignment)
    ver = []
    if args.verify:
        pick = rng.choice(names, size=min(args.verify, len(names)), replace=False)
        envs = {}
        for n in pick:  # each demo is replayed in ITS OWN scene (multi-scene datasets)
            at = eps[n]["attrs"]
            sj = at.get("scene_json")
            scene = json.loads(sj.decode() if isinstance(sj, bytes) else sj) if sj is not None else None
            key = scene["scene_id"] if scene else "default"
            if key not in envs:
                envs[key] = make_env(args.task, scene=scene, seed=0, dr=None)
            r = replay_check(envs[key], eps[n], at)
            r["demo"], r["scene"] = n, key
            ver.append(r)
            print("replay", r)
        for e in envs.values():
            e.close()

    T = [len(eps[n]["actions"]) for n in names]
    meta = {
        "dataset": "marble_home native scripted demonstrations", "format_version": "marble_home.demo.v1",
        "source": "generated in-repo by scripts/collect_demos.py (ScriptedExpert, privileged state)",
        "license": "generated data; no third-party content", "robot_embodiment": "Fourier GR1 (GR1FixedLowerBody), simulated",
        "action_representation": "4-D ee_abs: right-hand xyz target (base frame, normalised to workspace) + scalar grip; "
                                 "executed 24-D ik actions also stored", "observation_modalities": ["proprio(58)", "state(69)"],
        "episodes": len(names), "steps": int(sum(T)), "mean_len": float(np.mean(T)), "sampling_hz": 20,
        "task_labels": [args.task], "success_rate_of_stored": float(np.mean([bool(eps[n]["attrs"]["success"]) for n in names])),
        "splits": {k: len(v) for k, v in split.items() if isinstance(v, list)},
        "required_conversions": "none (native). Third-party datasets (RoboMimic/MimicGen/DROID/OXE) use other embodiments and "
                               "action spaces and are NOT mixed in; see docs/datasets.md",
        "scene": info["scene"], "versions": info["versions"], "checksum_sha256": provenance.file_sha256(out),
        "replay_verification": ver,
    }
    provenance.write_json(d / "dataset_meta.json", meta)
    print(json.dumps({k: meta[k] for k in ("episodes", "steps", "mean_len", "splits", "checksum_sha256")}, indent=1))
    if ver and not all(v["ok"] for v in ver):
        print("WARNING: some replays did not verify within tolerance")


if __name__ == "__main__":
    main()
