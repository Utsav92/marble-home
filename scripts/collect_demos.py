"""Collect scripted-expert demonstrations into versioned HDF5 shards (multi-process).

    python scripts/collect_demos.py --task pick_up_cup --episodes 100
    python scripts/collect_demos.py --task pick_up_cup --episodes 300 --workers 3
Writes data/demonstrations/<task>/shard_<k>.hdf5 ; run scripts/prepare_dataset.py to merge/split/normalise.
Teleoperation: RoboSuite's keyboard/spacemouse devices drive OSC/ee controllers; this repo's demos are
SCRIPTED (privileged-state expert). Replay verification: python scripts/prepare_dataset.py --verify N
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

from marble_home.utils import provenance
from marble_home.utils.paths import DATA, ensure


def _work(args):
    task, n, out, seed0, noise, scene, keep, dr = args
    from marble_home.datasets.collect import collect

    return collect(task, n, out, seed0=seed0, noise=noise, scene=scene, keep_failures=keep, dr=dr, log=lambda m: print(m, flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_up_cup")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000, help="first episode seed (instance seeds are seed0+i)")
    ap.add_argument("--noise", type=float, default=0.004, help="std (m) of Gaussian noise on expert targets")
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--tag", default="", help="subfolder tag (e.g. a scene id)")
    ap.add_argument("--dataset", default=None, help="dataset name (default: the task); keeps experiments from mixing")
    ap.add_argument("--scene-spec", default=None, help="path to a scene_spec.json (procedural/Marble-derived scene)")
    ap.add_argument("--dr", default=None, help="domain randomisation preset recorded into the demos (none|visual|physics|sensor|combined)")
    args = ap.parse_args()

    out_dir = DATA / "demonstrations" / (args.dataset or args.task) / args.tag
    ensure(out_dir)
    per = [args.episodes // args.workers + (1 if i < args.episodes % args.workers else 0) for i in range(args.workers)]
    import json as _json
    scene = _json.loads(open(args.scene_spec).read()) if args.scene_spec else None
    jobs = [(args.task, per[i], str(out_dir / f"shard_{i}.hdf5"), args.seed + i * 100_000, args.noise, scene, args.keep_failures, args.dr)
            for i in range(args.workers) if per[i] > 0]
    if len(jobs) == 1:
        res = [_work(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as p:
            res = p.map(_work, jobs)
    tot = {"episodes": sum(r["kept"] for r in res), "tried": sum(r["tried"] for r in res),
           "steps": sum(r["steps"] for r in res), "seconds": max(r["seconds"] for r in res)}
    print("DONE", tot)
    provenance.write_json(out_dir / "collection.json", {"args": vars(args), "shards": res, "total": tot,
                                                         "manifest": provenance.run_manifest()})


if __name__ == "__main__":
    main()
