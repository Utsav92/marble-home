"""Create an image dataset by replaying a state dataset's demos with cameras (optionally under domain randomisation).

    python scripts/render_dataset.py --task pick_up_cup --obs-mode rgb --n-demos 100 --out pick_up_cup_rgb64
    python scripts/render_dataset.py --task pick_up_cup --obs-mode rgb --n-demos 100 --dr visual --out pick_up_cup_rgb64_visdr
The new dataset copies the SOURCE split (train/val membership by demo order) and writes its own stats.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from marble_home.datasets.hdf5 import DemoFile
from marble_home.datasets.render import render_dataset
from marble_home.utils import provenance
from marble_home.utils.paths import DATA, ensure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_up_cup")
    ap.add_argument("--src-dataset", default=None, help="state dataset name (default: task)")
    ap.add_argument("--obs-mode", default="rgb", choices=["rgb", "rgbd", "multicam"])
    ap.add_argument("--hw", type=int, default=64)
    ap.add_argument("--n-demos", type=int, default=100, help="number of TRAIN demos (+ all val demos) to render")
    ap.add_argument("--dr", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src_name = args.src_dataset or args.task
    src = DATA / "demonstrations" / src_name / "dataset.hdf5"
    split = json.loads((DATA / "splits" / f"{src_name}.json").read_text())
    train = split["train"][: args.n_demos]
    val = split["val"]
    names = train + val
    out = DATA / "demonstrations" / args.out / "dataset.hdf5"
    ensure(out.parent)
    t0 = time.time()
    summ = render_dataset(src, out, args.task, names, args.obs_mode, args.hw, args.dr, args.workers)
    # the merged file renumbers demos demo_0..; rebuild the split accordingly (train first, then val)
    new = {"train": [f"demo_{i}" for i in range(len(train))], "val": [f"demo_{len(train) + i}" for i in range(len(val))], "test": [],
           "source_dataset": src_name, "source_names": names}
    ensure(DATA / "splits")
    provenance.write_json(DATA / "splits" / f"{args.out}.json", new)
    with DemoFile(out) as df:
        prop = np.concatenate([df.demo(n)["obs"]["proprio"] for n in new["train"]], 0)
    st = {"proprio": {"mean": prop.mean(0).tolist(), "std": np.maximum(prop.std(0), 1e-3).tolist(), "n": int(len(prop))}}
    provenance.write_json(DATA / "splits" / f"{args.out}_stats.json", st)
    provenance.write_json(out.parent / "dataset_meta.json", {"source": src_name, "obs_mode": args.obs_mode, "hw": args.hw, "dr": args.dr, "summary": summ,
                                                          "seconds": time.time() - t0, "checksum_sha256": provenance.file_sha256(out)})
    print(summ, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
