"""Policy-only inference latency (no simulator), single torch thread; reports min / median over repeats.

    python scripts/measure_latency.py
The minimum over repeats is the least contention-sensitive estimate on a shared CPU; it is still a lower-bound-ish number,
measured with the other jobs on the machine running, and says nothing about GPU latency.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from marble_home.policies.agents import load_checkpoint
from marble_home.utils import provenance
from marble_home.utils.paths import RESULTS, ensure

torch.set_num_threads(1)


def timeit(fn, reps=15, inner=10):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(inner):
            fn()
        ts.append((time.perf_counter() - t) / inner * 1000)
    return float(np.min(ts)), float(np.median(ts))


def main():
    out = {"threads": 1, "rows": []}
    m, _ = load_checkpoint("checkpoints/bc_mlp_v2delta_pick_up_cup_s0.pt")
    x = torch.randn(1, 69)
    with torch.no_grad():
        mn, md = timeit(lambda: m.act(x), inner=50)
    out["rows"].append({"policy": "bc_mlp", "ddim_steps": None, "min_ms": mn, "median_ms": md, "per_env_step_min_ms": mn})
    print(f"BC-MLP: min {mn:.3f} ms median {md:.3f} ms", flush=True)
    dm, _ = load_checkpoint("checkpoints/diffusion_v2d_pick_up_cup_s0.pt")
    xo = torch.randn(1, dm.n_obs, 69)
    for k in (1, 2, 5, 10, 20):
        with torch.no_grad():
            mn, md = timeit(lambda: dm.sample(xo, ddim_steps=k), reps=10, inner=3)
        out["rows"].append({"policy": "diffusion", "ddim_steps": k, "min_ms": mn, "median_ms": md, "per_env_step_min_ms": mn / dm.n_act, "n_act": dm.n_act})
        print(f"diffusion DDIM-{k}: min {mn:.1f} ms/plan median {md:.1f} ({mn / dm.n_act:.2f} ms per env step, n_act={dm.n_act})", flush=True)
    ensure(RESULTS / "experiments")
    provenance.write_json(RESULTS / "experiments" / "latency_v2.json", {**out, "manifest": provenance.run_manifest()})


if __name__ == "__main__":
    main()
