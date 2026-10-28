"""Uncertainty estimates for benchmark reporting."""
from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (good for small n / extreme p)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))  # clamp: float rounding can land at -1e-17 when k=0


def bootstrap_ci(x, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return (float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2)))


def mcnemar_exact(a_wins_b_loses: int, b_wins_a_loses: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes on identical task instances."""
    n = a_wins_b_loses + b_wins_a_loses
    if n == 0:
        return 1.0
    k = min(a_wins_b_loses, b_wins_a_loses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * p)


def summarize(episodes: list[dict]) -> dict:
    """Aggregate per-episode records (see evaluation.rollout) into the spec's headline metrics."""
    n = len(episodes)
    succ = sum(e["success"] for e in episodes)
    lo, hi = wilson(succ, n)
    out = {"n": n, "success_rate": succ / max(n, 1), "success_ci95": [lo, hi]}
    for k in ("grasp_success", "collision_free"):
        out[k] = float(np.mean([e[k] for e in episodes])) if n else float("nan")
    for k in ("place_error", "eef_error", "steps", "invalid_steps", "action_smoothness", "joint_jerk", "latency_ms", "return"):
        vals = [e[k] for e in episodes if e.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
        out[k + "_ci95"] = list(bootstrap_ci(vals)) if vals else None
    ok_steps = [e["steps"] for e in episodes if e["success"]]
    out["completion_time_s"] = float(np.mean(ok_steps)) / 20.0 if ok_steps else None
    return out
