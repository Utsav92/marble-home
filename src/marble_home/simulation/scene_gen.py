"""Seeded procedural scene generator for the five household categories (PROCEDURAL STAND-INS, not Marble output).

Every scene keeps the same interactive work surface (a counter-height table in front of the robot) so the cup
tasks are defined in every category; what changes is room furniture, dimensions, colours and the spawn region.
This is the "independent procedural-scene fallback" of the spec and the source of the multi-scene splits used in the
generalisation experiments. Real Marble-derived scenes arrive through ``geometry.pipeline`` with the same scene schema.
"""
from __future__ import annotations

import numpy as np

CATEGORIES = ("kitchen", "living_room", "bedroom", "laundry_room", "multiroom_apartment")

# 3-4 distinct palettes -> visual diversity
PALETTES = [
    {"floor": [0.62, 0.55, 0.47, 1], "wall": [0.92, 0.91, 0.88, 1], "counter": [0.45, 0.42, 0.40, 1]},
    {"floor": [0.40, 0.42, 0.45, 1], "wall": [0.80, 0.86, 0.90, 1], "counter": [0.70, 0.62, 0.50, 1]},
    {"floor": [0.55, 0.40, 0.30, 1], "wall": [0.95, 0.90, 0.80, 1], "counter": [0.25, 0.28, 0.30, 1]},
    {"floor": [0.70, 0.70, 0.68, 1], "wall": [0.75, 0.80, 0.72, 1], "counter": [0.60, 0.35, 0.25, 1]},
]


def _extras(category: str, rng: np.random.Generator) -> list[dict]:
    j = lambda a, b: float(rng.uniform(a, b))
    if category == "kitchen":
        return [{"name": "fridge", "pos": [0.05, 1.25, 0.9], "half": [0.35, 0.3, 0.9], "rgba": [0.85, 0.85, 0.88, 1]},
                {"name": "stool", "pos": [-1.1, j(-0.9, -0.6), 0.3], "half": [0.17, 0.17, 0.3], "rgba": [0.3, 0.3, 0.35, 1]}]
    if category == "living_room":
        return [{"name": "sofa_seat", "pos": [-1.9, 0.0, 0.22], "half": [0.45, 1.0, 0.22], "rgba": [0.35, 0.4, 0.55, 1]},
                {"name": "sofa_back", "pos": [-2.3, 0.0, 0.6], "half": [0.1, 1.0, 0.3], "rgba": [0.35, 0.4, 0.55, 1]},
                {"name": "coffee_table", "pos": [-1.0, j(-0.4, 0.4), 0.2], "half": [0.3, 0.5, 0.2], "rgba": [0.5, 0.35, 0.2, 1]},
                {"name": "shelf", "pos": [0.15, -1.25, 0.9], "half": [0.2, 0.35, 0.9], "rgba": [0.55, 0.4, 0.25, 1]}]
    if category == "bedroom":
        return [{"name": "bed", "pos": [-2.0, 0.6, 0.28], "half": [1.0, 0.8, 0.28], "rgba": [0.8, 0.8, 0.9, 1]},
                {"name": "wardrobe", "pos": [0.1, 1.3, 1.0], "half": [0.3, 0.5, 1.0], "rgba": [0.5, 0.38, 0.28, 1]},
                {"name": "laundry_pile", "pos": [-1.2, j(-0.9, -0.5), 0.06], "half": [0.2, 0.2, 0.06], "rgba": [0.7, 0.3, 0.3, 1]}]
    if category == "laundry_room":
        return [{"name": "washer", "pos": [0.05, 1.2, 0.45], "half": [0.3, 0.3, 0.45], "rgba": [0.9, 0.9, 0.92, 1]},
                {"name": "dryer", "pos": [0.05, -1.25, 0.45], "half": [0.3, 0.3, 0.45], "rgba": [0.9, 0.9, 0.92, 1]},
                {"name": "basket", "pos": [-1.3, j(-0.7, -0.4), 0.2], "half": [0.25, 0.25, 0.2], "rgba": [0.3, 0.6, 0.8, 1]}]
    if category == "multiroom_apartment":
        return [{"name": "partition_a", "pos": [-1.6, 1.0, 1.35], "half": [0.05, 0.9, 1.35], "rgba": [0.9, 0.9, 0.88, 1]},
                {"name": "partition_b", "pos": [-1.6, -1.2, 1.35], "half": [0.05, 0.6, 1.35], "rgba": [0.9, 0.9, 0.88, 1]},
                {"name": "console", "pos": [0.15, -1.3, 0.4], "half": [0.2, 0.4, 0.4], "rgba": [0.5, 0.35, 0.25, 1]}]
    raise ValueError(category)


def make_scene(category: str = "kitchen", seed: int = 0) -> dict:
    """A reproducible scene spec (see simulation.kitchen_arena.DEFAULT_SCENE)."""
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    rng = np.random.default_rng(seed)
    pal = PALETTES[int(rng.integers(len(PALETTES)))]
    h = float(rng.uniform(0.82, 0.92))  # counter height
    L, W = float(rng.uniform(0.62, 0.78)), float(rng.uniform(1.0, 1.3))
    cup_x0 = float(rng.uniform(-0.05, 0.0))
    return {
        "scene_id": f"proc_{category}_{seed:04d}", "kind": category, "source": "procedural",
        "counter_height": h, "counter_size": [L, W], "counter_x": 0.0,
        "cup_spawn": {"x": [cup_x0, cup_x0 + 0.08], "y": [-0.21, -0.09]},
        "place_target": {"x": float(rng.uniform(0.0, 0.05)), "y": float(rng.uniform(-0.36, -0.32)), "radius": 0.045},
        "decor": {"back_wall": True, "upper_cabinets": category in ("kitchen", "laundry_room"),
                  "dishwasher": False, "dining_table": category in ("kitchen", "multiroom_apartment"),
                  "chairs": int(rng.integers(0, 4)) if category in ("kitchen", "multiroom_apartment") else 0,
                  "extras": _extras(category, rng)},
        "visual": {"floor_rgba": pal["floor"], "wall_rgba": pal["wall"], "counter_rgba": pal["counter"]},
        "static_geoms": None, "cameras": {},
    }


def split_scenes(n_train=10, n_val=3, n_test=5, seed=0) -> dict:
    """Scene-identity splits (spec 13): disjoint seeds, categories balanced round-robin."""
    out, k = {"train": [], "val": [], "test": []}, 0
    for name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for _ in range(n):
            out[name].append(make_scene(CATEGORIES[k % len(CATEGORIES)], seed * 10_000 + k))
            k += 1
    return out
