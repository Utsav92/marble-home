"""Task layouts for the multi-item household tasks (clear_table, load_dishwasher, laundry_basket).

Each layout adds to a scene spec: per-item spawn boxes (sorted by x so the item nearest the robot is handled first and
never blocks a later approach), per-item placement ``slots`` (xy + height of the receptacle surface above the counter) and
static ``fixtures`` (receptacle geometry). Everything is a rigid-body PROXY:

* clear_table      - cups -> a flat serving tray                                   (no table-clearing locomotion)
* load_dishwasher  - cups -> the rack plate of a counter-top dishwasher            (door / rack sliding NOT modelled)
* laundry_basket   - cylindrical "laundry rolls" -> a tote with end rims on the counter (cloth is NOT simulated; items are rigid)

Reachable world-frame workspace of the right hand is x in [-0.35, 0.15], y in [-0.55, -0.05] (measured from the IK
workspace box); the grasp site sits 8 cm behind and 7 cm outboard of an item, so item x <= ~0.2 and y >= ~-0.42 keep both
the grasp and the placement inside it.
"""
from __future__ import annotations

import copy

from marble_home.tasks.definitions import MULTI_ITEM

SLOT_Y = -0.35
# y band per item, increasing with x: the open hand reaches ~0.2 m ahead of the grasp site along +x and ~5 cm to either side of
# (cup_y - 0.07), so a later item must sit at larger y than the previous item's approach corridor
SPAWN_Y = {3: [[-0.20, -0.17], [-0.13, -0.10], [-0.06, -0.03]], 2: [[-0.17, -0.14], [-0.10, -0.07]]}
SPAWN_X = {3: [-0.10, -0.015, 0.07], 2: [-0.06, 0.03]}
JIT = 0.008

_GREY = [0.62, 0.64, 0.68, 1.0]
ITEM_RGBA = {
    "clear_table": [[0.92, 0.92, 0.97, 1], [0.95, 0.80, 0.55, 1], [0.60, 0.80, 0.95, 1]],
    "load_dishwasher": [[0.92, 0.92, 0.97, 1], [0.95, 0.80, 0.55, 1]],
    "laundry_basket": [[0.85, 0.25, 0.25, 1], [0.25, 0.45, 0.85, 1], [0.95, 0.85, 0.30, 1]],
}


def _fx(name, pos, half, rgba, collide=True):
    return {"name": name, "pos": [float(v) for v in pos], "half": [float(v) for v in half], "rgba": rgba, "collide": collide}


def apply_layout(task: str, scene: dict) -> dict:
    """Return a copy of ``scene`` with the task layout applied (idempotent; no-op for single-item tasks)."""
    if task not in MULTI_ITEM or scene.get("layout") == task:
        return scene
    sc = copy.deepcopy(scene)
    n, h = MULTI_ITEM[task], sc["counter_height"]
    cx = sc["counter_x"]
    sx = SPAWN_X[n]
    sc["item_spawn"] = [{"x": [cx + x - JIT, cx + x + JIT], "y": SPAWN_Y[n][k]} for k, x in enumerate(sx)]
    sc["item_rgba"] = ITEM_RGBA[task]
    fx = []
    if task == "clear_table":
        slot_x = [0.11, -0.01, -0.13]  # fill far slot first: the wrist trails the item toward -x, so earlier items must be in FRONT
        top = 0.012
        fx.append(_fx("tray", [cx + 0.02, SLOT_Y, h + 0.006], [0.17, 0.065, 0.006], [0.72, 0.60, 0.42, 1]))
    elif task == "load_dishwasher":
        slot_x = [0.11, -0.01]
        top = 0.086
        xd, yd = 0.06, SLOT_Y
        fx.append(_fx("dw_rack", [cx + xd, yd, h + 0.08], [0.15, 0.062, 0.006], [0.55, 0.57, 0.62, 1]))
        fx.append(_fx("dw_back", [cx + xd + 0.158, yd, h + 0.14], [0.008, 0.062, 0.14], _GREY))
        for k in (0, 2):  # wire-rack legs at the back only (front legs would sit in the hand approach corridor)
            fx.append(_fx(f"dw_leg{k}", [cx + xd + 0.14, yd + (-0.052 if k < 2 else 0.052), h + 0.037], [0.006, 0.006, 0.037], [0.35, 0.36, 0.40, 1]))
        fx.append(_fx("dw_side_in", [cx + xd, yd + 0.068, h + 0.14], [0.15, 0.003, 0.14], _GREY, collide=False))  # visual-only walls
        fx.append(_fx("dw_side_out", [cx + xd, yd - 0.068, h + 0.14], [0.15, 0.003, 0.14], _GREY, collide=False))
    else:  # laundry_basket
        slot_x = [0.11, -0.01, -0.13]  # fill far slot first: the wrist trails the item toward -x, so earlier items must be in FRONT
        top = 0.012
        xb, yb = 0.0, SLOT_Y
        fx.append(_fx("tote_floor", [cx + xb, yb, h + 0.006], [0.19, 0.065, 0.006], [0.78, 0.66, 0.40, 1]))
        fx.append(_fx("tote_back", [cx + xb + 0.196, yb, h + 0.035], [0.006, 0.065, 0.035], [0.70, 0.58, 0.34, 1]))  # only collidable wall
        fx.append(_fx("tote_front", [cx + xb - 0.196, yb, h + 0.035], [0.006, 0.065, 0.035], [0.70, 0.58, 0.34, 1], collide=False))  # visual-only
        fx.append(_fx("tote_outer", [cx + xb, yb - 0.071, h + 0.035], [0.19, 0.006, 0.035], [0.70, 0.58, 0.34, 1], collide=False))
    sc["slots"] = [{"x": cx + x, "y": SLOT_Y, "dz": top} for x in slot_x[:n]]
    sc["fixtures"] = fx
    sc["place_target"] = {"x": sc["slots"][0]["x"], "y": SLOT_Y, "radius": 0.045}
    sc["layout"] = task
    return sc
