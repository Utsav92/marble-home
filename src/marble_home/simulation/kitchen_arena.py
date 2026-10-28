"""Procedural kitchen arena for RoboSuite.

A ``TableArena`` provides floor + the working counter; ``add_kitchen_decor`` bolts static
furniture on as plain MJCF geoms. All dimensions are metres. The same scene-spec dict is what
the Marble pipeline emits for *dynamic* content, so procedural and Marble scenes share a schema.

Nothing here is measured from a real kitchen: sizes follow common residential conventions
(counter 0.9 m, dining table 0.75 m, dishwasher 0.6 m wide) and are configurable.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import new_geom

DEFAULT_SCENE: dict[str, Any] = {
    "scene_id": "proc_kitchen_default",
    "kind": "kitchen",
    "counter_height": 0.85,  # work-surface height above floor
    "counter_size": [0.70, 1.20],  # x (depth, toward robot), y (width)
    "counter_x": 0.0,
    # GR1's RIGHT hand is on -y (left on +y): right-arm workspace is y<0. World xy ranges.
    "cup_spawn": {"x": [-0.05, 0.05], "y": [-0.22, -0.08]},
    "place_target": {"x": 0.02, "y": -0.34, "radius": 0.045},
    "decor": {"back_wall": True, "upper_cabinets": True, "dishwasher": False, "dining_table": True, "chairs": 2},
    # Simplification: a *counter-top* open dishwasher (rack plate 0.08 m above the counter), because an
    # under-counter rack (z~0.5 m) is outside the reach of the fixed-base GR1 torso. Static, not articulated.
    "dishwasher": {"x": 0.10, "y": -0.42, "rack_above_counter": 0.08},
    "visual": {"floor_rgba": [0.62, 0.55, 0.47, 1], "wall_rgba": [0.92, 0.91, 0.88, 1],
               "counter_rgba": [0.45, 0.42, 0.40, 1]},
    # Marble-derived static geometry in the TASK frame (see geometry.pipeline): {"boxes": [{pos, half}],
    # "hulls": [{name, file}], "visual_mesh": path|None}. None for procedural scenes.
    "static_geoms": None,
    "cameras": {},  # extra cameras: name -> {"pos": [...], "lookat": [...]}
}


def merge_scene(spec: dict | None) -> dict:
    out = copy.deepcopy(DEFAULT_SCENE)
    if spec:
        for k, v in spec.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k].update(v)
            else:
                out[k] = v
    return out


def _box(arena, name, pos, half, rgba, collide=True, friction=None, visible=True):
    # RoboSuite renders only group-1 geoms and collides group-0 geoms, so a solid needs both.
    if collide:
        kw = dict(type="box", name=name, pos=pos, size=half, rgba=rgba, group=0, contype=1, conaffinity=1)
        if friction is not None:
            kw["friction"] = friction
        arena.worldbody.append(new_geom(**kw))
    if visible:
        arena.worldbody.append(
            new_geom(type="box", name=name + "_vis", pos=pos, size=half, rgba=rgba, group=1, contype=0, conaffinity=0)
        )


def _lookat_quat(pos, target, up=(0, 0, 1)):
    """MuJoCo camera quaternion (wxyz): camera looks down its -z axis, +y is image-up."""
    from scipy.spatial.transform import Rotation as R

    pos, target, up = (np.asarray(v, float) for v in (pos, target, up))
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:  # looking straight along `up`: pick any perpendicular
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = R.from_matrix(np.stack([x, y, z], axis=1)).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


EVAL_CAMERAS = {
    # name: (pos, lookat) -- all inside the room (the back wall hides RoboSuite's default frontview/agentview)
    "eval_side": ((-1.25, -1.05, 1.55), (-0.05, -0.12, 0.95)),
    "eval_front3q": ((-1.35, 0.95, 1.55), (-0.05, -0.12, 0.95)),
    "eval_top": ((-0.15, -0.1, 2.25), (-0.1, -0.1, 0.9)),
    "eval_close": ((-0.95, -0.95, 1.32), (0.0, -0.25, 0.92)),  # table + right arm filling the frame (video footage)
}


def _add_static_geoms(arena, sg: dict):
    """Marble-derived static geometry. Boxes: collision + visual twin. Hulls: convex mesh geoms (MuJoCo collides
    meshes as convex hulls, so only compact components are ever exported as meshes). Visual mesh: no collision."""
    import xml.etree.ElementTree as ET

    for i, b in enumerate(sg.get("boxes", [])):
        ceiling = b["pos"][2] - b["half"][2] >= 2.3  # keep collision, hide from cameras so overview shots see inside
        _box(arena, f"mb_box_{i}", b["pos"], b["half"], [0.78, 0.74, 0.68, 1], visible=not ceiling)
    for h in sg.get("hulls", []):
        ET.SubElement(arena.asset, "mesh", name=h["name"], file=h["file"])
        arena.worldbody.append(new_geom(type="mesh", name=h["name"] + "_col", mesh=h["name"], size=[1, 1, 1], pos=[0, 0, 0],
                                        rgba=[0.7, 0.6, 0.5, 1], group=0, contype=1, conaffinity=1))
        arena.worldbody.append(new_geom(type="mesh", name=h["name"] + "_vis", mesh=h["name"], size=[1, 1, 1], pos=[0, 0, 0],
                                        rgba=[0.7, 0.6, 0.5, 1], group=1, contype=0, conaffinity=0))


def build_arena(scene: dict) -> TableArena:
    sc = merge_scene(scene)
    cx, h = sc["counter_x"], sc["counter_height"]
    L, W = sc["counter_size"]
    arena = TableArena(
        table_full_size=(L, W, 0.04),
        table_friction=(1.0, 5e-3, 1e-4),
        table_offset=(cx, 0, h),
        has_legs=False,
    )
    vis = sc["visual"]
    arena.table_visual.set("rgba", " ".join(map(str, vis["counter_rgba"])))
    for cam, (cpos, look) in EVAL_CAMERAS.items():
        arena.set_camera(camera_name=cam, pos=list(cpos), quat=list(_lookat_quat(cpos, look)))
    for cam, c in (sc.get("cameras") or {}).items():
        arena.set_camera(camera_name=cam, pos=list(c["pos"]), quat=list(_lookat_quat(c["pos"], c["lookat"])))
    if sc.get("static_geoms"):
        _add_static_geoms(arena, sc["static_geoms"])
    dec = sc["decor"]
    top = h
    # base cabinet carcass under the work counter (collision: keeps the robot from clipping the counter front)
    _box(arena, "counter_carcass", [cx, 0, (top - 0.04) / 2], [L / 2 - 0.02, W / 2 - 0.02, (top - 0.04) / 2],
         [0.80, 0.78, 0.74, 1])
    if dec.get("back_wall"):
        _box(arena, "back_wall", [cx + L / 2 + 0.05, 0, 1.35], [0.05, 2.0, 1.35], vis["wall_rgba"])
    if dec.get("upper_cabinets"):
        _box(arena, "upper_cabinet_r", [cx + L / 2 - 0.10, 0.62, 1.70], [0.17, 0.55, 0.30], [0.85, 0.83, 0.80, 1])
        _box(arena, "upper_cabinet_l", [cx + L / 2 - 0.10, -0.62, 1.70], [0.17, 0.55, 0.30], [0.85, 0.83, 0.80, 1])
    if dec.get("dishwasher"):
        d = sc["dishwasher"]
        # open counter-top dishwasher: back + two side walls + a rack plate (static)
        x, y = d["x"], d["y"]
        rk = top + d["rack_above_counter"]
        wz = top + 0.14
        _box(arena, "dw_side_a", [x, y - 0.17, wz], [0.17, 0.008, 0.14], [0.75, 0.75, 0.78, 1])
        _box(arena, "dw_side_b", [x, y + 0.17, wz], [0.17, 0.008, 0.14], [0.75, 0.75, 0.78, 1])
        _box(arena, "dw_back", [x + 0.17, y, wz], [0.008, 0.17, 0.14], [0.75, 0.75, 0.78, 1])
        _box(arena, "dw_rack", [x, y, rk], [0.16, 0.16, 0.006], [0.55, 0.57, 0.62, 1], friction=[1.0, 5e-3, 1e-4])
    if dec.get("dining_table"):
        _box(arena, "dining_top", [cx - 1.40, 0.0, 0.75], [0.45, 0.45, 0.02], [0.55, 0.38, 0.25, 1])
        for i, (lx, ly) in enumerate([(-1.0, 0.4), (-1.0, -0.4), (-1.8, 0.4), (-1.8, -0.4)]):
            _box(arena, f"dining_leg{i}", [lx, ly, 0.37], [0.025, 0.025, 0.37], [0.4, 0.28, 0.2, 1])
    for item in dec.get("extras", []):  # category furniture (sofa, bed, washer, ...): name, pos, half, rgba (static boxes)
        _box(arena, item["name"], item["pos"], item["half"], item.get("rgba", [0.6, 0.6, 0.6, 1]))
    for f in sc.get("fixtures") or []:  # task receptacles (tray / dishwasher rack / tote), see simulation.layouts
        _box(arena, f["name"], f["pos"], f["half"], f["rgba"], collide=f.get("collide", True), friction=[1.0, 5e-3, 1e-4] if f.get("collide", True) else None)
    for i in range(int(dec.get("chairs", 0))):
        sy = 0.65 if i % 2 == 0 else -0.65
        _box(arena, f"chair{i}_seat", [cx - 1.40, sy, 0.45], [0.20, 0.20, 0.02], [0.3, 0.3, 0.35, 1])
        _box(arena, f"chair{i}_back", [cx - 1.40, sy + np.sign(sy) * 0.19, 0.75], [0.20, 0.015, 0.28],
             [0.3, 0.3, 0.35, 1])
    return arena
