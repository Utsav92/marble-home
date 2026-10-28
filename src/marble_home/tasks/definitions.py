"""Pure task logic: success criteria, shaped reward, phases. No simulator imports.

Success is computed from physical quantities only and is deliberately independent of the
shaped reward (so reward hacking cannot fake success).  All thresholds live in ``TaskConfig``
and are overridable from ``configs/tasks/*.yaml``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# multi-item pick-and-place variants: item count per task (rigid-body proxies; no articulation, no deformables)
MULTI_ITEM = {"clear_table": 3, "load_dishwasher": 2, "laundry_basket": 3}
TASKS = ("reach_cup", "pick_up_cup", "place_cup") + tuple(MULTI_ITEM)
PHASES = ("approach", "grasp", "lift", "transport", "release", "done")


@dataclass
class TaskConfig:
    task: str = "pick_up_cup"
    horizon: int = 200
    arm: str = "right"
    n_items: int = 1          # items that must each be picked and placed (multi-item tasks)
    retreat: tuple = (0.0, -0.12, 0.0)  # expert hand retreat after release (site frame offsets, world)
    slot_tol: float = 0.05    # a finished item must still sit within this xy distance of its slot at the end
    # reach
    # grip-site position at which the right hand encloses the cup, relative to the cup (world frame).
    # Measured empirically from GR1 hand geometry (palm 4.8 cm ahead of the site, fingers curl toward +y):
    # cup sits 8 cm ahead and 7 cm to the +y side of the site -> site = cup - (0.08, 0.07, 0).
    approach_offset: tuple = (-0.08, -0.07, 0.0)
    reach_tol: float = 0.03   # v2: tightened from 0.05 (v1 baseline). The v2 expert reaches ~0.7 cm.
    reach_hold_steps: int = 5  # must STAY within tolerance (a fly-through does not count)
    # pick
    lift_height: float = 0.08
    hold_steps: int = 10
    # place
    place_radius: float = 0.03   # v2: tightened from 0.045 (v1). The v2 expert places within ~0.5 cm.
    place_hold_steps: int = 10
    place_max_speed: float = 0.05
    upright_cos: float = 0.9
    # contacts
    invalid_force: float = 20.0  # N: robot<->scene contact above this counts as an invalid collision step
    max_invalid_steps: int = 5
    max_peak_force: float = 150.0
    min_grasp_contacts: int = 2
    # reward weights
    w_reach: float = 1.0
    w_grasp: float = 0.25
    w_lift: float = 1.0
    w_place: float = 1.0
    w_success: float = 5.0
    w_action: float = 1e-3
    w_collision: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class TaskState:
    """Everything success/reward/phase need; filled by the env each step."""
    eef_pos: np.ndarray
    cup_pos: np.ndarray
    cup_init_pos: np.ndarray
    cup_up_cos: float
    cup_speed: float
    n_grasp_contacts: int
    target_xy: np.ndarray
    table_z: float
    cup_rest_z: float
    max_scene_force: float  # this step, robot<->non-cup contacts
    # episode accumulators maintained by the env
    items_done: int = 0
    all_placed_ok: bool = True
    lift_hold: int = 0
    place_hold: int = 0
    reach_hold: int = 0
    was_lifted: bool = False
    invalid_steps: int = 0
    peak_force: float = 0.0


def approach_pose(cfg: TaskConfig, cup_pos: np.ndarray) -> np.ndarray:
    return cup_pos + np.asarray(cfg.approach_offset)


def is_lifted(cfg: TaskConfig, s: TaskState) -> bool:
    return bool(s.cup_pos[2] - s.cup_init_pos[2] > cfg.lift_height)


def in_place_region(cfg: TaskConfig, s: TaskState) -> bool:
    return bool(np.linalg.norm(s.cup_pos[:2] - s.target_xy) < cfg.place_radius)


def placed_stable(cfg: TaskConfig, s: TaskState) -> bool:
    return bool(
        in_place_region(cfg, s)
        and abs(s.cup_pos[2] - s.cup_rest_z) < 0.02
        and s.cup_speed < cfg.place_max_speed
        and s.cup_up_cos > cfg.upright_cos
        and s.n_grasp_contacts == 0
    )


def collisions_ok(cfg: TaskConfig, s: TaskState) -> bool:
    return s.invalid_steps <= cfg.max_invalid_steps and s.peak_force <= cfg.max_peak_force


def check_success(cfg: TaskConfig, s: TaskState) -> bool:
    if cfg.task == "reach_cup":
        return bool(s.reach_hold >= cfg.reach_hold_steps and collisions_ok(cfg, s))
    if cfg.task == "pick_up_cup":
        return bool(s.lift_hold >= cfg.hold_steps and s.n_grasp_contacts >= 1 and collisions_ok(cfg, s))
    if cfg.task == "place_cup":
        return bool(s.was_lifted and s.place_hold >= cfg.place_hold_steps and collisions_ok(cfg, s))
    if cfg.task in MULTI_ITEM:  # every item placed (the last one counted by the env this step) and none knocked out since
        return bool(s.items_done >= cfg.n_items and s.all_placed_ok and collisions_ok(cfg, s))
    raise ValueError(cfg.task)


def current_phase(cfg: TaskConfig, s: TaskState, success: bool) -> int:
    if success:
        return PHASES.index("done")
    grasped = s.n_grasp_contacts >= cfg.min_grasp_contacts
    if cfg.task == "reach_cup":
        return PHASES.index("approach")
    if not grasped and not s.was_lifted:
        near = np.linalg.norm(s.eef_pos - approach_pose(cfg, s.cup_pos)) < 0.08
        return PHASES.index("grasp" if near else "approach")
    if cfg.task == "pick_up_cup":
        return PHASES.index("lift")
    if grasped and not is_lifted(cfg, s):
        return PHASES.index("grasp")
    if grasped:
        return PHASES.index("transport" if not in_place_region(cfg, s) else "release")
    return PHASES.index("release" if s.was_lifted else "approach")


def shaped_reward(cfg: TaskConfig, s: TaskState, action: np.ndarray, success: bool) -> dict:
    d_reach = np.linalg.norm(s.eef_pos - approach_pose(cfg, s.cup_pos))
    parts = {"reach": cfg.w_reach * (1 - np.tanh(10.0 * d_reach)), "grasp": 0.0, "lift": 0.0, "place": 0.0}
    if cfg.task != "reach_cup":
        grasped = s.n_grasp_contacts >= cfg.min_grasp_contacts
        parts["grasp"] = cfg.w_grasp if grasped else 0.0
        if grasped:
            parts["lift"] = cfg.w_lift * float(np.clip((s.cup_pos[2] - s.cup_init_pos[2]) / cfg.lift_height, 0, 1))
    if cfg.task not in ("reach_cup", "pick_up_cup") and s.was_lifted:
        d_xy = np.linalg.norm(s.cup_pos[:2] - s.target_xy)
        parts["place"] = cfg.w_place * (1 - np.tanh(10.0 * d_xy))
    parts["action"] = -cfg.w_action * float(np.mean(np.square(action)))
    parts["collision"] = -cfg.w_collision * float(s.max_scene_force > cfg.invalid_force)
    parts["success"] = cfg.w_success if success else 0.0
    parts["total"] = float(sum(parts.values()))
    return parts


def default_task_config(task: str, **overrides) -> TaskConfig:
    base = {"task": task}
    if task == "reach_cup":
        base["horizon"] = 120
    elif task == "pick_up_cup":
        base["horizon"] = 220
    elif task == "place_cup":
        base["horizon"] = 320
    elif task in MULTI_ITEM:
        n = MULTI_ITEM[task]
        base.update(n_items=n, horizon=340 * n + 60)
    base.update(overrides)
    return TaskConfig.from_dict(base)
