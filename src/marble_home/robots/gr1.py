"""GR1 humanoid embodiment adapter (RoboSuite 1.5.x).

Only ``GR1FixedLowerBody`` (Level A: fixed-base, torso+head+both arms+hands) is used for
learned manipulation. ``GR1FloatingBody`` / ``GR1`` are exposed for Level B/C probing but
no learned locomotion exists in this repo (see docs/limitations.md).
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass

import robosuite
from robosuite.controllers import load_composite_controller_config

LEVEL_ROBOTS = {"A": "GR1FixedLowerBody", "B": "GR1FloatingBody", "C": "GR1"}

# control modes == action representations compared in Experiment 6
CONTROL_MODES = ("ik", "osc", "joint")


@dataclass(frozen=True)
class RobotSpec:
    robot: str = "GR1FixedLowerBody"
    control_mode: str = "ik"
    # EE reference sites of the Fourier hands (RoboSuite naming)
    right_site: str = "gripper0_right_grip_site"
    left_site: str = "gripper0_left_grip_site"


_REGISTERED = False

# Controller v2 (default): the whole-body IK may use the waist (yaw/pitch posture weight 1 instead of RoboSuite's 10).
# Measured effect (scripts/expert_map.py): scripted pick succeeds at ALL cup positions in the spawn region (was 81% on a 21-point
# grid, failing deterministically in the far-outboard corners) and end-effector error drops from ~0.015-0.02 m to ~0.006-0.008 m.
# Reproduce the v1 baseline with overrides={"ik_posture_weights": {"robot0_torso_waist_yaw": 10.0, "robot0_torso_waist_pitch": 10.0}}.
CONTROLLER_V2_OVERRIDES = {"ik_posture_weights": {"robot0_torso_waist_yaw": 1.0, "robot0_torso_waist_pitch": 1.0}}
V1_OVERRIDES = {"ik_posture_weights": {"robot0_torso_waist_yaw": 10.0, "robot0_torso_waist_pitch": 10.0}}


def register_mink_ik() -> bool:
    """Import the vendored WholeBodyMinkIK so ``WHOLE_BODY_MINK_IK`` is registered.

    Returns True on success. The RoboSuite 1.5.2 wheel omits ``robosuite.examples`` so the
    stock import in ``robosuite/__init__`` silently fails.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    from marble_home.robots.vendor import mink_controller  # noqa: F401  (registers on import)

    _REGISTERED = True
    return True


def _robot_cfg_path(robot: str) -> str:
    import robosuite.controllers as C

    name = {
        "GR1FixedLowerBody": "default_gr1_fixed_lower_body.json",
        "GR1FloatingBody": "default_gr1_floating_body.json",
        "GR1": "default_gr1.json",
    }[robot]
    return os.path.join(os.path.dirname(C.__file__), "config", "robots", name)


def make_controller_config(robot: str = "GR1FixedLowerBody", control_mode: str = "ik", overrides: dict | None = None) -> dict:
    """Return a composite-controller config dict for ``suite.make(controller_configs=...)``.

    ``overrides`` (ik mode only) updates ``composite_controller_specific_configs``; e.g. {"ik_posture_weights": {...}} to let the
    waist move more (lower weight = freer joint).
    """
    if control_mode not in CONTROL_MODES:
        raise ValueError(f"control_mode must be one of {CONTROL_MODES}")
    if control_mode == "ik":
        register_mink_ik()
        cfg = load_composite_controller_config(robot=robot)
        spec = cfg["composite_controller_specific_configs"]
        merged = {**CONTROLLER_V2_OVERRIDES, **(overrides or {})}
        merged["ik_posture_weights"] = {**CONTROLLER_V2_OVERRIDES["ik_posture_weights"], **(overrides or {}).get("ik_posture_weights", {})}
        if merged:
            for k, v in merged.items():
                if isinstance(v, dict) and isinstance(spec.get(k), dict):
                    spec[k] = {**spec[k], **v}
                else:
                    spec[k] = v
        return cfg
    if control_mode == "osc":
        return load_composite_controller_config(controller="BASIC", robot=robot)
    # joint: the robot's own default json already declares JOINT_POSITION body parts (used by
    # the IK composite as its fallback); run it through the loader (it flattens the nested
    # "arms" key) and demote the composite to BASIC so those parts are driven directly.
    register_mink_ik()
    cfg = copy.deepcopy(load_composite_controller_config(robot=robot))
    cfg["type"] = "BASIC"
    cfg.pop("composite_controller_specific_configs", None)
    return cfg


def robosuite_version() -> str:
    return robosuite.__version__
