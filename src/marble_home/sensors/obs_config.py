"""Observation configuration: which signals a policy may see, with shapes/units.

Privileged simulator state (``PRIVILEGED_KEYS``) is only exposed in ``state`` mode (documented as a
low-dimensional teacher/oracle setting, like RoboMimic low_dim). Vision modes ('rgb', 'rgbd',
'multicam') see ONLY proprioception + camera pixels (+ the goal xy, which in a real system would come
from the task specification, not perception -- flagged ``goal``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

OBS_MODES = ("state", "rgb", "rgbd", "multicam")

# policy-visible proprioception (all from joint encoders / FK; no object info). Units in comments.
PROPRIO_KEYS = (
    ("robot0_joint_pos", 20),                   # rad: waist(3)+head(3)+r arm(7)+l arm(7)
    ("robot0_right_gripper_qpos", 11),          # rad: finger joints
    ("robot0_left_gripper_qpos", 11),           # rad
    ("robot0_base_to_right_eef_pos", 3),        # m, base frame
    ("robot0_base_to_right_eef_quat_site", 4),  # xyzw
    ("robot0_base_to_left_eef_pos", 3),
    ("robot0_base_to_left_eef_quat_site", 4),
)
PROPRIO_DIM = sum(d for _, d in PROPRIO_KEYS)  # 56

# privileged object/task state (simulator ground truth)
PRIVILEGED_KEYS = (("cup_pos", 3), ("cup_quat", 4), ("cup_to_eef", 3))
GOAL_KEYS = (("target_xy", 2),)  # task goal (placement coaster xy); given by the task spec

CAMERAS = {  # short name -> robosuite camera
    "head": "robot0_robotview",
    "wrist_r": "robot0_eye_in_right_hand",
    "wrist_l": "robot0_eye_in_left_hand",
}


@dataclass
class ObsConfig:
    mode: str = "state"
    image_hw: int = 64
    cameras: tuple = ("head",)
    include_goal: bool = True
    sampling_hz: float = 20.0  # == control_freq; every observable sampled once per control step

    def __post_init__(self):
        if self.mode not in OBS_MODES:
            raise ValueError(self.mode)
        if self.mode == "rgb" or self.mode == "rgbd":
            self.cameras = ("head",)
        elif self.mode == "multicam":
            self.cameras = ("head", "wrist_r", "wrist_l")

    @property
    def uses_images(self) -> bool:
        return self.mode != "state"

    @property
    def depth(self) -> bool:
        return self.mode in ("rgbd", "multicam")  # multicam includes head depth only (see gym_env)

    def describe(self) -> dict:
        d = {"mode": self.mode, "proprio_dim": PROPRIO_DIM, "sampling_hz": self.sampling_hz,
             "proprio": {k: n for k, n in PROPRIO_KEYS}}
        if self.mode == "state":
            d["privileged"] = {k: n for k, n in PRIVILEGED_KEYS}
        if self.include_goal:
            d["goal"] = {k: n for k, n in GOAL_KEYS}
        if self.uses_images:
            d["images"] = {c: f"uint8 {self.image_hw}x{self.image_hw}x3 (RGB)" for c in self.cameras}
            if self.depth:
                d["depth"] = f"float32 {self.image_hw}x{self.image_hw}x1, metres (head camera)"
        return d
