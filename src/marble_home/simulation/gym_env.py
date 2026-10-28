"""Gymnasium adapter around KitchenManip.

Policy observation = what ``ObsConfig`` allows. Everything else (cup pose, phase, reward parts, ...)
is returned under ``info["privileged"]`` and must not be fed to vision policies.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from robosuite.utils import transform_utils as RT
from robosuite.utils.camera_utils import get_real_depth_map

from marble_home.robots import action_space as AS
from marble_home.sensors.obs_config import CAMERAS, GOAL_KEYS, PRIVILEGED_KEYS, PROPRIO_KEYS, ObsConfig
from marble_home.simulation.kitchen_env import KitchenManip
from marble_home.tasks import definitions as TD


def _flip(img):  # robosuite/MuJoCo images are bottom-up
    return np.ascontiguousarray(img[::-1])


class HouseholdEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str = "pick_up_cup",
        obs_mode: str = "state",
        action_mode: str = "ee_abs",
        scene: dict | None = None,
        task_cfg: dict | None = None,
        image_hw: int = 64,
        reward_shaping: bool = True,
        seed: int | None = 0,
        randomizer=None,
        dr: str | None = None,
        render_camera: str = "eval_side",
        controller_overrides: dict | None = None,
    ):
        self.obs_cfg = ObsConfig(mode=obs_mode, image_hw=image_hw)
        self.action_mode = action_mode
        self.reward_shaping = reward_shaping
        if randomizer is None and dr and dr != "none":
            from marble_home.simulation.randomization import DRConfig, Randomizer

            randomizer = Randomizer(DRConfig.preset(dr), seed=seed or 0)
        self.randomizer = randomizer
        self.render_camera = render_camera
        cams = [CAMERAS[c] for c in self.obs_cfg.cameras] if self.obs_cfg.uses_images else None
        depths = False
        if self.obs_cfg.depth:
            depths = [True] + [False] * (len(cams) - 1)
        self.env = KitchenManip(
            task=task, scene=scene, task_cfg=task_cfg, control_mode="ik",
            use_camera_obs=self.obs_cfg.uses_images, has_offscreen_renderer=True,
            camera_names=cams if cams else "robot0_robotview", camera_heights=image_hw, camera_widths=image_hw,
            camera_depths=depths, seed=seed, controller_overrides=controller_overrides,
        )
        self.task = task
        self.horizon = self.env.task_cfg.horizon
        self.action_space = spaces.Box(-1.0, 1.0, shape=(AS.ACTION_DIM,), dtype=np.float32)
        proprio_dim = sum(d for _, d in PROPRIO_KEYS) + (sum(d for _, d in GOAL_KEYS) if self.obs_cfg.include_goal else 0)
        if self.obs_cfg.mode == "state":
            dim = proprio_dim + sum(d for _, d in PRIVILEGED_KEYS) + 1  # + progress
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        else:
            sp = {"proprio": spaces.Box(-np.inf, np.inf, shape=(proprio_dim,), dtype=np.float32)}
            for c in self.obs_cfg.cameras:
                sp[f"rgb_{c}"] = spaces.Box(0, 255, shape=(image_hw, image_hw, 3), dtype=np.uint8)
            if self.obs_cfg.depth:
                sp["depth_head"] = spaces.Box(0.0, 10.0, shape=(image_hw, image_hw, 1), dtype=np.float32)
            self.observation_space = spaces.Dict(sp)
        self._aa_right = None
        self._left6 = None
        self._target = None
        self._raw = None

    # ---------------------------------------------------------------- helpers
    def _proprio(self, raw) -> np.ndarray:
        parts = [np.asarray(raw[k], dtype=np.float32).ravel() for k, _ in PROPRIO_KEYS]
        if self.obs_cfg.include_goal:
            parts += [np.asarray(raw[k], dtype=np.float32).ravel() for k, _ in GOAL_KEYS]
        return np.concatenate(parts)

    def _policy_obs(self, raw):
        proprio = self._proprio(raw)
        if self.obs_cfg.mode == "state":
            priv = [np.asarray(raw[k], dtype=np.float32).ravel() for k, _ in PRIVILEGED_KEYS]
            return np.concatenate([proprio] + priv + [np.asarray(raw["progress"], dtype=np.float32)])
        obs = {"proprio": proprio}
        for c in self.obs_cfg.cameras:
            obs[f"rgb_{c}"] = _flip(raw[f"{CAMERAS[c]}_image"]).astype(np.uint8)
        if self.obs_cfg.depth:
            d = _flip(raw[f"{CAMERAS['head']}_depth"])
            obs["depth_head"] = get_real_depth_map(self.env.sim, d).astype(np.float32).reshape(*d.shape[:2], 1)
        return obs

    def privileged(self, raw) -> dict:
        return {
            "cup_pos": np.asarray(raw["cup_pos"], np.float32), "cup_quat": np.asarray(raw["cup_quat"], np.float32),
            "task_phase": int(raw["task_phase"][0]), "grasp_flag": float(raw["grasp_flag"][0]),
            "eef_pos_world": self.env.eef_pos_world().astype(np.float32),
        }

    # -------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:  # per-instance reseed (in place: samplers hold a reference to env.rng)
            self.env.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        if self.randomizer is not None:
            self.randomizer.apply(self.env, seed)
        raw = self.env.reset()
        self._raw = raw
        self._aa_right = RT.quat2axisangle(raw["robot0_base_to_right_eef_quat_site"])
        self._left6 = np.concatenate([raw["robot0_base_to_left_eef_pos"],
                                      RT.quat2axisangle(raw["robot0_base_to_left_eef_quat_site"])])
        self._target = np.asarray(raw["robot0_base_to_right_eef_pos"], dtype=float).copy()
        return self._obs_and_info(raw)

    def _obs_and_info(self, raw, extra=None):
        info = {"privileged": self.privileged(raw), "success": bool(self.env._success)}
        if extra:
            info.update(extra)
        obs = self._policy_obs(raw)
        if self.randomizer is not None:  # sensor noise/delay/dropout affect the POLICY observation only
            obs = self.randomizer.perturb_obs(obs)
        return obs, info

    def full_action(self, a) -> np.ndarray:
        full, self._target = AS.reduced_to_full(
            a, self._aa_right, self._left6,
            p_base=np.asarray(self._raw["robot0_base_to_right_eef_pos"], dtype=float),
            mode=self.action_mode, prev_target=self._target,
        )
        return full

    def step(self, action):
        full = self.full_action(action)
        raw, r, done, einfo = self.env.step(full)
        self._raw = raw
        success = bool(einfo["success"])
        reward = float(einfo["reward_parts"]["total"]) if self.reward_shaping else float(success)
        terminated = success
        truncated = (not terminated) and self.env.timestep >= self.horizon
        obs, info = self._obs_and_info(raw, extra=einfo)
        info.update(full_action=full, ever_success=einfo["ever_success"])
        return obs, reward, terminated, truncated, info

    def render(self, width: int = 480, height: int = 360, camera: str | None = None):
        return _flip(self.env.sim.render(width=width, height=height, camera_name=camera or self.render_camera))

    def close(self):
        self.env.close()
