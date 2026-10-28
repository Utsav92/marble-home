"""Domain randomisation with independent visual / physics / sensor groups (for controlled ablations).

All ranges are bounded and keep scenes physically valid (mass/friction/damping scaled, never zeroed). Model
parameters are re-sampled from the ORIGINAL values at every reset (no random walk). Geometry randomisation
(counter height, layout, chairs) is done at scene level by ``scene_gen``; object spawn pose by the env sampler.

Presets: none | visual | physics | sensor | combined.  Visual randomisation only affects camera pixels, so it is
irrelevant to state-based policies (documented in the ablation report).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

PRESETS = {
    "none": dict(visual=False, physics=False, sensor=False),
    "visual": dict(visual=True, physics=False, sensor=False),
    "physics": dict(visual=False, physics=True, sensor=False),
    "sensor": dict(visual=False, physics=False, sensor=True),
    "combined": dict(visual=True, physics=True, sensor=True),
}


@dataclass
class DRConfig:
    visual: bool = False
    physics: bool = False
    sensor: bool = False
    # visual
    light_scale: tuple = (0.6, 1.4)
    light_dir_jitter: float = 0.3
    color_jitter: float = 0.15
    exposure: tuple = (0.75, 1.25)
    # physics
    cup_mass_scale: tuple = (0.6, 1.6)
    friction_scale: tuple = (0.7, 1.3)
    damping_scale: tuple = (0.7, 1.4)
    # sensor
    proprio_noise: float = 0.002
    rgb_noise: float = 4.0
    depth_noise: float = 0.01
    cam_pos_jitter: float = 0.01
    obs_delay: int = 1          # frames (max); sampled uniformly 0..obs_delay per episode
    frame_dropout: float = 0.02  # prob. a camera frame repeats the previous one

    @classmethod
    def preset(cls, name: str, **kw):
        return cls(**{**PRESETS[name], **kw})


class Randomizer:
    def __init__(self, cfg: DRConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._orig = None
        self._buf: deque = deque()
        self._delay = 0
        self._last_img = {}

    # ------------------------------------------------------------------ model-level (reset time)
    def _snapshot(self, env):
        m = env.sim.model._model
        self._orig = {
            "light_diffuse": m.light_diffuse.copy(), "light_dir": m.light_dir.copy(), "geom_rgba": m.geom_rgba.copy(),
            "body_mass": m.body_mass.copy(), "geom_friction": m.geom_friction.copy(), "dof_damping": m.dof_damping.copy(),
            "cam_pos": m.cam_pos.copy(),
        }

    def apply(self, env, seed: int | None = None):
        """Re-sample model parameters (call BEFORE env.reset)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed + 7919)
        m = env.sim.model._model
        if self._orig is None:
            self._snapshot(env)
        o, c, r = self._orig, self.cfg, self.rng
        m.light_diffuse[:] = o["light_diffuse"]
        m.light_dir[:] = o["light_dir"]
        m.geom_rgba[:] = o["geom_rgba"]
        m.body_mass[:] = o["body_mass"]
        m.geom_friction[:] = o["geom_friction"]
        m.dof_damping[:] = o["dof_damping"]
        m.cam_pos[:] = o["cam_pos"]
        self._exposure = 1.0
        if c.visual:
            m.light_diffuse[:] = o["light_diffuse"] * r.uniform(*c.light_scale)
            d = o["light_dir"] + r.normal(0, c.light_dir_jitter, o["light_dir"].shape)
            m.light_dir[:] = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
            for g in range(m.ngeom):  # visual-only geoms (group 1): colour jitter, alpha untouched
                if m.geom_group[g] == 1 and m.geom_rgba[g, 3] > 0.99:
                    m.geom_rgba[g, :3] = np.clip(o["geom_rgba"][g, :3] + r.normal(0, c.color_jitter, 3), 0.02, 1.0)
            self._exposure = float(r.uniform(*c.exposure))
        if c.physics:
            cup = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cup_main")
            if cup < 0:
                cup = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, env.cup.root_body)
            if cup >= 0:
                m.body_mass[cup] = o["body_mass"][cup] * r.uniform(*c.cup_mass_scale)
            fs = r.uniform(*c.friction_scale)
            for g in env._cup_geoms | env._hand_geoms:
                m.geom_friction[g, 0] = o["geom_friction"][g, 0] * fs
            m.dof_damping[:] = o["dof_damping"] * r.uniform(*c.damping_scale)
        if c.sensor:
            m.cam_pos[:] = o["cam_pos"] + r.normal(0, c.cam_pos_jitter, o["cam_pos"].shape)
        self._delay = int(r.integers(0, c.obs_delay + 1)) if c.sensor else 0
        self._buf.clear()
        self._last_img = {}

    # ------------------------------------------------------------------ observation-level (every step)
    def perturb_obs(self, obs):
        """Apply sensor noise / exposure / dropout / delay to a policy observation (vector or dict)."""
        c, r = self.cfg, self.rng
        if not (c.sensor or c.visual):
            return obs
        if isinstance(obs, dict):
            out = dict(obs)
            if c.sensor:
                out["proprio"] = obs["proprio"] + r.normal(0, c.proprio_noise, obs["proprio"].shape).astype(np.float32)
            for k in list(out):
                if k.startswith("rgb_"):
                    img = out[k].astype(np.float32)
                    if c.visual:
                        img = img * self._exposure
                    if c.sensor:
                        img = img + r.normal(0, c.rgb_noise, img.shape)
                        if k in self._last_img and r.random() < c.frame_dropout:
                            img = self._last_img[k]
                    img = np.clip(img, 0, 255)
                    self._last_img[k] = img
                    out[k] = img.astype(np.uint8)
                elif k.startswith("depth") and c.sensor:
                    out[k] = np.clip(out[k] + r.normal(0, c.depth_noise, out[k].shape), 0.05, 10).astype(np.float32)
            return self._delayed(out)
        if c.sensor:
            noisy = obs + r.normal(0, c.proprio_noise, obs.shape).astype(np.float32)
            return self._delayed(noisy)
        return obs

    def _delayed(self, obs):
        if self._delay == 0:
            return obs
        self._buf.append(obs)
        while len(self._buf) > self._delay + 1:
            self._buf.popleft()
        return self._buf[0]
