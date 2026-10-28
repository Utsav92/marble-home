"""Scripted expert for the GR1 kitchen tasks (IK control mode).

Uses *privileged* simulator state (cup pose, target) -- it is a teacher / data generator, never a
deployable policy. Output actions follow the ``ik`` layout:
    [right_pose6 | left_pose6 | right_grip6 | left_grip6]   (poses absolute, base frame; grip -1 open .. +1 closed)
"""
from __future__ import annotations

import numpy as np
from robosuite.utils import transform_utils as T

from marble_home.tasks import definitions as TD

PRE_BACKOFF = 0.14  # hand starts this far behind the grasp point so open fingers run alongside the cup
LIFT = 0.14
CARRY_Z = 0.10  # extra clearance while transporting
HI = 0.14  # multi-item: approach the next item from this far above, so the sweep clears placed/other items


def _smooth(u: float) -> float:  # min-jerk
    return u * u * u * (10 + u * (-15 + 6 * u))


class ScriptedExpert:
    def __init__(self, env, noise: float = 0.0, speed: float = 1.0, rng: np.random.Generator | None = None, start_high: bool = False):
        self.env = env
        self.start_high = start_high
        self.cfg = env.task_cfg
        self.noise = noise
        self.speed = speed
        self.rng = rng or np.random.default_rng(0)

    # ------------------------------------------------------------------ setup
    def reset(self, obs: dict):
        env = self.env
        self.bp, self.br = env.base_pose()
        self.aa_right = T.quat2axisangle(obs["robot0_base_to_right_eef_quat_site"])
        self.left6 = np.concatenate(
            [obs["robot0_base_to_left_eef_pos"], T.quat2axisangle(obs["robot0_base_to_left_eef_quat_site"])]
        )
        self.cup0 = np.array(obs["cup_pos"], dtype=float)
        self.off = -np.asarray(self.cfg.approach_offset)  # cup relative to site (fx, fy, fz)
        self.grasp = self.cup0 - self.off
        self.ctx_rel = None
        self.t = 0
        self.seg = 0
        self.seg_t = 0
        self.seg_start = env.eef_pos_world().copy()
        task = self.cfg.task
        n = lambda k: max(2, int(round(k / self.speed)))
        pre = lambda c: c["grasp"] - np.array([PRE_BACKOFF, 0, 0])
        segs = []
        if self.start_high:
            segs += [("approach", lambda c: c["last"] + np.array([0, 0, HI]), -1.0, n(15)),  # straight up first: the hand is still beside the placed item
                     ("approach", lambda c: pre(c) + np.array([0, 0, HI]), -1.0, n(25)),
                     ("approach", lambda c: pre(c), -1.0, n(20))]
        else:
            segs += [("approach", lambda c: pre(c), -1.0, n(30))]
        segs += [("approach", lambda c: c["grasp"], -1.0, n(25))]
        ret = np.asarray(self.cfg.retreat, float)
        up = np.array([0, 0, HI]) if self.cfg.task in TD.MULTI_ITEM else ret  # legacy single-item: second retreat leg continues outward
        if task != "reach_cup":
            segs += [
                ("grasp", lambda c: c["grasp"], +1.0, n(20)),
                ("lift", lambda c: c["grasp"] + np.array([0, 0, LIFT]), +1.0, n(25)),
            ]
            if task == "pick_up_cup":
                segs += [("lift", lambda c: c["grasp"] + np.array([0, 0, LIFT]), +1.0, n(20))]
            else:  # place_cup
                segs += [
                    ("transport", lambda c: self._site_for_cup_xy(c, 0.0), +1.0, n(40)),
                    # closed-loop trim: IK lags by 1-2 cm, so re-aim using the measured cup position
                    ("transport", lambda c: self._trim_to_target(c), +1.0, n(14)),
                    ("release", lambda c: self._site_for_cup_xy(c, 0.004, lower=True), +1.0, n(25)),
                    ("release", lambda c: c["last"], -1.0, n(15)),
                    ("release", lambda c: c["last"] + ret, -1.0, n(20)),
                    ("release", lambda c: c["last"] + up, -1.0, n(15)),
                ]
        self.segs = segs
        self.ctx = {"grasp": self.grasp, "last": self.seg_start}
        self.seg_target = self.segs[0][1](self.ctx)
        self.phase_name = "approach"

    def _trim_to_target(self, c):
        env = self.env
        err = np.append(env.target_xy - env._last_state.cup_pos[:2], 0.0)
        c["rel"] = env._last_state.cup_pos - env.eef_pos_world()  # refresh in-hand offset
        return env.eef_pos_world() + err

    def _site_for_cup_xy(self, c, z_extra, lower=False):
        env = self.env
        rel = c["rel"]  # cup - site measured at end of lift
        goal_cup_xy = env.target_xy
        if lower:  # cup bottom just above the counter given where the cup actually sits in the hand
            z_site = env.cup_rest_z + 0.004 - rel[2]
        else:
            z_site = self.grasp[2] + self.cup_hold_dz + z_extra
        return np.array([goal_cup_xy[0] - rel[0], goal_cup_xy[1] - rel[1], z_site])

    # -------------------------------------------------------------------- act
    def act(self, obs: dict | None = None) -> np.ndarray:
        env = self.env
        name, _, grip, nsteps = self.segs[min(self.seg, len(self.segs) - 1)]
        done_all = self.seg >= len(self.segs)
        if done_all:
            pos = self.seg_target
            grip = self.segs[-1][2]
        else:
            u = _smooth(min(1.0, (self.seg_t + 1) / nsteps))
            pos = self.seg_start + u * (self.seg_target - self.seg_start)
            self.phase_name = name
            self.seg_t += 1
            if self.seg_t >= nsteps:
                self.seg_start = self.seg_target.copy()
                self.ctx["last"] = self.seg_target.copy()
                self.seg += 1
                self.seg_t = 0
                if self.seg < len(self.segs):
                    if name == "lift":  # measure how the cup actually sits in the hand
                        self.ctx["rel"] = env._last_state.cup_pos - env.eef_pos_world()
                        self.cup_hold_dz = float(env.eef_pos_world()[2] - self.grasp[2])
                    self.seg_target = self.segs[self.seg][1](self.ctx)
        if self.noise > 0:
            pos = pos + self.rng.normal(0, self.noise, 3)
        self.t += 1
        right6 = np.concatenate([self.br.T @ (pos - self.bp), self.aa_right])
        return np.concatenate([right6, self.left6, np.full(6, grip), np.full(6, -1.0)])

    @property
    def finished(self) -> bool:
        return self.seg >= len(self.segs)


class MultiItemExpert:
    """Runs the single-item place expert once per item; re-plans whenever the env hands over to the next item."""

    def __init__(self, env, noise: float = 0.0, speed: float = 1.0, rng: np.random.Generator | None = None):
        self.env, self.noise, self.speed, self.rng = env, noise, speed, rng or np.random.default_rng(0)
        self.ex, self.k = None, -1

    def _build(self, obs: dict, k: int):
        o = dict(obs)
        o["cup_pos"] = np.array(self.env.sim.data.body_xpos[self.env.cup_body_id], dtype=float)  # never a stale previous item
        self.ex = ScriptedExpert(self.env, self.noise, self.speed, self.rng, start_high=k > 0)
        self.ex.reset(o)
        self.k = k

    def reset(self, obs: dict):
        self._build(obs, 0)

    def act(self, obs: dict | None = None) -> np.ndarray:
        if obs is not None and self.env.active != self.k:
            self._build(obs, self.env.active)
        return self.ex.act(obs)

    @property
    def finished(self) -> bool:
        return self.ex.finished and self.k >= self.env.n_items - 1

    @property
    def phase_name(self) -> str:
        return self.ex.phase_name


def make_expert(env, noise: float = 0.0, speed: float = 1.0, rng: np.random.Generator | None = None):
    """The right scripted teacher for ``env.task_cfg.task`` (single-item ScriptedExpert or per-item MultiItemExpert)."""
    cls = MultiItemExpert if env.task_cfg.task in TD.MULTI_ITEM else ScriptedExpert
    return cls(env, noise=noise, speed=speed, rng=rng)
