"""KitchenManip: RoboSuite environment with a GR1 humanoid in a procedural kitchen.

Tasks: reach_cup, pick_up_cup, place_cup and the multi-item clear_table / load_dishwasher / laundry_basket (see ``marble_home.tasks.definitions``).
Physics is plain MuJoCo; rendering is only used for camera observations.

Observation groups returned by ``step``/``reset`` (flat dict):
  * proprioception / sensors  -> ``robot0_*``                 (policy-visible)
  * images / depth            -> ``<cam>_image``, ``<cam>_depth``  (policy-visible if enabled)
  * object / task state       -> ``cup_*``, ``target_xy``, ``task_phase``, ``progress``  (PRIVILEGED)
The Gymnasium wrapper decides which of these a given policy may see.
"""
from __future__ import annotations

from collections import OrderedDict, deque

import mujoco
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.objects import CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, new_geom
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from marble_home.robots.gr1 import make_controller_config
from marble_home.simulation.kitchen_arena import build_arena, merge_scene
from marble_home.simulation.layouts import apply_layout
from marble_home.tasks import definitions as T

CUP_RADIUS, CUP_HALF_H = 0.033, 0.045


class KitchenManip(ManipulationEnv):
    def __init__(
        self,
        task: str = "pick_up_cup",
        scene: dict | None = None,
        task_cfg: dict | None = None,
        robots: str = "GR1FixedLowerBody",
        control_mode: str = "ik",
        controller_configs=None,
        initialization_noise=None,
        use_camera_obs: bool = False,
        has_renderer: bool = False,
        has_offscreen_renderer: bool | None = None,
        render_camera: str = "robot0_robotview",
        control_freq: int = 20,
        horizon: int | None = None,
        ignore_done: bool = False,
        hard_reset: bool = False,
        camera_names="robot0_robotview",
        camera_heights=96,
        camera_widths=96,
        camera_depths=False,
        terminate_on_success: bool = True,
        seed: int | None = None,
        controller_overrides: dict | None = None,
        **kwargs,
    ):
        self.scene = apply_layout(task, merge_scene(scene))
        self.task_cfg = T.default_task_config(task, **(task_cfg or {}))
        if horizon is not None:
            self.task_cfg.horizon = horizon
        self.control_mode = control_mode
        self.terminate_on_success = terminate_on_success
        self.table_offset = np.array((self.scene["counter_x"], 0.0, self.scene["counter_height"]))
        self.counter_len = self.scene["counter_size"][0]
        if controller_configs is None:
            controller_configs = make_controller_config(robots, control_mode, controller_overrides)
        if has_offscreen_renderer is None:
            has_offscreen_renderer = use_camera_obs
        self._reset_accumulators()
        super().__init__(
            robots=robots,
            env_configuration="default",
            controller_configs=controller_configs,
            base_types="default",
            gripper_types="default",
            initialization_noise=initialization_noise if initialization_noise is not None else None,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            control_freq=control_freq,
            lite_physics=True,
            horizon=self.task_cfg.horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            seed=seed,
            **kwargs,
        )

    # ------------------------------------------------------------------ model
    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.counter_len)
        xpos = (xpos[0] + self.scene["counter_x"], xpos[1], xpos[2])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = build_arena(self.scene)
        arena.set_origin([0, 0, 0])
        self.n_items = int(self.task_cfg.n_items)
        tgt = self.scene["place_target"]
        self.slots = self.scene.get("slots") or [{"x": tgt["x"], "y": tgt["y"], "dz": 0.0}]
        for k, sl in enumerate(self.slots[: self.n_items]):
            arena.worldbody.append(  # visual-only coaster marking each placement region
                new_geom(type="cylinder", name=f"place_marker{k}", pos=[sl["x"], sl["y"], self.scene["counter_height"] + sl["dz"] + 0.0015],
                         size=[tgt["radius"] if self.n_items == 1 else 0.038, 0.0015], rgba=[0.1, 0.8, 0.3, 0.6], group=1, contype=0, conaffinity=0)
            )
        rgbas = self.scene.get("item_rgba") or []
        self.items = []
        for k in range(self.n_items):
            self.items.append(CylinderObject(
                name="cup" if self.n_items == 1 else f"cup{k}", size=[CUP_RADIUS, CUP_HALF_H],
                rgba=rgbas[k] if k < len(rgbas) else [0.92, 0.92, 0.97, 1.0],
                friction=(2.0, 0.005, 0.0001), density=600, rng=self.rng,
            ))
        self.cup = self.items[0]
        spawns = self.scene.get("item_spawn") or [self.scene["cup_spawn"]] * self.n_items
        self.placement_initializers = [
            UniformRandomSampler(
                name=f"CupSampler{k}", mujoco_objects=self.items[k],
                x_range=[sp["x"][0] - self.table_offset[0], sp["x"][1] - self.table_offset[0]],
                y_range=sp["y"], rotation=None, rotation_axis="z", ensure_object_boundary_in_range=False,
                ensure_valid_placement=True, reference_pos=self.table_offset, z_offset=0.002, rng=self.rng,
            ) for k, sp in enumerate(spawns)
        ]
        self.placement_initializer = self.placement_initializers[0]
        self.model = ManipulationTask(
            mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots], mujoco_objects=self.items
        )

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model._model
        self._mj_model = m
        self.item_body_ids = [self.sim.model.body_name2id(o.root_body) for o in self.items]
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
        cup_b = {i for i, n in enumerate(names) if n.startswith("cup")}
        rob_b = {i for i, n in enumerate(names) if n.startswith(("robot0_", "gripper0_"))}
        hand_b = {i for i, n in enumerate(names) if n.startswith(f"gripper0_{self.task_cfg.arm}_")}
        gb = m.geom_bodyid
        collidable = (m.geom_contype > 0) | (m.geom_conaffinity > 0)
        self._all_item_geoms = [{g for g in range(m.ngeom) if gb[g] == b} for b in self.item_body_ids]
        self._cup_geoms = set().union(*self._all_item_geoms)
        self._robot_geoms = {g for g in range(m.ngeom) if gb[g] in rob_b and collidable[g]}
        self._hand_geoms = {g for g in range(m.ngeom) if gb[g] in hand_b and collidable[g]}
        self.grip_site = f"gripper0_{self.task_cfg.arm}_grip_site"
        self.grip_site_id = self.sim.model.site_name2id(self.grip_site)
        self.base_body_id = self.sim.model.body_name2id("robot0_base")
        self.table_z = float(self.scene["counter_height"])
        self._items_done = 0
        self._set_active(0)

    def _set_active(self, k: int):
        """Point the per-item references (cup body, grasp geoms, slot xy, rest height) at item ``k``."""
        self.active = k
        self.cup_body_id = self.item_body_ids[k]
        self._active_geoms = self._all_item_geoms[k]
        sl = self.slots[min(k, len(self.slots) - 1)]
        self.target_xy = np.array([sl["x"], sl["y"]])
        self.cup_rest_z = self.table_z + sl["dz"] + CUP_HALF_H

    # ------------------------------------------------------------- observables
    def _setup_observables(self):
        observables = super()._setup_observables()

        @sensor(modality="object")
        def cup_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.cup_body_id])

        @sensor(modality="object")
        def cup_quat(obs_cache):
            return convert_quat(np.array(self.sim.data.body_xquat[self.cup_body_id]), to="xyzw")

        @sensor(modality="object")
        def cup_to_eef(obs_cache):
            return self.eef_pos_world() - np.array(self.sim.data.body_xpos[self.cup_body_id])

        for s in (cup_pos, cup_quat, cup_to_eef):
            observables[s.__name__] = Observable(name=s.__name__, sensor=s, sampling_rate=self.control_freq)
        return observables

    # ------------------------------------------------------------------ state
    def _reset_accumulators(self):
        self._lift_hold = 0
        self._place_hold = 0
        self._reach_hold = 0
        self._was_lifted = False
        self._invalid_steps = 0
        self._peak_force = 0.0
        self._cup_init_pos = None
        self._last_state = None
        self._success = False
        self._ever_success = False
        self._items_done = 0
        self._cup_hist = deque(maxlen=5)
        self._last_reward_parts = {}

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_accumulators()
        if not self.deterministic_reset:
            for init in self.placement_initializers:
                for obj_pos, obj_quat, obj in init.sample().values():
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        self._set_active(0)
        self.sim.forward()
        self._cup_init_pos = np.array(self.sim.data.body_xpos[self.cup_body_id]).copy()

    def eef_pos_world(self, arm: str | None = None) -> np.ndarray:
        if arm is None or arm == self.task_cfg.arm:
            return np.array(self.sim.data.site_xpos[self.grip_site_id])
        return np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(f"gripper0_{arm}_grip_site")])

    def base_pose(self):
        """(pos, 3x3 rot) of the robot base in world frame."""
        return (np.array(self.sim.data.body_xpos[self.base_body_id]),
                np.array(self.sim.data.body_xmat[self.base_body_id]).reshape(3, 3))

    def _contact_stats(self):
        """(#distinct hand geoms touching cup, max normal force robot<->scene excluding cup)."""
        m, d = self._mj_model, self.sim.data._data
        hand_touch, max_scene = set(), 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = g1 in self._robot_geoms, g2 in self._robot_geoms
            if not (r1 or r2) or (r1 and r2):
                continue  # robot-vs-world only (ignores self contacts, which RoboSuite tunes out)
            other = g2 if r1 else g1
            mine = g1 if r1 else g2
            if other in self._active_geoms:
                if mine in self._hand_geoms:
                    hand_touch.add(mine)
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            max_scene = max(max_scene, abs(float(f6[0])))
        return len(hand_touch), max_scene

    def _finished_items_ok(self) -> bool:
        """Every already-placed item still rests on its slot (not knocked off / dragged away by a later move)."""
        for k in range(self._items_done):
            p = np.array(self.sim.data.body_xpos[self.item_body_ids[k]])
            sl = self.slots[min(k, len(self.slots) - 1)]
            if np.linalg.norm(p[:2] - np.array([sl["x"], sl["y"]])) > self.task_cfg.slot_tol or abs(p[2] - (self.table_z + sl["dz"] + CUP_HALF_H)) > 0.03:
                return False
        return True

    def _compute_state(self) -> T.TaskState:
        cup_pos = np.array(self.sim.data.body_xpos[self.cup_body_id])
        xmat = np.array(self.sim.data.body_xmat[self.cup_body_id]).reshape(3, 3)
        # mean speed over the last <=5 control steps (instantaneous qvel is dominated by rim-rocking jitter)
        hist = self._cup_hist
        speed = 0.0
        if len(hist) >= 2:
            speed = float(np.linalg.norm(cup_pos - hist[0]) / (len(hist) * self.control_timestep))
        n_con, f_scene = self._contact_stats()
        return T.TaskState(
            eef_pos=self.eef_pos_world(), cup_pos=cup_pos, cup_init_pos=self._cup_init_pos,
            cup_up_cos=float(xmat[2, 2]), cup_speed=speed, n_grasp_contacts=n_con,
            target_xy=self.target_xy, table_z=self.table_z, cup_rest_z=self.cup_rest_z, max_scene_force=f_scene,
            lift_hold=self._lift_hold, place_hold=self._place_hold, reach_hold=self._reach_hold, was_lifted=self._was_lifted,
            invalid_steps=self._invalid_steps, peak_force=self._peak_force, items_done=self._items_done,
        )

    def _update_task_state(self, action):
        cfg = self.task_cfg
        s = self._compute_state()
        lifted = T.is_lifted(cfg, s)
        self._lift_hold = self._lift_hold + 1 if (lifted and s.n_grasp_contacts >= 1) else 0
        self._was_lifted = self._was_lifted or (lifted and s.n_grasp_contacts >= 1)
        self._invalid_steps += int(s.max_scene_force > cfg.invalid_force)
        self._peak_force = max(self._peak_force, s.max_scene_force)
        s.lift_hold, s.was_lifted = self._lift_hold, self._was_lifted
        s.invalid_steps, s.peak_force = self._invalid_steps, self._peak_force
        self._place_hold = self._place_hold + 1 if (self._was_lifted and T.placed_stable(cfg, s)) else 0
        s.place_hold = self._place_hold
        if cfg.task in T.MULTI_ITEM and self._items_done < cfg.n_items and self._place_hold >= cfg.place_hold_steps:  # item finished: count it, hand over to the next
            self._items_done += 1
            nxt_done = self._items_done >= cfg.n_items
            if not nxt_done:
                self._set_active(self._items_done)
                self._lift_hold = self._place_hold = self._reach_hold = 0
                self._was_lifted = False
                self._cup_hist.clear()
                self._cup_init_pos = np.array(self.sim.data.body_xpos[self.cup_body_id]).copy()
        s.items_done = self._items_done
        s.all_placed_ok = self._finished_items_ok()
        within = float(np.linalg.norm(s.eef_pos - T.approach_pose(cfg, s.cup_pos))) < cfg.reach_tol
        self._reach_hold = self._reach_hold + 1 if within else 0
        s.reach_hold = self._reach_hold
        self._success = T.check_success(cfg, s)
        self._cup_hist.append(s.cup_pos.copy())
        self._last_state = s
        self._last_reward_parts = T.shaped_reward(cfg, s, np.asarray(action, dtype=float), self._success)

    # --------------------------------------------------------- robosuite hooks
    def _post_action(self, action):
        self._update_task_state(action)
        reward, done, info = super()._post_action(action)
        s = self._last_state
        info.update(
            success=bool(self._success),
            phase=T.current_phase(self.task_cfg, s, self._success),
            n_grasp_contacts=s.n_grasp_contacts,
            invalid_steps=s.invalid_steps,
            peak_force=s.peak_force,
            reward_parts=dict(self._last_reward_parts),
            eef_error=float(np.linalg.norm(s.eef_pos - T.approach_pose(self.task_cfg, s.cup_pos))),
            place_error=float(np.linalg.norm(s.cup_pos[:2] - s.target_xy)),
            items_done=self._items_done,
        )
        return reward, done, info

    def reward(self, action=None):
        return float(self._last_reward_parts.get("total", 0.0))

    def _check_success(self):
        return bool(self._success)

    def _check_terminated(self):
        return bool(self.terminate_on_success and self._success)

    # ------------------------------------------------- privileged task obs
    def _task_obs(self) -> OrderedDict:
        s = self._last_state or self._compute_state()
        ph = T.current_phase(self.task_cfg, s, self._success)
        return OrderedDict(
            target_xy=self.target_xy.astype(np.float32),
            task_phase=np.array([ph], dtype=np.float32),
            progress=np.array([self.timestep / max(1, self.task_cfg.horizon)], dtype=np.float32),
            grasp_flag=np.array([float(s.n_grasp_contacts >= self.task_cfg.min_grasp_contacts)], dtype=np.float32),
        )

    def reset(self):
        obs = super().reset()
        self._update_task_state(np.zeros(self.action_dim))
        obs.update(self._task_obs())
        return obs

    def step(self, action):
        obs, r, d, info = super().step(action)
        obs.update(self._task_obs())
        self._ever_success = self._ever_success or bool(info.get("success"))
        info["ever_success"] = self._ever_success
        if self.terminate_on_success and info.get("success"):
            d = True  # RoboSuite's own `done` only reflects the horizon
        return obs, r, d, info
