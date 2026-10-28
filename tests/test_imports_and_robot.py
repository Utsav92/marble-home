import mujoco
import numpy as np
import pytest

from marble_home.robots import action_space as AS
from marble_home.sensors.obs_config import PROPRIO_DIM, ObsConfig

pytestmark = pytest.mark.slow


def test_dependency_imports():
    import gymnasium, h5py, robosuite, scipy, torch, trimesh  # noqa: F401
    import stable_baselines3  # noqa: F401


def test_mujoco_pin_exposes_qM():
    # robosuite 1.5.2 requires MjData.qM (removed in mujoco >= 3.4)
    d = mujoco.MjData(mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><joint type='free'/><geom size='.1'/></body></worldbody></mujoco>"))
    assert hasattr(d, "qM")


def test_mink_controller_registered():
    from marble_home.robots.gr1 import register_mink_ik

    assert register_mink_ik()


def test_robot_model_compiles(env_state):
    m = env_state.env.sim.model._model
    assert m.nu > 20 and m.njnt > 30 and m.opt.timestep == pytest.approx(0.002)


def test_joint_limits_valid_and_reset_inside(env_state):
    env_state.reset(seed=0)
    m, d = env_state.env.sim.model._model, env_state.env.sim.data._data
    for j in range(m.njnt):
        if m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
            assert lo < hi
            assert lo - 1e-6 <= d.qpos[m.jnt_qposadr[j]] <= hi + 1e-6


def test_reset_deterministic_per_seed(env_state):
    a, _ = env_state.reset(seed=11)
    b, _ = env_state.reset(seed=11)
    c, _ = env_state.reset(seed=12)
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_stepping_100_steps_finite(env_state):
    env_state.reset(seed=1)
    for _ in range(100):
        obs, r, term, trunc, info = env_state.step(np.zeros(4, np.float32))
        assert np.isfinite(obs).all() and np.isfinite(r)
        if term or trunc:
            env_state.reset(seed=1)


def test_state_observation_shape_matches_space(env_state):
    obs, _ = env_state.reset(seed=2)
    assert obs.shape == env_state.observation_space.shape == (PROPRIO_DIM + 2 + 10 + 1,)


def test_rgbd_observation_shapes_and_sensor_output(env_rgbd):
    obs, info = env_rgbd.reset(seed=0)
    assert obs["rgb_head"].shape == (64, 64, 3) and obs["rgb_head"].dtype == np.uint8
    assert obs["depth_head"].shape == (64, 64, 1)
    assert obs["rgb_head"].std() > 1.0  # not a blank frame
    d = obs["depth_head"]
    assert 0.1 < d.min() and d.max() < 10.0  # metres
    assert "cup_pos" in info["privileged"]  # privileged state is NOT inside the policy obs
    assert not any("cup" in k for k in obs)


def test_vision_modes_hide_privileged_state():
    cfg = ObsConfig(mode="rgb")
    assert "privileged" not in cfg.describe()
    assert "privileged" in ObsConfig(mode="state").describe()


def test_action_bounds_and_expansion(env_state):
    assert env_state.action_space.low.min() == -1 and env_state.action_space.high.max() == 1
    env_state.reset(seed=3)
    full = env_state.full_action(np.array([5.0, -5.0, 0.0, 9.0]))  # out-of-range input must be clipped
    assert full.shape == (AS.IK_ACTION_DIM,)
    assert np.all(full[12:18] == 1.0) and np.all(full[18:24] == -1.0)
    lo, hi = AS.WS_CENTER - AS.WS_HALF, AS.WS_CENTER + AS.WS_HALF
    assert np.all(full[:3] >= lo - 1e-9) and np.all(full[:3] <= hi + 1e-9)


def test_reduced_action_roundtrip():
    p = np.array([0.35, -0.25, 0.05])
    a = np.concatenate([AS.pos_to_unit(p), [1.0]])
    full, tgt = AS.reduced_to_full(a, np.zeros(3), np.zeros(6))
    assert np.allclose(tgt, p, atol=1e-9)
    assert np.allclose(AS.full_to_reduced(full)[:3], a[:3], atol=1e-6)


def test_contact_sensing_hand_cup_during_grasp():
    from marble_home.policies.agents import ScriptedAgent
    from marble_home.simulation.gym_env import HouseholdEnv

    env = HouseholdEnv(task="pick_up_cup", obs_mode="state", seed=0)
    obs, info = env.reset(seed=3)
    ag = ScriptedAgent()
    ag.reset(env)
    seen = 0
    for _ in range(80):
        obs, r, term, trunc, info = env.step(ag.act(obs))
        seen = max(seen, info["n_grasp_contacts"])
        if term or trunc:
            break
    env.close()
    assert seen >= 2


def test_invalid_collision_is_detected():
    """Driving the right hand down into the counter must register as an invalid robot<->scene contact."""
    from marble_home.simulation.gym_env import HouseholdEnv

    env = HouseholdEnv(task="reach_cup", obs_mode="state", seed=0)
    env.reset(seed=0)
    for _ in range(5):  # free motion: no contact yet
        obs, r, term, trunc, info = env.step(np.array([0.2, 0.0, 0.0, -1.0], np.float32))
    assert info["invalid_steps"] == 0 and info["peak_force"] == 0.0
    for _ in range(60):  # command the hand to the bottom of the workspace (below the counter top)
        obs, r, term, trunc, info = env.step(np.array([0.2, 0.0, -1.0, -1.0], np.float32))
    env.close()
    # measured on the dev machine: ~770 N peak and ~50 invalid steps; assert conservative bounds
    assert info["peak_force"] > 100.0
    assert info["invalid_steps"] > env.env.task_cfg.max_invalid_steps
    assert not env.env._success  # a reach that exceeds the collision budget must not count as success
