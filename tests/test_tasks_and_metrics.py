"""Pure-logic tests (no simulator): task success/reward/phase, metrics."""
import numpy as np
import pytest

from marble_home.evaluation.metrics import bootstrap_ci, mcnemar_exact, summarize, wilson
from marble_home.tasks import definitions as TD


def state(**kw):
    base = dict(eef_pos=np.array([0.0, -0.3, 0.9]), cup_pos=np.array([0.05, -0.2, 0.897]), cup_init_pos=np.array([0.05, -0.2, 0.897]),
                cup_up_cos=1.0, cup_speed=0.0, n_grasp_contacts=0, target_xy=np.array([0.02, -0.34]), table_z=0.85,
                cup_rest_z=0.895, max_scene_force=0.0)
    base.update(kw)
    return TD.TaskState(**base)


def test_reach_success_uses_approach_pose_and_collision_budget():
    cfg = TD.default_task_config("reach_cup")
    s = state()
    s.eef_pos = s.cup_pos + np.asarray(cfg.approach_offset) + np.array([0.01, 0, 0])
    s.reach_hold = cfg.reach_hold_steps - 1
    assert not TD.check_success(cfg, s)  # a fly-through is not a reach: must hold
    s.reach_hold = cfg.reach_hold_steps
    assert TD.check_success(cfg, s)
    s.reach_hold = 0
    assert not TD.check_success(cfg, s)  # moved away -> hold counter reset
    s.reach_hold = cfg.reach_hold_steps
    s.invalid_steps = cfg.max_invalid_steps + 1
    assert not TD.check_success(cfg, s)  # held, but too many invalid collisions


def test_pick_requires_hold_duration_and_contact():
    cfg = TD.default_task_config("pick_up_cup")
    s = state(cup_pos=np.array([0.05, -0.2, 0.897 + 0.12]), n_grasp_contacts=4, lift_hold=cfg.hold_steps - 1)
    assert TD.is_lifted(cfg, s) and not TD.check_success(cfg, s)
    s.lift_hold = cfg.hold_steps
    assert TD.check_success(cfg, s)
    s.n_grasp_contacts = 0  # cup in the air without hand contact (thrown) does not count
    assert not TD.check_success(cfg, s)


def test_place_requires_region_rest_release_and_upright():
    cfg = TD.default_task_config("place_cup")
    s = state(cup_pos=np.array([0.02, -0.34, 0.8955]), was_lifted=True, place_hold=cfg.place_hold_steps)
    assert TD.placed_stable(cfg, s) and TD.check_success(cfg, s)
    for mutate in (dict(cup_pos=np.array([0.2, -0.34, 0.8955])),        # wrong place
                   dict(n_grasp_contacts=2),                              # still held
                   dict(cup_up_cos=0.3),                                  # knocked over
                   dict(cup_speed=0.5)):                                  # still moving
        s2 = state(cup_pos=np.array([0.02, -0.34, 0.8955]), was_lifted=True, place_hold=cfg.place_hold_steps)
        for k, v in mutate.items():
            setattr(s2, k, v)
        assert not TD.placed_stable(cfg, s2), mutate
    s3 = state(cup_pos=np.array([0.02, -0.34, 0.8955]), was_lifted=False, place_hold=cfg.place_hold_steps)
    assert not TD.check_success(cfg, s3)  # was never lifted (e.g. cup already at target)


def test_success_is_independent_of_shaped_reward():
    cfg = TD.default_task_config("pick_up_cup")
    s = state(cup_pos=np.array([0.05, -0.2, 0.897]), n_grasp_contacts=5)  # perfect shaping signals, no lift
    parts = TD.shaped_reward(cfg, s, np.zeros(4), success=False)
    assert parts["success"] == 0.0 and not TD.check_success(cfg, s)
    assert TD.shaped_reward(cfg, s, np.zeros(4), success=True)["total"] > parts["total"]


def test_reward_monotone_in_distance_and_penalises_collisions():
    cfg = TD.default_task_config("reach_cup")
    near, far = state(), state()
    near.eef_pos = near.cup_pos + np.asarray(cfg.approach_offset)
    far.eef_pos = near.eef_pos + np.array([0.3, 0, 0])
    assert TD.shaped_reward(cfg, near, np.zeros(4), False)["reach"] > TD.shaped_reward(cfg, far, np.zeros(4), False)["reach"]
    hit = state(max_scene_force=cfg.invalid_force + 1)
    assert TD.shaped_reward(cfg, hit, np.zeros(4), False)["collision"] < 0


def test_phase_progression():
    cfg = TD.default_task_config("place_cup")
    far = state(eef_pos=np.array([0.4, -0.3, 1.1]))  # default test state is already within 8 cm of the approach pose
    assert TD.PHASES[TD.current_phase(cfg, far, False)] == "approach"
    near = state()
    near.eef_pos = near.cup_pos + np.asarray(cfg.approach_offset)
    assert TD.PHASES[TD.current_phase(cfg, near, False)] == "grasp"
    carrying = state(cup_pos=np.array([0.05, -0.2, 1.0]), n_grasp_contacts=4, was_lifted=True)
    assert TD.PHASES[TD.current_phase(cfg, carrying, False)] == "transport"
    assert TD.PHASES[TD.current_phase(cfg, carrying, True)] == "done"


def test_wilson_interval_known_values():
    lo, hi = wilson(30, 30)
    assert lo == pytest.approx(0.8864, abs=2e-3) and hi == pytest.approx(1.0, abs=1e-6)
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=2e-3) and hi == pytest.approx(0.7634, abs=2e-3)
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2           # no negative lower bound
    lo, hi = wilson(20, 20)
    assert hi == pytest.approx(1.0, abs=1e-12) and lo < 1.0


def test_mcnemar_and_bootstrap():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    lo, hi = bootstrap_ci(np.arange(100.0))
    assert lo < 49.5 < hi


def test_summarize_shapes():
    eps = [dict(success=bool(i % 2), grasp_success=True, collision_free=True, place_error=0.1, eef_error=0.02, steps=100 + i,
                invalid_steps=0, action_smoothness=0.1, joint_jerk=1.0, latency_ms=5.0, **{"return": 1.0}) for i in range(10)]
    s = summarize(eps)
    assert s["n"] == 10 and s["success_rate"] == 0.5 and s["success_ci95"][0] < 0.5 < s["success_ci95"][1]
    assert s["completion_time_s"] is not None
