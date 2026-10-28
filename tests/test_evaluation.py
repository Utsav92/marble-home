import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_evaluation_is_reproducible_for_a_seed(env_state):
    from marble_home.evaluation.rollout import run_episode
    from marble_home.policies.agents import ScriptedAgent

    a = run_episode(env_state, ScriptedAgent(), seed=424242)
    b = run_episode(env_state, ScriptedAgent(), seed=424242)
    for k in ("success", "steps", "grasp_success", "invalid_steps"):
        assert a[k] == b[k], k
    assert a["place_error"] == pytest.approx(b["place_error"], abs=2e-3)
    assert a["eef_error"] == pytest.approx(b["eef_error"], abs=2e-3)


def test_random_policy_does_not_solve_pick(env_state):
    from marble_home.evaluation.rollout import run_episode
    from marble_home.policies.agents import RandomAgent

    ok = sum(run_episode(env_state, RandomAgent(s), seed=s)["success"] for s in range(3))
    assert ok == 0


def test_scripted_expert_terminates_on_success_before_horizon(env_state):
    from marble_home.evaluation.rollout import run_episode
    from marble_home.policies.agents import ScriptedAgent

    r = run_episode(env_state, ScriptedAgent(), seed=900001)
    assert r["success"] and r["steps"] < env_state.horizon
