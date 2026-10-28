import json

import numpy as np
import pytest
import torch

from marble_home.datasets.hdf5 import DemoFile, DemoWriter, merge
from marble_home.datasets.torch_ds import SeqDataset, StepDataset
from marble_home.policies.agents import BCAgent, DiffusionAgent, load_checkpoint, save_checkpoint
from marble_home.policies.bc import MLPPolicy
from marble_home.policies.diffusion import DiffusionPolicy


def fake_episode(T=30, seed=0):
    r = np.random.default_rng(seed)
    return {"actions": r.uniform(-1, 1, (T, 4)).astype(np.float32), "actions_full": r.normal(size=(T, 24)).astype(np.float32),
            "rewards": r.normal(size=T).astype(np.float32), "dones": np.r_[np.zeros(T - 1), 1].astype(np.float32),
            "task_phase": np.zeros(T, np.float32), "eef_error": np.ones(T, np.float32), "place_error": np.ones(T, np.float32),
            "obs": {"state": r.normal(size=(T, 69)).astype(np.float32), "proprio": r.normal(size=(T, 58)).astype(np.float32)},
            "init_state": r.normal(size=100)}


def write(path, n=3):
    with DemoWriter(path, "pick_up_cup", {"scene_id": "x"}, {"a": 1}, {"mujoco": "3.3.7"}, {"robot": "GR1"}) as w:
        for i in range(n):
            w.add(fake_episode(30 + i, i), seed=i, scene_id="x", task="pick_up_cup", success=True, expert_noise=0.0, speed=1.0)


def test_demo_serialization_roundtrip(tmp_path):
    p = tmp_path / "d.hdf5"
    write(p)
    with DemoFile(p) as df:
        assert df.attrs["format_version"] == "marble_home.demo.v1" and len(df.demos) == 3
        ep = df.demo("demo_1")
        ref = fake_episode(31, 1)
        assert np.array_equal(ep["actions"], ref["actions"]) and np.array_equal(ep["obs"]["state"], ref["obs"]["state"])
        assert ep["attrs"]["seed"] == 1 and ep["attrs"]["num_samples"] == 31
        for k in ("versions", "robot_config", "scene", "env_args"):  # provenance fields required by the spec
            json.loads(df.attrs[k])


def test_merge_preserves_episodes(tmp_path):
    write(tmp_path / "a.hdf5", 2)
    write(tmp_path / "b.hdf5", 3)
    s = merge([tmp_path / "a.hdf5", tmp_path / "b.hdf5"], tmp_path / "m.hdf5")
    assert s["episodes"] == 5 and s["steps"] == (30 + 31) + (30 + 31 + 32)


def test_step_and_seq_datasets_shapes():
    eps = [{"obs": {"state": e["obs"]["state"]}, "act": e["actions"]} for e in (fake_episode(30, 0), fake_episode(25, 1))]
    sd = StepDataset(eps)
    o, a = sd[0]
    assert len(sd) == 55 and o["state"].shape == (69,) and a.shape == (4,)
    q = SeqDataset(eps, n_obs=2, horizon=16)
    o, a = q[0]
    assert o["state"].shape == (2, 69) and a.shape == (16, 4)
    o, a = q[len(q) - 1]  # last step of the last episode: edge-padded, still full-size
    assert a.shape == (16, 4) and torch.isfinite(a).all()


def test_bc_checkpoint_roundtrip_and_bounds(tmp_path):
    m = MLPPolicy(69, 4, (32, 32), {"mean": np.zeros(69), "std": np.ones(69)})
    sha = save_checkpoint(tmp_path / "bc.pt", m, {"task": "t"})
    m2, meta = load_checkpoint(tmp_path / "bc.pt")
    x = torch.randn(5, 69)
    assert torch.allclose(m.eval()(x), m2(x), atol=1e-6) and len(sha) == 64
    a = BCAgent(m2).act(np.random.randn(69).astype(np.float32))
    assert a.shape == (4,) and np.all(np.abs(a) <= 1.0)


def test_diffusion_checkpoint_roundtrip_inference_and_bounds(tmp_path):
    m = DiffusionPolicy(69, 4, n_obs=2, horizon=16, n_act=8, ddim_steps=4, widths=(16, 32), obs_feat=32,
                        stats={"mean": np.zeros(69), "std": np.ones(69)})
    loss = m.loss(torch.randn(6, 2, 69), torch.rand(6, 16, 4) * 2 - 1)
    loss.backward()
    assert torch.isfinite(loss)
    save_checkpoint(tmp_path / "d.pt", m, {"task": "t"})
    m2, _ = load_checkpoint(tmp_path / "d.pt")
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    x = torch.randn(3, 2, 69)
    assert torch.allclose(m.eval().sample(x, generator=g1), m2.sample(x, generator=g2), atol=1e-5)  # same weights+noise -> same actions
    ag = DiffusionAgent(m2, ddim_steps=2)
    acts = [ag.act(np.random.randn(69).astype(np.float32)) for _ in range(10)]  # >8 forces a re-plan
    assert all(a.shape == (4,) and np.all(np.abs(a) <= 1.0) for a in acts)


def test_diffusion_horizon_must_be_divisible_by_4():
    with pytest.raises(AssertionError):
        DiffusionPolicy(10, 4, horizon=10)


@pytest.mark.slow
def test_demo_replay_alignment_on_real_sim(tmp_path):
    """Collect 1 short reach demo with the real sim, then replay it from the saved initial state."""
    from marble_home.datasets.collect import make_env, replay_check, run_expert_episode

    env = make_env("reach_cup", seed=0)
    ep = run_expert_episode(env, seed=7, noise=0.0, speed=1.0)
    assert ep["success"] and len(ep["actions"]) > 20
    attrs = {"seed": 7, "init_state": ep["init_state"], "success": True}
    r = replay_check(env, ep, attrs)
    env.close()
    assert r["ok"], r


def test_bc_agent_applies_dropped_input_dims(tmp_path):
    """Regression: a checkpoint trained with obs_drop must be fed only the kept dims (69-d env obs -> 67-d model input)."""
    keep = [i for i in range(69) if i not in (56, 57)]
    m = MLPPolicy(len(keep), 4, (16, 16), {"mean": np.zeros(len(keep)), "std": np.ones(len(keep))})
    save_checkpoint(tmp_path / "k.pt", m, {"task": "t", "obs_key": "state", "obs_keep": keep, "action_mode": "ee_delta"})
    from marble_home.policies.agents import make_agent

    ag = make_agent({"type": "checkpoint", "path": str(tmp_path / "k.pt")})
    assert ag.env_kwargs == {"action_mode": "ee_delta"}
    a = ag.act(np.random.randn(69).astype(np.float32))  # would raise a size mismatch if the mask were ignored
    assert a.shape == (4,) and np.all(np.abs(a) <= 1.0)
