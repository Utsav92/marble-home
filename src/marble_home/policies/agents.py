"""Checkpoint I/O and the uniform ``Agent`` interface used by evaluation / rendering / benchmarking.

Agent: ``reset(env)``, ``act(obs) -> np.ndarray(4)``.  Kinds: random, scripted (privileged teacher),
checkpoint (bc_mlp / bc_cnn / diffusion).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from marble_home.policies.bc import CNNPolicy, MLPPolicy
from marble_home.policies.diffusion import DiffusionPolicy
from marble_home.policies.scripted import make_expert
from marble_home.robots import action_space as AS
from marble_home.utils import provenance


def save_checkpoint(path: str | Path, model: torch.nn.Module, meta: dict) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": model.kind, "cfg": model.cfg, "state_dict": model.state_dict(), "meta": meta}, path)
    return provenance.file_sha256(path)


def load_checkpoint(path: str | Path) -> tuple[torch.nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, kind = ck["cfg"], ck["kind"]
    if kind == "bc_mlp":
        m = MLPPolicy(cfg["obs_dim"], cfg["act_dim"], tuple(cfg["hidden"]))
    elif kind == "bc_cnn":
        m = CNNPolicy(cfg["proprio_dim"], cfg["act_dim"], tuple(cfg["cameras"]), cfg["depth"], cfg["feat"], tuple(cfg["hidden"]))
    elif kind == "diffusion":
        m = DiffusionPolicy(cfg["obs_dim"], cfg["act_dim"], cfg["n_obs"], cfg["horizon"], cfg["n_act"], cfg["train_steps"],
                            cfg["ddim_steps"], tuple(cfg["widths"]), cfg["obs_feat"], None, tuple(cfg["cameras"]),
                            cfg["depth"], cfg["img_feat"])
    else:
        raise ValueError(kind)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["meta"]


def _to_t(obs):
    if isinstance(obs, dict):
        return {k: torch.from_numpy(np.asarray(v))[None] for k, v in obs.items()}
    return torch.from_numpy(np.asarray(obs, dtype=np.float32))[None]


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env=None):
        pass

    def act(self, obs):
        return self.rng.uniform(-1, 1, AS.ACTION_DIM).astype(np.float32)


class ScriptedAgent:
    name = "scripted_expert"

    def __init__(self, noise: float = 0.0):
        self.noise = noise
        self.ex = None
        self.env = None

    def reset(self, env=None):
        self.env = env
        self.ex = make_expert(env.env, noise=self.noise)
        self.ex.reset(env._raw)

    def act(self, obs):
        return AS.full_to_reduced(self.ex.act(self.env._raw))


class BCAgent:
    def __init__(self, model, name="bc", ddim_steps=None, obs_mode=None, obs_key="state", action_mode=None, obs_keep=None):
        self.m, self.name = model, name
        self.keep = np.asarray(obs_keep, dtype=int) if obs_keep is not None else None
        self.slice = 58 if obs_key == "proprio" else None  # state vector = proprio(56)+goal(2) | privileged cup info
        self.env_kwargs = {"obs_mode": obs_mode} if obs_mode else {}
        if action_mode:
            self.env_kwargs["action_mode"] = action_mode  # the env must use the parametrisation the policy was trained in

    def reset(self, env=None):
        pass

    def act(self, obs):
        if self.slice is not None:
            obs = obs[: self.slice]
        if self.keep is not None:
            obs = obs[self.keep]
        with torch.no_grad():
            return self.m.act(_to_t(obs))[0].numpy().astype(np.float32)


class DiffusionAgent:
    def __init__(self, model: DiffusionPolicy, name="diffusion", ddim_steps: int | None = None, seed: int = 0, action_mode: str | None = None):
        self.m, self.name, self.k = model, name, ddim_steps
        self.env_kwargs = {"action_mode": action_mode} if action_mode else {}
        self.hist: deque = deque(maxlen=model.n_obs)
        self.queue: deque = deque()
        self.gen = torch.Generator().manual_seed(seed)
        self.n_act = model.n_act

    def reset(self, env=None):
        self.hist.clear()
        self.queue.clear()

    def act(self, obs):
        self.hist.append(obs)
        if not self.queue:
            frames = list(self.hist)
            while len(frames) < self.m.n_obs:  # pad history with the first frame
                frames.insert(0, frames[0])
            if isinstance(frames[0], dict):
                batch = {k: torch.from_numpy(np.stack([f[k] for f in frames]))[None] for k in frames[0]}
            else:
                batch = torch.from_numpy(np.stack(frames).astype(np.float32))[None]
            with torch.no_grad():
                traj = self.m.sample(batch, ddim_steps=self.k, generator=self.gen)[0].numpy()
            s = self.m.n_obs - 1
            self.queue.extend(traj[s:s + self.n_act])
        return self.queue.popleft().astype(np.float32)


class PPOAgent:
    """Stable-Baselines3 PPO checkpoint (<name>.zip + <name>.stats.json with obs normalisation and action mode)."""

    def __init__(self, zip_path: str | Path, name: str | None = None):
        import json

        from stable_baselines3 import PPO

        p = Path(zip_path)
        self.model = PPO.load(str(p), device="cpu")
        st = json.loads(p.with_suffix(".stats.json").read_text())
        self.mean, self.std = np.asarray(st["mean"], np.float32), np.asarray(st["std"], np.float32)
        self.env_kwargs = {"action_mode": st["action_mode"]}  # evaluation must use the same action parametrisation
        self.name = name or p.stem

    def reset(self, env=None):
        pass

    def act(self, obs):
        a, _ = self.model.predict(((obs - self.mean) / self.std).astype(np.float32), deterministic=True)
        return np.clip(a, -1, 1).astype(np.float32)


def make_agent(spec: dict):
    """spec: {'type': 'random'|'scripted'|'ppo'|'checkpoint', 'path':..., 'ddim_steps':..., 'name':...}"""
    t = spec["type"]
    if t == "random":
        return RandomAgent(spec.get("seed", 0))
    if t == "scripted":
        return ScriptedAgent(spec.get("noise", 0.0))
    if t == "ppo" or str(spec.get("path", "")).endswith(".zip"):
        return PPOAgent(spec["path"], spec.get("name"))
    m, meta = load_checkpoint(spec["path"])
    name = spec.get("name") or Path(spec["path"]).stem
    if m.kind == "diffusion":
        return DiffusionAgent(m, name, spec.get("ddim_steps"), spec.get("seed", 0), action_mode=meta.get("action_mode"))
    return BCAgent(m, name, obs_mode=meta.get("obs") if m.kind == "bc_cnn" else None, obs_key=meta.get("obs_key", "state"), action_mode=meta.get("action_mode"),
                   obs_keep=meta.get("obs_keep"))
