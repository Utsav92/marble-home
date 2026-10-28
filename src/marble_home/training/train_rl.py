"""PPO (Stable-Baselines3) on the fixed-base GR1 tasks, with reward-component diagnostics.

Imitation -> RL: a diffusion-policy checkpoint cannot be loaded into a PPO actor (different architecture and a
stochastic multi-step sampler), so the supported route here is ``bc_init``: supervised pre-training of the PPO
*actor* on the demonstrations (distillation of the demonstrator), followed by PPO fine-tuning with a small initial
action std. ``bc_init: null`` gives the PPO-from-scratch baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from marble_home.datasets.torch_ds import load_episodes, load_stats
from marble_home.simulation.gym_env import HouseholdEnv
from marble_home.training.common import Run, set_seed
from marble_home.utils.paths import CHECKPOINTS, RESULTS


class NormObs(gym.ObservationWrapper):
    """Fixed observation normalisation (stats from the task's demonstration dataset, train split)."""

    def __init__(self, env, mean, std):
        super().__init__(env)
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    def observation(self, obs):
        return ((obs - self.mean) / self.std).astype(np.float32)


class StageControl(gym.Wrapper):
    """Lets a curriculum change the cup-spawn region between episodes (takes effect at the next reset)."""

    def set_stage(self, stage: dict):
        sam = self.env.unwrapped.env.placement_initializer
        sam.x_range = list(stage["cup_x"])  # table x offset is 0, so world == relative
        sam.y_range = list(stage["cup_y"])
        return True


def make_env_fn(task, action_mode, mean, std, seed, rank, initial_stage=None):
    def _f():
        env = HouseholdEnv(task=task, obs_mode="state", action_mode=action_mode, seed=seed + rank)
        env = NormObs(env, mean, std)
        env = StageControl(env)
        if initial_stage is not None:
            env.set_stage(initial_stage)
        return Monitor(env, info_keywords=("ever_success",))
    return _f


class CurriculumCallback(BaseCallback):
    """Feeds finished-episode outcomes to a CurriculumScheduler and pushes stage changes to every env."""

    def __init__(self, scheduler, mode: str, switch_step: int | None = None):
        super().__init__()
        self.sch, self.mode, self.switch_step = scheduler, mode, switch_step
        self.manual_done = False

    def _push(self):
        self.training_env.env_method("set_stage", self.sch.stage)

    def _on_step(self) -> bool:
        if self.mode == "manual":
            if not self.manual_done and self.num_timesteps >= self.switch_step:
                frm = self.sch.stage["id"]
                self.sch.idx = min(self.sch.idx + 1, len(self.sch.stages) - 1)
                self.sch.log.append({"step": self.num_timesteps, "event": "manual_advance", "from_stage": frm,
                                     "to_stage": self.sch.stage["id"], "criterion": f"fixed schedule at {self.switch_step} steps"})
                self.manual_done = True
                self._push()
            return True
        for info in self.locals["infos"]:
            if "episode" in info:
                ev = self.sch.record(bool(info.get("ever_success", False)), self.num_timesteps)
                if ev:
                    self._push()
        return True


class DiagCallback(BaseCallback):
    """Logs episode return, success rate and mean per-step reward components at every rollout end."""

    def __init__(self, run: Run):
        super().__init__()
        self.run = run
        self.parts: dict[str, list] = {}

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for k, v in info.get("reward_parts", {}).items():
                self.parts.setdefault(k, []).append(v)
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        row = {"timesteps": self.num_timesteps}
        if len(buf):
            row["ep_return"] = float(np.mean([e["r"] for e in buf]))
            row["ep_len"] = float(np.mean([e["l"] for e in buf]))
            row["success_rate"] = float(np.mean([e.get("ever_success", 0.0) for e in buf]))
        for k, v in self.parts.items():
            row[f"rew_{k}"] = float(np.mean(v))
        self.parts = {}
        self.run.log(self.num_timesteps, **row)


def bc_pretrain_actor(model: PPO, episodes, epochs: int = 40, lr: float = 1e-3, log=print):
    """Distil demonstrations into the PPO actor mean (MSE on actions) and shrink the exploration std."""
    obs = torch.as_tensor(np.concatenate([e["obs"]["state_n"] for e in episodes]), dtype=torch.float32)
    act = torch.as_tensor(np.concatenate([e["act"] for e in episodes]), dtype=torch.float32)
    pol = model.policy
    params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    n = len(obs)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            mean = pol.action_net(pol.mlp_extractor.policy_net(obs[idx]))
            loss = torch.nn.functional.mse_loss(mean, act[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  bc-init epoch {ep}: mse {tot / n:.5f}")
    with torch.no_grad():
        pol.log_std.fill_(np.log(0.1))  # explore near the demonstrator
    return tot / n


def train_ppo(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    task = cfg["task"]
    ds = cfg.get("dataset") or task
    st = load_stats(ds)["state"]
    mean, std = np.asarray(st["mean"], np.float32), np.asarray(st["std"], np.float32)
    name = cfg.get("run_name") or f"ppo_{task}_{'bcinit' if cfg.get('bc_init') else 'scratch'}_s{cfg['seed']}"
    run = Run(name, cfg)
    cur_mode = cfg.get("curriculum")  # None | "adaptive" | "manual"
    sch = None
    if cur_mode:
        from marble_home.training.curriculum import STAGES, CurriculumScheduler

        sch = CurriculumScheduler(stages=[STAGES[0], STAGES[1]], min_episodes=cfg.get("cur_min_episodes", 30),
                                  advance_lcb=cfg.get("cur_advance_lcb", 0.70), rollback_ucb=cfg.get("cur_rollback_ucb", 0.10))
    fns = [make_env_fn(task, cfg["action_mode"], mean, std, cfg["seed"] * 1000, i, sch.stage if sch else None) for i in range(cfg["n_envs"])]
    venv = SubprocVecEnv(fns, start_method="spawn") if cfg["n_envs"] > 1 else DummyVecEnv(fns)
    model = PPO("MlpPolicy", venv, n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
                learning_rate=cfg["learning_rate"], gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"], ent_coef=cfg["ent_coef"],
                clip_range=cfg["clip_range"], policy_kwargs={"net_arch": {"pi": cfg["net_arch"], "vf": cfg["net_arch"]}, "log_std_init": cfg.get("log_std_init", 0.0)},
                seed=cfg["seed"], verbose=0, device="cpu")
    if cfg.get("bc_init"):
        eps = load_episodes(ds, "train", ("state",), max_episodes=cfg.get("bc_init_demos"))
        for e in eps:
            e["obs"]["state_n"] = ((e["obs"]["state"] - mean) / std).astype(np.float32)
        if cfg["action_mode"] != "ee_abs":
            raise ValueError("bc_init uses demonstration actions, which are ee_abs; set action_mode: ee_abs")
        bc_pretrain_actor(model, eps)
    cbs = [DiagCallback(run)]
    if sch:
        cbs.append(CurriculumCallback(sch, cur_mode, cfg.get("manual_switch_step", cfg["total_timesteps"] // 3)))
    model.learn(total_timesteps=cfg["total_timesteps"], callback=cbs, progress_bar=False)
    if sch:
        sch.dump(RESULTS / "curriculum" / f"{name}.json")
    venv.close()
    CHECKPOINTS.mkdir(exist_ok=True)
    zip_path = CHECKPOINTS / f"{name}.zip"
    model.save(zip_path)
    (CHECKPOINTS / f"{name}.stats.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(),
                                                                "action_mode": cfg["action_mode"], "task": task}))
    run.ckpt = zip_path
    from marble_home.utils import provenance

    run.finish(provenance.file_sha256(zip_path), total_timesteps=cfg["total_timesteps"])
    return {"name": name, "ckpt": str(zip_path), "run_dir": str(run.dir)}
