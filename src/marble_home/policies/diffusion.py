"""Conditional diffusion policy (compact re-implementation after Chi et al., "Diffusion Policy", RSS 2023).

Predicts a short future action trajectory a[0:H] conditioned on the last ``n_obs`` observations.
Training: epsilon-prediction DDPM with a cosine schedule. Inference: deterministic DDIM with
``ddim_steps`` (configurable; fewer steps = lower latency, measured in Experiment 7).
This is NOT the official repository code; it is a small independent implementation whose
behaviour is validated only by the rollouts in results/.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from marble_home.policies.bc import Normalizer
from marble_home.policies.networks import ConvEncoder, CondUNet1D, mlp, random_shift


def cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, T, T + 1)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    return torch.clip(1 - ac[1:] / ac[:-1], 1e-4, 0.999)


class DiffusionPolicy(nn.Module):
    kind = "diffusion"

    def __init__(self, obs_dim: int, act_dim: int = 4, n_obs: int = 2, horizon: int = 16, n_act: int = 8,
                 train_steps: int = 100, ddim_steps: int = 10, widths=(64, 128), obs_feat: int = 128,
                 stats: dict | None = None, cameras: tuple = (), depth: bool = False, img_feat: int = 64):
        super().__init__()
        assert horizon % 4 == 0 and n_act <= horizon - n_obs + 1
        s = stats or {"mean": [0.0] * obs_dim, "std": [1.0] * obs_dim}
        self.norm = Normalizer(s["mean"], s["std"])
        self.n_obs, self.horizon, self.n_act, self.act_dim = n_obs, horizon, n_act, act_dim
        self.train_steps, self.ddim_steps = train_steps, ddim_steps
        self.cameras, self.depth = tuple(cameras), depth
        self.enc = nn.ModuleDict({c: ConvEncoder(4 if (depth and c == "head") else 3, img_feat) for c in self.cameras})
        self.obs_mlp = mlp(obs_dim * n_obs + img_feat * len(self.cameras) * n_obs, (256,), obs_feat)
        self.unet = CondUNet1D(act_dim, obs_feat, widths)
        b = cosine_betas(train_steps)
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - b, 0))
        self.cfg = {"obs_dim": obs_dim, "act_dim": act_dim, "n_obs": n_obs, "horizon": horizon, "n_act": n_act,
                    "train_steps": train_steps, "ddim_steps": ddim_steps, "widths": list(widths), "obs_feat": obs_feat,
                    "cameras": list(cameras), "depth": depth, "img_feat": img_feat}

    # ------------------------------------------------------------------ conditioning
    def encode_obs(self, obs) -> torch.Tensor:
        """obs: Tensor (B,n_obs,D) for state policies, or dict {'proprio':(B,n_obs,D), 'rgb_<cam>':(B,n_obs,H,W,3), ...}."""
        if isinstance(obs, dict):
            prop = self.norm(obs["proprio"])
            B, To = prop.shape[:2]
            feats = [prop.reshape(B, -1)]
            for c in self.cameras:
                x = obs[f"rgb_{c}"].float().div(255.0).flatten(0, 1).permute(0, 3, 1, 2)
                if self.depth and c == "head":
                    x = torch.cat([x, obs["depth_head"].flatten(0, 1).permute(0, 3, 1, 2)], 1)
                if self.training:
                    x = random_shift(x)
                feats.append(self.enc[c](x).reshape(B, -1))
            return self.obs_mlp(torch.cat(feats, -1))
        return self.obs_mlp(self.norm(obs).reshape(obs.shape[0], -1))

    # ---------------------------------------------------------------------- training
    def loss(self, obs, actions: torch.Tensor) -> torch.Tensor:
        cond = self.encode_obs(obs)
        B = actions.shape[0]
        k = torch.randint(0, self.train_steps, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        ac = self.alphas_cumprod[k].view(B, 1, 1)
        noisy = ac.sqrt() * actions + (1 - ac).sqrt() * noise
        return nn.functional.mse_loss(self.unet(noisy, k, cond), noise)

    # --------------------------------------------------------------------- inference
    @torch.no_grad()
    def sample(self, obs, ddim_steps: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        cond = self.encode_obs(obs)
        B = cond.shape[0]
        K = ddim_steps or self.ddim_steps
        a = torch.randn(B, self.horizon, self.act_dim, generator=generator)
        steps = torch.linspace(self.train_steps - 1, 0, K).round().long()
        for i, k in enumerate(steps):
            kb = torch.full((B,), int(k))
            eps = self.unet(a, kb, cond)
            ac = self.alphas_cumprod[k]
            ac_prev = self.alphas_cumprod[steps[i + 1]] if i + 1 < K else torch.tensor(1.0)
            x0 = (a - (1 - ac).sqrt() * eps) / ac.sqrt()
            x0 = x0.clamp(-1, 1)
            a = ac_prev.sqrt() * x0 + (1 - ac_prev).sqrt() * eps
        return a.clamp(-1, 1)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)
