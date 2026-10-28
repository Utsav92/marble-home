"""Behaviour-cloning policies: MLP on state vectors, CNN (+proprio) on images."""
from __future__ import annotations

import torch
import torch.nn as nn

from marble_home.policies.networks import ConvEncoder, mlp, random_shift


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class MLPPolicy(nn.Module):
    kind = "bc_mlp"

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), stats: dict | None = None):
        super().__init__()
        s = stats or {"mean": [0.0] * obs_dim, "std": [1.0] * obs_dim}
        self.norm = Normalizer(s["mean"], s["std"])
        self.net = mlp(obs_dim, tuple(hidden), act_dim, norm=True)
        self.cfg = {"obs_dim": obs_dim, "act_dim": act_dim, "hidden": list(hidden)}

    def forward(self, obs):
        return torch.tanh(self.net(self.norm(obs)))

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)


class CNNPolicy(nn.Module):
    """Per-camera ConvEncoder (+depth channel for the head camera) + proprio MLP -> fused MLP head."""
    kind = "bc_cnn"

    def __init__(self, proprio_dim: int, act_dim: int, cameras: tuple[str, ...] = ("head",), depth: bool = False,
                 feat: int = 64, hidden=(256, 256), stats: dict | None = None, aug: bool = True):
        super().__init__()
        s = stats or {"mean": [0.0] * proprio_dim, "std": [1.0] * proprio_dim}
        self.norm = Normalizer(s["mean"], s["std"])
        self.cameras, self.depth, self.aug = tuple(cameras), depth, aug
        self.enc = nn.ModuleDict({c: ConvEncoder(4 if (depth and c == "head") else 3, feat) for c in self.cameras})
        self.prop = mlp(proprio_dim, (128,), 64)
        self.head = mlp(feat * len(self.cameras) + 64, tuple(hidden), act_dim, norm=True)
        self.cfg = {"proprio_dim": proprio_dim, "act_dim": act_dim, "cameras": list(cameras), "depth": depth,
                    "feat": feat, "hidden": list(hidden)}

    def _img(self, obs, c):
        x = obs[f"rgb_{c}"].float() / 255.0  # (B,H,W,3)
        x = x.permute(0, 3, 1, 2)
        if self.depth and c == "head":
            d = obs["depth_head"].permute(0, 3, 1, 2) / 1.0
            x = torch.cat([x, d], 1)
        if self.training and self.aug:
            x = random_shift(x)
        return x

    def forward(self, obs: dict):
        feats = [self.enc[c](self._img(obs, c)) for c in self.cameras]
        feats.append(self.prop(self.norm(obs["proprio"])))
        return torch.tanh(self.head(torch.cat(feats, -1)))

    @torch.no_grad()
    def act(self, obs: dict) -> torch.Tensor:
        self.eval()
        return self.forward(obs)
