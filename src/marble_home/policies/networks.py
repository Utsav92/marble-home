"""Small PyTorch building blocks (CPU-friendly): MLPs, a spatial-softmax conv encoder, a conditional 1-D UNet."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.ReLU, norm: bool = False) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h)] + ([nn.LayerNorm(h)] if norm else []) + [act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class SpatialSoftmax(nn.Module):
    """Feature map -> expected (x, y) keypoints per channel (as in RoboMimic / Levine et al.)."""

    def forward(self, x):
        b, c, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        p = F.softmax(x.reshape(b, c, h * w), dim=-1).reshape(b, c, h, w)
        ex = (p.sum(2) * xs).sum(-1)
        ey = (p.sum(3) * ys).sum(-1)
        return torch.cat([ex, ey], dim=-1)  # (b, 2c)


class ConvEncoder(nn.Module):
    """64x64 (or 96x96) image -> feature vector. ``in_ch`` = 3 (RGB) or 4 (RGB-D)."""

    def __init__(self, in_ch: int = 3, out_dim: int = 64, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 5, 2, 2), nn.GroupNorm(8, width), nn.ReLU(),
            nn.Conv2d(width, width * 2, 3, 1, 1), nn.GroupNorm(8, width * 2), nn.ReLU(),
            nn.Conv2d(width * 2, width * 2, 3, 2, 1), nn.GroupNorm(8, width * 2), nn.ReLU(),
            nn.Conv2d(width * 2, 32, 3, 1, 1),
        )
        self.pool = SpatialSoftmax()
        self.fc = nn.Linear(64, out_dim)  # 32 keypoints * 2

    def forward(self, img):
        return self.fc(self.pool(self.net(img)))


def random_shift(img: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ-style random-shift augmentation, img float (B,C,H,W)."""
    b, c, h, w = img.shape
    img = F.pad(img, (pad,) * 4, mode="replicate")
    ix = torch.randint(0, 2 * pad + 1, (b,))
    iy = torch.randint(0, 2 * pad + 1, (b,))
    out = torch.empty(b, c, h, w, dtype=img.dtype)
    for i in range(b):
        out[i] = img[i, :, iy[i]:iy[i] + h, ix[i]:ix[i] + w]
    return out


# --------------------------------------------------------------------------- diffusion UNet
class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = math.log(10000) / (half - 1)
        f = torch.exp(torch.arange(half, device=t.device) * -f)
        e = t.float()[:, None] * f[None]
        return torch.cat([e.sin(), e.cos()], dim=-1)


class CondBlock(nn.Module):
    def __init__(self, cin, cout, cond_dim, k=5, groups=8):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(groups, cout), nn.Mish())
        self.c2 = nn.Sequential(nn.Conv1d(cout, cout, k, padding=k // 2), nn.GroupNorm(groups, cout), nn.Mish())
        self.film = nn.Linear(cond_dim, cout * 2)  # FiLM scale+bias
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.c1(x)
        s, b = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        h = self.c2(h * (1 + s) + b)
        return h + self.res(x)


class CondUNet1D(nn.Module):
    """Noise predictor eps(a_t, k | obs). Input (B,H,A); H must be divisible by 4."""

    def __init__(self, act_dim: int, cond_dim: int, widths=(64, 128), emb_dim: int = 64):
        super().__init__()
        self.temb = nn.Sequential(SinusoidalEmb(emb_dim), nn.Linear(emb_dim, emb_dim * 2), nn.Mish(),
                                  nn.Linear(emb_dim * 2, emb_dim))
        cd = emb_dim + cond_dim
        w0, w1 = widths
        self.d0 = CondBlock(act_dim, w0, cd)
        self.down0 = nn.Conv1d(w0, w0, 3, 2, 1)
        self.d1 = CondBlock(w0, w1, cd)
        self.down1 = nn.Conv1d(w1, w1, 3, 2, 1)
        self.mid = CondBlock(w1, w1, cd)
        self.up1 = nn.ConvTranspose1d(w1, w1, 4, 2, 1)
        self.u1 = CondBlock(w1 * 2, w0, cd)
        self.up0 = nn.ConvTranspose1d(w0, w0, 4, 2, 1)
        self.u0 = CondBlock(w0 * 2, w0, cd)
        self.out = nn.Conv1d(w0, act_dim, 1)

    def forward(self, a, k, obs_cond):
        x = a.transpose(1, 2)
        c = torch.cat([self.temb(k), obs_cond], dim=-1)
        h0 = self.d0(x, c)
        h1 = self.d1(self.down0(h0), c)
        m = self.mid(self.down1(h1), c)
        u1 = self.u1(torch.cat([self.up1(m), h1], dim=1), c)
        u0 = self.u0(torch.cat([self.up0(u1), h0], dim=1), c)
        return self.out(u0).transpose(1, 2)
