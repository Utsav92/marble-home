"""Minimal CPU point-splat renderer for the Marble `splat.ply` files (3D Gaussian splats, 497,664 per world).

Not a full Gaussian rasteriser: every splat is drawn as a small opaque square whose size follows its projected scale, painter's-sorted far to
near. It is enough to show what the generated world looks like; it is NOT the quality of a splat viewer and does not claim to be.
Colours: DC spherical-harmonic coefficient -> rgb = 0.5 + 0.2821 * f_dc (view-dependent terms are not in the file).
"""
from __future__ import annotations

import numpy as np

SH0 = 0.28209479177387814


def load_splat(path, min_opacity=0.35, max_scale=-3.0):
    raw = open(path, "rb").read()
    a = np.frombuffer(raw[raw.index(b"end_header") + 11:], dtype=np.float32).reshape(-1, 14)
    pos = a[:, 0:3].astype(np.float64)
    rgb = np.clip(0.5 + SH0 * a[:, 3:6], 0, 1)
    op = 1.0 / (1.0 + np.exp(-np.clip(a[:, 6], -30, 30)))
    sc = a[:, 7:10].max(1)  # log-scale, largest axis
    keep = (op > min_opacity) & (sc < max_scale)
    return pos[keep], rgb[keep].astype(np.float32), np.exp(sc[keep])


def look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])  # rows: right, up, forward


def render(pos, rgb, size_w, eye, target, up, W=1280, H=720, fov_deg=75.0, bg=(12, 12, 14), px_gain=1.0, max_r=3):
    R = look_at(np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float))
    pc = (pos - eye) @ R.T  # x right, y up, z forward
    z = pc[:, 2]
    ok = z > 0.05
    pc, z, col, sw = pc[ok], z[ok], rgb[ok], size_w[ok]
    fpx = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    u = (pc[:, 0] / z * fpx + W / 2).astype(np.int32)
    v = (-pc[:, 1] / z * fpx + H / 2).astype(np.int32)
    rad = np.clip((sw / z * fpx * px_gain).astype(np.int32), 0, max_r)
    inb = (u >= -max_r) & (u < W + max_r) & (v >= -max_r) & (v < H + max_r)
    u, v, z, col, rad = u[inb], v[inb], z[inb], col[inb], rad[inb]
    order = np.argsort(-z)  # far -> near, later writes win
    u, v, col, rad = u[order], v[order], col[order], rad[order]
    img = np.empty((H, W, 3), np.uint8)
    img[:] = np.array(bg, np.uint8)
    c8 = (col * 255).astype(np.uint8)
    for r in range(max_r + 1):
        sel = rad == r
        if not sel.any():
            continue
        uu, vv, cc = u[sel], v[sel], c8[sel]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = uu + dx, vv + dy
                m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                img[y[m], x[m]] = cc[m]
    return img
