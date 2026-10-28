"""Slow camera pans through the five REAL Marble worlds (point-splat render of the downloaded 500k splat files)."""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat_render import load_splat, render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "footage"
WORLDS = ["marble_kitchen_001", "marble_living_room_001", "marble_bedroom_001", "marble_laundry_room_001", "marble_multiroom_apartment_001"]
YAW0 = {"marble_kitchen_001": 0.2, "marble_living_room_001": 1.7, "marble_bedroom_001": 0.0, "marble_laundry_room_001": 0.0,
        "marble_multiroom_apartment_001": 0.0}


def fly(world, n=90, W=1280, H=720):
    pos, rgb, sw = load_splat(ROOT / "data" / "marble_raw" / world / "splat.ply", min_opacity=0.25, max_scale=-2.5)
    c = np.median(pos, axis=0)
    c[1] = np.percentile(pos[:, 1], 55)  # a little below mid-height (y is down)
    up = np.array([0.0, -1.0, 0.0])
    y0 = YAW0[world]
    out = OUT / f"world_{world}.mp4"
    with imageio.get_writer(out, fps=30, macro_block_size=1, quality=8) as w:
        for i in range(n):
            u = i / (n - 1)
            yaw = y0 + 0.9 * u
            eye = c + np.array([np.sin(yaw), 0, np.cos(yaw)]) * (-0.35 + 0.35 * u)
            tgt = eye + np.array([np.sin(yaw), -0.03, np.cos(yaw)])
            w.append_data(render(pos, rgb, sw, eye, tgt, up, W=W, H=H, fov_deg=78, max_r=4))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for wd in (sys.argv[1:] or WORLDS):
        print(fly(wd), flush=True)
