"""Concatenate the final rollout videos into one annotated reel (title card before each clip).

    python scripts/make_reel.py            -> results/videos/FINAL_reel.mp4
All clips are produced by scripts/render_episode.py from checkpoints in checkpoints/; nothing is synthesised.
"""
from __future__ import annotations

import glob
import json

import cv2
import imageio.v2 as imageio
import numpy as np

from marble_home.utils.paths import RESULTS

V = RESULTS / "videos"
W, H, FPS = 480, 360, 20

CLIPS = [
    ("01_random_pick.mp4", "Random policy (baseline)", "pick_up_cup | fails: 0/30 on the benchmark"),
    ("02_scripted_expert_pick.mp4", "Scripted expert", "pick_up_cup | 100% (30/30) | hand error 0.6 cm"),
    ("03_bc_pick.mp4", "Behaviour cloning (MLP, delta actions)", "pick_up_cup | 100% (30/30) held-out"),
    ("04b_diffusion_pick_success_seed900006.mp4", "Diffusion policy - a success", "pick_up_cup | 73% (22/30) overall"),
    ("04_diffusion_pick.mp4", "Diffusion policy - a typical FAILURE", "same policy, seed 900003: ends 15 cm short"),
    ("05_bc_reach.mp4", "BC: reach_cup", "100% (30/30) under a 3 cm + hold rule"),
    ("06_bc_place.mp4", "BC: place_cup", "100% (30/30) | cup 0.16 cm from target here"),
    ("07_multiscene_bc_heldout_livingroom.mp4", "Multi-scene BC on a HELD-OUT scene", "100% on 5 unseen scenes (50/50)"),
]
CLIPS += [(f"08_marble_{w}_bc_pick.mp4", f"Inside a real Marble world: {w.replace('_001', '').replace('_', ' ')}",
           "marble-1.0-draft | scale ASSUMED | same policy, same task")
          for w in ("kitchen_001", "living_room_001", "bedroom_001", "laundry_room_001", "multiroom_apartment_001")]


def _household_stats():
    """Numbers for the multi-item clips come from results/experiments/household_tasks.json (never typed in here)."""
    f = RESULTS / "experiments" / "household_tasks.json"
    if not f.exists():
        return {}
    rows = json.loads(f.read_text())["rows"]
    out = {}
    for t in ("clear_table", "laundry_basket", "load_dishwasher"):
        bc = [r for r in rows if r["task"] == t and r["policy"] == "bc_mlp"]
        ex = [r for r in rows if r["task"] == t and r["policy"] == "scripted_expert"]
        if bc:
            k = sum(round(r["success"] * r["n"]) for r in bc)
            n = sum(r["n"] for r in bc)
            out[t] = f"learned BC-MLP {k}/{n} held-out over {len(bc)} seed(s)" + (f" | scripted expert {ex[0]['success']:.0%}" if ex else "")
    return out


_HH = _household_stats()
HOUSEHOLD = [
    ("09_marble_living_room_bc_clear_table.mp4", "clear_table: 3 cups -> tray (learned)", "clear_table", "Marble living room, scale ASSUMED"),
    ("09_marble_laundry_room_bc_laundry.mp4", "laundry_basket: 3 laundry rolls -> tote (learned)", "laundry_basket", "Marble laundry room; rigid proxies, no cloth"),
    ("09_kitchen_bc_load_dishwasher.mp4", "load_dishwasher: 2 cups -> rack (learned)", "load_dishwasher", "kitchen; static rack, no door/sliding"),
    ("09_marble_apartment_bc_load_dishwasher.mp4", "load_dishwasher in a Marble apartment (learned)", "load_dishwasher", "Marble multiroom apartment, scale ASSUMED"),
]
CLIPS += [(fn, title, f"{_HH.get(t, 'results pending')} | {note}") for fn, title, t, note in HOUSEHOLD]


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


def card(title, sub, n=FPS * 2):
    img = np.full((H, W, 3), 18, np.uint8)
    y = 140
    for line in _wrap(title, 34):
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32
    y += 10
    for line in _wrap(sub, 44):
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 220, 255), 1, cv2.LINE_AA)
        y += 24
    return [img] * n


def main():
    frames = card("marble-home: final results", "GR1 humanoid | MuJoCo + RoboSuite | controller v2 + delta actions", FPS * 3)
    used = []
    for fn, title, sub in CLIPS:
        p = V / fn
        if not p.exists():
            print("missing", fn)
            continue
        frames += card(title, sub)
        for f in imageio.get_reader(p):
            frames.append(cv2.resize(np.asarray(f), (W, H)))
        used.append(fn)
    frames += card("Caveats", "state policies see item poses; one work-surface height (0.85 m); items are rigid proxies; Marble scales assumed", FPS * 3)
    out = V / "FINAL_reel.mp4"
    imageio.mimsave(out, frames, fps=FPS, macro_block_size=1)
    print(f"wrote {out}: {len(used)} clips, {len(frames) / FPS:.0f} s")


if __name__ == "__main__":
    main()
