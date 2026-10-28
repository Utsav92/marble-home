"""Rollout rendering with a status overlay (task / policy / reward / success / phase / EE error)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import imageio
import numpy as np

from marble_home.tasks.definitions import PHASES


def overlay(frame: np.ndarray, lines: list[str], ok: bool | None = None) -> np.ndarray:
    img = np.ascontiguousarray(frame.copy())
    h = 18 * len(lines) + 8
    bar = img[:h].astype(np.float32) * 0.35
    img[:h] = bar.astype(np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (6, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if ok is not None:
        col = (60, 220, 90) if ok else (230, 80, 70)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), col, 3)
    return img


def rollout_video(env, policy: Callable, path: str | Path, policy_name: str, seed: int | None = None,
                  camera: str = "eval_side", size=(480, 360), fps: int = 20, max_steps: int | None = None,
                  on_reset: Callable | None = None) -> dict:
    """Run one episode with ``policy(obs)->action`` in a HouseholdEnv; write an annotated MP4. Returns stats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obs, info = env.reset(seed=seed)
    if on_reset:
        on_reset(obs, info)
    frames, ret, done, t = [], 0.0, False, 0
    ever = False
    last = {}
    limit = max_steps or env.horizon
    while not done and t < limit:
        a = policy(obs)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        done = term or trunc
        ever = ever or bool(info.get("success"))
        last = info
        ph = PHASES[min(int(info.get("phase", 0)), len(PHASES) - 1)]
        frames.append(overlay(env.render(*size, camera=camera), [
            f"{env.task} | {policy_name}  t={t:03d}",
            f"return={ret:6.2f}  success={ever}  phase={ph}",
            f"ee_err={info.get('eef_error', 0):.3f} m  place_err={info.get('place_error', 0):.3f} m",
        ], ok=True if ever else None))
        t += 1
    for _ in range(int(fps * 0.8)):  # hold last frame
        frames.append(frames[-1])
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return {"video": str(path), "steps": t, "return": ret, "success": ever,
            "place_error": last.get("place_error"), "eef_error": last.get("eef_error")}
