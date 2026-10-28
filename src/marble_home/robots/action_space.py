"""Policy-facing action spaces for the GR1 right-arm manipulation tasks (IK controller).

The 24-D ``ik`` controller action is mostly constant for these tasks (left arm held, hand
orientation held, six identical finger-grip channels). Policies therefore act in a reduced
4-D space and the wrapper expands it:

    a = [x, y, z, grip]  in [-1, 1]
      ee_abs    : xyz is an ABSOLUTE right-hand target inside ``WORKSPACE`` (base frame)
      ee_delta  : xyz is a delta (``DELTA_SCALE`` m/step) applied to the previous target

Limitation (documented in docs/limitations.md): hand orientation and the left arm are not
learned; the policy only controls hand position and a scalar grip.
"""
from __future__ import annotations

import numpy as np

# right-hand workspace in the robot BASE frame (x forward, y left, z up); centre/half-extent
WS_CENTER = np.array([0.40, -0.30, 0.08])
WS_HALF = np.array([0.25, 0.25, 0.20])
DELTA_SCALE = 0.03  # m per control step at |a|=1
ACTION_DIM = 4
IK_ACTION_DIM = 24


def pos_to_unit(p_base: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(p_base) - WS_CENTER) / WS_HALF, -1.0, 1.0)


def unit_to_pos(u: np.ndarray) -> np.ndarray:
    return WS_CENTER + np.clip(u, -1, 1) * WS_HALF


def full_to_reduced(full: np.ndarray) -> np.ndarray:
    """24-D ik action -> 4-D ee_abs action (xyz absolute target, mean right grip)."""
    full = np.asarray(full, dtype=float)
    return np.concatenate([pos_to_unit(full[0:3]), [float(np.clip(full[12:18].mean(), -1, 1))]]).astype(np.float32)


def reduced_to_full(a: np.ndarray, aa_right: np.ndarray, left6: np.ndarray, p_base: np.ndarray | None = None,
                    mode: str = "ee_abs", prev_target: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Expand a 4-D policy action to the 24-D ik action. Returns (full_action, new_target_base_xyz)."""
    a = np.asarray(a, dtype=float).reshape(-1)
    if mode == "ee_abs":
        target = unit_to_pos(a[:3])
    elif mode == "ee_delta":
        base = prev_target if prev_target is not None else p_base
        target = np.clip(base + np.clip(a[:3], -1, 1) * DELTA_SCALE, WS_CENTER - WS_HALF, WS_CENTER + WS_HALF)
    else:
        raise ValueError(mode)
    grip = float(np.clip(a[3], -1, 1))
    full = np.concatenate([target, aa_right, left6, np.full(6, grip), np.full(6, -1.0)])
    return full, target
