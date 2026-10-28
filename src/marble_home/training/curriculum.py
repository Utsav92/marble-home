"""Adaptive success-based curriculum with confidence intervals, rollback, and a full transition log.

Advance when the Wilson LOWER bound of the recent success rate (>= ``min_episodes`` episodes) clears
``advance_lcb``; roll back when the UPPER bound falls below ``rollback_ucb``. Using the interval (not the point
estimate) prevents advancing on a lucky streak. Every transition records the exact criteria that triggered it.

Stage list mirrors the spec (section 11). Stages needing capabilities this repo does not have are flagged
``available: False`` and are skipped by ``CurriculumScheduler`` unless forced.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from marble_home.evaluation.metrics import wilson

STAGES = [
    {"id": 1, "name": "single cup, fixed robot, known location", "cup_x": (0.0, 0.0), "cup_y": (-0.15, -0.15), "distractors": 0, "scenes": "single", "available": True},
    {"id": 2, "name": "randomised cup position/orientation", "cup_x": (-0.05, 0.05), "cup_y": (-0.21, -0.09), "distractors": 0, "scenes": "single", "available": True},
    {"id": 3, "name": "multiple objects with distractors", "cup_x": (-0.05, 0.05), "cup_y": (-0.21, -0.09), "distractors": 2, "scenes": "single", "available": True},
    {"id": 4, "name": "randomised kitchen layouts", "cup_x": (-0.05, 0.05), "cup_y": (-0.21, -0.09), "distractors": 2, "scenes": "procedural", "available": True},
    {"id": 5, "name": "different Marble-generated kitchens", "distractors": 2, "scenes": "marble", "available": False,
     "why": "needs real Marble exports (no WORLDLABS_API_KEY); synthetic fixtures substitute only for pipeline tests"},
    {"id": 6, "name": "long-horizon sequential manipulation (pick then place)", "distractors": 2, "scenes": "procedural", "task": "place_cup", "available": True},
    {"id": 7, "name": "combined navigation and manipulation", "available": False,
     "why": "Level C (learned locomotion) is blocked; base is fixed (Level A)"},
]


@dataclass
class CurriculumScheduler:
    stages: list = field(default_factory=lambda: [s for s in STAGES if s["available"]])
    window: int = 100
    min_episodes: int = 30
    advance_lcb: float = 0.70
    rollback_ucb: float = 0.35
    idx: int = 0
    log: list = field(default_factory=list)
    _recent: deque = field(default_factory=deque)
    step: int = 0

    def __post_init__(self):
        self._recent = deque(maxlen=self.window)

    @property
    def stage(self) -> dict:
        return self.stages[self.idx]

    def record(self, success: bool, step: int | None = None) -> str | None:
        """Add an episode outcome; returns 'advance' / 'rollback' / None."""
        self._recent.append(bool(success))
        if step is not None:
            self.step = step
        n, k = len(self._recent), sum(self._recent)
        if n < self.min_episodes:
            return None
        lo, hi = wilson(k, n)
        if lo >= self.advance_lcb and self.idx < len(self.stages) - 1:
            return self._move(+1, "advance", k, n, lo, hi, f"wilson_lcb {lo:.3f} >= {self.advance_lcb}")
        if hi < self.rollback_ucb and self.idx > 0:
            return self._move(-1, "rollback", k, n, lo, hi, f"wilson_ucb {hi:.3f} < {self.rollback_ucb}")
        return None

    def _move(self, d, kind, k, n, lo, hi, why):
        frm = self.stage["id"]
        self.idx += d
        self.log.append({"step": self.step, "event": kind, "from_stage": frm, "to_stage": self.stage["id"], "successes": k, "episodes": n,
                         "success_rate": k / n, "wilson95": [lo, hi], "criterion": why,
                         "thresholds": {"advance_lcb": self.advance_lcb, "rollback_ucb": self.rollback_ucb, "min_episodes": self.min_episodes}})
        self._recent.clear()  # re-evaluate from scratch on the new stage
        return kind

    def dump(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"final_stage": self.stage["id"], "transitions": self.log,
                                          "skipped_unavailable": [{"id": s["id"], "why": s["why"]} for s in STAGES if not s["available"]]}, indent=2))
