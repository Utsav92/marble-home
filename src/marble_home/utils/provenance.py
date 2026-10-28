"""Run provenance: everything section 19 of the spec requires a training/eval run to record."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def hardware() -> dict:
    info = {"machine": platform.machine(), "processor": platform.processor(), "os": platform.platform(),
            "cpu_count": os.cpu_count()}
    try:
        import psutil  # optional

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    try:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("mujoco", "robosuite", "torch", "numpy", "gymnasium", "stable_baselines3", "trimesh", "h5py", "mink"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = None
    return out


def pip_lock_hash() -> str | None:
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        return hashlib.sha256(freeze.encode()).hexdigest()[:16]
    except Exception:
        return None


def run_manifest(**extra) -> dict:
    m = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "versions": versions(),
        "hardware": hardware(),
        "pip_freeze_sha": pip_lock_hash(),
    }
    m.update(extra)
    return m


def write_json(path: str | os.PathLike, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
