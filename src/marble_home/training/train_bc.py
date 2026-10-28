"""Behaviour-cloning trainer (state MLP). Image CNN variant lives in train_bc_images (needs rendered demos)."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from marble_home.datasets.torch_ds import StepDataset, load_episodes
from marble_home.policies.agents import save_checkpoint
from marble_home.policies.bc import MLPPolicy
from marble_home.training.common import Run, dataset_action_mode, dataset_path, episode_stats, set_seed


def _forward(model, o, cnn: bool):
    return model(o) if cnn else model(o["state"])


def train_bc_images(cfg: dict) -> dict:
    """CNN behaviour cloning on rendered datasets (obs: rgb | rgbd | multicam)."""
    import numpy as np

    from marble_home.policies.bc import CNNPolicy

    set_seed(cfg["seed"])
    task, ds = cfg["task"], cfg["dataset"]
    mode = cfg.get("obs", "rgb")
    keys = ["proprio", "rgb_head"] + (["depth_head"] if mode in ("rgbd", "multicam") else []) + (["rgb_wrist_r", "rgb_wrist_l"] if mode == "multicam" else [])
    cams = ("head",) + (("wrist_r", "wrist_l") if mode == "multicam" else ())
    train = load_episodes(ds, "train", tuple(keys), max_episodes=cfg.get("n_demos"))
    val = load_episodes(ds, "val", tuple(keys))
    st = episode_stats(train, "proprio")
    model = CNNPolicy(train[0]["obs"]["proprio"].shape[1], 4, cams, depth=mode in ("rgbd", "multicam"), feat=cfg.get("feat", 64),
                      hidden=tuple(cfg["hidden"]), stats={"mean": st["mean"], "std": st["std"]})
    dl = DataLoader(StepDataset(train), batch_size=cfg["batch_size"], shuffle=True)
    vl = DataLoader(StepDataset(val), batch_size=256)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    name = cfg.get("run_name") or f"bc_cnn_{mode}_{ds}_n{len(train)}_s{cfg['seed']}"
    run = Run(name, cfg, dataset_path(ds))
    best, bad, best_state = float("inf"), 0, None
    for ep in range(cfg["epochs"]):
        model.train()
        tl, n = 0.0, 0
        for o, a in dl:
            loss = torch.nn.functional.mse_loss(model(o), a)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tl += loss.item() * len(a)
            n += len(a)
        sched.step()
        model.eval()
        with torch.no_grad():
            vs = [(torch.nn.functional.mse_loss(model(o), a, reduction="sum").item(), len(a)) for o, a in vl]
        vmse = sum(x for x, _ in vs) / sum(k for _, k in vs) / 4
        run.log(ep, train_mse=tl / n, val_mse=vmse)
        if vmse < best - 1e-6:
            best, bad, best_state = vmse, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    sha = save_checkpoint(run.ckpt, model, {"task": task, "obs": mode, "dataset": ds, "action_mode": dataset_action_mode(ds), "n_train_episodes": len(train), "cfg": cfg,
                                            "best_val_mse": best, "epochs_run": ep + 1})
    run.finish(sha, best_val_mse=best, epochs_run=ep + 1, n_train_episodes=len(train))
    return {"name": name, "ckpt": str(run.ckpt), "best_val_mse": best, "epochs": ep + 1, "run_dir": str(run.dir)}


def train_bc(cfg: dict) -> dict:
    if cfg.get("model") == "cnn":
        return train_bc_images(cfg)
    set_seed(cfg["seed"])
    task = cfg["task"]
    ds = cfg.get("dataset") or task  # dataset name (== task unless a multi-scene/other dataset is used)
    key = cfg.get("obs_key", "state")  # "state" (privileged cup pose included) | "proprio" (robot-only, no object information)
    train = load_episodes(ds, "train", (key,), max_episodes=cfg.get("n_demos"))
    val = load_episodes(ds, "val", (key,))
    drop = set(cfg.get("obs_drop") or [])
    keep = [i for i in range(train[0]["obs"][key].shape[1]) if i not in drop]
    if drop:  # remove inputs that are irrelevant to the task (e.g. placement-goal xy for pick) -> no spurious correlations
        for e in train + val:
            e["obs"][key] = e["obs"][key][:, keep]
    st = episode_stats(train, key)
    obs_dim = train[0]["obs"][key].shape[1]
    model = MLPPolicy(obs_dim, 4, tuple(cfg["hidden"]), {"mean": st["mean"], "std": st["std"]})
    dl = DataLoader(StepDataset(train), batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    vl = DataLoader(StepDataset(val), batch_size=1024)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    name = cfg.get("run_name") or f"bc_mlp_{task}_n{len(train)}_s{cfg['seed']}"
    run = Run(name, cfg, dataset_path(ds))
    best, bad, best_state = float("inf"), 0, None
    for ep in range(cfg["epochs"]):
        model.train()
        tl, n = 0.0, 0
        for o, a in dl:
            loss = torch.nn.functional.mse_loss(model(o[key]), a)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tl += loss.item() * len(a)
            n += len(a)
        sched.step()
        model.eval()
        with torch.no_grad():
            vs = [(torch.nn.functional.mse_loss(model(o[key]), a, reduction="sum").item(), len(a)) for o, a in vl]
        vmse = sum(s for s, _ in vs) / sum(k for _, k in vs) / 4
        run.log(ep, train_mse=tl / n, val_mse=vmse, lr=sched.get_last_lr()[0])
        if vmse < best - 1e-6:
            best, bad = vmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    sha = save_checkpoint(run.ckpt, model, {"task": task, "obs": "state", "obs_key": key, "obs_keep": keep if drop else None, "action_mode": dataset_action_mode(ds), "n_train_episodes": len(train), "cfg": cfg,
                                            "best_val_mse": best, "epochs_run": ep + 1})
    run.finish(sha, best_val_mse=best, epochs_run=ep + 1, n_train_episodes=len(train),
               n_train_samples=int(sum(len(e["act"]) for e in train)))
    return {"name": name, "ckpt": str(run.ckpt), "best_val_mse": best, "epochs": ep + 1, "run_dir": str(run.dir)}
