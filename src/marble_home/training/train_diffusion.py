"""Diffusion-policy trainer (state observations; image variant via cameras in cfg when rendered demos exist)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from marble_home.datasets.torch_ds import SeqDataset, load_episodes
from marble_home.policies.agents import save_checkpoint
from marble_home.policies.diffusion import EMA, DiffusionPolicy
from marble_home.training.common import Run, dataset_action_mode, dataset_path, episode_stats, set_seed


def _obs_of(o):
    return o["state"]


@torch.no_grad()
def val_losses(ema_model, dl, n_sample_batches: int = 2, ddim_steps: int = 10):
    """(noise-prediction loss, action-chunk MSE of DDIM samples vs ground truth) on validation windows."""
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    nl, n = 0.0, 0
    am, an = 0.0, 0
    for i, (o, a) in enumerate(dl):
        obs = _obs_of(o)
        nl += ema_model.loss(obs, a).item() * len(a)
        n += len(a)
        if i < n_sample_batches:
            pred = ema_model.sample(obs, ddim_steps=ddim_steps, generator=g)
            am += ((pred - a) ** 2).mean().item() * len(a)
            an += len(a)
    return nl / n, am / max(an, 1)


def train_diffusion(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    task = cfg["task"]
    To, H = cfg["n_obs"], cfg["horizon"]
    ds = cfg.get("dataset") or task
    train = load_episodes(ds, "train", ("state",), max_episodes=cfg.get("n_demos"))
    val = load_episodes(ds, "val", ("state",))
    st = episode_stats(train, "state")
    obs_dim = train[0]["obs"]["state"].shape[1]
    model = DiffusionPolicy(obs_dim, 4, To, H, cfg["n_act"], cfg["train_steps"], cfg["ddim_steps"], tuple(cfg["widths"]),
                            cfg["obs_feat"], {"mean": st["mean"], "std": st["std"]})
    ema = EMA(model, cfg.get("ema_decay", 0.995))
    dl = DataLoader(SeqDataset(train, To, H), batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    vdl = DataLoader(SeqDataset(val, To, H), batch_size=256, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total, pct_start=0.05)
    name = cfg.get("run_name") or f"diffusion_{task}_n{len(train)}_s{cfg['seed']}"
    run = Run(name, cfg, dataset_path(ds))
    best, bad, best_state, step = float("inf"), 0, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        tl, n = 0.0, 0
        for o, a in dl:
            loss = model.loss(_obs_of(o), a)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            ema.update(model)
            tl += loss.item() * len(a)
            n += len(a)
            step += 1
        vloss, vmse = val_losses(ema.model, vdl, ddim_steps=cfg["ddim_steps"])
        run.log(ep, train_loss=tl / n, val_noise_loss=vloss, val_action_mse=vmse, lr=sched.get_last_lr()[0])
        if vmse < best - 1e-6:
            best, bad = vmse, 0
            best_state = {k: v.clone() for k, v in ema.model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    ema.model.load_state_dict(best_state)
    sha = save_checkpoint(run.ckpt, ema.model, {"task": task, "obs": "state", "action_mode": dataset_action_mode(ds), "n_train_episodes": len(train), "cfg": cfg,
                                                "best_val_action_mse": best, "epochs_run": ep + 1, "weights": "ema"})
    run.finish(sha, best_val_action_mse=best, epochs_run=ep + 1, n_train_episodes=len(train),
               n_params=int(sum(p.numel() for p in model.parameters())))
    return {"name": name, "ckpt": str(run.ckpt), "best_val_action_mse": best, "epochs": ep + 1, "run_dir": str(run.dir)}
