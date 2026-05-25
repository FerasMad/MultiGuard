"""LR scheduler builder per V3.1 section 5.5: StepLR decay 0.1 at epoch 30."""

from __future__ import annotations

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    kind = cfg.get("type", "steplr")
    if kind == "steplr":
        step_size = int(cfg.get("lr_step", 30))
        gamma = float(cfg.get("lr_gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if kind == "cosine":
        t_max = int(cfg.get("t_max", 50))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    if kind == "none":
        return None
    raise ValueError(f"unknown scheduler: {kind}")
