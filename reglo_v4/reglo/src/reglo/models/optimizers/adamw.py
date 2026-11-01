"""AdamW — the conventional optimizer of M1 and M2 — and its learning-rate schedule."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

from reglo.models.common import penalty_mask

__all__ = ["build_adamw", "build_scheduler"]


def build_adamw(
    cfg: DictConfig,
    model: nn.Module,
    decoupled_l2: float = 0.0,
    exclude: Sequence[str] = ("bias", "norm"),
) -> torch.optim.AdamW:
    """AdamW from ``configs/optimizers/adamw.yaml``.

    ``decoupled_l2 > 0`` (M2-L2 with ``mode: decoupled``) adds that weight decay to the
    parameters an L2 penalty would touch, leaving biases and normalisation layers alone.
    """
    base_decay = float(cfg.weight_decay)
    params = list(model.parameters())
    if decoupled_l2 > 0:
        mask = penalty_mask(model, exclude)
        groups = [
            {"params": [p for p, m in zip(params, mask) if m], "weight_decay": base_decay + float(decoupled_l2)},
            {"params": [p for p, m in zip(params, mask) if not m], "weight_decay": base_decay},
        ]
        groups = [g for g in groups if g["params"]]
    else:
        groups = [{"params": params, "weight_decay": base_decay}]
    return torch.optim.AdamW(groups, lr=float(cfg.lr), betas=tuple(cfg.betas), eps=float(cfg.eps))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    total_steps: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up, then cosine decay to ``min_lr`` (or constant), stepped once per optimizer step.

    ``cfg`` is the optimizer config (it holds ``lr`` and the ``scheduler`` block).
    """
    sched = cfg.scheduler
    if sched.name not in ("cosine", "constant"):
        raise ValueError(f"scheduler.name must be cosine or constant, got '{sched.name}'")
    warmup = int(round(float(sched.warmup_epochs) * steps_per_epoch))
    floor = float(sched.min_lr) / float(cfg.lr) if float(cfg.lr) > 0 else 0.0
    total = max(int(total_steps), 1)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if sched.name == "constant":
            return 1.0
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
