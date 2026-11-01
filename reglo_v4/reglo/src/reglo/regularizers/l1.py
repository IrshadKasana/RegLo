"""L1 penalty  R(θ) = ‖θ‖₁ — sparsity                     (configs/regularizers/l1.yaml)

M2: ``penalty`` mode adds λ‖θ‖₁ to the loss; ``proximal`` mode soft-thresholds the weights
    after each AdamW step, which produces exact zeros.
M4: on the learned optimizer's weights (``meta.space: phi``, the study's setting): L_reg = λ‖φ‖₁
    with M2's λ (``meta.normalize: explicit``); ``mean`` uses ‖φ‖₁ / N, ``sum`` the bare ‖φ‖₁.
    ``meta.space: theta`` puts the same penalty on the optimizee's end point θ_T(φ) instead.
Biases and normalisation-layer parameters are excluded (of θ and of φ).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

from reglo.registry import REGULARIZERS
from reglo.regularizers.base import ExplicitResult, LossFn, MetaPenalty, Regularizer, batch_loss

__all__ = ["L1"]


@REGULARIZERS.register("l1")
class L1(Regularizer):
    name = "l1"

    def __init__(self, cfg: DictConfig, model: Optional[nn.Module] = None) -> None:
        super().__init__(cfg, model)
        self.coef = float(cfg.explicit.coef)
        self.mode = str(cfg.explicit.mode)
        if self.mode not in ("penalty", "proximal"):
            raise ValueError(f"l1 explicit.mode must be penalty or proximal, got '{self.mode}'")
        self.normalize = str(cfg.meta.normalize)
        if self.normalize not in ("explicit", "mean", "sum"):
            raise ValueError(f"l1 meta.normalize must be explicit, mean or sum, got '{self.normalize}'")

    def norm(self, params: Sequence[torch.Tensor]) -> torch.Tensor:
        return sum(p.abs().sum() for p in self.masked(params))

    def explicit_step(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> ExplicitResult:
        loss = batch_loss(model, x, y)
        if self.mode == "proximal":
            loss.backward()
            return ExplicitResult(loss.item())
        penalty = self.coef * self.norm(list(model.parameters()))
        (loss + penalty).backward()
        return ExplicitResult(loss.item(), penalty.item())

    @torch.no_grad()
    def after_step(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Proximal mode: θ ← sign(θ)·max(|θ| − lr·λ, 0)."""
        if self.mode != "proximal":
            return
        lr_of = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
        for p in self.masked(list(model.parameters())):
            shrink = lr_of[id(p)] * self.coef
            p.copy_(torch.sign(p) * torch.clamp(p.abs() - shrink, min=0.0))

    def _penalty(self, chosen: Sequence[torch.Tensor]) -> MetaPenalty:
        total = sum(p.abs().sum() for p in chosen)
        if self.normalize == "mean":
            total = total / sum(p.numel() for p in chosen)
        elif self.normalize == "explicit":
            total = self.coef * total
        return MetaPenalty(total, total.item())

    def meta_penalty(self, loss_fn: LossFn, params: Sequence[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> MetaPenalty:
        return self._penalty(self.masked(params))

    def phi_penalty(self, named_phi) -> MetaPenalty:
        return self._penalty(self.phi_weights(named_phi))
