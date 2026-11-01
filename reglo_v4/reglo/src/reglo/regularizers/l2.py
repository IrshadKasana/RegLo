"""L2 penalty  R(θ) = ½‖θ‖₂² — weight-norm shrinkage       (configs/regularizers/l2.yaml)

M2: ``penalty`` mode adds ½λ‖θ‖₂² to the loss (the same R as M4); ``decoupled`` mode uses
    AdamW's own weight decay instead (set up when the optimizer is built).
M4: on the learned optimizer's weights (``meta.space: phi``, the study's setting): L_reg = ½λ‖φ‖₂²
    with M2's λ (``meta.normalize: explicit``); ``mean`` uses ½‖φ‖₂² / N, ``sum`` the bare ½‖φ‖₂².
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

__all__ = ["L2"]


@REGULARIZERS.register("l2")
class L2(Regularizer):
    name = "l2"

    def __init__(self, cfg: DictConfig, model: Optional[nn.Module] = None) -> None:
        super().__init__(cfg, model)
        self.coef = float(cfg.explicit.coef)
        self.mode = str(cfg.explicit.mode)
        if self.mode not in ("penalty", "decoupled"):
            raise ValueError(f"l2 explicit.mode must be penalty or decoupled, got '{self.mode}'")
        self.decoupled_coef = float(cfg.explicit.get("decoupled_coef", self.coef))
        self.normalize = str(cfg.meta.normalize)
        if self.normalize not in ("explicit", "mean", "sum"):
            raise ValueError(f"l2 meta.normalize must be explicit, mean or sum, got '{self.normalize}'")

    def half_squared_norm(self, params: Sequence[torch.Tensor]) -> torch.Tensor:
        return 0.5 * sum(p.pow(2).sum() for p in self.masked(params))

    def explicit_step(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> ExplicitResult:
        loss = batch_loss(model, x, y)
        if self.mode == "decoupled":
            loss.backward()
            return ExplicitResult(loss.item())
        penalty = self.coef * self.half_squared_norm(list(model.parameters()))
        (loss + penalty).backward()
        return ExplicitResult(loss.item(), penalty.item())

    def _penalty(self, chosen: Sequence[torch.Tensor]) -> MetaPenalty:
        total = 0.5 * sum(p.pow(2).sum() for p in chosen)
        if self.normalize == "mean":
            total = total / sum(p.numel() for p in chosen)
        elif self.normalize == "explicit":
            total = self.coef * total
        return MetaPenalty(total, total.item())

    def meta_penalty(self, loss_fn: LossFn, params: Sequence[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> MetaPenalty:
        return self._penalty(self.masked(params))

    def phi_penalty(self, named_phi) -> MetaPenalty:
        return self._penalty(self.phi_weights(named_phi))
