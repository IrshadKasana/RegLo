"""SAM — Sharpness-Aware Minimization (Foret et al. 2021).

    R(θ) = max_{‖ε‖₂≤ρ} L(θ+ε) − L(θ),   one-step inner max  ε̂ = ρ·∇L/‖∇L‖
    (ASAM, ``adaptive: true``:  ε̂ = ρ·|θ|²∇L / ‖|θ|∇L‖)

M2: ascend to θ+ε̂, take the gradient there, step from θ (two passes).
M4: L_reg = L(θ_T + ε̂) with ε̂ held fixed — the perturbed loss, as Algorithm 1 writes it;
    the logged value is the sharpness L(θ_T+ε̂) − L(θ_T).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

from reglo.registry import REGULARIZERS
from reglo.regularizers.base import (
    _TINY,
    ExplicitResult,
    LossFn,
    MetaPenalty,
    Regularizer,
    batch_loss,
    displaced,
    frozen_bn_stats,
    global_norm,
    grads_of,
    trainable,
)

__all__ = ["SAM", "sam_perturbation"]


def sam_perturbation(
    grads: Sequence[torch.Tensor], rho: float, params: Optional[Sequence[torch.Tensor]] = None, adaptive: bool = False
) -> List[torch.Tensor]:
    """ε̂ for SAM (or ASAM when ``adaptive`` and ``params`` are given)."""
    if adaptive:
        scaled = [p.detach().abs() * g for p, g in zip(params, grads)]
        norm = global_norm(scaled)
        return [rho * p.detach().pow(2) * g / (norm + _TINY) for p, g in zip(params, grads)]
    norm = global_norm(grads)
    return [rho * g / (norm + _TINY) for g in grads]


@REGULARIZERS.register("sam")
class SAM(Regularizer):
    name = "sam"
    replays = True  # meta.space phi: perturb φ along ∇_φ L_meta, replay the unroll at φ + ε̂

    def __init__(self, cfg: DictConfig, model: Optional[nn.Module] = None) -> None:
        super().__init__(cfg, model)
        self.rho = float(cfg.rho)
        self.adaptive = bool(cfg.explicit.get("adaptive", False))

    @property
    def passes(self) -> int:
        return 2

    def explicit_step(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> ExplicitResult:
        params = trainable(model)
        loss = batch_loss(model, x, y)
        loss.backward()
        with torch.no_grad():
            eps = sam_perturbation(grads_of(params), self.rho, params, self.adaptive)
        model.zero_grad(set_to_none=True)
        with displaced(params, eps), frozen_bn_stats(model):
            perturbed = batch_loss(model, x, y)
            perturbed.backward()  # p.grad = ∇L(θ + ε̂); the weights return to θ on exit
        return ExplicitResult(loss.item(), perturbed.item() - loss.item())

    def meta_penalty(self, loss_fn: LossFn, params: Sequence[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> MetaPenalty:
        loss = loss_fn(params, x, y)
        grads = torch.autograd.grad(loss, params)
        eps = sam_perturbation(grads, self.rho, params, self.adaptive)
        perturbed = loss_fn([p + e for p, e in zip(params, eps)], x, y)  # ε̂ is a constant here
        return MetaPenalty(perturbed, perturbed.item() - loss.item())

    def phi_perturbation(self, grads: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return sam_perturbation(grads, self.rho)

    def phi_gradient(self, grads: Sequence[torch.Tensor], perturbed: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return list(perturbed)  # ∇_φ L_meta(φ + ε̂)  — Algorithm 1, line 22
