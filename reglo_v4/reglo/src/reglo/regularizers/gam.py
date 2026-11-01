"""GAM — Gradient-norm Aware Minimization (Zhang et al. 2023): first-order flatness.

    form gradnorm (protocol):  R(θ) = ρ·‖∇L(θ)‖₂
    form max      (paper):     R(θ) = ρ·max_{‖ε‖≤ρ} ‖∇L(θ+ε)‖₂, one ascent step on ‖∇L‖

M2 uses finite-difference Hessian-vector products, H·v ≈ (∇L(θ + r·v) − ∇L(θ)) / r:
    gradnorm:  ∇[L + ρ‖∇L‖] = ∇L + ρ·H·∇L/‖∇L‖              — 2 passes
    max:       the same at the ascent point θ + ρ·a, a = H∇L/‖H∇L‖ — 4 passes
M4 differentiates ‖∇L(θ_T)‖ exactly (double backpropagation) into φ.
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
    set_grads,
    trainable,
)

__all__ = ["GAM"]


@REGULARIZERS.register("gam")
class GAM(Regularizer):
    name = "gam"
    replays = True  # meta.space phi: perturb φ along ∇_φ L_meta, replay the unroll at φ + ε̂

    def __init__(self, cfg: DictConfig, model: Optional[nn.Module] = None) -> None:
        super().__init__(cfg, model)
        self.rho = float(cfg.rho)
        self.form = str(cfg.form)
        if self.form not in ("gradnorm", "max"):
            raise ValueError(f"gam form must be gradnorm or max, got '{self.form}'")
        self.fd_radius = float(cfg.explicit.fd_radius)

    @property
    def passes(self) -> int:
        return 2 if self.form == "gradnorm" else 4

    # ---------------------------------------------------------------- M2
    def _grad_at(self, model: nn.Module, params, x, y, first: bool) -> tuple:
        model.zero_grad(set_to_none=True)
        if first:
            loss = batch_loss(model, x, y)
            loss.backward()
        else:
            with frozen_bn_stats(model):
                loss = batch_loss(model, x, y)
                loss.backward()
        return loss.item(), grads_of(params)

    def _hvp_direction(self, model, params, x, y, grads) -> List[torch.Tensor]:
        """H·v for v = ∇L/‖∇L‖ at the current weights, by finite differences (one extra pass)."""
        r = self.fd_radius
        norm = global_norm(grads) + _TINY
        with displaced(params, [r * g / norm for g in grads]):
            _, shifted = self._grad_at(model, params, x, y, first=False)
        return [(gs - g) / r for gs, g in zip(shifted, grads)]

    def explicit_step(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> ExplicitResult:
        params = trainable(model)
        loss, grads = self._grad_at(model, params, x, y, first=True)
        hv = self._hvp_direction(model, params, x, y, grads)
        if self.form == "gradnorm":
            set_grads(params, [g + self.rho * h for g, h in zip(grads, hv)])
            return ExplicitResult(loss, self.rho * global_norm(grads).item())
        ascent = [self.rho * h / (global_norm(hv) + _TINY) for h in hv]
        with displaced(params, ascent):
            _, adv_grads = self._grad_at(model, params, x, y, first=False)
            adv_hv = self._hvp_direction(model, params, x, y, adv_grads)
        set_grads(params, [g + self.rho * h for g, h in zip(grads, adv_hv)])
        return ExplicitResult(loss, self.rho * global_norm(adv_grads).item())

    # ---------------------------------------------------------------- M4
    def meta_penalty(self, loss_fn: LossFn, params: Sequence[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> MetaPenalty:
        point = list(params)
        if self.form == "max":
            loss = loss_fn(params, x, y)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            hv = torch.autograd.grad(torch.sqrt(sum(g.pow(2).sum() for g in grads)), params)
            norm = global_norm(hv)
            point = [p + self.rho * h / (norm + _TINY) for p, h in zip(params, hv)]  # ascent step held fixed
        loss = loss_fn(point, x, y)
        grads = torch.autograd.grad(loss, point, create_graph=True)
        term = self.rho * torch.sqrt(sum(g.pow(2).sum() for g in grads) + _TINY)
        return MetaPenalty(term, term.item())

    # ---------------------------------------------------------------- M4, meta.space: phi
    def phi_perturbation(self, grads: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        norm = global_norm(grads)
        return [self.fd_radius * g / (norm + _TINY) for g in grads]

    def phi_gradient(self, grads: Sequence[torch.Tensor], perturbed: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [self.rho * (gp - g) / self.fd_radius for g, gp in zip(grads, perturbed)]  # ρ·H·∇/‖∇‖ in φ
