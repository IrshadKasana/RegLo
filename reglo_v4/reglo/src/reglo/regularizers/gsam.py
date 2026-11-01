"""GSAM — Surrogate Gap Guided Sharpness-Aware Minimization (Zhuang et al. 2022).

    perturbed loss  L_p(θ) = L(θ + ε̂),      surrogate gap  h(θ) = L_p(θ) − L(θ)
    update direction       d = ∇L_p − α·∇⊥L,
    where ∇⊥L is the part of ∇L(θ) orthogonal to ∇L_p: descending d lowers L_p, and the
    orthogonal ascent shrinks h without raising L_p.

M2: two passes, then ``p.grad = d``.
M4: d at θ_T is backpropagated into φ (through the surrogate ⟨d, θ_T⟩, whose gradient is d);
    the logged value is the surrogate gap h(θ_T).
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
    grads_of,
    set_grads,
    trainable,
)
from reglo.regularizers.sam import sam_perturbation

__all__ = ["GSAM", "gsam_direction"]


def gsam_direction(
    grads: Sequence[torch.Tensor], perturbed_grads: Sequence[torch.Tensor], alpha: float
) -> List[torch.Tensor]:
    """d = ∇L_p − α·(∇L − proj_{∇L_p} ∇L)."""
    dot = sum((g * gp).sum() for g, gp in zip(grads, perturbed_grads))
    norm_sq = sum(gp.pow(2).sum() for gp in perturbed_grads)
    coef = dot / (norm_sq + _TINY)
    return [gp - alpha * (g - coef * gp) for g, gp in zip(grads, perturbed_grads)]


@REGULARIZERS.register("gsam")
class GSAM(Regularizer):
    name = "gsam"
    replays = True  # meta.space phi: perturb φ along ∇_φ L_meta, replay the unroll at φ + ε̂

    def __init__(self, cfg: DictConfig, model: Optional[nn.Module] = None) -> None:
        super().__init__(cfg, model)
        self.rho = float(cfg.rho)
        self.alpha = float(cfg.alpha)

    @property
    def passes(self) -> int:
        return 2

    def explicit_step(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> ExplicitResult:
        params = trainable(model)
        loss = batch_loss(model, x, y)
        loss.backward()
        grads = grads_of(params)
        eps = sam_perturbation(grads, self.rho)
        model.zero_grad(set_to_none=True)
        with displaced(params, eps), frozen_bn_stats(model):
            perturbed = batch_loss(model, x, y)
            perturbed.backward()
        perturbed_grads = grads_of(params)
        set_grads(params, gsam_direction(grads, perturbed_grads, self.alpha))
        return ExplicitResult(loss.item(), perturbed.item() - loss.item())

    def meta_penalty(self, loss_fn: LossFn, params: Sequence[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> MetaPenalty:
        loss = loss_fn(params, x, y)
        grads = torch.autograd.grad(loss, params)
        eps = sam_perturbation(grads, self.rho)
        perturbed = loss_fn([p + e for p, e in zip(params, eps)], x, y)
        perturbed_grads = torch.autograd.grad(perturbed, params)
        direction = gsam_direction(grads, perturbed_grads, self.alpha)
        term = sum((d.detach() * p).sum() for d, p in zip(direction, params))  # ∂term/∂θ_T = d
        return MetaPenalty(term, perturbed.item() - loss.item())

    def phi_perturbation(self, grads: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return sam_perturbation(grads, self.rho)

    def phi_gradient(self, grads: Sequence[torch.Tensor], perturbed: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return gsam_direction(grads, perturbed, self.alpha)
