"""M4's meta-objective (Algorithm 1, lines 16–23) and the weights w_t of L_meta.

    L_meta,reg(φ) = L_meta(φ) + λ_smooth·L_smooth(φ) + λ_reg·L_reg(θ_T(φ))
    L_meta(φ)     = Σ_t w_t·L(θ_t),   Σ_t w_t = 1

The weights are normalised so L_meta is on the scale of the training loss itself, which keeps
λ_smooth and λ_reg meaningful whatever the unroll length: L_reg is the regularized part of the
objective M2 trains with, at θ_T (so λ_reg = 1 gives M2's strength), and L_smooth is averaged
over the inner steps where it is computed. M3 is the special case λ_smooth = λ_reg = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from reglo.regularizers.base import MetaPenalty

__all__ = ["MetaObjective", "build_objective", "loss_weights"]


def loss_weights(kind: str, steps: int, device=None) -> torch.Tensor:
    """w_1 … w_T, summing to one: uniform | linear (later steps weigh more) | exponential."""
    t = torch.arange(1, steps + 1, dtype=torch.float32, device=device)
    if kind == "uniform":
        w = torch.ones_like(t)
    elif kind == "linear":
        w = t
    elif kind == "exponential":
        w = torch.exp(t / steps)
    else:
        raise ValueError(f"outer.loss_weights must be uniform, linear or exponential, got '{kind}'")
    return w / w.sum()


@dataclass
class MetaObjective:
    total: torch.Tensor  # what is backpropagated into φ
    parts: Dict[str, float]  # l_meta, l_smooth, l_reg and the logged objective
    l_meta: torch.Tensor  # kept for the meta.space=phi path, which needs ∇_φ L_meta alone


def build_objective(
    losses: List[torch.Tensor],
    weights: torch.Tensor,
    smooth_terms: List[torch.Tensor],
    smooth_coef: float,
    penalty: Optional[MetaPenalty],
    reg_coef: float,
) -> MetaObjective:
    l_meta = sum(w * loss for w, loss in zip(weights.to(losses[0].device), losses))
    total = l_meta
    parts = {"l_meta": l_meta.item(), "l_smooth": 0.0, "l_reg": 0.0}
    if smooth_terms:
        l_smooth = torch.stack(smooth_terms).mean()
        total = total + smooth_coef * l_smooth
        parts["l_smooth"] = l_smooth.item()
    if penalty is not None:
        total = total + reg_coef * penalty.term
        parts["l_reg"] = penalty.value
    parts["objective"] = parts["l_meta"] + smooth_coef * parts["l_smooth"] + reg_coef * parts["l_reg"]
    return MetaObjective(total, parts, l_meta)
