"""The regularizer interface.

Each regularizer r ∈ {L1, L2, SAM, GAM, GSAM} plays two roles in this project:

* **M2 — explicit, at every training step.** ``explicit_step(model, x, y)`` leaves in
  ``p.grad`` the gradient of the regularized objective; the caller then steps AdamW.
  SAM, GAM and GSAM need two forward–backward passes (``passes``); L1 and L2 one.

* **M4 — inside Algorithm 1's meta-objective**, on the learned optimizer's weights φ
  (``meta.space: phi``, the study's setting — Algorithm 1, lines 21–22):
  L1 / L2 penalise φ itself (``phi_penalty``); SAM, GAM and GSAM perturb φ along
  ∇_φ L_meta and replay the unroll at φ + ε̂ (``replays``; ``phi_perturbation`` /
  ``phi_gradient`` give the meta-trainer the two pieces it needs).
  With ``meta.space: theta`` instead, ``meta_penalty(loss_fn, θ_T, x, y)`` returns L_reg at
  the unroll's end point θ_T(φ), differentiable through the unroll into φ.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from reglo.models.common import penalty_mask
from reglo.registry import REGULARIZERS

__all__ = [
    "ExplicitResult",
    "LossFn",
    "MetaPenalty",
    "NoRegularizer",
    "Regularizer",
    "frozen_bn_stats",
    "global_norm",
]

#: (params, x, y) → scalar loss, e.g. ``FunctionalModel.loss``
LossFn = Callable[[Sequence[torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]

_TINY = 1e-12


@dataclass
class ExplicitResult:
    loss: float  # training loss L(θ) on the batch, from the first pass
    penalty: Optional[float] = None  # the regularizer's own number for this step (logged)


@dataclass
class MetaPenalty:
    term: torch.Tensor  # λ_reg · term is backpropagated into φ
    value: float  # R(θ_T) for the logs


def global_norm(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """‖(t_1, …, t_n)‖₂ over all tensors together."""
    tensors = list(tensors)
    if not tensors:
        return torch.zeros(())
    return torch.sqrt(sum(t.pow(2).sum() for t in tensors))


@contextmanager
def displaced(params: Sequence[torch.Tensor], deltas: Sequence[torch.Tensor]) -> Iterator[None]:
    """Temporarily move the weights by ``deltas``; afterwards copy back the exact originals."""
    saved = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, d in zip(params, deltas):
            p.add_(d)
    try:
        yield
    finally:
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.copy_(s)


@contextmanager
def frozen_bn_stats(model: nn.Module) -> Iterator[None]:
    """Extra forward passes (SAM's perturbed pass, …) must not update BatchNorm running statistics twice."""
    saved = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            saved.append((module, module.momentum))
            module.momentum = 0.0
    try:
        yield
    finally:
        for module, momentum in saved:
            module.momentum = momentum


def trainable(model: nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def batch_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(model(x), y)


def grads_of(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]


def set_grads(params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor]) -> None:
    for p, g in zip(params, grads):
        p.grad = g


class Regularizer:
    """No regularizer (M1, M3): plain gradients, no meta-penalty."""

    name = "none"

    def __init__(self, cfg: DictConfig, model: Optional[nn.Module] = None) -> None:
        self.cfg = cfg
        exclude = list(cfg.get("exclude", ["bias", "norm"]))
        self.mask: Optional[List[bool]] = penalty_mask(model, exclude) if model is not None else None
        meta = cfg.get("meta", None)
        self.meta_coef = float(meta.coef) if meta is not None else 0.0
        self.meta_space = str(meta.get("space", "theta")) if meta is not None else "theta"
        if self.meta_space not in ("theta", "phi"):
            raise ValueError(f"{cfg.get('name')}: meta.space must be theta or phi, got '{self.meta_space}'")

    # ---------------------------------------------------------------- M2
    @property
    def passes(self) -> int:
        """Forward–backward passes per training step in M2 (for compute accounting)."""
        return 1

    def explicit_step(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> ExplicitResult:
        loss = batch_loss(model, x, y)
        loss.backward()
        return ExplicitResult(loss.item())

    def after_step(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Hook run after the optimizer step (L1's proximal mode)."""

    # ---------------------------------------------------------------- M4
    def meta_penalty(self, loss_fn: LossFn, params: Sequence[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> MetaPenalty:
        zero = params[0].new_zeros(())
        return MetaPenalty(zero, 0.0)

    #: meta.space phi: True — perturb φ and replay the unroll (SAM, GAM, GSAM); False — phi_penalty (L1, L2)
    replays = False

    def phi_penalty(self, named_phi: Iterable[Tuple[str, torch.Tensor]]) -> MetaPenalty:
        """L_reg on the LSTM's own weights φ (``meta.space: phi``) — the norm regularizers."""
        raise NotImplementedError(f"meta.space: phi has no penalty form for {self.name}")

    def phi_perturbation(self, grads: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError(f"meta.space: phi perturbs φ only for sam, gam and gsam, not {self.name}")

    def phi_gradient(self, grads: Sequence[torch.Tensor], perturbed: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError(f"meta.space: phi perturbs φ only for sam, gam and gsam, not {self.name}")

    # ---------------------------------------------------------------- helpers
    def phi_weights(self, named_phi: Iterable[Tuple[str, torch.Tensor]]) -> List[torch.Tensor]:
        """The LSTM weights an L1/L2 penalty on φ applies to: the ``exclude`` list (biases) left out, as for θ."""
        exclude = list(self.cfg.get("exclude", ["bias", "norm"]))
        return [p for name, p in named_phi if not any(word in name for word in exclude)]

    def masked(self, params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """The parameters an L1/L2 penalty applies to (all of them if no mask is known)."""
        if self.mask is None:
            return list(params)
        if len(self.mask) != len(params):
            raise ValueError(f"penalty mask covers {len(self.mask)} parameters, got {len(params)}")
        return [p for p, keep in zip(params, self.mask) if keep]


@REGULARIZERS.register("none")
class NoRegularizer(Regularizer):
    """M1 and M3 (configs/regularizers/none.yaml)."""
