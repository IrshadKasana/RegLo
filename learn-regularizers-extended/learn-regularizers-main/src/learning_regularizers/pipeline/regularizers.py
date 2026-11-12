"""Regularizers R in {L1, L2, SAM, GAM, GSAM}.

* L1 / L2 are additive penalties  lambda * R(w)  with
      R_L1(w) = sum |w|,   R_L2(w) = sum w^2
  over weight tensors (ndim > 1; biases and BatchNorm affine parameters are excluded).
* SAM / GAM / GSAM keep their method-specific formulations: the classes from the base
  ``dmo/regularizers`` package are reused unchanged.

``BaselineStepper`` implements one training step of methods 1-2 (B or B + lambda R) on an
``nn.Module`` optimizee, following ``dmo/optimizee_train.fit_w_normal_optimizer``.
"""
from __future__ import annotations

from argparse import Namespace
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from . import DMO_DIR  # noqa: F401  (puts the base DMO sources on sys.path)
from regularizers.optimizers import GAM, GSAM, SAM  # base implementations
from regularizers.schedulers import (CosineScheduler, ProportionScheduler, StepLR, disable_running_stats,
                                     enable_running_stats)

from .flops import FlopMeter

PENALTY_REGULARIZERS = ("l1", "l2")
SHARPNESS_REGULARIZERS = ("sam", "gam", "gsam")


def weight_penalty(tensors: Iterable[torch.Tensor], kind: str) -> torch.Tensor:
    """R(w) without lambda. ``tensors`` may be nn.Parameters or functional parameter tensors."""
    weights = [t for t in tensors if t.dim() > 1]
    if not weights:
        raise ValueError("No weight tensors to regularize")
    if kind == "l1":
        return sum(w.abs().sum() for w in weights)
    if kind == "l2":
        return sum((w * w).sum() for w in weights)
    raise ValueError(f"'{kind}' is not a penalty regularizer")


def build_base_optimizer(params, args: Namespace) -> torch.optim.Optimizer:
    name = str(args.base_optimizer).lower()
    if name == "sgd":
        return optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    raise NotImplementedError(f"base optimizer '{args.base_optimizer}'")


def build_gam_or_gsam(model: torch.nn.Module, args: Namespace, total_steps: int):
    """Same construction as ``OPZEE_get_optim_and_schedulers`` in the base optimizee_train.py."""
    base = build_base_optimizer(model.parameters(), args)
    if args.regularizer == "gam":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(base, T_max=total_steps)
        grad_rho = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                       max_value=args.grad_rho_max, min_value=args.grad_rho_min)
        grad_norm_rho = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                            max_value=args.grad_norm_rho_max, min_value=args.grad_norm_rho_min)
        return GAM(params=model.parameters(), base_optimizer=base, model=model,
                   grad_rho_scheduler=grad_rho, grad_norm_rho_scheduler=grad_norm_rho,
                   adaptive=args.adaptive, args=args)
    if args.regularizer == "gsam":
        scheduler = CosineScheduler(T_max=total_steps, max_value=args.lr, min_value=0.0, optimizer=base)
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.lr, min_lr=0.0,
                                            max_value=args.rho_max, min_value=args.rho_min)
        return GSAM(params=model.parameters(), base_optimizer=base, model=model, gsam_alpha=args.alpha,
                    rho_scheduler=rho_scheduler, adaptive=args.adaptive, args=args)
    raise ValueError(args.regularizer)


class BaselineStepper:
    """One step of B (``regularizer='none'``) or B + lambda R on an nn.Module optimizee."""

    def __init__(self, model: torch.nn.Module, args: Namespace, total_steps: int):
        self.model = model
        self.args = args
        self.reg = getattr(args, "regularizer", "none") or "none"
        self.loss_fn = F.nll_loss
        if self.reg in ("none", "l1", "l2"):
            self.optimizer = build_base_optimizer(model.parameters(), args)
            self.lam = float(getattr(args, "reg_lambda", 0.0)) if self.reg in PENALTY_REGULARIZERS else 0.0
        elif self.reg == "sam":
            args.SAMonOptimizer = True   # on an optimizee SAM must apply its base step
            self.optimizer = SAM(model.parameters(), args=args)
            self.scheduler = StepLR(self.optimizer, args.lr, total_steps)
        elif self.reg in ("gam", "gsam"):
            self.optimizer = build_gam_or_gsam(model, args, total_steps)
        else:
            raise ValueError(f"Unknown regularizer '{self.reg}'")

    def step(self, x: torch.Tensor, y: torch.Tensor, step_idx: int, meter: FlopMeter) -> float:
        model, loss_fn = self.model, self.loss_fn
        if self.reg in ("none", "l1", "l2"):
            with meter.track():
                loss = loss_fn(model(x), y)
                total = loss + self.lam * weight_penalty(model.parameters(), self.reg) if self.lam else loss
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                self.optimizer.step()
            return float(loss.detach())

        if self.reg == "sam":
            with meter.track():
                loss = loss_fn(model(x), y)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                loss_fn(model(x), y).backward()
                self.optimizer.second_step()
            self.scheduler(step_idx)
            return float(loss.detach())

        # GAM / GSAM: the closure performs every forward/backward the method needs
        self.optimizer.set_closure(loss_fn, x, y)
        self.optimizer.update_rho_t()
        with meter.track():
            self.optimizer.step()
        # logging only: not counted as training compute, and BN running stats are left untouched
        disable_running_stats(model)
        try:
            with torch.no_grad():
                return float(loss_fn(model(x), y))
        finally:
            enable_running_stats(model)
