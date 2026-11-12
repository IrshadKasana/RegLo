"""Training an optimizee with a fixed optimizer.

Three kinds of optimizer can drive the same loop:
    baseline    B or B + lambda R            (methods 1-2)          -> regularizers.BaselineStepper
    lo          a frozen learned optimizer   (methods 3-6, phase 2) -> learned_optimizer.FrozenLOStepper

The loop follows the base ``optimizee_train.py`` protocol (``optimizee_epochs`` minibatch steps,
batch size and seeds from ``optimizee_eval.yaml``), records a learning curve, and returns final
train/test metrics plus the FLOPs spent on training.
"""
from __future__ import annotations

import math
import time
from argparse import Namespace
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import InfiniteLoader
from .evaluation import evaluate, summarize
from .flops import FlopMeter
from .learned_optimizer import FrozenLOStepper, LSTMLearnedOptimizer
from .regularizers import BaselineStepper

StepFn = Callable[[torch.Tensor, torch.Tensor, int], float]


def make_baseline_step_fn(model: nn.Module, args: Namespace, total_steps: int, meter: FlopMeter) -> StepFn:
    stepper = BaselineStepper(model, args, total_steps)
    return lambda x, y, t: stepper.step(x, y, t, meter)


def make_lo_step_fn(model: nn.Module, lo: LSTMLearnedOptimizer, meter: FlopMeter, chunk_size: int) -> StepFn:
    """Frozen LO: no regularizer, no meta-gradients; the LSTM state persists across steps."""
    for p in lo.parameters():
        p.requires_grad_(False)
    stepper = FrozenLOStepper(lo, list(model.parameters()), chunk_size=chunk_size)

    def step(x, y, t):
        with meter.track():
            loss = F.nll_loss(model(x), y)
            stepper.zero_grad()
            loss.backward()
            stepper.step()
        return float(loss.detach())

    return step


def train_optimizee(model: nn.Module, step_fn: StepFn, train_loader: DataLoader, test_loader: DataLoader,
                    train_eval_loader: DataLoader, steps: int, eval_every: int, device, meter: FlopMeter,
                    gap_kind: str = "accuracy", log_every: int = 0, log_prefix: str = "") -> Dict:
    model.to(device).train()
    stream = InfiniteLoader(train_loader)
    curve = {"step": [], "train_loss": [], "eval_step": [], "test_loss": [], "test_acc": []}
    flops_before = meter.total
    diverged = False
    t0 = time.time()
    for t in range(int(steps)):
        x, y = stream.next()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = step_fn(x, y, t)
        curve["step"].append(t)
        curve["train_loss"].append(loss)
        if not math.isfinite(loss):
            diverged = True
            print(f"{log_prefix} step {t}: non-finite loss, stopping this run (marked diverged)", flush=True)
            break
        if log_every and (t % log_every == 0):
            print(f"{log_prefix} step {t}/{steps} train_loss {loss:.4f}", flush=True)
        if eval_every and ((t + 1) % eval_every == 0) and (t + 1) < steps:
            te_loss, te_acc = evaluate(model, test_loader, device)
            curve["eval_step"].append(t + 1)
            curve["test_loss"].append(te_loss)
            curve["test_acc"].append(te_acc)
    train_flops = meter.total - flops_before
    wall = time.time() - t0

    tr_loss, tr_acc = evaluate(model, train_eval_loader, device)
    te_loss, te_acc = evaluate(model, test_loader, device)
    curve["eval_step"].append(len(curve["step"]))
    curve["test_loss"].append(te_loss)
    curve["test_acc"].append(te_acc)
    final = summarize(tr_loss, tr_acc, te_loss, te_acc, gap_kind)
    final.update({"train_flops": int(train_flops), "steps_done": len(curve["step"]), "diverged": diverged,
                  "wall_time_s": wall})
    return {"final": final, "curve": curve}
