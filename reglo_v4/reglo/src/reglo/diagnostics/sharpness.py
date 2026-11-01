"""The paper's point-of-convergence (PoC) property tests (Learning Regularizers, Figs. 4–6).

Around trained weights θ*, projected gradient ascent searches an ℓ∞ ball of radius ε:

    SAM property    max_{‖δ‖∞≤ε} L(θ* + δ)            "maximum loss in a neighbourhood"
    GSAM property   that maximum − L(θ*)              "surrogate gap"
    GAM property    max_{‖δ‖∞≤ε} ‖∇L(θ* + δ)‖₂        "maximum gradient norm in a neighbourhood"

As in the paper: a Gaussian start, then ``steps`` sign-ascent steps of size ε·step_ratio
(ε/10), each clipped back into the ball; radii 0.001 … 0.1; measured at convergence and at
completion. ``relative=True`` scales the ball per weight by |θ*| — useful on large networks,
where an absolute ε per weight is a very large perturbation in total.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterator, List, Sequence

import torch

from reglo.diagnostics.hessian import LossTerm, gradient, gradient_with_graph, total_loss

__all__ = ["max_grad_norm_in_ball", "max_loss_in_ball"]


@contextmanager
def _restored(params: Sequence[torch.Tensor]) -> Iterator[List[torch.Tensor]]:
    originals = [p.detach().clone() for p in params]
    try:
        yield originals
    finally:
        with torch.no_grad():
            for p, o in zip(params, originals):
                p.copy_(o)


def _set(params: Sequence[torch.Tensor], originals: Sequence[torch.Tensor], delta: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, o, d in zip(params, originals, delta):
            p.copy_(o + d)


def _ball(originals: Sequence[torch.Tensor], radius: float, relative: bool) -> List[torch.Tensor]:
    return [radius * (o.abs() if relative else torch.ones_like(o)) for o in originals]


def _start(bounds: Sequence[torch.Tensor], init_std_ratio: float, seed: int) -> List[torch.Tensor]:
    gen = torch.Generator(device=bounds[0].device).manual_seed(seed)
    noise = [torch.randn(b.shape, generator=gen, device=b.device, dtype=b.dtype) * init_std_ratio * b for b in bounds]
    return [torch.max(torch.min(n, b), -b) for n, b in zip(noise, bounds)]


def _ascend(delta, direction, bounds, step_ratio):
    return [torch.max(torch.min(d + step_ratio * b * torch.sign(g), b), -b) for d, g, b in zip(delta, direction, bounds)]


def max_loss_in_ball(
    terms: Sequence[LossTerm],
    params: Sequence[torch.Tensor],
    radius: float,
    steps: int = 10,
    step_ratio: float = 0.1,
    init_std_ratio: float = 0.5,
    relative: bool = False,
    seed: int = 0,
) -> Dict[str, float]:
    """Largest loss found within the ℓ∞ ball (SAM property) and the surrogate gap (GSAM property)."""
    with _restored(params) as originals:
        base = total_loss(terms)
        bounds = _ball(originals, radius, relative)
        delta = _start(bounds, init_std_ratio, seed)
        best = base
        for _ in range(int(steps)):
            _set(params, originals, delta)
            grads, value = gradient(terms, params)
            best = max(best, value)
            delta = _ascend(delta, grads, bounds, step_ratio)
        _set(params, originals, delta)
        best = max(best, total_loss(terms))
    if not math.isfinite(best):
        best = math.inf
    return {"loss": base, "max_loss": best, "surrogate_gap": best - base}


def max_grad_norm_in_ball(
    terms: Sequence[LossTerm],
    params: Sequence[torch.Tensor],
    radius: float,
    steps: int = 10,
    step_ratio: float = 0.1,
    init_std_ratio: float = 0.5,
    relative: bool = False,
    seed: int = 0,
) -> Dict[str, float]:
    """Largest gradient norm found within the ℓ∞ ball (GAM property)."""

    def norm_with_graph():
        grads = gradient_with_graph(terms, params)
        return torch.sqrt(sum(g.pow(2).sum() for g in grads))

    with _restored(params) as originals:
        base = float(norm_with_graph().detach())
        bounds = _ball(originals, radius, relative)
        delta = _start(bounds, init_std_ratio, seed)
        best = base
        for _ in range(int(steps)):
            _set(params, originals, delta)
            norm = norm_with_graph()
            best = max(best, norm.item())
            direction = torch.autograd.grad(norm, params)
            delta = _ascend(delta, direction, bounds, step_ratio)
        _set(params, originals, delta)
        best = max(best, float(norm_with_graph().detach()))
    return {"grad_norm": base, "max_grad_norm": best}
