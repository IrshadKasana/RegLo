"""Hessian spectrum probes (Exp 1.4, 2.3, 3.3, 4.1).

    λ_max(H)   power iteration on Hessian-vector products (protocol: 30 iterations)
    Tr(H)      Hutchinson's estimator  E[vᵀHv], v Rademacher (protocol: 50 probes)

Both work on a fixed set of loss terms — one closure per mini-batch, weighted by its share
of the examples — so memory stays at one batch's double backward however many batches are
averaged. Models are evaluated in eval mode (BatchNorm uses its running statistics), which
makes the Hessian a deterministic function of the weights.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LossTerm",
    "gradient",
    "gradient_with_graph",
    "hessian_trace",
    "hvp",
    "loss_terms",
    "top_eigenvalue",
    "total_loss",
]

LossTerm = Tuple[Callable[[], torch.Tensor], float]


def loss_terms(model: nn.Module, batches: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> List[LossTerm]:
    """Cross-entropy on each batch, weighted so the weighted sum is the mean over all examples."""
    total = sum(len(y) for _, y in batches)
    return [((lambda x=x, y=y: F.cross_entropy(model(x), y)), len(y) / total) for x, y in batches]


def total_loss(terms: Sequence[LossTerm]) -> float:
    with torch.no_grad():
        return sum(w * f().item() for f, w in terms)


def gradient(terms: Sequence[LossTerm], params: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
    """(∇L, L) of the weighted loss; no graph is kept."""
    grads = [torch.zeros_like(p) for p in params]
    value = 0.0
    for f, w in terms:
        loss = f()
        value += w * loss.item()
        for acc, g in zip(grads, torch.autograd.grad(loss, params)):
            acc.add_(g, alpha=w)
    return grads, value


def gradient_with_graph(terms: Sequence[LossTerm], params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """∇L with its graph kept (for gradients of the gradient norm)."""
    total = None
    for f, w in terms:
        grads = torch.autograd.grad(f(), params, create_graph=True)
        scaled = [w * g for g in grads]
        total = scaled if total is None else [a + b for a, b in zip(total, scaled)]
    return total


def hvp(terms: Sequence[LossTerm], params: Sequence[torch.Tensor], vector: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """H·v of the weighted loss, one batch at a time."""
    result = [torch.zeros_like(p) for p in params]
    for f, w in terms:
        grads = torch.autograd.grad(f(), params, create_graph=True)
        dot = sum((g * v).sum() for g, v in zip(grads, vector))
        for acc, h in zip(result, torch.autograd.grad(dot, params)):
            acc.add_(h, alpha=w)
    return result


def _dot(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> float:
    return float(sum((x * y).sum() for x, y in zip(a, b)))


def _normalised(vector: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    norm = math.sqrt(_dot(vector, vector)) + 1e-30
    return [v / norm for v in vector]


def top_eigenvalue(
    terms: Sequence[LossTerm], params: Sequence[torch.Tensor], iterations: int = 30, tol: float = 1e-3, seed: int = 0
) -> Dict[str, float]:
    """Largest-magnitude eigenvalue of H by power iteration (Rayleigh quotient of the last iterate)."""
    gen = torch.Generator(device=params[0].device).manual_seed(seed)
    v = _normalised([torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) for p in params])
    eigenvalue, used = 0.0, 0
    for used in range(1, int(iterations) + 1):
        hv = hvp(terms, params, v)
        new = _dot(hv, v)
        v = _normalised(hv)
        converged = used > 1 and abs(new - eigenvalue) <= tol * (abs(eigenvalue) + 1e-12)
        eigenvalue = new
        if converged:
            break
    return {"lambda_max": eigenvalue, "power_iterations": used}


def hessian_trace(terms: Sequence[LossTerm], params: Sequence[torch.Tensor], samples: int = 50, seed: int = 0) -> Dict[str, float]:
    """Hutchinson: mean of vᵀHv over Rademacher probes v, with its standard error."""
    gen = torch.Generator(device=params[0].device).manual_seed(seed + 1)
    estimates = []
    for _ in range(int(samples)):
        v = [torch.randint(0, 2, p.shape, generator=gen, device=p.device).to(p.dtype) * 2 - 1 for p in params]
        estimates.append(_dot(hvp(terms, params, v), v))
    values = torch.tensor(estimates, dtype=torch.float64)
    stderr = float(values.std(unbiased=True) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"trace": float(values.mean()), "trace_stderr": stderr, "hutchinson_samples": len(values)}
