"""The differentiable inner loop of Algorithm 1 (lines 5–15).

For t = 0 … T−1, on mini-batch t:

    L_t   = L(θ_t)                         line 9
    g_t   = ∇_θ L_t                        line 10   (detached when first_order)
    s_t   = features(g_t)                  the LSTM's input
    u_t, (h, c) ← F_φ(s_t, (h, c))         lines 12, 14
    θ_{t+1} = θ_t + u_t                    line 13

θ_{t+1} is an ordinary tensor that depends on φ, so the losses L(θ_1) … L(θ_T) — evaluated
on mini-batches 1 … T — carry gradients back into φ. With ``first_order`` (the standard
L2O approximation of Andrychowicz et al.) the LSTM's inputs are detached: φ is reached only
through the updates u_t, never through ∂g_t/∂φ.

Every unroll starts from a *detached* ``InnerState`` and returns its end state; the meta-
trainer detaches it again before the next truncation, so no graph ever spans two
truncations (truncated BPTT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from reglo.models.functional import FunctionalModel
from reglo.models.optimizers.features import GradientFeatures
from reglo.models.optimizers.lstm_optimizer import LSTMOptimizer
from reglo.regularizers.smooth import SmoothnessPenalty

__all__ = ["InnerState", "UnrollOutput", "unroll"]

Batch = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class InnerState:
    """Where an optimizee trajectory stands: θ_t, the LSTM state and the feature state."""

    params: List[torch.Tensor]
    h: torch.Tensor
    c: torch.Tensor
    features: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    step: int = 0  # inner steps taken since this optimizee was initialised

    def detached(self) -> "InnerState":
        """Cut every graph edge: the start of the next truncation (Algorithm 1, TBPTT)."""
        return InnerState(
            params=[p.detach().requires_grad_(True) for p in self.params],
            h=self.h.detach(),
            c=self.c.detach(),
            features={k: [t.detach() for t in v] for k, v in self.features.items()},
            step=self.step,
        )

    @property
    def is_detached(self) -> bool:
        tensors = list(self.params) + [self.h, self.c] + [t for v in self.features.values() for t in v]
        return all(t.grad_fn is None for t in tensors)


@dataclass
class UnrollOutput:
    losses: List[torch.Tensor]  # L(θ_1) … L(θ_T): the terms of L_meta (with graph to φ)
    smooth_terms: List[torch.Tensor]  # L_smooth at the steps where it was computed
    final: InnerState  # θ_T and the LSTM state after T steps (with graph)
    batches: List[Batch]  # the T + 1 mini-batches used, so the unroll can be replayed
    train_losses: List[float]  # L(θ_0) … L(θ_T) as numbers

    @property
    def finite(self) -> bool:
        return all(math.isfinite(v) for v in self.train_losses)


def _split(update: torch.Tensor, like: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    out, offset = [], 0
    for p in like:
        n = p.numel()
        out.append(update[offset : offset + n].view_as(p))
        offset += n
    return out


def unroll(
    fmodel: FunctionalModel,
    lstm: LSTMOptimizer,
    features: GradientFeatures,
    start: InnerState,
    batches: Iterator[Batch],
    steps: int,
    *,
    first_order: bool = True,
    smooth: Optional[SmoothnessPenalty] = None,
    checkpoint_lstm: bool = False,
    generator: Optional[torch.Generator] = None,
) -> UnrollOutput:
    """Run ``steps`` inner steps of the optimizee with the LSTM optimizer, keeping the graph to φ."""
    params, h, c, feature_state = list(start.params), start.h, start.c, start.features
    used: List[Batch] = []

    x, y = next(batches)
    used.append((x, y))
    loss = fmodel.loss(params, x, y)  # L(θ_0): not a meta-loss term (θ_0 does not depend on φ)
    train_losses = [loss.item()]
    losses: List[torch.Tensor] = []
    smooth_terms: List[torch.Tensor] = []

    for t in range(steps):
        grads = torch.autograd.grad(loss, params, retain_graph=t > 0 or not first_order, create_graph=not first_order)
        if first_order:
            inputs, feature_state = features([g.detach() for g in grads], [p.detach() for p in params], feature_state)
            inputs = inputs.detach()
        else:
            inputs, feature_state = features(list(grads), params, feature_state)
        if smooth is not None and smooth.applies_at(start.step + t):
            smooth_terms.append(smooth(lstm, inputs, h, c, generator))
        if checkpoint_lstm:
            raw, h, c = checkpoint(lstm, inputs, h, c, use_reentrant=False)
        else:
            raw, h, c = lstm(inputs, h, c)
        params = [p + u for p, u in zip(params, _split(raw * lstm.output_scale, params))]

        x, y = next(batches)
        used.append((x, y))
        loss = fmodel.loss(params, x, y)  # L(θ_{t+1})
        losses.append(loss)
        train_losses.append(loss.item())

    final = InnerState(params, h, c, feature_state, start.step + steps)
    return UnrollOutput(losses, smooth_terms, final, used, train_losses)
