"""Run a network with its parameters passed in explicitly.

Meta-training (Algorithm 1) needs θ_t to be ordinary tensors that depend on the LSTM's
weights φ — θ_{t+1} = θ_t + F_φ(g_t, h_t) — so the meta-loss can be backpropagated into φ.
``FunctionalModel`` evaluates the optimizee at any such list of tensors, using
``torch.func.functional_call``; the module's own parameters are only a template.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

__all__ = ["FunctionalModel"]


class FunctionalModel:
    """An optimizee whose parameters are supplied on every call.

    The parameter order is that of ``module.named_parameters()``. Buffers (BatchNorm running
    statistics) stay the module's own and are updated in place in training mode.
    """

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.names: List[str] = [name for name, _ in module.named_parameters()]

    def initial_params(self) -> List[torch.Tensor]:
        """Detached copies of the module's current parameters, ready to be differentiated."""
        return [p.detach().clone().requires_grad_(True) for p in self.module.parameters()]

    def logits(self, params: Sequence[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        tensors = dict(zip(self.names, params))
        tensors.update(dict(self.module.named_buffers()))
        return functional_call(self.module, tensors, (x,))

    def loss(self, params: Sequence[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.logits(params, x), y)

    def train(self, mode: bool = True) -> "FunctionalModel":
        self.module.train(mode)
        return self

    def eval(self) -> "FunctionalModel":
        return self.train(False)

    def load(self, params: Sequence[torch.Tensor]) -> None:
        """Copy ``params`` into the module (e.g. to evaluate or save a meta-trained optimizee)."""
        with torch.no_grad():
            for target, value in zip(self.module.parameters(), params):
                target.copy_(value)
