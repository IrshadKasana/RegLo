"""A meta-trained LSTM optimizer used like any ``torch.optim`` optimizer (M3 and M4).

φ is frozen: ``step()`` reads each parameter's ``.grad``, builds the LSTM inputs, runs F_φ
without autograd (in chunks, so ResNet-18 / WRN-28 fit) and adds the update to the weights.
No regularizer is computed here — for M4, whatever regularization φ*_r learned in Stage 1
acts only through these updates.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from reglo.models.optimizers.features import GradientFeatures
from reglo.models.optimizers.lstm_optimizer import LSTMOptimizer

__all__ = ["FrozenLSTMOptimizer"]


class FrozenLSTMOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lstm: LSTMOptimizer,
        features: GradientFeatures,
        chunk_size: int = 1 << 20,
    ) -> None:
        super().__init__(params, defaults={})
        self.lstm = lstm
        self.lstm.eval()
        self.lstm.requires_grad_(False)
        self.features = features
        self.chunk_size = int(chunk_size)
        self.h: Optional[torch.Tensor] = None
        self.c: Optional[torch.Tensor] = None
        self.feature_state: Dict[str, List[torch.Tensor]] = {}
        self.steps = 0

    def _params(self) -> List[torch.Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    def _ensure_state(self, params: List[torch.Tensor]) -> None:
        if self.h is None:
            total = sum(p.numel() for p in params)
            device = params[0].device
            self.lstm.to(device)
            self.h, self.c = self.lstm.init_state(total, device=device)
            self.feature_state = self.features.init_state(params)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = self._params()
        self._ensure_state(params)
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        inputs, self.feature_state = self.features(grads, params, self.feature_state)
        update = self.lstm.step_inplace(inputs, self.h, self.c, self.chunk_size)
        offset = 0
        for p in params:
            n = p.numel()
            p.add_(update[offset : offset + n].view_as(p))
            offset += n
        self.steps += 1
        return loss

    # The LSTM's per-coordinate state replaces torch.optim's per-parameter state.
    def state_dict(self) -> Dict[str, Any]:
        return {
            "h": self.h,
            "c": self.c,
            "feature_state": self.feature_state,
            "steps": self.steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        device = self._params()[0].device
        self.lstm.to(device)
        self.h = None if state["h"] is None else state["h"].to(device)
        self.c = None if state["c"] is None else state["c"].to(device)
        self.feature_state = {k: [t.to(device) for t in v] for k, v in state["feature_state"].items()}
        self.steps = int(state["steps"])
