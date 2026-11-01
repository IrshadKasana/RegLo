"""The paper's "point of convergence" (PoC).

Learning Regularizers measures each property twice: *at convergence*, "defined by a stopping
criterion where the optimizee's training loss does not consistently decrease for 100
consecutive steps", and *at completion* (after the fixed budget). This tracker watches an
exponential moving average of the training loss and fires once, the first time it has gone
``patience`` steps without a new minimum; the trainer then saves ``model_converged.pt``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

__all__ = ["ConvergenceTracker"]


class ConvergenceTracker:
    def __init__(self, patience: int = 100, smoothing: float = 0.9) -> None:
        self.patience = int(patience)
        self.smoothing = float(smoothing)
        self.ema: Optional[float] = None
        self.best = math.inf
        self.since_best = 0
        self.converged_step: Optional[int] = None

    def update(self, step: int, loss: float) -> bool:
        """Feed one training loss; True exactly once, at the step convergence is declared."""
        if self.converged_step is not None or self.patience <= 0:
            return False
        self.ema = loss if self.ema is None else self.smoothing * self.ema + (1.0 - self.smoothing) * loss
        if self.ema < self.best:
            self.best, self.since_best = self.ema, 0
            return False
        self.since_best += 1
        if self.since_best >= self.patience:
            self.converged_step = step
            return True
        return False

    def state_dict(self) -> Dict[str, Any]:
        return {"ema": self.ema, "best": self.best, "since_best": self.since_best, "converged_step": self.converged_step}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ema, self.best = state["ema"], state["best"]
        self.since_best, self.converged_step = state["since_best"], state["converged_step"]
