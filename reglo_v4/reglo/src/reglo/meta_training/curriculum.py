"""Trajectory-length curriculum (Chen et al. 2020, "Training Stronger Baselines for L2O").

Meta-training starts on short optimizee trajectories and moves to the next, longer length
once meta-validation has failed to improve ``patience`` times in a row. This keeps the LSTM
from over-fitting to one horizon — the paper uses it for the same reason.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

__all__ = ["Curriculum"]


class Curriculum:
    def __init__(self, lengths: Sequence[int], patience: int, enabled: bool, default_length: int) -> None:
        self.enabled = bool(enabled)
        self.lengths = [int(n) for n in lengths] if self.enabled else [int(default_length)]
        if not self.lengths or min(self.lengths) < 1:
            raise ValueError("trajectory lengths must be positive")
        self.patience = max(int(patience), 1)
        self.stage = 0
        self.best = math.inf
        self.bad = 0

    @property
    def length(self) -> int:
        return self.lengths[self.stage]

    @property
    def finished(self) -> bool:
        return self.stage == len(self.lengths) - 1

    def report(self, val_loss: float) -> bool:
        """Record a meta-validation result; returns True when the curriculum moves on."""
        if val_loss < self.best:
            self.best, self.bad = val_loss, 0
            return False
        self.bad += 1
        if self.enabled and self.bad >= self.patience and not self.finished:
            self.stage += 1
            self.best, self.bad = math.inf, 0
            return True
        return False

    def state_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "best": self.best, "bad": self.bad}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.stage = min(int(state["stage"]), len(self.lengths) - 1)
        self.best = float(state["best"])
        self.bad = int(state["bad"])
