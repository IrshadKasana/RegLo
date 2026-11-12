"""FLOP accounting.

``FlopMeter.track()`` wraps a region in ``torch.utils.flop_counter.FlopCounterMode`` and adds
the counted FLOPs (matmul / conv / attention, forward *and* backward) to a running total.
Element-wise ops (activations, BatchNorm, optimizer arithmetic) are not counted, which is the
usual convention. The learned optimizer computes its LSTM with explicit matmuls
(see ``learned_optimizer.py``) because ``nn.LSTM`` / cuDNN kernels are invisible to the counter,
so LO overhead is counted identically on CPU and GPU.
"""
from __future__ import annotations

from contextlib import contextmanager

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:  # torch < 2.1
    FlopCounterMode = None


class FlopMeter:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled) and FlopCounterMode is not None
        self.total = 0
        self._depth = 0

    @contextmanager
    def track(self):
        if not self.enabled or self._depth > 0:
            # nested regions are already being counted by the outer context
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self._depth += 1
        counter = FlopCounterMode(display=False)
        try:
            with counter:
                yield
        finally:
            self._depth -= 1
            self.total += int(counter.get_total_flops())

    def add(self, flops: int) -> None:
        if self.enabled:
            self.total += int(flops)

    def reset(self) -> int:
        total, self.total = self.total, 0
        return total


class NullMeter(FlopMeter):
    def __init__(self):
        super().__init__(enabled=False)
