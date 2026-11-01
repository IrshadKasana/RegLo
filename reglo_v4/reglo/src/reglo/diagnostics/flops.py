"""FLOPs and wall-clock time per training step (Stage 4).

FLOPs are counted with ``torch.utils.flop_counter.FlopCounterMode``, which sees the forward
*and* the backward pass and every extra pass a method makes (SAM's perturbed pass, GAM's
finite differences) — and also the LSTM optimizer's own step, which a forward-only counter
such as fvcore misses. It counts matrix multiplications and convolutions (the bulk of the
work); element-wise operations such as AdamW's update are not included.
"""

from __future__ import annotations

import time
from typing import Callable, Dict

import torch

from reglo.models.optimizers.lstm_optimizer import LSTMOptimizer

__all__ = ["count_flops", "lstm_flops_per_coordinate", "synchronize", "time_steps"]


def count_flops(fn: Callable[[], object]) -> int:
    """FLOPs (multiply and add counted separately) executed by ``fn()``."""
    from torch.utils.flop_counter import FlopCounterMode  # PyTorch ≥ 2.1

    with FlopCounterMode(display=False) as counter:
        fn()
    return int(counter.get_total_flops())


def lstm_flops_per_coordinate(lstm: LSTMOptimizer) -> int:
    """Analytic cost of one LSTM-optimizer step for one optimizee parameter (matrix products only)."""
    cfg = lstm.config()
    hidden, inputs = cfg["hidden_size"], cfg["num_inputs"]
    macs = 0
    for layer in range(cfg["num_layers"]):
        width = inputs if layer == 0 else hidden
        macs += 4 * hidden * (width + hidden)
    width = hidden
    for _ in range(cfg["head_layers"] - 1):
        macs += width * cfg["head_hidden"]
        width = cfg["head_hidden"]
    macs += width
    return 2 * macs


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_steps(step: Callable[[], object], device: torch.device, warmup: int = 10, timed: int = 50) -> Dict[str, float]:
    """Mean wall-clock seconds of ``step()`` after ``warmup`` untimed calls (GPU-synchronised)."""
    for _ in range(warmup):
        step()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(timed):
        step()
    synchronize(device)
    total = time.perf_counter() - start
    return {"seconds_per_step": total / max(timed, 1), "timed_steps": timed, "warmup_steps": warmup}
