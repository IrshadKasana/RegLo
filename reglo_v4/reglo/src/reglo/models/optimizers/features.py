"""What the LSTM optimizer sees for each parameter coordinate: its input s_t.

Algorithm 1 feeds the gradient g_t. Before reaching the LSTM, each feature is normalised so
the same small network works for every layer and every architecture:

    rms         each tensor divided by its own RMS (default; scale-free across layers)
    global_rms  every tensor divided by one RMS over all parameters (Xiong & Hsieh's code)
    log_sign    (log|x|/p, sign x) for |x| ≥ e^-p, else (−1, e^p·x), p = 10 (Andrychowicz et al.)
    none        raw values

Optional extra features: ``momentum`` (an exponential moving average of g) and ``param``
(θ_t itself, as in the paper's Eq. 2).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

__all__ = ["FEATURES", "NORMALIZATIONS", "GradientFeatures"]

FEATURES = ("grad", "momentum", "param")
NORMALIZATIONS = ("rms", "global_rms", "log_sign", "none")
_P = 10.0  # Andrychowicz et al.'s preprocessing constant

State = Dict[str, List[torch.Tensor]]


class GradientFeatures:
    """Builds the (num_coordinates, num_inputs) input matrix of the LSTM optimizer."""

    def __init__(
        self,
        features: Sequence[str] = ("grad",),
        normalization: str = "rms",
        momentum_beta: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        unknown = set(features) - set(FEATURES)
        if unknown or not features:
            raise ValueError(f"features must be a non-empty subset of {FEATURES}, got {list(features)}")
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got '{normalization}'")
        self.features = tuple(features)
        self.normalization = normalization
        self.momentum_beta = float(momentum_beta)
        self.eps = float(eps)

    @property
    def num_inputs(self) -> int:
        return len(self.features) * (2 if self.normalization == "log_sign" else 1)

    def config(self) -> Dict[str, object]:
        return {
            "features": list(self.features),
            "normalization": self.normalization,
            "momentum_beta": self.momentum_beta,
            "eps": self.eps,
        }

    def init_state(self, params: Sequence[torch.Tensor]) -> State:
        if "momentum" in self.features:
            return {"momentum": [torch.zeros_like(p) for p in params]}
        return {}

    def __call__(
        self, grads: Sequence[torch.Tensor], params: Sequence[torch.Tensor], state: State
    ) -> Tuple[torch.Tensor, State]:
        """Inputs for this step and the updated feature state (the inputs are not detached here)."""
        columns: List[torch.Tensor] = []
        new_state: State = dict(state)
        for name in self.features:
            if name == "grad":
                tensors = list(grads)
            elif name == "momentum":
                beta = self.momentum_beta
                tensors = [beta * m + (1.0 - beta) * g for m, g in zip(state["momentum"], grads)]
                new_state["momentum"] = tensors
            else:  # "param"
                tensors = list(params)
            columns.extend(self._encode(tensors))
        return torch.stack(columns, dim=1), new_state

    def _encode(self, tensors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if self.normalization == "rms":
            return [torch.cat([t.reshape(-1) / (t.pow(2).mean().sqrt() + self.eps) for t in tensors])]
        flat = torch.cat([t.reshape(-1) for t in tensors])
        if self.normalization == "none":
            return [flat]
        if self.normalization == "global_rms":
            return [flat / torch.sqrt(flat.pow(2).mean() + 1e-5)]
        magnitude = flat.abs()
        big = magnitude >= math.exp(-_P)
        log_part = torch.where(big, torch.log(magnitude.clamp_min(1e-30)) / _P, torch.full_like(flat, -1.0))
        sign_part = torch.where(big, torch.sign(flat), math.exp(_P) * flat)
        return [log_part, sign_part]
