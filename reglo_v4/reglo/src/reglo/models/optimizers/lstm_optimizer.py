"""The learned optimizer F_φ of M3 and M4: a coordinate-wise LSTM (Andrychowicz et al. 2016).

Every parameter coordinate of the optimizee is one row of a batch: the same LSTM weights φ
process all coordinates, and each coordinate keeps its own hidden state (h, c) per layer.

    raw_t, (h, c) ← LSTM_φ(s_t, (h, c));      u_t = output_scale · head_φ(h_t)
    θ_{t+1} = θ_t + u_t                                        (Algorithm 1, lines 12–14)

The cell is written with plain linear layers so that FLOP counters see every operation and
double backpropagation works everywhere. State memory: layers × 2 × hidden_size floats per
optimizee parameter (2 layers × 20 units → 320 bytes per parameter).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from reglo.models.optimizers.features import GradientFeatures

__all__ = [
    "LSTMOptimizer",
    "build_lstm_optimizer",
    "load_learned_optimizer",
    "save_learned_optimizer",
]

_FORMAT = "reglo-lstm-optimizer/1"


class LSTMCell(nn.Module):
    """One LSTM layer; gates i, f, g, o from W_ih·x + W_hh·h (+ b)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        with torch.no_grad():  # forget-gate bias 1: remember by default (standard LSTM practice)
            self.ih.bias.zero_()
            self.ih.bias[hidden_size : 2 * hidden_size].fill_(1.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        i, f, g, o = (self.ih(x) + self.hh(h)).chunk(4, dim=-1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class LSTMOptimizer(nn.Module):
    """F_φ: per-coordinate inputs → per-coordinate parameter update."""

    def __init__(
        self,
        num_inputs: int = 1,
        num_layers: int = 2,
        hidden_size: int = 20,
        head_layers: int = 2,
        head_hidden: int = 20,
        output_scale: float = 1e-3,
        head_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1 or head_layers < 1:
            raise ValueError("num_layers and head_layers must be ≥ 1")
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_scale = float(output_scale)
        self._kwargs = dict(
            num_inputs=num_inputs,
            num_layers=num_layers,
            hidden_size=hidden_size,
            head_layers=head_layers,
            head_hidden=head_hidden,
            output_scale=float(output_scale),
            head_init_scale=float(head_init_scale),
        )
        self.cells = nn.ModuleList(
            LSTMCell(num_inputs if layer == 0 else hidden_size, hidden_size) for layer in range(num_layers)
        )
        head, width = [], hidden_size
        for _ in range(head_layers - 1):
            head += [nn.Linear(width, head_hidden), nn.ReLU()]
            width = head_hidden
        last = nn.Linear(width, 1)
        with torch.no_grad():
            last.weight.mul_(head_init_scale)
            last.bias.zero_()
        self.head = nn.Sequential(*head, last)

    def config(self) -> Dict[str, Any]:
        return dict(self._kwargs)

    def init_state(self, num_coords: int, device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero (h, c), each of shape (num_layers, num_coords, hidden_size)."""
        shape = (self.num_layers, num_coords, self.hidden_size)
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)

    def forward(
        self, inputs: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (raw output (P,), new h, new c). The update is ``output_scale · raw``."""
        x, hs, cs = inputs, [], []
        for layer, cell in enumerate(self.cells):
            h_l, c_l = cell(x, h[layer], c[layer])
            hs.append(h_l)
            cs.append(c_l)
            x = h_l
        return self.head(x).squeeze(-1), torch.stack(hs), torch.stack(cs)

    @torch.no_grad()
    def step_inplace(self, inputs: torch.Tensor, h: torch.Tensor, c: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Inference without autograd: process ``chunk_size`` coordinates at a time, writing
        the new state into ``h`` and ``c`` in place. Returns the update u_t (P,)."""
        total = inputs.shape[0]
        update = torch.empty(total, device=inputs.device, dtype=inputs.dtype)
        for start in range(0, total, max(int(chunk_size), 1)):
            stop = min(total, start + int(chunk_size))
            raw, h_new, c_new = self.forward(inputs[start:stop], h[:, start:stop], c[:, start:stop])
            h[:, start:stop] = h_new
            c[:, start:stop] = c_new
            update[start:stop] = raw
        return update.mul_(self.output_scale)


def build_lstm_optimizer(cfg: DictConfig) -> Tuple[LSTMOptimizer, GradientFeatures]:
    """F_φ and its input features from ``configs/optimizers/lstm.yaml`` (fresh, untrained weights)."""
    features = GradientFeatures(list(cfg.features), str(cfg.normalization), float(cfg.momentum_beta))
    lstm = LSTMOptimizer(
        num_inputs=features.num_inputs,
        num_layers=int(cfg.num_layers),
        hidden_size=int(cfg.hidden_size),
        head_layers=int(cfg.head_layers),
        head_hidden=int(cfg.head_hidden),
        output_scale=float(cfg.output_scale),
        head_init_scale=float(cfg.head_init_scale),
    )
    return lstm, features


def save_learned_optimizer(
    path: Union[str, Path], lstm: LSTMOptimizer, features: GradientFeatures, **meta: Any
) -> Path:
    """Write φ with everything needed to rebuild it (architecture, input features, provenance)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format": _FORMAT,
        "lstm": lstm.config(),
        "features": features.config(),
        "state_dict": {k: v.detach().cpu() for k, v in lstm.state_dict().items()},
        "meta": {k: (OmegaConf.to_container(v) if isinstance(v, DictConfig) else v) for k, v in meta.items()},
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)
    return path


def load_learned_optimizer(
    path: Union[str, Path], device: Optional[Union[str, torch.device]] = None
) -> Tuple[LSTMOptimizer, GradientFeatures, Dict[str, Any]]:
    """Rebuild a saved φ exactly as it was meta-trained."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No meta-trained LSTM optimizer at {path}. Run Stage 1 first "
            f"(scripts/meta_train.py), or point method.checkpoint=<path> at another one."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != _FORMAT:
        raise ValueError(f"{path} is not a {_FORMAT} checkpoint")
    lstm = LSTMOptimizer(**blob["lstm"])
    lstm.load_state_dict(blob["state_dict"])
    features = GradientFeatures(**blob["features"])
    if device is not None:
        lstm.to(device)
    return lstm, features, blob.get("meta", {})
