"""Weight norms and sparsity (Exp 1.2–1.3, 2.2, 3.2, 4.2).

    l1, l2              ‖θ‖₁ and ‖θ‖₂ (Frobenius) over the penalised weights — the ones an
                        L1/L2 regularizer acts on (biases and normalisation layers excluded)
    l1_all, l2_all      the same over every parameter
    sparsity            share of penalised weights with |w| < threshold (protocol: 1e-3)
    layers              per-tensor ‖·‖₂ and sparsity (layerwise Frobenius norms, Exp 4.2)
    histogram           counts of log10|w| in half-decade bins from 1e-10 to 10; the first bin
                        also holds every smaller weight, exact zeros included
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from reglo.models.common import penalty_mask

__all__ = ["weight_statistics"]

_BINS = np.arange(-10.0, 1.01, 0.5)  # log10|w| from 1e-10 to 10


@torch.no_grad()
def weight_statistics(
    model: nn.Module,
    threshold: float = 1e-3,
    exclude: Sequence[str] = ("bias", "norm"),
    mask: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    mask = mask if mask is not None else penalty_mask(model, exclude)
    named = list(model.named_parameters())
    chosen = [p.detach().flatten() for (_, p), keep in zip(named, mask) if keep]
    everything = [p.detach().flatten() for _, p in named]
    flat = torch.cat(chosen) if chosen else torch.zeros(0)
    flat_all = torch.cat(everything)
    magnitudes = flat.abs().double().cpu().numpy()
    log_magnitudes = np.clip(np.log10(np.maximum(magnitudes, 1e-30)), _BINS[0], _BINS[-1])  # zeros → first bin
    counts, _ = np.histogram(log_magnitudes, bins=_BINS)
    return {
        "num_params": int(flat_all.numel()),
        "num_penalised": int(flat.numel()),
        "l1": float(flat.abs().sum()),
        "l2": float(flat.norm()),
        "l1_all": float(flat_all.abs().sum()),
        "l2_all": float(flat_all.norm()),
        "sparsity": float((flat.abs() < threshold).double().mean()) if flat.numel() else 0.0,
        "sparsity_threshold": float(threshold),
        "layers": [
            {
                "name": name,
                "numel": p.numel(),
                "l2": float(p.detach().norm()),
                "sparsity": float((p.detach().abs() < threshold).double().mean()),
                "penalised": bool(keep),
            }
            for (name, p), keep in zip(named, mask)
        ],
        "histogram": {"log10_edges": _BINS.tolist(), "counts": counts.tolist()},
    }
