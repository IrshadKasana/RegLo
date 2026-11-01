"""Evaluation: loss and top-k accuracy over a whole batch stream, in eval mode."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from reglo.data.loaders import BatchStream

__all__ = ["evaluate"]


@torch.no_grad()
def evaluate(model: nn.Module, stream: BatchStream, topk: Sequence[int] = (1, 5)) -> Dict[str, float]:
    """Mean cross-entropy and top-k accuracy (fractions in [0, 1]) over every example of ``stream``."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total = torch.zeros((), dtype=torch.float64, device=device)
    hits = {k: torch.zeros((), dtype=torch.long, device=device) for k in topk}
    count = 0
    for x, y in stream:
        logits = model(x)
        total += F.cross_entropy(logits, y, reduction="sum").double()
        ranked = logits.topk(min(max(topk), logits.shape[1]), dim=1).indices
        for k in topk:
            hits[k] += (ranked[:, : min(k, logits.shape[1])] == y[:, None]).any(dim=1).sum()
        count += y.numel()
    model.train(was_training)
    if count == 0:
        raise ValueError("cannot evaluate on an empty stream")
    result = {"loss": total.item() / count, "n": count}
    for k in topk:
        result[f"top{k}"] = hits[k].item() / count
    return result
