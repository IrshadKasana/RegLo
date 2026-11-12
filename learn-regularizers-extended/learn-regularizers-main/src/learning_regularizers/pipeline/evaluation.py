"""Metrics: accuracy, loss, generalization gap, and the frozen-backbone linear probe.

Generalization gap (``metrics.gap``):
    accuracy -> train_acc - test_acc      (in percentage points)
    loss     -> test_loss - train_loss
Lower is better in both cases.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .flops import FlopMeter, NullMeter


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, meter: Optional[FlopMeter] = None) -> Tuple[float, float]:
    """Mean NLL and accuracy (%) of ``model`` (eval mode) on ``loader``."""
    meter = meter or NullMeter()
    was_training = model.training
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with meter.track():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            total_loss += F.nll_loss(out, y, reduction="sum").item()
            correct += (out.argmax(dim=1) == y).sum().item()
            count += y.numel()
    model.train(was_training)
    if count == 0:
        return float("nan"), float("nan")
    return total_loss / count, 100.0 * correct / count


def generalization_gap(train_loss: float, train_acc: float, test_loss: float, test_acc: float, kind: str) -> float:
    if kind == "accuracy":
        return train_acc - test_acc
    if kind == "loss":
        return test_loss - train_loss
    raise ValueError(f"Unknown gap metric '{kind}'")


def summarize(train_loss, train_acc, test_loss, test_acc, gap_kind) -> Dict[str, float]:
    return {"train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc,
            "gen_gap": generalization_gap(train_loss, train_acc, test_loss, test_acc, gap_kind),
            "gen_gap_acc": train_acc - test_acc, "gen_gap_loss": test_loss - train_loss}


# ------------------------------------------------------------------------- linear probe

@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device, meter: FlopMeter):
    was_training = model.training
    model.eval()
    feats, labels = [], []
    with meter.track():
        for x, y in loader:
            feats.append(model.features(x.to(device, non_blocking=True)).float())
            labels.append(y.to(device))
    model.train(was_training)
    return torch.cat(feats), torch.cat(labels)


class LinearHead(nn.Module):
    """New classifier on frozen features; log-probabilities like every optimizee."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, f):
        return F.log_softmax(self.linear(f), dim=-1)


def linear_probe(backbone: nn.Module, train_loader: DataLoader, test_loader: DataLoader, num_classes: int,
                 device, probe_cfg, gap_kind: str, seed: int, meter: FlopMeter,
                 make_step_fn: Optional[Callable] = None) -> Dict[str, float]:
    """Freeze ``backbone``; fit a new linear head on its features; report train/test metrics.

    ``make_step_fn(head, total_steps)`` (optional) returns ``step(x, y, t) -> loss`` that trains the
    head with the method's own frozen optimizer; otherwise a fixed Adam probe is used so the
    comparison isolates the quality of the frozen optimizee.
    """
    for p in backbone.parameters():
        p.requires_grad_(False)
    f_train, y_train = extract_features(backbone, train_loader, device, meter)
    f_test, y_test = extract_features(backbone, test_loader, device, meter)

    torch.manual_seed(seed)
    head = LinearHead(f_train.shape[1], num_classes).to(device)
    gen = torch.Generator().manual_seed(seed)
    batch = int(probe_cfg.batch_size)
    epochs = int(probe_cfg.epochs)
    n = f_train.shape[0]
    steps_per_epoch = max(1, (n + batch - 1) // batch)
    total_steps = epochs * steps_per_epoch

    if make_step_fn is None:
        opt = torch.optim.Adam(head.parameters(), lr=float(probe_cfg.lr))

        def step(x, y, t):
            with meter.track():
                loss = F.nll_loss(head(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            return float(loss.detach())
    else:
        step = make_step_fn(head, total_steps)

    t = 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(device)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            step(f_train[idx], y_train[idx], t)
            t += 1

    def feat_metrics(f, y):
        with torch.no_grad():
            out = head(f)
            return F.nll_loss(out, y).item(), 100.0 * (out.argmax(1) == y).float().mean().item()

    tr_loss, tr_acc = feat_metrics(f_train, y_train)
    te_loss, te_acc = feat_metrics(f_test, y_test)
    res = summarize(tr_loss, tr_acc, te_loss, te_acc, gap_kind)
    res.update({"probe_train_samples": int(n), "probe_test_samples": int(f_test.shape[0]),
                "probe_steps": int(total_steps)})
    return res
