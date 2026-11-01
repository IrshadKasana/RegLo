"""
The model under study and the flatness diagnostics.

Default architecture: MLP 784 -> 20 -> 10.

    fc1 : 784*20 + 20 = 15,700
    fc2 :  20*10 + 10 =    210
    total             = 15,910

Note this is UNDERPARAMETERIZED for 55,000 training examples: it cannot
interpolate the training set, so it will not find the sharp interpolating
minima that sharpness-aware methods exist to avoid. Expect small effect sizes
on clean MNIST at this width. `--hidden 200` (159k params) gives an
overparameterized regime where the effects are much larger; running both is
the cleanest way to show the effect scales with capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid,
               "gelu": F.gelu}


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=20, out_dim=10, act="relu"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        if act not in ACTIVATIONS:
            raise ValueError(f"unknown activation {act!r}")
        self.act_name = act
        self.act = ACTIVATIONS[act]

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def make_model(seed: int, device, hidden: int = 20, act: str = "relu") -> MLP:
    """
    Deterministic init from `seed`. Every arm at a given seed gets bit-identical
    initial weights, which is what makes the seed-paired significance test in
    `analyze` legitimate.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return MLP(hidden=hidden, act=act).to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, x, y, batch: int = 10000):
    was_training = model.training
    model.eval()
    loss_sum, correct = 0.0, 0
    for i in range(0, x.shape[0], batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        out = model(xb)
        loss_sum += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    n = x.shape[0]
    if was_training:
        model.train()
    return loss_sum / n, correct / n


# --------------------------------------------------------------------------- #
# flatness diagnostics -- these carry the actual scientific claim
# --------------------------------------------------------------------------- #
def grad_norm_at(model, x, y, batch: int = 10000) -> float:
    """||grad L||_2 over the given set. Smaller => flatter to first order."""
    model.zero_grad(set_to_none=True)
    n = x.shape[0]
    for i in range(0, n, batch):
        loss = F.cross_entropy(model(x[i:i + batch]), y[i:i + batch],
                               reduction="sum") / n
        loss.backward()
    sq = sum((p.grad ** 2).sum().item()
             for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return sq ** 0.5


def sam_gap(model, x, y, rho: float = 0.05, n_ascent: int = 5,
            batch: int = 10000) -> float:
    """
    Empirical sharpness:  max_{||e||<=rho} L(w+e) - L(w),
    estimated by projected gradient ascent inside the rho-ball.

    This is the quantity SAM bounds. If you want to argue that a method
    "finds flatter minima", this is the number to show -- it is far more
    direct evidence than a test-accuracy delta, and it stays interpretable
    even when the accuracy difference is too small to resolve statistically.
    """
    base_loss, _ = evaluate(model, x, y, batch)
    orig = [p.detach().clone() for p in model.parameters()]
    n = x.shape[0]

    for _ in range(n_ascent):
        model.zero_grad(set_to_none=True)
        for i in range(0, n, batch):
            loss = F.cross_entropy(model(x[i:i + batch]), y[i:i + batch],
                                   reduction="sum") / n
            loss.backward()
        with torch.no_grad():
            gn = torch.sqrt(sum((p.grad ** 2).sum()
                                for p in model.parameters())) + 1e-12
            for p in model.parameters():
                p.add_(p.grad * (rho / n_ascent) / gn)
            disp = torch.sqrt(sum(((p - o) ** 2).sum()
                                  for p, o in zip(model.parameters(), orig)))
            if disp > rho:
                for p, o in zip(model.parameters(), orig):
                    p.copy_(o + (p - o) * (rho / disp))

    adv_loss, _ = evaluate(model, x, y, batch)
    with torch.no_grad():
        for p, o in zip(model.parameters(), orig):
            p.copy_(o)
    model.zero_grad(set_to_none=True)
    return adv_loss - base_loss


def hessian_top_eigenvalue(model, x, y, iters: int = 20,
                           batch: int = 10000) -> float:
    """
    Largest Hessian eigenvalue by power iteration on Hessian-vector products.

    lambda_max is the classical curvature-based sharpness measure and is the
    second diagnostic reviewers look for alongside the SAM gap. Uses a data
    subset for tractability; exact value matters less than the ordering
    between arms.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    xb, yb = x[:batch], y[:batch]

    v = [torch.randn_like(p) for p in params]
    nv = torch.sqrt(sum((t ** 2).sum() for t in v))
    v = [t / nv for t in v]

    eig = 0.0
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        gv = sum((g * t).sum() for g, t in zip(grads, v))
        hv = torch.autograd.grad(gv, params, retain_graph=False)
        hv = [h.detach() for h in hv]
        eig = float(sum((h * t).sum() for h, t in zip(hv, v)))
        nh = torch.sqrt(sum((h ** 2).sum() for h in hv)) + 1e-12
        v = [h / nh for h in hv]

    model.zero_grad(set_to_none=True)
    return eig
