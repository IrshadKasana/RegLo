"""Coordinate-wise LSTM learned optimizer (LO) with a differentiable unroll.

Checkpoint compatibility
------------------------
``LSTMLearnedOptimizer`` owns the same sub-modules as the base ``dmo.meta_optimizer.MetaOptimizerLSTM``
(``first_order``: ``nn.LSTM(2, hidden, layers, bias=False, batch_first=True)``;
``outputer``: ``nn.Linear(hidden, 1, bias=False)`` scaled by 0.1) and the same gradient
pre-processing, so ``state_dict()`` files are interchangeable in both directions. A checkpoint
written here can be loaded by the base ``optimizee_train.py --saved_model ...``.

Why a new class
---------------
The base ``MetaModel.set_flat_params`` writes ``param.data = ...``, which detaches the
optimizee parameters from the LSTM output, so ``meta_loss.backward()`` never reaches the LSTM
(all its ``.grad`` stay ``None``). Here the optimizee is evaluated functionally
(``torch.func.functional_call``) on parameter tensors that remain connected to the LSTM
updates, which is what truncated BPTT requires.

The LSTM recurrence is computed with explicit matmuls on the ``nn.LSTM`` weights. This is
numerically identical to ``nn.LSTM`` (gate order i, f, g, o) and makes the LO's cost visible
to ``FlopCounterMode``.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

State = Tuple[torch.Tensor, torch.Tensor]   # (h, c), each (num_layers, n_coords, hidden)


class LSTMLearnedOptimizer(nn.Module):
    def __init__(self, num_layers: int = 2, hidden_size: int = 20, preproc: bool = True,
                 preproc_factor: float = 10.0):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.preproc = preproc
        self.first_order = nn.LSTM(2 if preproc else 1, self.hidden_size, self.num_layers,
                                   batch_first=True, bias=False)
        self.outputer = nn.Linear(self.hidden_size, 1, bias=False)
        self.outputer.weight.data.mul_(0.1)
        self.preproc_factor = float(preproc_factor)
        self.preproc_threshold = float(np.exp(-preproc_factor))

    # ------------------------------------------------------------------ core maths
    def preprocess(self, grads: torch.Tensor) -> torch.Tensor:
        """(N,) gradients -> (N, 2) inputs, identical to the base MetaOptimizerLSTM.forward."""
        if not self.preproc:
            return grads.unsqueeze(-1)
        keep = grads.abs() >= self.preproc_threshold
        log_mag = torch.log(grads.abs() + 1e-8) / self.preproc_factor
        first = torch.where(keep, log_mag, torch.full_like(grads, -1.0))
        second = torch.where(keep, torch.sign(grads), grads * float(np.exp(self.preproc_factor)))
        return torch.stack([first, second], dim=-1)

    def init_state(self, n_coords: int, device=None, dtype=torch.float32) -> State:
        shape = (self.num_layers, n_coords, self.hidden_size)
        return (torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype))

    def _cell(self, inputs: torch.Tensor, state: State) -> Tuple[torch.Tensor, State]:
        h, c = state
        x = inputs
        new_h, new_c = [], []
        for layer in range(self.num_layers):
            w_ih = getattr(self.first_order, f"weight_ih_l{layer}")
            w_hh = getattr(self.first_order, f"weight_hh_l{layer}")
            gates = x @ w_ih.t() + h[layer] @ w_hh.t()
            i, f, g, o = gates.chunk(4, dim=1)
            c_l = torch.sigmoid(f) * c[layer] + torch.sigmoid(i) * torch.tanh(g)
            h_l = torch.sigmoid(o) * torch.tanh(c_l)
            new_h.append(h_l)
            new_c.append(c_l)
            x = h_l
        update = (x @ self.outputer.weight.t()).squeeze(-1)
        return update, (torch.stack(new_h), torch.stack(new_c))

    def forward(self, grads: torch.Tensor, state: State) -> Tuple[torch.Tensor, State]:
        """Pure step: flat gradients (N,) and state -> (update (N,), new state)."""
        return self._cell(self.preprocess(grads), state)

    @torch.no_grad()
    def forward_chunked(self, grads: torch.Tensor, state: State, chunk_size: int) -> torch.Tensor:
        """In-place, memory-bounded version for a frozen LO on large optimizees (e.g. ResNet-18).

        Coordinates are independent, so chunking is exact. ``state`` is updated in place.
        """
        h, c = state
        update = torch.empty_like(grads)
        n = grads.numel()
        for start in range(0, n, chunk_size):
            sl = slice(start, min(n, start + chunk_size))
            u, (h_new, c_new) = self._cell(self.preprocess(grads[sl]), (h[:, sl], c[:, sl]))
            update[sl] = u
            h[:, sl] = h_new
            c[:, sl] = c_new
        return update

    # ------------------------------------------------------------------ checkpoints
    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def from_checkpoint(cls, path, num_layers: int, hidden_size: int, map_location="cpu"):
        lo = cls(num_layers=num_layers, hidden_size=hidden_size)
        lo.load_state_dict(torch.load(path, map_location=map_location))
        return lo


# ====================================================================== R_smooth

def smooth_penalty(lo: LSTMLearnedOptimizer, grads: torch.Tensor, state: State, clean_update: torch.Tensor,
                   eps: float, num_steps: int) -> torch.Tensor:
    """Adversarial smoothness (Lipschitz) regularizer of the SmoothedOptimizer reference.

        R_smooth = max_{||delta||_inf <= eps} || LO(g + delta) - LO(g) ||^2   (MSE)

    The inner max is approximated with ``num_steps`` signed projected-gradient-ascent steps
    of size eps/5 from a small random start, exactly as in ``SmoothedOptimizer/optimization/
    meta_optimizer.py``. As there, the perturbation lives in RMS-normalised gradient space
    (g / sqrt(mean(g^2) + 1e-5)), so ``eps`` is scale-free; it is mapped back to raw-gradient
    space before the LO sees it, so the LO gets the same kind of input as in its real update.
    Both LO evaluations start from the same hidden state; neither advances it.

    ``clean_update`` must be LO(g, state) from the actual step (it is reused, not recomputed).
    """
    g = grads.detach()
    rms = torch.sqrt((g ** 2).mean() + 1e-5)
    z = g / rms
    detached_state = (state[0].detach(), state[1].detach())
    target = clean_update.detach()
    adv = z + 0.01 * torch.randn_like(z)
    step = eps / 5.0
    for _ in range(int(num_steps)):
        adv = adv.detach().requires_grad_(True)
        with torch.enable_grad():
            pert, _ = lo(adv * rms, detached_state)
            dist = F.mse_loss(pert, target)
            (grad_adv,) = torch.autograd.grad(dist, [adv])
        adv = adv.detach() + step * torch.sign(grad_adv)
        adv = torch.max(torch.min(adv, z + eps), z - eps)
    perturbed_update, _ = lo(adv.detach() * rms, state)
    return F.mse_loss(clean_update, perturbed_update)


# ====================================================================== flat-parameter helpers

def flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten(flat: torch.Tensor, shapes: Sequence[torch.Size]) -> List[torch.Tensor]:
    out, offset = [], 0
    for shape in shapes:
        n = math.prod(shape) if len(shape) else 1
        out.append(flat[offset:offset + n].view(shape))
        offset += n
    return out


class FrozenLOStepper:
    """Applies a frozen LO to a real ``nn.Module`` whose ``.grad`` fields are populated.

    Used in phase 2 (optimizer evaluation) and by the phase-3 probe when probe.optimizer=method.
    No autograd graph is built; the LSTM state is kept per optimizee.
    """

    def __init__(self, lo: LSTMLearnedOptimizer, params: Sequence[torch.nn.Parameter], chunk_size: int = 1 << 20):
        self.lo = lo.eval()
        self.params = [p for p in params if p.requires_grad]
        n = sum(p.numel() for p in self.params)
        device = self.params[0].device
        self.state = lo.init_state(n, device=device)
        self.chunk_size = int(chunk_size)

    @torch.no_grad()
    def step(self) -> None:
        grads = flatten([p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params])
        update = self.lo.forward_chunked(grads, self.state, self.chunk_size)
        offset = 0
        for p in self.params:
            n = p.numel()
            p.add_(update[offset:offset + n].view_as(p))
            offset += n

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
