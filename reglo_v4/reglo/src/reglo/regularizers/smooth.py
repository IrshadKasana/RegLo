"""L_smooth (Xiong & Hsieh 2022) — keeps the LSTM optimizer's update rule smooth.

    L_smooth(φ) = max_{s' ∈ B∞(s_t, ε)} ‖raw(s_t) − raw(s')‖²₂,   hidden state (h_t, c_t) fixed

where s_t is the LSTM's input at step t (one row per optimizee coordinate) and ``raw`` its
output before the fixed ``output_scale`` (the penalty is measured in the LSTM's own units,
independent of the step size). The squared norm runs over every coordinate (``reduction:
sum``; ``mean`` divides by their number). The maximiser is found by projected gradient
ascent, as in Xiong & Hsieh's released code: a small Gaussian start (std ε·init_ratio), then
``pga_steps`` sign steps of size ε·step_ratio, each clipped back into the ℓ∞ ball.

Part of M4's meta-objective only; it is never computed when an optimizee is trained.
"""

from __future__ import annotations

from typing import Optional

import torch
from omegaconf import DictConfig

from reglo.models.optimizers.lstm_optimizer import LSTMOptimizer

__all__ = ["SmoothnessPenalty"]


class SmoothnessPenalty:
    def __init__(self, cfg: DictConfig) -> None:
        self.coef = float(cfg.coef)
        self.epsilon = float(cfg.epsilon)
        self.pga_steps = int(cfg.pga_steps)
        self.step_size = float(cfg.step_ratio) * self.epsilon
        self.init_std = float(cfg.init_ratio) * self.epsilon
        self.every = max(int(cfg.every), 1)
        self.reduction = str(cfg.get("reduction", "sum"))
        if self.reduction not in ("sum", "mean"):
            raise ValueError(f"smooth.reduction must be sum or mean, got '{self.reduction}'")

    def _reduce(self, squared: torch.Tensor) -> torch.Tensor:
        return squared.sum() if self.reduction == "sum" else squared.mean()

    def applies_at(self, step: int) -> bool:
        return step % self.every == 0

    def __call__(
        self,
        lstm: LSTMOptimizer,
        inputs: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """L_smooth at one inner step; differentiable with respect to φ only."""
        s, h, c = inputs.detach(), h.detach(), c.detach()
        clean, _, _ = lstm(s, h, c)
        target = clean.detach()
        noise = torch.randn(s.shape, generator=generator, device=s.device, dtype=s.dtype)
        adversarial = s + (self.init_std * noise).clamp(-self.epsilon, self.epsilon)  # start inside the ball
        for _ in range(self.pga_steps):
            adversarial = adversarial.detach().requires_grad_(True)
            with torch.enable_grad():
                distance = (lstm(adversarial, h, c)[0] - target).pow(2).sum()  # the ascent direction only
                (direction,) = torch.autograd.grad(distance, adversarial)
            adversarial = adversarial.detach() + self.step_size * direction.sign()
            adversarial = torch.max(torch.min(adversarial, s + self.epsilon), s - self.epsilon)
        perturbed, _, _ = lstm(adversarial.detach(), h, c)
        return self._reduce((clean - perturbed).pow(2))
