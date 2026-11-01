"""M1–M4: what happens in one training step of an optimizee.

    M1  AdamW                   loss.backward()                   → AdamW step
    M2  AdamW + regularizer r   r.explicit_step()  (1, 2 or 4 passes) → AdamW step
    M3  frozen LSTM φ           loss.backward()                   → LSTM step
    M4  frozen LSTM φ*_r        loss.backward()                   → LSTM step   (r acted only in Stage 1)

M2 computes its regularizer at every step wherever it is trained — Stage 1 on each round's source
pair, and every evaluation pair of Stages 2–3 (the source-pair runs serve both).

M3 and M4 refuse a checkpoint that was meta-trained for something else (an M4-SAM run
cannot silently load the L1 optimizer, or M3's), or on another source pair than its round's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from reglo.models.optimizers import (
    FrozenLSTMOptimizer,
    build_adamw,
    build_scheduler,
    load_learned_optimizer,
)
from reglo.regularizers import L2, Regularizer, build_regularizer
from reglo.utils.paths import regularizer_of

__all__ = ["CheckpointMismatch", "Method", "StepResult", "build_method"]

log = logging.getLogger("reglo")


class CheckpointMismatch(ValueError):
    """An M3/M4 run was pointed at an LSTM optimizer meta-trained for another method."""


@dataclass
class StepResult:
    loss: float  # training loss on the batch, before the update
    penalty: Optional[float] = None  # M2: the regularizer's own number for this step


class Method:
    """One of M1–M4, bound to a model: ``step(x, y)`` performs one full training step."""

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        self.label = str(cfg.method.label)
        self.name = str(cfg.method.name)
        self.regularizer_name = regularizer_of(cfg)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regularizer = regularizer
        clip = cfg.optimizer.get("grad_clip", None)
        self.grad_clip = float(clip) if clip else None
        self.phi_source: Optional[str] = None  # M3/M4: "<optimizee>-<dataset>" φ was meta-trained on

    @property
    def passes(self) -> int:
        """Forward–backward passes per step (the LSTM's own step comes on top for M3/M4)."""
        return self.regularizer.passes if self.regularizer is not None else 1

    @property
    def lr(self) -> Optional[float]:
        return self.optimizer.param_groups[0].get("lr") if self.scheduler is not None else None

    def step(self, x: torch.Tensor, y: torch.Tensor) -> StepResult:
        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer is None:
            loss = F.cross_entropy(self.model(x), y)
            loss.backward()
            result = StepResult(loss.item())
        else:
            out = self.regularizer.explicit_step(self.model, x, y)
            result = StepResult(out.loss, out.penalty)
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.regularizer is not None:
            self.regularizer.after_step(self.model, self.optimizer)
        if self.scheduler is not None:
            self.scheduler.step()
        return result

    def state_dict(self) -> Dict[str, Any]:
        state = {"optimizer": self.optimizer.state_dict()}
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])


def _check_checkpoint(cfg: DictConfig, meta: Dict[str, Any], path: str) -> None:
    wanted_entry = "m3_lstm" if cfg.method.name == "m3_lstm" else "m4_lstm_reg"
    wanted_reg = regularizer_of(cfg)
    got_entry, got_reg = meta.get("meta_training"), meta.get("regularizer")
    if got_entry is not None and got_entry != wanted_entry:
        raise CheckpointMismatch(f"{cfg.method.label} needs a φ meta-trained by {wanted_entry}, but {path} came from {got_entry}")
    if got_reg is not None and got_reg != wanted_reg:
        raise CheckpointMismatch(
            f"{cfg.method.label} with regularizer={wanted_reg} loaded a φ meta-trained with regularizer={got_reg} ({path})"
        )


def _phi_source(cfg: DictConfig, meta: Dict[str, Any], path: str) -> Optional[str]:
    """The pair φ was meta-trained on (from its checkpoint); it must be the source of the run's round."""
    from reglo.rounds import get_round

    selected = (meta.get("config") or {}).get("selected") or {}
    optimizee, dataset = selected.get("optimizee"), selected.get("dataset")
    if optimizee is None or dataset is None:
        return None
    rnd = get_round(str(cfg.meta_tag))
    if rnd is not None and (optimizee, dataset) != (rnd.optimizee, rnd.dataset):
        raise CheckpointMismatch(
            f"round {rnd.name} is meta-trained on {rnd.optimizee} / {rnd.dataset} (configs/rounds.yaml), but {path} "
            f"was meta-trained on {optimizee} / {dataset}"
        )
    return f"{optimizee}-{dataset}"


def build_method(cfg: DictConfig, model: nn.Module, steps_per_epoch: int, total_steps: int, device: torch.device) -> Method:
    """M1/M2: AdamW (+ regularizer) with its schedule. M3/M4: the frozen meta-trained LSTM."""
    m = cfg.method
    if m.optimizer_kind == "conventional":
        regularizer = build_regularizer(cfg.regularizer, model) if m.regularizer_applied == "every_step" else None
        decoupled = regularizer.decoupled_coef if isinstance(regularizer, L2) and regularizer.mode == "decoupled" else 0.0
        exclude = list(cfg.regularizer.get("exclude", ["bias", "norm"]))
        optimizer = build_adamw(cfg.optimizer, model, decoupled_l2=decoupled, exclude=exclude)
        scheduler = build_scheduler(optimizer, cfg.optimizer, total_steps, steps_per_epoch)
        return Method(cfg, model, optimizer, scheduler, regularizer)

    lstm, features, meta = load_learned_optimizer(m.checkpoint, device)
    _check_checkpoint(cfg, meta, str(m.checkpoint))
    source = _phi_source(cfg, meta, str(m.checkpoint))
    configured = {k: v for k, v in OmegaConf.to_container(cfg.optimizer).items() if k in ("num_layers", "hidden_size", "features", "normalization")}
    stored = {**lstm.config(), **features.config()}
    differs = {k: (v, stored.get(k)) for k, v in configured.items() if k in stored and stored[k] != v}
    if differs:
        log.warning("configs/optimizers/lstm.yaml differs from the checkpoint %s; the checkpoint is used", differs)
    optimizer = FrozenLSTMOptimizer(model.parameters(), lstm, features, chunk_size=int(cfg.optimizer.chunk_size))
    method = Method(cfg, model, optimizer)
    method.phi_source = source
    return method
