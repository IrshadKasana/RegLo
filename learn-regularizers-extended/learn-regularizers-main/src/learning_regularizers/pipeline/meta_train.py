"""Phase 1 for learned optimizers (methods 3-6): meta-train the LO on the seen task.

Meta-objective of one truncation (T = truncated_bptt_step inner steps) starting at (theta_0, s_0):

    theta_{t+1} = theta_t + LO_phi(grad f(theta_t; B_t), s_t)          (first-order: grads detached,
                                                                         as in the base code)
    L_task      = f(theta_T; B_{T-1})                (meta_loss='last', base code)
               or mean_t f(theta_{t+1}; B_t)         (meta_loss='sum', SmoothedOptimizer / L2L)
    L_meta      = scale * ( L_task
                            + lambda_reg    * R(theta_T)                       if R in {l1, l2}
                            + lambda_smooth * mean_t R_smooth(phi; g_t, s_t) )  if method uses R_smooth

    phi <- OuterStep(phi, grad_phi L_meta)   with OuterStep = SGD/Adam             (no R / l1 / l2)
                                                         = SAM / GAM / GSAM on phi (R in {sam, gam, gsam})

SAM/GAM/GSAM re-evaluate L_meta at perturbed phi through a closure that replays the *same*
truncation (same start point, LSTM state, minibatches and BatchNorm statistics), which is what
those methods require. The optimizee then continues from the trajectory computed at the
unperturbed phi (as in the base code).

Hyperparameters: ``meta_train.yaml`` + ``smooth.yaml`` + ``<R>.yaml`` (see config.py).
The curriculum (unroll lengths 100 -> 200 -> 500 -> 1000, ``n_period`` evaluations per stage)
follows the base ``train.py``; the only change is that the "improved" flag is reset at each
new stage, so the documented stopping rule ("no improvement with curriculum -> stop") applies.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from . import DMO_DIR  # noqa: F401
from regularizers.optimizers import SAM, get_optim_and_schedulers  # base implementations

from .artifacts import lo_checkpoint_path, write_json
from .config import REPO_ROOT, meta_models_cfg, meta_training_args, namespace_to_dict
from .data import InfiniteLoader, get_dataset, make_loader, num_classes
from .flops import FlopMeter
from .learned_optimizer import FrozenLOStepper, LSTMLearnedOptimizer, flatten, smooth_penalty, unflatten
from .methods import RunKey
from .models import build_optimizee
from .regularizers import PENALTY_REGULARIZERS, build_base_optimizer, weight_penalty

CURRICULUM = [100, 200, 500, 1000]   # base train.py


class MetaTrainer:
    def __init__(self, cfg, run_key: RunKey, device, log=print):
        assert run_key.spec.uses_lo, "MetaTrainer is for methods 3-6"
        self.cfg, self.key, self.device, self.log = cfg, run_key, device, log
        self.args = a = meta_training_args(cfg, run_key.spec, run_key.regularizer)
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)

        self.n_classes = num_classes(a.dataset)
        dataset = get_dataset(a.dataset, "meta_train", cfg)
        self.loader = make_loader(dataset, a.batch_size, True, a.seed, cfg, drop_last=True)
        self.stream = InfiniteLoader(self.loader)
        self.eval_dataset = dataset

        self.lo = LSTMLearnedOptimizer(a.num_layers, a.hidden_size).to(device)
        if getattr(a, "if_pretrained", False):
            path = REPO_ROOT / "artifacts" / "dmo" / "checkpoints" / f"{a.pretrained_model_name}.pt"
            self.log(f"[{run_key.run_id}] fine-tuning from pretrained LO {path}")
            self.lo.load_state_dict(torch.load(path, map_location=device))

        self.reg = a.regularizer
        self.outer = self._build_outer_optimizer()
        self.meter = FlopMeter(cfg.metrics.count_flops)
        self.eval_meter = FlopMeter(cfg.metrics.count_flops)
        self.scale = 1.0 / len(self.loader) if a.meta_loss_scale == "loader_len" else 1.0
        self.optimizee_counter = 0
        self.ckpt_path = lo_checkpoint_path(cfg, run_key.run_id)
        self.models_cfg = meta_models_cfg(cfg)   # models + meta_training.models (proxy widths)
        self._warn_if_too_big()

    # ------------------------------------------------------------------ setup
    BYTES_PER_COORD_STEP = 1.4e3   # measured: LSTM graph kept for BPTT, per optimizee weight per unroll step

    def _warn_if_too_big(self) -> None:
        """The unrolled LO graph grows with (#optimizee weights x truncated_bptt_step); warn before an OOM."""
        a = self.args
        with torch.random.fork_rng(devices=[]):   # building a model must not shift the training RNG
            n = sum(p.numel() for p in build_optimizee(a.optimizee_model_name, self.n_classes,
                                                       self.models_cfg).parameters())
        need = self.BYTES_PER_COORD_STEP * n * a.truncated_bptt_step
        msg = (f"[{self.key.run_id}] meta-training {a.optimizee_model_name} ({n:,} weights) with "
               f"truncated_bptt_step={a.truncated_bptt_step} needs roughly {need / 2**30:.1f} GiB")
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            if need > 0.9 * free:
                self.log(f"WARNING {msg}, but only {free / 2**30:.1f} GiB is free on {self.device}. Lower "
                         f"meta_training.overrides.truncated_bptt_step or use a narrower proxy via "
                         f"meta_training.models (e.g. {{resnet18: {{width: 16}}}}).")
                return
        self.log(msg)

    def _build_outer_optimizer(self):
        a = self.args
        if self.reg in ("none",) + PENALTY_REGULARIZERS:
            return build_base_optimizer(self.lo.parameters(), a)
        if self.reg == "sam":
            a.SAMonOptimizer = True          # SAM on the optimizer's parameters (sam.yaml)
            return SAM(self.lo.parameters(), args=a)
        if self.reg in ("gam", "gsam"):
            return get_optim_and_schedulers(self.lo, args=a)
        raise ValueError(f"Unknown regularizer '{self.reg}'")

    def _new_optimizee(self, seed: int):
        torch.manual_seed(seed)
        model = build_optimizee(self.args.optimizee_model_name, self.n_classes, self.models_cfg).to(self.device)
        model.train()
        names = [n for n, _ in model.named_parameters()]
        params = [p.detach().clone() for p in model.parameters()]
        for p in model.parameters():      # the module only provides structure + buffers
            p.requires_grad_(False)
        return model, names, params

    # ------------------------------------------------------------------ unroll
    def _unroll(self, model, names, params0: List[torch.Tensor], state0, batches):
        a = self.args
        shapes = [p.shape for p in params0]
        params, state = list(params0), state0
        task_terms, smooth_terms = [], []
        last_loss = float("nan")
        for j, (x, y) in enumerate(batches):
            leaf = [p.detach().requires_grad_(True) for p in params]
            with torch.enable_grad():   # inner gradients are always needed (even under no_grad callers)
                loss = F.nll_loss(functional_call(model, dict(zip(names, leaf)), (x,)), y)
                grads = torch.autograd.grad(loss, leaf)
            last_loss = float(loss.detach())
            flat_g = flatten(grads).detach()
            update, new_state = self.lo(flat_g, state)
            if a.if_smooth and j % max(1, a.smooth_every) == 0:
                smooth_terms.append(smooth_penalty(self.lo, flat_g, state, update,
                                                   eps=a.smooth_eps, num_steps=a.smooth_num_pga_steps))
            state = new_state
            params = [p + u for p, u in zip(params, unflatten(update, shapes))]
            if a.meta_loss == "sum" or j == len(batches) - 1:
                task_terms.append(F.nll_loss(functional_call(model, dict(zip(names, params)), (x,)), y))

        total = torch.stack(task_terms).mean() if a.meta_loss == "sum" else task_terms[-1]
        if self.reg in PENALTY_REGULARIZERS:
            total = total + float(a.reg_lambda) * weight_penalty(params, self.reg)
        if a.if_smooth and smooth_terms:
            total = total + float(a.smooth_lamb) * torch.stack(smooth_terms).mean()
        return total * self.scale, params, state, last_loss

    def _truncation(self, model, names, params, state) -> Tuple[List[torch.Tensor], tuple, float, float, bool]:
        a = self.args
        batches = []
        for _ in range(a.truncated_bptt_step):
            x, y = self.stream.next()
            batches.append((x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)))
        buffers = list(model.buffers())
        snapshot = [b.detach().clone() for b in buffers]
        committed: Dict = {}
        calls = [0]
        finite = [True]

        def closure():
            calls[0] += 1
            for b, s in zip(buffers, snapshot):
                b.copy_(s)
            self.lo.zero_grad(set_to_none=True)
            with torch.enable_grad():
                total, new_params, new_state, last_loss = self._unroll(model, names, params, state, batches)
                if not torch.isfinite(total):
                    finite[0] = False
                    for p in self.lo.parameters():
                        p.grad = torch.zeros_like(p)
                    return total.detach()
                total.backward()
            if a.meta_grad_clip:
                for p in self.lo.parameters():
                    if p.grad is not None:
                        p.grad.clamp_(-float(a.meta_grad_clip), float(a.meta_grad_clip))
            if calls[0] == 1:   # trajectory at the unperturbed phi is the one the optimizee follows
                committed.update(params=[p.detach() for p in new_params],
                                 state=(new_state[0].detach(), new_state[1].detach()),
                                 loss=last_loss, meta=float(total.detach()),
                                 buffers=[b.detach().clone() for b in buffers])
            return total.detach()

        with self.meter.track():
            if self.reg in ("none",) + PENALTY_REGULARIZERS:
                closure()
                if finite[0]:
                    self.outer.step()
            elif self.reg == "sam":
                closure()
                if finite[0]:
                    self.outer.first_step(zero_grad=True)
                    closure()
                    self.outer.second_step(zero_grad=False)
            else:  # gam / gsam: the method calls the closure as many times as it needs
                self.outer.update_rho_t()
                self.outer.step(closure)

        if not committed:  # non-finite on the first evaluation
            return params, state, float("nan"), float("nan"), False
        for b, s in zip(buffers, committed["buffers"]):
            b.copy_(s)
        return committed["params"], committed["state"], committed["loss"], committed["meta"], finite[0]

    # ------------------------------------------------------------------ train / eval
    def train_period(self, num_unroll: int) -> float:
        a = self.args
        n_trunc = max(1, num_unroll // a.truncated_bptt_step)
        metas = []
        for epoch in range(a.train_period):
            self.optimizee_counter += 1
            model, names, params = self._new_optimizee(seed=a.seed * 100003 + self.optimizee_counter)
            state = self.lo.init_state(sum(p.numel() for p in params), device=self.device)
            last = float("nan")
            for k in range(n_trunc):
                params, state, last, meta, ok = self._truncation(model, names, params, state)
                if not ok:
                    self.log(f"[{self.key.run_id}] non-finite meta-loss; abandoning this optimizee")
                    break
                metas.append(meta)
            self.log(f"[{self.key.run_id}] optimizee {epoch + 1}/{a.train_period} unroll={num_unroll} "
                     f"last inner loss {last:.4f} meta-loss {np.mean(metas[-n_trunc:]) if metas else float('nan'):.5f}")
        return float(np.mean(metas)) if metas else float("nan")

    @torch.no_grad()
    def evaluate(self, num_unroll: int, lo: Optional[LSTMLearnedOptimizer] = None) -> float:
        """Average training loss of fresh optimizees driven by the (frozen) LO for num_unroll steps."""
        a = self.args
        lo = lo or self.lo
        total = 0.0
        # optimizees are initialised on the CPU, so forking the CPU RNG keeps training RNG untouched
        with torch.random.fork_rng(devices=[]):
            for ep in range(a.eval_epochs):
                seed = int(self.cfg.meta_training.eval_seeds_offset) + ep
                torch.manual_seed(seed)
                model = build_optimizee(a.optimizee_model_name, self.n_classes, self.models_cfg).to(self.device)
                model.train()
                stepper = FrozenLOStepper(lo, list(model.parameters()),
                                          chunk_size=int(self.cfg.models.lo_chunk_size))
                stream = InfiniteLoader(make_loader(self.eval_dataset, a.batch_size, True, seed, self.cfg,
                                                    drop_last=True))
                loss_sum, steps = 0.0, 0
                for _ in range(num_unroll):
                    x, y = stream.next()
                    x, y = x.to(self.device), y.to(self.device)
                    with self.eval_meter.track():
                        with torch.enable_grad():
                            loss = F.nll_loss(model(x), y)
                            stepper.zero_grad()
                            loss.backward()
                        stepper.step()
                    if not math.isfinite(loss.item()):
                        loss_sum, steps = float("inf"), 1
                        break
                    loss_sum += loss.item()
                    steps += 1
                total += loss_sum / max(steps, 1)
        self.lo.train()
        return total / a.eval_epochs

    def fit(self) -> Dict:
        a = self.args
        t0 = time.time()
        history = []
        if a.if_cl:
            stage, n_train = 0, CURRICULUM[0]
            n_eval = CURRICULUM[1]
        else:
            stage, n_train, n_eval = None, a.optimizee_steps, a.optimizee_steps
        loss_min, best_state = float("inf"), None
        improved, n = False, 1

        for outer in range(a.max_epochs // a.train_period):
            meta = self.train_period(n_train)
            n += 1
            loss_eval = self.evaluate(n_eval)
            record = {"outer_iter": outer, "stage": stage, "unroll_train": n_train, "unroll_eval": n_eval,
                      "meta_loss": meta, "eval_loss": loss_eval, "meta_train_flops": self.meter.total,
                      "wall_time_s": time.time() - t0}
            if loss_eval < loss_min:
                loss_min = loss_eval
                improved = True
                best_state = copy.deepcopy(self.lo.state_dict())
                self.lo.save(self.ckpt_path)
                record["saved"] = True
            history.append(record)
            self.log(f"[{self.key.run_id}] outer {outer} stage {stage} eval loss {loss_eval:.4f} "
                     f"(best {loss_min:.4f})")

            if not a.if_cl:
                continue
            if n >= a.n_period and improved:
                self.log(f"[{self.key.run_id}] curriculum stage {stage} completed")
                n, improved = 1, False
                stage = stage + 1 if stage + 1 < len(CURRICULUM) else len(CURRICULUM) - 1
                n_train = n_eval = CURRICULUM[stage]
                best_lo = LSTMLearnedOptimizer(a.num_layers, a.hidden_size).to(self.device)
                best_lo.load_state_dict(best_state)
                loss_min = self.evaluate(n_eval, lo=best_lo)
            elif n >= a.n_period and not improved:
                self.log(f"[{self.key.run_id}] no improvement in curriculum stage {stage}: stopping")
                break

        if best_state is None:   # never improved (e.g. diverged): keep the last state
            self.lo.save(self.ckpt_path)
        summary = {
            "run": self.key.as_dict(), "checkpoint": str(self.ckpt_path),
            "best_eval_loss": loss_min, "meta_train_flops": self.meter.total,
            "meta_eval_flops": self.eval_meter.total, "wall_time_s": time.time() - t0,
            "history": history, "args": namespace_to_dict(a),
            "fingerprint": meta_fingerprint(self.cfg, self.key),
            "checkpoint_format": "state_dict compatible with dmo.meta_optimizer.MetaOptimizerLSTM "
                                 f"(num_layers={a.num_layers}, hidden_size={a.hidden_size})",
        }
        write_json(self.ckpt_path.with_suffix(".json"), summary)
        return summary


def meta_fingerprint(cfg, run_key: RunKey) -> str:
    """Hash of everything that determines a meta-trained LO (task, model widths, hyperparameters, data).

    Phase 1 re-uses a saved LO only when its fingerprint matches, so changing e.g. the
    meta-training optimizee/dataset under the same experiment name re-trains instead of
    silently re-using an LO trained on another task.
    """
    payload = {"args": namespace_to_dict(meta_training_args(cfg, run_key.spec, run_key.regularizer)),
               "synthetic": bool(cfg.experiment.get("synthetic_data", False)),
               "data_seed": int(cfg.experiment.seed), "eval_seeds_offset": cfg.meta_training.eval_seeds_offset}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_learned_optimizer(cfg, run_key: RunKey, device) -> LSTMLearnedOptimizer:
    a = meta_training_args(cfg, run_key.spec, run_key.regularizer)
    path = lo_checkpoint_path(cfg, run_key.run_id)
    if not path.exists():
        raise FileNotFoundError(f"No learned-optimizer checkpoint for {run_key.run_id} at {path}; run phase 1 first")
    return LSTMLearnedOptimizer.from_checkpoint(path, a.num_layers, a.hidden_size, map_location=device).to(device)
