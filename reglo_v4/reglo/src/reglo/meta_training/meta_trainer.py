"""Stage 1: meta-training the LSTM optimizer — the outer loop of Algorithm 1.

    for each trajectory k = 1 … K:                                 (line 1)
        sample a fresh optimizee θ_0, zero LSTM state               (lines 2–4)
        for each truncation of T inner steps:
            unroll T steps with F_φ                                 (lines 5–15)
            L_meta,reg = Σ w_t L(θ_t) + λ_smooth·L_smooth + λ_reg·L_reg(φ)     (16–23)
                L_reg on φ: L1 λ‖φ‖₁ · L2 ½λ‖φ‖² · SAM L_meta(φ+ε̂), ε̂ = ρ∇_φL_meta/‖∇_φL_meta‖
                · GAM ρ‖∇_φL_meta‖ · GSAM its direction (meta.space: phi; theta: at θ_T)
            φ ← Adam step on ∇_φ L_meta,reg; detach θ and the state  (24, truncated BPTT)
        every few trajectories: train fresh optimizees with the frozen φ, score them on the
        held-out meta images, keep the best φ (phi_best.pt), advance the curriculum

M3's optimizer is the case λ_smooth = λ_reg = 0 (meta_training=m3_lstm); M4's add L_smooth
and one L_reg (meta_training=m4_lstm_reg regularizer=<r>).

The optimizee and dataset are the source pair of the round named by ``meta_tag``
(configs/rounds.yaml: A = ConvNet-4 / SVHN, B = MLP / FMNIST), on the meta-training half of
the dataset's training images.

Outputs (artifacts/<profile>/meta_training/<round>/<m3_lstm | m4_lstm_reg-<r>>/): config.yaml,
environment.json, meta_log.jsonl, phi_best.pt (used by M3/M4 runs), phi_last.pt (resume),
summary.json. With ``max_hours`` the run stops cleanly before the limit (phi_last.pt saved,
status ``interrupted``); the same command continues it.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from reglo.config import save_config
from reglo.data.loaders import build_meta_data
from reglo.meta_training.curriculum import Curriculum
from reglo.meta_training.objective import MetaObjective, build_objective, loss_weights
from reglo.meta_training.unroll import InnerState, unroll
from reglo.models.functional import FunctionalModel
from reglo.models.optimizees import build_optimizee
from reglo.models.optimizers import FrozenLSTMOptimizer, build_lstm_optimizer, save_learned_optimizer
from reglo.regularizers import SmoothnessPenalty, build_regularizer
from reglo.regularizers.base import displaced
from reglo.training.evaluate import evaluate
from reglo.utils.checkpoint import load_checkpoint, rng_state, save_checkpoint, set_rng_state
from reglo.utils.logging import MetricsWriter, read_json, write_json
from reglo.utils.paths import meta_run_dir, regularizer_of
from reglo.utils.runtime import environment_info, resolve_device

__all__ = ["MetaTrainer"]


def _zeros_for_none(grads, params) -> List[torch.Tensor]:
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]


class MetaTrainer:
    def __init__(self, cfg: DictConfig, device: Optional[torch.device] = None, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.mt = mt = cfg.meta_training
        self.device = torch.device(device) if device is not None else resolve_device(cfg.device)
        self.log = logger or logging.getLogger("reglo")
        self.out_dir = meta_run_dir(cfg)

        self.data = build_meta_data(cfg, self.device)
        self.lstm, self.features = build_lstm_optimizer(cfg.optimizer)
        self.lstm.to(self.device)
        self.outer = torch.optim.Adam(self.lstm.parameters(), lr=float(mt.outer.lr))

        self.T = int(mt.inner.unroll_steps)
        self.smooth = SmoothnessPenalty(cfg.smooth) if mt.objective.smooth else None
        self.regularizer = (
            build_regularizer(cfg.regularizer, build_optimizee(cfg.optimizee)) if mt.objective.regularizer else None
        )
        self.reg_coef = self.regularizer.meta_coef if self.regularizer is not None else 0.0
        self.reg_space = self.regularizer.meta_space if self.regularizer is not None else "theta"
        sched = mt.schedule
        self.curriculum = Curriculum(
            sched.curriculum.trajectory_steps, sched.curriculum.patience, sched.curriculum.enabled, sched.trajectory_steps
        )

        self.batches = self.data.train.forever()
        self.noise = torch.Generator(device=self.device)
        self.noise.manual_seed(int(cfg.seed) + 17)
        self.trajectory = 0
        self.meta_steps = 0
        self.elapsed = 0.0
        self.best: Dict[str, Any] = {"val_loss": math.inf, "trajectory": None}

    # ------------------------------------------------------------------ public

    def fit(self) -> Dict[str, Any]:
        started_run = time.monotonic()
        summary_path = self.out_dir / "summary.json"
        if summary_path.is_file() and not self.mt.rerun:
            self.log.info("Already finished: %s (meta_training.rerun=true to redo)", summary_path)
            return read_json(summary_path)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.mt.rerun:  # start over: an earlier φ must never be mistaken for this run's
            for stale in ("summary.json", "phi_best.pt", "phi_last.pt", "meta_log.jsonl"):
                (self.out_dir / stale).unlink(missing_ok=True)
        save_config(self.cfg, self.out_dir / "config.yaml")
        write_json(self.out_dir / "environment.json", environment_info(self.device))
        last = self.out_dir / "phi_last.pt"
        if self.mt.resume and last.is_file():
            self._load_last(last)
            self.log.info("Resuming after trajectory %d from %s", self.trajectory, last)

        total = int(self.mt.schedule.num_trajectories)
        every = max(int(self.mt.validation.every), 1)
        hours = float(self.cfg.max_hours)
        deadline = started_run + 3600.0 * hours if hours > 0 else None
        slowest = {"trajectory": 0.0, "validation": 0.0}
        with MetricsWriter(self.out_dir / "meta_log.jsonl") as metrics:
            while self.trajectory < total:
                started = time.perf_counter()
                row = self._train_trajectory(metrics)
                self.trajectory += 1
                spent = time.perf_counter() - started
                self.elapsed += spent
                slowest["trajectory"] = max(slowest["trajectory"], spent)
                self.log.info(
                    "trajectory %d/%d  steps %d  L_meta %.4f  L_smooth %.4g  L_reg %.4g  loss(θ_T) %.4f",
                    self.trajectory, total, self.curriculum.length, row["l_meta"], row["l_smooth"], row["l_reg"], row["loss_T"],
                )
                if self.trajectory % every == 0 or self.trajectory == total:
                    started = time.perf_counter()
                    self._validate(metrics)
                    slowest["validation"] = max(slowest["validation"], time.perf_counter() - started)
                saved = False
                if self.trajectory % max(int(self.mt.checkpoint_every), 1) == 0 or self.trajectory == total:
                    self._save_last(last)
                    saved = True
                if deadline is not None and self.trajectory < total:
                    upcoming = slowest["trajectory"] + (slowest["validation"] if (self.trajectory + 1) % every == 0 else 0.0)
                    if time.monotonic() + 1.5 * upcoming > deadline:
                        if not saved:
                            self._save_last(last)
                        self.log.warning("max_hours=%s reached after trajectory %d of %d: phi_last.pt saved, stopping; "
                                         "run the same command again to continue", hours, self.trajectory, total)
                        return {"status": "interrupted", "trajectories": self.trajectory, "checkpoint": last.as_posix()}

        best_path = self.out_dir / "phi_best.pt"
        if not best_path.is_file():  # no validation was finite: keep the pipeline going, but say so
            self.log.warning("no meta-validation finished with a finite loss; the last φ becomes phi_best.pt")
            save_learned_optimizer(best_path, self.lstm, self.features, meta_training=str(self.mt.name),
                                   regularizer=regularizer_of(self.cfg), trajectory=self.trajectory,
                                   val_loss=math.inf, val_acc=None, selected="last", config=self.cfg)

        summary = {
            "status": "completed",
            "meta_training": self.mt.name,
            "regularizer": regularizer_of(self.cfg),
            "round": str(self.cfg.meta_tag),
            "source": f"{self.cfg.selected.optimizee}-{self.cfg.selected.dataset}",
            "smooth": self.smooth is not None,
            "reg_space": self.reg_space if self.regularizer is not None else None,
            "trajectories": self.trajectory,
            "meta_steps": self.meta_steps,
            "best_trajectory": self.best["trajectory"],
            "best_val_loss": self.best["val_loss"],
            "best_val_acc": self.best.get("val_acc"),
            "selected": "best validation" if self.best["trajectory"] is not None else "last (no finite validation)",
            "profile": str(self.cfg.profile),
            "curriculum_stage": self.curriculum.stage,
            "wall_clock_s": round(self.elapsed, 1),
            "checkpoint": (self.out_dir / "phi_best.pt").as_posix(),
        }
        write_json(summary_path, summary)
        self.log.info("Best φ from trajectory %s (val loss %.4f) → %s", summary["best_trajectory"], summary["best_val_loss"], summary["checkpoint"])
        return summary

    # ------------------------------------------------------------------ one trajectory

    def new_optimizee(self, index: int) -> Tuple[FunctionalModel, InnerState]:
        """A freshly initialised optimizee (Algorithm 1, lines 2–4), reproducible per trajectory index."""
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.cfg.seed) * 1_000_003 + index)
            model = build_optimizee(self.cfg.optimizee).to(self.device)
        model.train()
        fmodel = FunctionalModel(model)
        params = fmodel.initial_params()
        h, c = self.lstm.init_state(sum(p.numel() for p in params), device=self.device)
        return fmodel, InnerState(params, h, c, self.features.init_state(params), 0)

    def _train_trajectory(self, metrics: MetricsWriter) -> Dict[str, Any]:
        fmodel, state = self.new_optimizee(self.trajectory)
        length, done, truncation, row = self.curriculum.length, 0, 0, {}
        while done < length:
            steps = min(self.T, length - done)
            state, row = self.meta_step(fmodel, state, steps)
            metrics.write(
                kind="meta_step", trajectory=self.trajectory, truncation=truncation, inner_step=state.step,
                trajectory_steps=length, meta_step=self.meta_steps, **row,
            )
            done += steps
            truncation += 1
            if row["diverged"]:
                self.log.warning("trajectory %d diverged at inner step %d; starting a new one", self.trajectory, state.step)
                break
        return row

    def meta_step(self, fmodel: FunctionalModel, state: InnerState, steps: int) -> Tuple[InnerState, Dict[str, Any]]:
        """One truncation: unroll ``steps`` inner steps, one update of φ, detach (Algorithm 1, lines 5–24)."""
        inner = self.mt.inner
        out = unroll(
            fmodel, self.lstm, self.features, state, self.batches, steps,
            first_order=bool(inner.first_order), smooth=self.smooth,
            checkpoint_lstm=bool(inner.checkpoint_lstm), generator=self.noise,
        )
        row: Dict[str, Any] = {"loss_0": out.train_losses[0], "loss_T": out.train_losses[-1], "diverged": not out.finite}
        if not out.finite:
            row.update(l_meta=math.nan, l_smooth=math.nan, l_reg=math.nan, objective=math.nan, grad_norm=math.nan, skipped=True)
            return out.final.detached(), row

        penalty = None
        replay = self.regularizer is not None and self.reg_space == "phi" and self.regularizer.replays
        if self.regularizer is not None and self.reg_space == "theta":
            x, y = next(self.batches) if self.mt.objective.reg_batch == "fresh" else out.batches[-1]
            penalty = self.regularizer.meta_penalty(fmodel.loss, out.final.params, x, y)
        elif self.regularizer is not None and not replay:  # L1 / L2 on the LSTM's own weights φ
            penalty = self.regularizer.phi_penalty(self.lstm.named_parameters())
        weights = loss_weights(str(self.mt.outer.loss_weights), steps, self.device)
        smooth_coef = self.smooth.coef if self.smooth is not None else 0.0
        objective = build_objective(out.losses, weights, out.smooth_terms, smooth_coef, penalty, self.reg_coef)

        self.outer.zero_grad(set_to_none=True)
        if replay:
            self._phi_space_backward(objective, fmodel, state, out.batches, steps, weights)
        else:
            objective.total.backward()
        clip = float(self.mt.outer.grad_clip)
        params = list(self.lstm.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(params, clip if clip > 0 else float("inf"))
        skipped = not torch.isfinite(grad_norm).item()
        if skipped:
            self.outer.zero_grad(set_to_none=True)
        else:
            self.outer.step()
        self.meta_steps += 1
        row.update(objective.parts, grad_norm=grad_norm.item(), skipped=skipped)
        return out.final.detached(), row

    def _phi_space_backward(self, objective: MetaObjective, fmodel, start, batches, steps, weights) -> None:
        """Algorithm 1, lines 21–22 (SAM, GAM, GSAM): perturb the LSTM's own weights φ.

        ε̂ comes from ∇_φ L_meta; the same truncation is replayed (same θ_0, state and batches)
        at φ + ε̂, and the regularizer combines the two gradients (SAM: the perturbed one).
        """
        phi = list(self.lstm.parameters())
        g_meta = _zeros_for_none(torch.autograd.grad(objective.l_meta, phi, retain_graph=True, allow_unused=True), phi)
        objective.total.backward()  # L_meta + λ_smooth·L_smooth
        buffers = {name: b.detach().clone() for name, b in fmodel.module.named_buffers()}
        with displaced(phi, self.regularizer.phi_perturbation(g_meta)):
            replay = unroll(fmodel, self.lstm, self.features, start, iter(batches), steps,
                            first_order=bool(self.mt.inner.first_order),
                            checkpoint_lstm=bool(self.mt.inner.checkpoint_lstm))
            l_meta_perturbed = sum(w * loss for w, loss in zip(weights, replay.losses))
            g_perturbed = _zeros_for_none(torch.autograd.grad(l_meta_perturbed, phi, allow_unused=True), phi)
        with torch.no_grad():
            for name, b in fmodel.module.named_buffers():
                b.copy_(buffers[name])
        for p, extra in zip(phi, self.regularizer.phi_gradient(g_meta, g_perturbed)):
            p.grad = self.reg_coef * extra if p.grad is None else p.grad + self.reg_coef * extra
        objective.parts["l_reg"] = l_meta_perturbed.item() - objective.parts["l_meta"]
        objective.parts["objective"] = objective.parts["l_meta"] + objective.parts.get("l_smooth", 0.0) * (
            self.smooth.coef if self.smooth else 0.0
        ) + self.reg_coef * objective.parts["l_reg"]

    # ------------------------------------------------------------------ validation and checkpoints

    def validate(self) -> Dict[str, Any]:
        """Train ``validation.optimizees`` fresh optimizees with a frozen copy of φ and score each on
        the held-out meta images; the mean decides (one diverging optimizee makes the loss infinite)."""
        vcfg = self.mt.validation
        losses, accs, ends, diverged = [], [], [], 0
        for k in range(max(int(vcfg.optimizees), 1)):
            result = self._validate_one(k, int(vcfg.steps))
            losses.append(result["val_loss"])
            accs.append(result["val_acc"])
            ends.append(result["train_loss_end"])
            diverged += int(result["diverged"])
        finite = [v for v in ends if math.isfinite(v)]
        return {
            "val_loss": math.inf if diverged else sum(losses) / len(losses),
            "val_acc": sum(accs) / len(accs),
            "val_losses": losses,
            "train_loss_end": sum(finite) / len(finite) if finite else math.nan,
            "diverged": diverged > 0,
            "optimizees_diverged": diverged,
        }

    def _validate_one(self, k: int, steps: int) -> Dict[str, Any]:
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.cfg.seed) + 424_242 + 7919 * k)  # the same k-th optimizee every validation
            model = build_optimizee(self.cfg.optimizee).to(self.device)
        optimizer = FrozenLSTMOptimizer(model.parameters(), copy.deepcopy(self.lstm), self.features)
        stream = self.data.val_train
        stream.generator.manual_seed(int(self.cfg.seed) + 1 + 7919 * k)  # the same batches every validation
        batches, last, diverged = stream.forever(), math.nan, False
        model.train()
        for _ in range(steps):
            x, y = next(batches)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            optimizer.step()
            last = loss.item()
        scores = evaluate(model, self.data.val) if not diverged else {"loss": math.inf, "top1": 0.0}
        if not math.isfinite(scores["loss"]):
            diverged = True
        return {
            "val_loss": scores["loss"] if not diverged else math.inf,
            "val_acc": scores["top1"] if not diverged else 0.0,
            "train_loss_end": last,
            "diverged": diverged,
        }

    def _validate(self, metrics: MetricsWriter) -> None:
        result = self.validate()
        improved = result["val_loss"] < self.best["val_loss"]
        if improved:
            self.best = {"val_loss": result["val_loss"], "val_acc": result["val_acc"], "trajectory": self.trajectory}
            save_learned_optimizer(
                self.out_dir / "phi_best.pt", self.lstm, self.features,
                meta_training=str(self.mt.name), regularizer=regularizer_of(self.cfg), trajectory=self.trajectory,
                val_loss=result["val_loss"], val_acc=result["val_acc"], config=self.cfg,
            )
        advanced = self.curriculum.report(result["val_loss"])
        metrics.write(kind="validation", trajectory=self.trajectory, improved=improved, curriculum_stage=self.curriculum.stage,
                      trajectory_steps=self.curriculum.length, **result)
        self.log.info(
            "validation after %d trajectories: loss %.4f  acc %.3f%s%s",
            self.trajectory, result["val_loss"], result["val_acc"], "  (best)" if improved else "",
            f"  → curriculum stage {self.curriculum.stage} ({self.curriculum.length} steps)" if advanced else "",
        )

    def _save_last(self, path) -> None:
        save_checkpoint(path, {
            "lstm": self.lstm.state_dict(),
            "outer": self.outer.state_dict(),
            "trajectory": self.trajectory,
            "meta_steps": self.meta_steps,
            "elapsed": self.elapsed,
            "best": self.best,
            "curriculum": self.curriculum.state_dict(),
            "stream": self.data.train.state_dict(),
            "noise": self.noise.get_state(),
            "rng": rng_state(),
        })

    def _load_last(self, path) -> None:
        state = load_checkpoint(path, map_location=self.device)
        self.lstm.load_state_dict(state["lstm"])
        self.outer.load_state_dict(state["outer"])
        self.trajectory = int(state["trajectory"])
        self.meta_steps = int(state["meta_steps"])
        self.elapsed = float(state["elapsed"])
        self.best = dict(state["best"])
        self.curriculum.load_state_dict(state["curriculum"])
        self.data.train.load_state_dict(state["stream"])
        self.batches = self.data.train.forever()
        self.noise.set_state(state["noise"].cpu())
        set_rng_state(state["rng"])
