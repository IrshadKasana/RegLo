"""Train one optimizee with M1, M2, M3 or M4 and record what the tables need — M2 with its
regularizer at every step (Stage 1 on each round's source pair, Stages 2–3 on every pair).

Per run (artifacts/<profile>/runs/<study>/<optimizee>-<dataset>/<method>[-<r>][@<round>]/seed<k>/):

    config.yaml           fully resolved configuration
    environment.json      GPU model, library versions, PBS job id
    metrics.jsonl         per-epoch training loss, test loss/accuracy, clean training loss, time
    final.json            the headline numbers (accuracy, generalization gap, norms, sparsity,
                          FLOPs, wall-clock) — what reporting/tables.py reads
    weights.json          layerwise norms and the |w| histogram
    model_final.pt        trained weights ("at completion")
    model_converged.pt    weights when the training loss stopped improving ("at convergence")
    last.pt               resumable checkpoint while the run is in progress (deleted at the end)

Wall-clock time counts training steps only — not evaluation, not checkpointing — and the
GPU is synchronised around each epoch so the numbers are real. The test set is evaluated every
``eval_every`` epochs, the clean training set every ``train_eval_every`` epochs, and both at the
end (the generalization gap). With ``max_hours`` the run stops cleanly — checkpoint, status
``interrupted``, no final.json — when the next epoch would not finish in time; the same command
continues it.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from reglo.config import save_config
from reglo.data.loaders import OptimizeeData, build_optimizee_data
from reglo.diagnostics.convergence import ConvergenceTracker
from reglo.diagnostics.flops import count_flops, synchronize
from reglo.diagnostics.norms import weight_statistics
from reglo.models.optimizees import build_optimizee
from reglo.training.evaluate import evaluate
from reglo.training.methods import Method, build_method
from reglo.utils.checkpoint import load_checkpoint, rng_state, save_checkpoint, set_rng_state
from reglo.utils.logging import MetricsWriter, read_json, write_json
from reglo.utils.paths import method_tag, regularizer_of, run_dir
from reglo.utils.runtime import environment_info, resolve_device

__all__ = ["OptimizeeTrainer"]


class OptimizeeTrainer:
    def __init__(self, cfg: DictConfig, device: Optional[torch.device] = None, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.tr = cfg.training
        self.device = torch.device(device) if device is not None else resolve_device(cfg.device)
        self.log = logger or logging.getLogger("reglo")
        self.out_dir = run_dir(cfg)

    # ------------------------------------------------------------------ setup

    def build_model(self) -> nn.Module:
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.cfg.seed))  # same seed → same initial weights for M1–M4
            return build_optimizee(self.cfg.optimizee).to(self.device)

    def count_step_flops(self, model: nn.Module, data: OptimizeeData) -> int:
        """FLOPs of one full training step, measured on a copy so the real run is untouched."""
        clone = copy.deepcopy(model)
        method = build_method(self.cfg, clone, 1, 1, self.device)
        ds = self.cfg.dataset
        x = torch.zeros(int(self.tr.batch_size), int(ds.channels), int(ds.image_size), int(ds.image_size), device=self.device)
        y = torch.zeros(int(self.tr.batch_size), dtype=torch.long, device=self.device)
        return count_flops(lambda: method.step(x, y))

    # ------------------------------------------------------------------ run

    def run(self) -> Dict[str, Any]:
        started = time.monotonic()
        final_path = self.out_dir / "final.json"
        if final_path.is_file() and not self.tr.rerun:
            self.log.info("Already finished: %s (training.rerun=true to redo)", final_path)
            return read_json(final_path)
        out = self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        if self.tr.rerun:  # start over: nothing of an earlier attempt may leak into this one
            for stale in ("final.json", "last.pt", "model_converged.pt", "model_final.pt", "metrics.jsonl", "weights.json"):
                (out / stale).unlink(missing_ok=True)
        save_config(self.cfg, out / "config.yaml")
        env = environment_info(self.device)
        write_json(out / "environment.json", env)

        data = build_optimizee_data(self.cfg, self.device)
        model = self.build_model()
        steps_per_epoch = len(data.train)
        epochs = int(self.tr.epochs)
        cap = None if self.tr.max_steps is None else int(self.tr.max_steps)
        total_steps = steps_per_epoch * epochs if cap is None else min(steps_per_epoch * epochs, cap)
        method = build_method(self.cfg, model, steps_per_epoch, total_steps, self.device)
        flops = self.count_step_flops(model, data) if self.tr.count_flops else None

        tracker = ConvergenceTracker(int(self.tr.convergence_patience))
        state = {"epoch": 0, "step": 0, "train_time": 0.0, "evaluation": None, "diverged": False}
        last_path = out / "last.pt"
        if self.tr.resume and last_path.is_file():
            self._resume(last_path, model, method, tracker, data, state)
            self.log.info("Resuming at epoch %d (step %d) from %s", state["epoch"], state["step"], last_path)

        self.log.info(
            "%s %s — %s on %s, %d epochs × %d steps, %d pass(es)/step%s",
            method.label, method_tag(self.cfg), self.cfg.selected.optimizee, self.cfg.selected.dataset, epochs,
            steps_per_epoch, method.passes, f", {flops / 1e9:.2f} GFLOPs/step" if flops else "",
        )
        deadline = started + 3600.0 * float(self.cfg.max_hours) if float(self.cfg.max_hours) > 0 else None
        with MetricsWriter(out / "metrics.jsonl") as metrics:
            while state["epoch"] < epochs and not state["diverged"]:
                if cap is not None and state["step"] >= cap:
                    break
                epoch_started = time.monotonic()
                row = self._train_epoch(method, data, tracker, state, cap, metrics)
                epoch = state["epoch"]
                if state["diverged"]:
                    metrics.write(row)
                    self.log.warning("Training diverged in epoch %d (non-finite loss)", epoch)
                    break
                last = epoch == epochs or (cap is not None and state["step"] >= cap)
                test = last or epoch % max(int(self.tr.eval_every), 1) == 0
                train = last or epoch % max(int(self.tr.train_eval_every), 1) == 0
                if test or train:
                    evaluation = self._evaluate(model, data, test=test, train=train)
                    row.update(evaluation)
                    if test and train:
                        state["evaluation"] = {**evaluation, "epoch": epoch}
                metrics.write(row)
                self._log_epoch(row, epochs)
                saved = False
                if epoch % max(int(self.tr.checkpoint_every), 1) == 0 and not last:
                    self._checkpoint(last_path, model, method, tracker, data, state)
                    saved = True
                if deadline is not None and not last and time.monotonic() + 1.5 * (time.monotonic() - epoch_started) > deadline:
                    if not saved:
                        self._checkpoint(last_path, model, method, tracker, data, state)
                    self.log.warning("max_hours=%s reached after epoch %d of %d: checkpoint saved, stopping; run the "
                                     "same command again to continue", self.cfg.max_hours, epoch, epochs)
                    return {"status": "interrupted", "epochs_completed": epoch, "checkpoint": last_path.as_posix()}

        evaluation = state["evaluation"]
        if not state["diverged"] and (evaluation is None or evaluation.get("epoch") != state["epoch"]):
            state["evaluation"] = {**self._evaluate(model, data), "epoch": state["epoch"]}
        torch.save(model.state_dict(), out / "model_final.pt")
        stats = weight_statistics(
            model, float(self.tr.sparsity_threshold), list(self.cfg.regularizer.get("exclude", ["bias", "norm"]))
        )
        write_json(out / "weights.json", stats)
        final = self._final(method, state, tracker, flops, stats, env, steps_per_epoch)
        write_json(final_path, final)
        if last_path.is_file():
            last_path.unlink()  # large for M3/M4; the run is complete
        self.log.info("Done: test top-1 %.4f, gap %.4f, %.1f s → %s", final["test_top1"] or math.nan,
                      final["gen_gap"] or math.nan, final["wall_clock_s"], final_path)
        return final

    # ------------------------------------------------------------------ pieces

    def _train_epoch(self, method: Method, data: OptimizeeData, tracker: ConvergenceTracker, state, cap, metrics) -> Dict[str, Any]:
        method.model.train()
        log_every = max(int(self.cfg.logging.log_every), 1)
        loss_sum, batches = 0.0, 0
        synchronize(self.device)
        started = time.perf_counter()
        for x, y in data.train:
            result = method.step(x, y)
            if not math.isfinite(result.loss):
                state["diverged"] = True
                break
            state["step"] += 1
            loss_sum += result.loss
            batches += 1
            if tracker.update(state["step"], result.loss):
                torch.save(method.model.state_dict(), self.out_dir / "model_converged.pt")
            if state["step"] % log_every == 0:
                metrics.write(kind="step", step=state["step"], epoch=state["epoch"] + 1, loss=result.loss,
                              penalty=result.penalty, lr=method.lr)
            if cap is not None and state["step"] >= cap:
                break
        synchronize(self.device)
        seconds = time.perf_counter() - started
        state["train_time"] += seconds
        state["epoch"] += 1
        return {
            "kind": "epoch",
            "epoch": state["epoch"],
            "step": state["step"],
            "train_loss_running": loss_sum / max(batches, 1),
            "epoch_seconds": seconds,
            "lr": method.lr,
            "diverged": state["diverged"],
        }

    def _evaluate(self, model: nn.Module, data: OptimizeeData, test: bool = True, train: bool = True) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if test:
            scores = evaluate(model, data.test)
            out.update(test_loss=scores["loss"], test_top1=scores["top1"], test_top5=scores["top5"])
        if train:
            scores = evaluate(model, data.train_eval)  # clean: same images, no augmentation
            out.update(train_loss=scores["loss"], train_top1=scores["top1"])
        if test and train:
            out.update(gen_gap=out["test_loss"] - out["train_loss"], gen_gap_top1=out["train_top1"] - out["test_top1"])
        return out

    def _log_epoch(self, row: Dict[str, Any], epochs: int) -> None:
        parts = [f"epoch {row['epoch']}/{epochs}", f"loss {row['train_loss_running']:.4f}"]
        if "test_top1" in row:
            parts.append(f"test acc {row['test_top1']:.4f}")
        if "gen_gap_top1" in row:
            parts.append(f"gap {row['gen_gap_top1']:+.4f}")
        parts.append(f"{row['epoch_seconds']:.1f}s")
        self.log.info("  ".join(parts))

    def _final(self, method: Method, state, tracker, flops, stats, env, steps_per_epoch) -> Dict[str, Any]:
        from reglo.rounds import get_round

        ev = state["evaluation"] or {}
        epochs_done = max(state["epoch"], 1)
        learned = self.cfg.method.optimizer_kind == "learned"
        rnd = get_round(str(self.cfg.meta_tag)) if learned else None
        return {
            "status": "diverged" if state["diverged"] else "completed",
            "study": str(self.cfg.study),
            "profile": str(self.cfg.profile),
            "optimizee": str(self.cfg.selected.optimizee),
            "dataset": str(self.cfg.selected.dataset),
            "method": method.name,
            "label": method.label,
            "regularizer": regularizer_of(self.cfg),
            "round": str(self.cfg.meta_tag) if learned else None,       # whose φ an M3/M4 run used
            "phi_source": method.phi_source,                            # the pair that φ was meta-trained on
            "role": rnd.role(str(self.cfg.selected.optimizee), str(self.cfg.selected.dataset)) if rnd else None,
            "regularizer_applied": str(self.cfg.method.regularizer_applied),
            "evaluated_on": "validation" if float(self.tr.get("validation_fraction", 0.0) or 0.0) > 0 else "test",
            "seed": int(self.cfg.seed),
            "epochs_completed": state["epoch"],
            "steps": state["step"],
            "steps_per_epoch": steps_per_epoch,
            "test_loss": ev.get("test_loss"),
            "test_top1": ev.get("test_top1"),
            "test_top5": ev.get("test_top5"),
            "train_loss": ev.get("train_loss"),
            "train_top1": ev.get("train_top1"),
            "gen_gap": ev.get("gen_gap"),
            "gen_gap_top1": ev.get("gen_gap_top1"),
            "wall_clock_s": round(state["train_time"], 3),
            "seconds_per_epoch": round(state["train_time"] / epochs_done, 3),
            "passes_per_step": method.passes,
            "flops_per_step": flops,
            "total_gflops": None if flops is None else flops * state["step"] / 1e9,
            "converged_step": tracker.converged_step,
            "weights": {k: stats[k] for k in ("num_params", "num_penalised", "l1", "l2", "l1_all", "l2_all", "sparsity")},
            "gpu": env.get("gpu"),
            "hostname": env.get("hostname"),
        }

    def _checkpoint(self, path, model, method, tracker, data, state) -> None:
        save_checkpoint(path, {
            "model": model.state_dict(),
            "method": method.state_dict(),
            "tracker": tracker.state_dict(),
            "stream": data.train.state_dict(),
            "state": state,
            "rng": rng_state(),
        })

    def _resume(self, path, model, method, tracker, data, state) -> None:
        saved = load_checkpoint(path, map_location=self.device)
        model.load_state_dict(saved["model"])
        method.load_state_dict(saved["method"])
        tracker.load_state_dict(saved["tracker"])
        data.train.load_state_dict(saved["stream"])
        state.update(saved["state"])
        set_rng_state(saved["rng"])
