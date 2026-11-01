"""
One training run = one (regularizer, hyperparameters, seed) triple.

SELECTION PROTOCOL -- read before changing anything
---------------------------------------------------
The test set is never used for any decision. Validation accuracy is tracked
every epoch; we record the epoch with the best validation accuracy and report
the test accuracy measured at that epoch. Hyperparameters are later selected on
validation too (see analyze.py). Selecting rho on test is the single most
common methodological complaint against papers in this area.

`fwd_bwd_passes` is recorded so a compute-matched table can be built later:
SAM and GSAM cost 2x the gradient evaluations per step, GAM costs 4x, so an
epoch-matched comparison hands the regularizers extra compute.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data import Mnist
from .device import (resolve_device, configure_backends, empty_cache,
                     peak_memory_mb, reset_peak_memory)
from .models import (make_model, count_params, evaluate, grad_norm_at,
                     sam_gap, hessian_top_eigenvalue)
from .optimizers import build_optimizer, PASSES_PER_STEP

DEFAULTS = dict(
    gamma=None, lam=None, decouple=None,
    regularizer="none", seed=0, epochs=100, max_steps=None, batch_size=128,
    lr=1e-3, weight_decay=0.01, hidden=20, act="relu", lr_schedule="cosine",
    warmup_steps=0, label_noise=0.0, grad_clip=None, m_sharpness=None,
    measure_sharpness=True, sharpness_radii=(0.05, 0.5), measure_hessian=False,
    keep_history=True, data_root="./data", deterministic=True, allow_tf32=False,
)


def _make_scheduler(opt, cfg, total_steps):
    kind = cfg.get("lr_schedule", "cosine")
    warm = int(cfg.get("warmup_steps", 0) or 0)
    base = opt.base_optimizer

    if kind == "constant" and warm == 0:
        return None

    def fn(step):
        if warm and step < warm:
            return (step + 1) / warm
        if kind == "constant":
            return 1.0
        prog = (step - warm) / max(total_steps - warm, 1)
        prog = min(max(prog, 0.0), 1.0)
        if kind == "cosine":
            import math
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))
        if kind == "step":
            return 0.1 ** int(prog * 3)
        raise ValueError(f"unknown lr_schedule {kind!r}")

    return torch.optim.lr_scheduler.LambdaLR(base, fn)


def run(cfg: dict, data: Mnist | None = None, device=None,
        progress: bool = False) -> dict:
    full = dict(DEFAULTS)
    full.update({k: v for k, v in cfg.items() if v is not None or k in cfg})
    cfg = full

    device = device or resolve_device(cfg.get("device", "auto"))
    configure_backends(deterministic=cfg["deterministic"],
                       allow_tf32=cfg["allow_tf32"])

    owns_data = data is None
    if owns_data:
        data = Mnist(cfg["data_root"], device, label_noise=cfg["label_noise"],
                     allow_synthetic=cfg.get("allow_synthetic", False),
                     eval_test=cfg.get("eval_test", False))

    reset_peak_memory(device)

    model = make_model(cfg["seed"], device, hidden=cfg["hidden"], act=cfg["act"])
    opt = build_optimizer(cfg["regularizer"], model.parameters(), cfg)

    bs = cfg["batch_size"]
    spe = data.steps_per_epoch(bs)
    epochs = int(cfg["epochs"])
    max_steps = int(cfg["max_steps"]) if cfg.get("max_steps") else epochs * spe
    sched = _make_scheduler(opt, cfg, max_steps)

    # Data-order RNG, seeded per run, independent of the init RNG.
    # CPU generator on purpose -- see Mnist.epoch_batches.
    gen = torch.Generator().manual_seed(10_000 + int(cfg["seed"]))

    clip = cfg.get("grad_clip")
    best = {"val_acc": -1.0, "epoch": -1, "test_acc": float("nan"),
            "val_loss": float("nan"), "test_loss": float("nan")}
    history, step, t0, stop = [], 0, time.time(), False

    for ep in range(epochs):
        model.train()
        for xb, yb in data.epoch_batches(bs, gen):
            def closure(microbatch=None):
                """
                Forward+backward on the current batch.

                Called with `microbatch=m` (only by SAM's m-sharpness path) it
                instead RETURNS A GENERATOR of sub-closures, one per chunk of
                m examples, so the optimizer can compute a separate ascent
                perturbation per chunk. Without this branch, setting
                `m_sharpness` in a config crashed with
                    TypeError: closure() got an unexpected keyword 'microbatch'
                i.e. the option was documented but unusable.
                """
                if microbatch is None:
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward()
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    return loss

                def _make(i):
                    xs, ys = xb[i:i + microbatch], yb[i:i + microbatch]

                    def _sub():
                        loss = F.cross_entropy(model(xs), ys)
                        loss.backward()
                        if clip:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), clip)
                        return loss
                    return _sub

                return (_make(i) for i in range(0, xb.shape[0], microbatch))
            opt.step(closure)
            if sched is not None:
                sched.step()
            step += 1
            if step >= max_steps:
                stop = True
                break

        trl, tra = evaluate(model, data.xtr, data.ytr)
        vl, va = evaluate(model, data.xva, data.yva)
        # Test is read ONLY when explicitly enabled. During a sweep the test
        # split is never loaded to the device, so it cannot leak into
        # selection even accidentally.
        if cfg.get("eval_test", False) and data.xte is not None:
            tl, ta = evaluate(model, data.xte, data.yte)
        else:
            tl, ta = float("nan"), float("nan")
        history.append(dict(epoch=ep, step=step, lr=opt.param_groups[0]["lr"],
                            train_loss=trl, train_acc=tra, val_loss=vl,
                            val_acc=va, test_loss=tl, test_acc=ta))
        if va > best["val_acc"]:
            best = dict(val_acc=va, val_loss=vl, epoch=ep,
                        test_acc=ta, test_loss=tl)
        if progress:
            print(f"  ep {ep:3d}  train {tra*100:5.2f}  val {va*100:5.2f}  "
                  f"test {ta*100:5.2f}", flush=True)
        if stop:
            break

    diag = {}
    if cfg["measure_sharpness"]:
        diag["final_grad_norm_train"] = grad_norm_at(model, data.xtr, data.ytr)
        for r in cfg["sharpness_radii"]:
            diag[f"sam_gap_rho{r}"] = sam_gap(model, data.xtr, data.ytr, rho=float(r))
    if cfg["measure_hessian"]:
        diag["hessian_lambda_max"] = hessian_top_eigenvalue(model, data.xtr, data.ytr)

    last = history[-1]
    out = {
        "config": {k: v for k, v in cfg.items()
                   if k not in ("device", "allow_synthetic")},
        "n_params": count_params(model),
        "steps": step,
        "epochs_run": len(history),
        "fwd_bwd_passes": step * PASSES_PER_STEP[cfg["regularizer"].lower()],
        "wall_seconds": round(time.time() - t0, 2),
        "peak_memory_mb": round(peak_memory_mb(device), 1),
        "best_val_acc": best["val_acc"],
        "best_val_epoch": best["epoch"],
        "test_acc_at_best_val": best["test_acc"],
        "test_loss_at_best_val": best["test_loss"],
        "final_val_acc": last["val_acc"],
        "final_test_acc": last["test_acc"],
        "final_train_acc": last["train_acc"],
        "final_train_loss": last["train_loss"],
        # Generalisation gap measured against the HELD-OUT split actually
        # available. With eval_test=False that is validation, which is a
        # perfectly valid held-out estimate of the gap; it just must not also
        # be the split used to pick the winner in the same breath. Selection
        # uses mean val ACCURACY; this gap is a separate diagnostic.
        "generalization_gap": (last["train_acc"] - last["test_acc"]
                               if cfg.get("eval_test", False)
                               else last["train_acc"] - last["val_acc"]),
        "gap_reference": "test" if cfg.get("eval_test", False) else "val",
        **diag,
        "history": history if cfg["keep_history"] else [],
    }

    del model, opt
    if owns_data:
        del data
    empty_cache(device)
    return out
