"""
Grid sweep runner: append-only, resumable, shardable.
"""

from __future__ import annotations

import itertools
import json
import os
import time

import yaml

from .data import Mnist, DATASET_MB
from .device import (resolve_device, device_info, require_memory,
                     cap_memory_fraction, configure_backends)
from .train import run

STAGE1_SEEDS = 3

# dataset/model MUST be in the resume key. Without them a previous
# CIFAR/ResNet run counts as "already done" for a MNIST/MLP run that
# happens to share optimizer settings, silently mixing two studies
# into one results file.
_KEY_FIELDS = ["dataset", "model",
               "arm", "regularizer", "seed", "lr", "rho", "alpha", "adaptive",
               "grad_rho", "grad_norm_rho", "grad_gamma", "m_sharpness",
               "gamma", "lam", "decouple",
               "epochs", "max_steps", "batch_size", "hidden", "label_noise",
               "weight_decay"]


def load_grid(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand(cfg: dict, stage1: bool = False, only=None, arms=None) -> list[dict]:
    """Expand the YAML grid into a flat list of run configs."""
    fixed = dict(cfg.get("fixed", {}))
    seeds = cfg["seeds"][:STAGE1_SEEDS] if stage1 else cfg["seeds"]
    lr_grid = cfg.get("lr_grid", [fixed.get("lr", 1e-3)])
    runs = []

    for arm in cfg["arms"]:
        if arms and arm["name"] not in arms:
            continue
        # Per-arm lr_grid overrides the global one. This exists because the
        # expensive arms (GAM at 4 passes/step) should not be forced to search
        # learning rates that cheaper arms have already ruled out -- it is the
        # difference between a 3-hour and a 40-hour refinement pass.
        arm_lrs = arm.get("lr_grid", lr_grid)
        grid = dict(arm.get("grid", {}))
        keys = list(grid)
        combos = list(itertools.product(*[grid[k] for k in keys])) or [()]

        for combo in combos:
            hp = dict(zip(keys, combo))
            for name, expr in arm.get("derive", {}).items():
                hp[name] = eval(expr, {"__builtins__": {}}, dict(hp))
            hp = {k: v for k, v in hp.items() if not k.startswith("_")}
            for lr in arm_lrs:
                for seed in seeds:
                    r = dict(fixed)
                    r.update(hp)
                    r.update(arm=arm["name"], regularizer=arm["regularizer"],
                             lr=lr, seed=seed)
                    if only is not None and not _matches(r, only):
                        continue
                    runs.append(r)
    return runs


def _matches(run_cfg: dict, winners: list) -> bool:
    return any(all(run_cfg.get(k) == v for k, v in w.items()) for w in winners)


def run_key(c: dict) -> str:
    return "|".join(f"{k}={c.get(k)}" for k in _KEY_FIELDS)


def load_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(run_key(json.loads(line)["config"]))
            except Exception:
                continue
    return done


def sweep(config_path: str, out: str, shard: str = "0/1", stage1: bool = False,
          stage2: str | None = None, device_spec: str = "auto",
          arms=None, dry_run: bool = False, memory_fraction: float | None = None,
          allow_synthetic: bool = False) -> None:
    cfg = load_grid(config_path)

    winners = None
    if stage2:
        with open(stage2) as f:
            winners = yaml.safe_load(f)["winners"]

    runs = expand(cfg, stage1=stage1, only=winners, arms=arms)
    i, n = (int(x) for x in shard.split("/"))
    runs = [r for j, r in enumerate(runs) if j % n == i]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = load_done(out)
    todo = [r for r in runs if run_key(r) not in done]

    print(f"[sweep] shard {i}/{n}: {len(runs)} runs total, "
          f"{len(runs) - len(todo)} already done, {len(todo)} to run")
    if dry_run:
        for r in todo[:25]:
            print("   ", run_key(r))
        if len(todo) > 25:
            print(f"    ... and {len(todo) - 25} more")
        return
    if not todo:
        print("[sweep] nothing to do")
        return

    device = resolve_device(device_spec)
    configure_backends(deterministic=cfg["fixed"].get("deterministic", True),
                       allow_tf32=cfg["fixed"].get("allow_tf32", False))
    cap_memory_fraction(device, memory_fraction)
    require_memory(device, DATASET_MB + 150)
    print(f"[sweep] device: {device_info(device)}")

    data = Mnist(cfg["fixed"].get("data_root", "./data"), device,
                 label_noise=cfg["fixed"].get("label_noise", 0.0),
                 allow_synthetic=allow_synthetic)
    if data.synthetic:
        print("[sweep] !! SYNTHETIC DATA - smoke test only, results meaningless")

    t0 = time.time()
    with open(out, "a") as fout:
        for k, rc in enumerate(todo):
            res = run(dict(rc), data=data, device=device)
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            el = time.time() - t0
            eta = el / (k + 1) * (len(todo) - k - 1)
            # Show the arm-specific knob, not just lr/seed. Without this two
            # runs differing only in rho print identically and you cannot tell
            # from the log which configuration a number belongs to.
            bits = []
            for k_ in ("rho", "alpha", "gamma", "grad_rho", "grad_gamma"):
                if rc.get(k_) is not None:
                    bits.append(f"{k_}={rc[k_]:g}")
            hp_s = " ".join(bits)
            print(f"[{k+1}/{len(todo)}] {rc['arm']:6s} lr={rc['lr']:<8g} "
                  f"{hp_s:<34s} seed={rc['seed']:<2d} -> "
                  f"val {res['best_val_acc']*100:5.2f} "
                  f"test {res['test_acc_at_best_val']*100:5.2f} "
                  f"({res['wall_seconds']:.0f}s, eta {eta/60:.0f}m)", flush=True)

    print(f"[sweep] shard {i}/{n} finished in {(time.time()-t0)/60:.1f} min -> {out}")
