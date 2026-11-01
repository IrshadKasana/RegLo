#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_compute.py
====================
Measures the compute trade-offs the reviewer asked for, per method:

    - wall-clock time per optimizee step (ms)
    - throughput (steps / second)
    - number of forward passes per step
    - number of backward passes per step
    - approximate FLOPs per step (forward+backward, via fvcore if available)

Methods benchmarked:
    AdamW + SAM   (explicit, no LO)   -> 2 fwd + 2 bwd per step
    No Reg LO     (learned optimizer) -> 1 fwd + 1 bwd (optimizee) + LSTM fwd
    SAM LO        (learned optimizer) -> 1 fwd + 1 bwd (optimizee) + LSTM fwd

Outputs:
    artifacts/dmo/results/compute_benchmark.json
    artifacts/dmo/results/compute_benchmark.txt

Run on a GPU node:
    python benchmark_compute.py
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "learning_regularizers" / "dmo"))

import torch
import torch.nn.functional as F

from model import initialize_model
from meta_optimizer import MetaOptimizerLSTM, MetaModel
from artifact_manager import get_checkpoint_path
from regularizers.optimizers import SAM

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts" / "dmo" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Benchmarking on device: {device}")

# ---- Benchmark settings ----
DATASET = 'mnist'
MODEL = 'mlp'
N_WARMUP = 20
N_STEPS = 200
BATCH = 128

# Names of your trained LO checkpoints
LO_CHECKPOINTS = {
    "No Reg LO":  "DMO_adam_280424",
    "SAM LO":     "AdSAM_sgd_LO_140424",
}


class _Args:
    """Minimal args object for the SAM optimizer constructor."""
    rho = 0.05
    adaptive = True
    base_optimizer = 'AdamW'   # SAM ctor branches on this; AdamW falls into adam path below
    lr = 0.001
    momentum = 0.0
    weight_decay = 0.01
    SAMonOptimizer = True


def make_batch():
    x = torch.randn(BATCH, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (BATCH,), device=device)
    return x, y


def try_flops(model, x):
    """Approximate forward FLOPs for one batch using fvcore, if installed."""
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
        model.train()
        return int(total)
    except Exception as e:
        print(f"  (fvcore FLOP counting unavailable: {e})")
        return None


def bench_adamw_sam():
    print("\n=== AdamW + SAM (explicit, no LO) ===")
    model = initialize_model(DATASET, MODEL).to(device)
    args = _Args()
    # SAM now natively supports base_optimizer='AdamW' (after setup_baseline_comparison.py
    # patches the SAM class). No manual swap needed.
    sam = SAM(model.parameters(), args=args)

    def step():
        x, y = make_batch()
        # first pass (ascent)
        F.nll_loss(model(x), y).backward()
        sam.first_step(zero_grad=True)
        # second pass (descent)
        F.nll_loss(model(x), y).backward()
        sam.second_step()
        model.zero_grad()

    for _ in range(N_WARMUP):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    x, _ = make_batch()
    fwd_flops = try_flops(model, x)
    return {
        "method": "AdamW + SAM",
        "uses_learned_optimizer": False,
        "forward_passes_per_step": 2,
        "backward_passes_per_step": 2,
        "ms_per_step": 1000 * dt / N_STEPS,
        "steps_per_sec": N_STEPS / dt,
        "approx_fwd_flops_per_batch": fwd_flops,
        # 2 fwd + 2 bwd; bwd ~ 2x fwd -> total ~ 2*(1) + 2*(2) = 6 fwd-equivalents
        "approx_total_flops_per_step": (fwd_flops * 6) if fwd_flops else None,
    }


def bench_lo(name, ckpt_key):
    print(f"\n=== {name} (learned optimizer) ===")
    model = initialize_model(DATASET, MODEL).to(device)
    meta_model = initialize_model(DATASET, MODEL).to(device)
    lo = MetaOptimizerLSTM(MetaModel(meta_model), 2, 20, preproc_factor=10.0, preproc=True)
    try:
        path = get_checkpoint_path(ckpt_key, experiment="dmo")
        lo.load_state_dict(torch.load(path, map_location=device))
        print(f"  loaded checkpoint: {path}")
    except Exception as e:
        print(f"  ! could not load {ckpt_key}: {e} - using random-init LO for timing only")
    lo.reset_lstm(keep_states=False, model=meta_model)
    lo = lo.to(device)

    def step():
        x, y = make_batch()
        F.nll_loss(model(x), y).backward()
        with torch.no_grad():
            lo.meta_update(model)
        model.zero_grad()

    for _ in range(N_WARMUP):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    x, _ = make_batch()
    fwd_flops = try_flops(model, x)
    return {
        "method": name,
        "uses_learned_optimizer": True,
        "forward_passes_per_step": 1,          # optimizee forward
        "backward_passes_per_step": 1,         # optimizee backward
        "ms_per_step": 1000 * dt / N_STEPS,
        "steps_per_sec": N_STEPS / dt,
        "approx_fwd_flops_per_batch": fwd_flops,
        # 1 fwd + 1 bwd (~2 fwd) for optimizee + LSTM overhead (not counted in FLOPs)
        "approx_total_flops_per_step": (fwd_flops * 3) if fwd_flops else None,
        "note": "Plus LSTM meta-optimizer forward (per-parameter); not included in optimizee FLOPs.",
    }


def main():
    results = []
    results.append(bench_adamw_sam())
    for name, key in LO_CHECKPOINTS.items():
        results.append(bench_lo(name, key))

    # Save JSON
    (OUT_DIR / "compute_benchmark.json").write_text(json.dumps(results, indent=2))

    # Pretty table
    lines = ["=" * 100,
             f"{'Method':<16} {'fwd/step':>9} {'bwd/step':>9} {'ms/step':>10} "
             f"{'steps/s':>10} {'fwd FLOPs/batch':>18} {'~FLOPs/step':>16}",
             "-" * 100]
    for r in results:
        ff = f"{r['approx_fwd_flops_per_batch']:,}" if r['approx_fwd_flops_per_batch'] else "n/a"
        tf = f"{r['approx_total_flops_per_step']:,}" if r['approx_total_flops_per_step'] else "n/a"
        lines.append(f"{r['method']:<16} {r['forward_passes_per_step']:>9} "
                     f"{r['backward_passes_per_step']:>9} {r['ms_per_step']:>10.3f} "
                     f"{r['steps_per_sec']:>10.1f} {ff:>18} {tf:>16}")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Notes:")
    lines.append("  - AdamW+SAM needs 2 forward + 2 backward passes per step (ascent + descent).")
    lines.append("  - Learned optimizers need 1 forward + 1 backward on the optimizee, PLUS an")
    lines.append("    LSTM meta-optimizer forward pass (per-parameter) not captured by fvcore.")
    lines.append("  - FLOP estimates use bwd ~ 2x fwd. Treat as order-of-magnitude, not exact.")

    table = "\n".join(lines)
    (OUT_DIR / "compute_benchmark.txt").write_text(table)
    print("\n" + table)
    print(f"\nSaved to: {OUT_DIR / 'compute_benchmark.json'}")
    print(f"          {OUT_DIR / 'compute_benchmark.txt'}")


if __name__ == "__main__":
    main()