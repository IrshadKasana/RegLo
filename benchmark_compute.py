"""
benchmark_compute.py (v2)
=========================
Per-step compute benchmark for EVERY method in the comparison
(user item 5 + Change.MD items 58/60/61):

    adamw_sam : explicit AdamW+SAM      (2 fwd + 2 bwd per step)
    no_reg / sam / gam / gsam : learned optimizer at meta-test
                                        (1 fwd + 1 bwd + LSTM per step)

Measures per method:  mean/median step wall-clock, throughput (steps/s and
samples/s), peak GPU memory, optimizee forward FLOPs (fvcore if installed,
analytic fallback), per-update pass counts, and — the previously missing,
reviewer-critical term — the ANALYTIC LSTM FLOPs of the coordinate-wise
learned optimizer:

    per-coordinate cell cost (2-layer LSTM, hidden H, input 2 after
    log-preprocessing, +1 output unit):
        layer1: 8*H*(2+H+1)   layer2: 8*H*(H+H+1)   head: 2*H     [FLOPs]
    LSTM FLOPs/step = n_params * (layer1 + layer2 + head)

so the efficiency table can no longer overstate the LO by omitting its own
network (Change.MD item 58). The honest headline: LO cost scales with
n_params, SAM cost with n_params x batch_size.

Outputs: <out>.json, <out>.csv, and a printed markdown table.

Example:
    python benchmark_compute.py --dataset mnist --model mlp --batch-size 128 \
        --methods adamw_sam no_reg sam \
        --lo-checkpoints no_reg:NOREGLO_mnist_mlp,sam:SAMLO_mnist_mlp \
        --out results/benchmark/exp1
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent / "src" / "learning_regularizers" / "dmo"))
sys.path.append(str(Path(__file__).resolve().parent))

from model import initialize_model                      # noqa: E402
from meta_optimizer import MetaOptimizerLSTM, MetaModel  # noqa: E402
from regularizers import SAM  # noqa: E402
from regularizers.optimizers import get_optim_and_schedulers  # noqa: E402
from artifact_manager import get_checkpoint_path        # noqa: E402

LO_METHODS = ("no_reg", "sam", "gam", "gsam")
# single source of truth shared with optimizee_train (user: 'FLOPs in the
# experiment') — evaluation JSONs and this table can never disagree.
from flops_utils import (lstm_flops_per_step, optimizee_fwd_flops,  # noqa: E402
                         LSTM_LAYERS, LSTM_HIDDEN)


class _Args:  # namespace covering SAM + GAM + GSAM meta-test optimizers
    def __init__(self, adaptive, rho, base, lr, momentum, wd):
        self.adaptive, self.rho = adaptive, rho
        self.base_optimizer, self.lr = base, lr
        self.momentum, self.weight_decay = momentum, wd
        self.SAMonOptimizer = True
        # GAM schedule constants (paper defaults, as in adamw_gam_*.yaml)
        self.grad_beta_0, self.grad_beta_1 = 1.0, 1.0
        self.grad_beta_2, self.grad_beta_3 = -1.0, 1.0
        self.grad_rho, self.grad_rho_max, self.grad_rho_min = 0.04, 0.04, 0.02
        self.grad_norm_rho = 0.2
        self.grad_norm_rho_max, self.grad_norm_rho_min = 0.4, 0.2
        self.grad_gamma = 0.03
        # GSAM schedule constants (as in adamw_gsam_*.yaml)
        self.rho_min, self.rho_max, self.alpha = 0.5, 2.0, 0.4
        self.optimizee_epochs = 1000
        self.regularizer = "sam"


def bench_reg_baseline(reg, model, batch, target, steps, device):
    """Time a REAL AdamW+GAM / AdamW+GSAM meta-test step (set_closure +
    update_rho_t + step, exactly the fit_w_normal_optimizer path)."""
    a = _Args(True, 0.05, "AdamW", 1e-3, 0.0, 0.01)
    a.regularizer = reg
    a.train_period = 1000   # scheduler horizon for the timing run
    opt = get_optim_and_schedulers(model, args=a)
    times = []
    for i in range(steps + 5):
        t0 = time.perf_counter()
        loss = F.nll_loss(model(batch), target)
        model.zero_grad()
        loss.backward()
        opt.set_closure(F.nll_loss, batch, target)
        opt.update_rho_t()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 5:
            times.append(time.perf_counter() - t0)
    # analytic pass counts: GAM needs grads at w, w+rho*g_hat, the
    # grad-norm ascent (HVP ~ fwd+bwd) and the combined step -> ~4F+4B
    # upper bound; GSAM is SAM-shaped: 2F+2B. Wall-clock is MEASURED.
    passes = {"gam": {"fwd": 4, "bwd": 4, "lstm": 0},
              "gsam": {"fwd": 2, "bwd": 2, "lstm": 0}}[reg]
    return times, passes


def bench_plain_adamw(model, batch, target, steps, device):
    """Method 8 of the comparison setup: plain AdamW, NO sharpness
    regularizer -> 1 fwd + 1 bwd per step. This branch was MISSING, so the
    manager's `--methods ... adamw_no_reg` died with
    `unknown method adamw_no_reg` (see logs/bench_exp2.err)."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    times = []
    for i in range(steps + 5):
        t0 = time.perf_counter()
        loss = F.nll_loss(model(batch), target)
        model.zero_grad()
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 5:
            times.append(time.perf_counter() - t0)
    return times, {"fwd": 1, "bwd": 1, "lstm": 0}


def bench_adamw_sam(model, batch, target, steps, device):
    sam = SAM(model.parameters(),
              args=_Args(True, 0.05, "AdamW", 1e-3, 0.0, 0.01))
    times = []
    for i in range(steps + 5):
        t0 = time.perf_counter()
        loss = F.nll_loss(model(batch), target)
        loss.backward()
        sam.first_step(zero_grad=True)
        F.nll_loss(model(batch), target).backward()
        sam.second_step(zero_grad=False)
        model.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 5:
            times.append(time.perf_counter() - t0)
    return times, {"fwd": 2, "bwd": 2, "lstm": 0}


def bench_lo(
        model,
        batch,
        target,
        steps,
        device,
        checkpoint=None,
        param_chunk_size=0,
):
    wrap = initialize_model_like(model).to(device)
    mo = MetaOptimizerLSTM(
        MetaModel(wrap),
        LSTM_LAYERS,
        LSTM_HIDDEN,
        param_chunk_size=param_chunk_size,
).to(device)
    if checkpoint:
        path = get_checkpoint_path(checkpoint, experiment="dmo")
        if Path(path).exists():
            mo.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"  [warn] checkpoint {checkpoint} not found — timing a "
                  f"random-init LO (identical cost)")
    mo.reset_lstm(keep_states=False, model=model)
    times = []
    for i in range(steps + 5):
        t0 = time.perf_counter()
        loss = F.nll_loss(model(batch), target)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            mo.meta_update(model)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 5:
            times.append(time.perf_counter() - t0)
    return times, {"fwd": 1, "bwd": 1, "lstm": 1}


_MODEL_SPEC = {}


def initialize_model_like(model):
    return initialize_model(*_MODEL_SPEC["ds_arch"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--methods", nargs="+",
                    default=["adamw_no_reg", "adamw_sam",
                             "adamw_gam", "adamw_gsam",
                             "no_reg", "sam", "gam", "gsam"])
    ap.add_argument("--lo-checkpoints", default="",
                    help="comma list method:checkpoint, e.g. "
                         "sam:SAMLO_mnist_mlp,no_reg:NOREGLO_mnist_mlp")
    ap.add_argument("--param-chunk-size", type=int, default=0)
    ap.add_argument("--out", default="results/benchmark/bench")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit("CUDA not available; pass --allow-cpu for a CPU "
                         "timing run (numbers not paper-grade).")

    _MODEL_SPEC["ds_arch"] = (args.dataset, args.model)
    ckpt_map = dict(kv.split(":") for kv in args.lo_checkpoints.split(",")
                    if ":" in kv)

    # pacs added 2026-07-31 (exp6): 224px RGB, 7 classes — without it the
    # generated bench_exp6* jobs died on shapes[args.dataset] KeyError.
    shapes = {"mnist": (1, 28, 28), "fmnist": (1, 28, 28),
              "cifar10": (3, 32, 32), "cifar100": (3, 32, 32),
              "pacs": (3, 224, 224)}
    from model import DATASET_NUM_CLASSES  # single source of truth
    n_classes = DATASET_NUM_CLASSES.get(args.dataset, 10)
    x = torch.randn(args.batch_size, *shapes[args.dataset], device=device)
    y = torch.randint(0, n_classes, (args.batch_size,), device=device)

    rows = []
    for m in args.methods:
        model = initialize_model(args.dataset, args.model).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if m == "sgd":
            # exp6 (PACS) includes the MetaReg paper baseline: plain SGD,
            # lr 1e-3, momentum 0.9 (sgd_baseline.yaml) -> 1F + 1B.
            # Added 2026-07-31; previously `unknown method sgd`.
            opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
            times = []
            for i in range(args.steps + 5):
                t0 = time.perf_counter()
                loss = F.nll_loss(model(x), y)
                model.zero_grad()
                loss.backward()
                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= 5:
                    times.append(time.perf_counter() - t0)
            passes = {"fwd": 1, "bwd": 1, "lstm": 0}
        elif m == "adamw_no_reg":
            times, passes = bench_plain_adamw(model, x, y, args.steps, device)
        elif m == "adamw_sam":
            times, passes = bench_adamw_sam(model, x, y, args.steps, device)
        elif m in ("adamw_gam", "adamw_gsam"):
            times, passes = bench_reg_baseline(m.split("_", 1)[1], model, x, y,
                                               args.steps, device)
        elif m in LO_METHODS:
            times, passes = bench_lo(
    model,
    x,
    y,
    args.steps,
    device,
    checkpoint=ckpt_map.get(m),
    param_chunk_size=args.param_chunk_size,
)
        else:
            raise SystemExit(f"unknown method {m}")

        fwd_flops = optimizee_fwd_flops(
            initialize_model(args.dataset, args.model).to(device), x)
        opt_flops = (passes["fwd"] + 2 * passes["bwd"]) * (fwd_flops or 0)
        lstm = lstm_flops_per_step(n_params) if passes["lstm"] else 0
        step = statistics.median(times)
        rows.append({
            "method": m,
            "dataset": args.dataset, "model": args.model,
            "batch_size": args.batch_size, "n_params": n_params,
            "passes_per_step": f"{passes['fwd']}F+{passes['bwd']}B"
                               + ("+LSTM" if passes["lstm"] else ""),
            "median_step_ms": round(step * 1e3, 3),
            "mean_step_ms": round(statistics.mean(times) * 1e3, 3),
            "throughput_steps_per_s": round(1.0 / step, 2),
            "throughput_samples_per_s": round(args.batch_size / step, 1),
            "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                                if device.type == "cuda" else None),
            "optimizee_flops_per_step": opt_flops or None,
            "lstm_flops_per_step": lstm,
            "total_flops_per_step": (opt_flops + lstm) if fwd_flops else None,
            "device": device.type,
        })
        print(f"[{m}] {rows[-1]['passes_per_step']:9s} "
              f"median {rows[-1]['median_step_ms']} ms | "
              f"LSTM FLOPs {lstm:,} | total "
              f"{rows[-1]['total_flops_per_step']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"config": vars(args), "rows": rows}, indent=2))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = ["method", "passes_per_step", "median_step_ms",
           "throughput_samples_per_s", "peak_gpu_mem_mb",
           "total_flops_per_step"]
    print("\n| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in hdr) + " |")
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
