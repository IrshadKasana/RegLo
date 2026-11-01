"""
Single entry point:  python -m sharpbench <command>

    check     verify GPU/CUDA/memory and benchmark this machine
    fetch     download MNIST (needs internet; run once)
    train     one training run, for debugging
    sweep     run a grid (resumable, shardable)
    analyze   selection report + significance table + LaTeX/CSV
    plot      generate all figures
"""

from __future__ import annotations

import argparse
import json
import sys


# --------------------------------------------------------------------------- #
def cmd_check(a):
    import time
    import torch
    import torch.nn.functional as F
    from .device import (resolve_device, device_info, configure_backends,
                         reset_peak_memory, peak_memory_mb)
    from .models import make_model, count_params
    from .optimizers import build_optimizer

    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    print(f"  python         : {sys.version.split()[0]}")
    print(f"  torch          : {torch.__version__}")
    print(f"  torch CUDA     : {torch.version.cuda}")
    print(f"  cuda available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device count   : {torch.cuda.device_count()}")

    device = resolve_device(a.device)
    info = device_info(device)
    print(f"  selected       : {info}")
    if not info.is_cuda:
        print("\n  !! Running on CPU. If you expected a GPU, you probably")
        print("     installed the CPU-only wheel. Reinstall with:")
        print("     pip install torch torchvision --index-url \\")
        print("         https://download.pytorch.org/whl/cu121\n")

    configure_backends(deterministic=True, allow_tf32=False)
    print(f"  TF32           : disabled (reproducibility; see device.py)")

    m = make_model(0, device, hidden=a.hidden)
    n = count_params(m)
    ok = "OK" if (a.hidden != 20 or n == 15910) else "!! expected 15,910"
    print(f"  model params   : {n:,}  {ok}")

    print("\n" + "=" * 70)
    print(f"THROUGHPUT  ({55000 // a.batch_size} steps = 1 epoch at "
          f"batch {a.batch_size})")
    print("=" * 70)

    x = torch.randn(55000, 784, device=device)
    y = torch.randint(0, 10, (55000,), device=device)
    reset_peak_memory(device)
    cost = {}

    for reg, hp in [("none", {}),
                    ("sam", {"rho": 0.05, "adaptive": False}),
                    ("gsam", {"rho": 0.05, "alpha": 0.1}),
                    ("gam", {"grad_rho": 0.04, "grad_norm_rho": 0.2,
                             "grad_gamma": 0.03, "adaptive": True})]:
        m = make_model(0, device, hidden=a.hidden)
        opt = build_optimizer(reg, m.parameters(),
                              dict(lr=1e-3, weight_decay=0.01, **hp))
        idx = torch.randperm(55000).to(device)

        def one_pass(limit):
            for i in range(0, limit, a.batch_size):
                s = idx[i:i + a.batch_size]
                xb, yb = x[s], y[s]
                opt.step(lambda: _bwd(m, xb, yb))

        one_pass(a.batch_size * 10)                    # warm up CUDA kernels
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        one_pass(55000 - a.batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        d = time.time() - t
        cost[reg] = d
        print(f"  {reg:5s} {d:6.2f} s/epoch    {d*100/60:6.2f} min "
              f"per 100-epoch run")

    peak = peak_memory_mb(device)
    if device.type == "cuda":
        print(f"\n  peak GPU memory: {peak:.0f} MB "
              f"(dataset {220:.0f} MB + model/optimizer)")
        usable = info.free_mb * 0.85
        print(f"  concurrent shards this card can hold: "
              f"~{max(1, int(usable // max(peak, 1)))}")

    p100 = {k: v * 100 for k, v in cost.items()}
    stage1 = 3 * 3 * (1 * p100["none"] + 6 * p100["sam"] + 5 * p100["sam"]
                      + 5 * p100["gam"] + 16 * p100["gsam"])
    stage2 = 10 * (p100["none"] + 2 * p100["sam"] + p100["gam"] + p100["gsam"])
    print("\n" + "=" * 70)
    print("SWEEP TIME ESTIMATE  (configs/sweep_grid.yaml, 100 epochs)")
    print("=" * 70)
    print(f"  stage 1  (~297 runs, 3 seeds)  : {stage1/3600:5.1f} h serial")
    print(f"  stage 2  (~50 runs, 10 seeds)  : {stage2/3600:5.1f} h serial")
    print(f"  TOTAL with 4 shards            : "
          f"{(stage1/4 + stage2)/3600:5.1f} h wall clock")
    print("\n  Too long? Set epochs: 50 in the config, or start with")
    print("  configs/quick.yaml to validate the pipeline first.")


def _bwd(m, xb, yb):
    import torch.nn.functional as F
    loss = F.cross_entropy(m(xb), yb)
    loss.backward()
    return loss


# --------------------------------------------------------------------------- #
def cmd_fetch(a):
    from torchvision import datasets
    datasets.MNIST(a.data_root, train=True, download=True)
    datasets.MNIST(a.data_root, train=False, download=True)
    print(f"MNIST ready under {a.data_root}")


def cmd_train(a):
    from .train import run
    cfg = {k: v for k, v in vars(a).items()
           if k not in ("func", "command", "out")}
    res = run(cfg, progress=True)
    slim = {k: v for k, v in res.items() if k != "history"}
    print(json.dumps(slim, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {a.out}")


def cmd_sweep(a):
    from .sweep import sweep
    sweep(a.config, a.out, shard=a.shard, stage1=a.stage1, stage2=a.stage2,
          device_spec=a.device, arms=a.arms, dry_run=a.dry_run,
          memory_fraction=a.memory_fraction, allow_synthetic=a.allow_synthetic)


def cmd_analyze(a):
    from .analyze import report
    report(a.results, control=a.control, top_k=a.top_k,
           emit_winners=a.emit_winners, latex=a.latex, csv=a.csv)


def cmd_plot(a):
    from .plots import make_all
    make_all(a.results, outdir=a.outdir, control=a.control, min_acc=a.min_acc)


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m sharpbench",
        description="Sharpness-aware regularizer benchmark (SAM / GAM / GSAM)")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="verify environment and benchmark")
    c.add_argument("--device", default="auto")
    c.add_argument("--hidden", type=int, default=20)
    c.add_argument("--batch-size", type=int, default=128)
    c.set_defaults(func=cmd_check)

    c = sub.add_parser("fetch", help="download MNIST")
    c.add_argument("--data-root", default="./data")
    c.set_defaults(func=cmd_fetch)

    c = sub.add_parser("train", help="one run")
    c.add_argument("--regularizer", default="none",
                   choices=["none", "sam", "gam", "gsam", "wsam"])
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--epochs", type=int, default=100)
    c.add_argument("--max-steps", type=int, default=None)
    c.add_argument("--batch-size", type=int, default=128)
    c.add_argument("--lr", type=float, default=1e-3)
    c.add_argument("--weight-decay", type=float, default=0.01)
    c.add_argument("--hidden", type=int, default=20)
    c.add_argument("--act", default="relu")
    c.add_argument("--lr-schedule", default="cosine",
                   choices=["cosine", "constant", "step"])
    c.add_argument("--warmup-steps", type=int, default=0)
    c.add_argument("--label-noise", type=float, default=0.0)
    c.add_argument("--grad-clip", type=float, default=None)
    c.add_argument("--rho", type=float, default=0.05)
    c.add_argument("--alpha", type=float, default=0.1)
    c.add_argument("--gamma", type=float, default=None,
                   help="WSAM sharpness weight in [0,1); lambda = g/(1-g)")
    c.add_argument("--lam", type=float, default=None,
                   help="WSAM weight as lambda directly (lam=1 is exactly SAM)")
    c.add_argument("--coupled", dest="decouple", action="store_false",
                   help="Coupled-WSAM ablation (default: weight-decoupled)")
    c.add_argument("--adaptive", action="store_true")
    c.add_argument("--m-sharpness", type=int, default=None)
    c.add_argument("--grad-rho", type=float, default=0.04)
    c.add_argument("--grad-norm-rho", type=float, default=0.2)
    c.add_argument("--grad-gamma", type=float, default=0.03)
    c.add_argument("--measure-hessian", action="store_true")
    c.add_argument("--data-root", default="./data")
    c.add_argument("--device", default="auto")
    c.add_argument("--allow-synthetic", action="store_true")
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_train)

    c = sub.add_parser("sweep", help="run a grid")
    c.add_argument("--config", default="configs/sweep_grid.yaml")
    c.add_argument("--out", default="results/sweep.jsonl")
    c.add_argument("--shard", default="0/1", help="i/n")
    c.add_argument("--stage1", action="store_true",
                   help="first 3 seeds only (hyperparameter search)")
    c.add_argument("--stage2", default=None, metavar="WINNERS_YAML")
    c.add_argument("--arms", nargs="*", default=None)
    c.add_argument("--device", default="auto")
    c.add_argument("--memory-fraction", type=float, default=None)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--allow-synthetic", action="store_true")
    c.set_defaults(func=cmd_sweep)

    c = sub.add_parser("analyze", help="selection report + final table")
    c.add_argument("--results", nargs="+", required=True)
    c.add_argument("--control", default="adamw")
    c.add_argument("--top-k", type=int, default=3)
    c.add_argument("--emit-winners", default=None)
    c.add_argument("--latex", default=None)
    c.add_argument("--csv", default=None)
    c.set_defaults(func=cmd_analyze)

    c = sub.add_parser("plot", help="generate all figures")
    c.add_argument("--results", nargs="+", required=True)
    c.add_argument("--outdir", default="figures")
    c.add_argument("--control", default="adamw")
    c.add_argument("--min-acc", type=float, default=90.0,
                   help="exclude diverged runs below this test accuracy "
                        "from the flatness correlation")
    c.set_defaults(func=cmd_plot)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
