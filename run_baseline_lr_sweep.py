#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baseline_lr_sweep.py
========================
Closes the two reviewer-fatal gaps of fixed-horizon (1000-step) comparison:

  (2) "Each baseline must be TUNED for that horizon."  AdamW's loss was
      still visibly descending at step 1000 with lr=1e-3 — i.e. the lr was
      chosen for a longer budget. This driver sweeps lr per baseline under
      the SAME horizon so each baseline is reported at its own best setting.

  (3) "The LO's horizon advantage must be stated out loud."  The LO is
      meta-trained on unrolls that top out at exactly the scoring horizon.
      --horizon N emits ONE out-of-horizon evaluation per method (e.g.
      N=3000) so the paper can show whether the gap holds beyond the
      meta-training horizon — either outcome is publishable.

Usage (Agastya):
    python run_baseline_lr_sweep.py --experiment exp2 --mode pbs
    for f in pbs_scripts/baseline_lr/*.pbs; do qsub $f; done
    python run_baseline_lr_sweep.py --experiment exp2 --report
    # then re-run each baseline's FINAL eval with its winning lr, e.g.:
    #   ... evaluate_optimizee ... --eval_tag exp2_adamw_sam --set lr=<best>

    python run_baseline_lr_sweep.py --experiment exp2 --horizon 3000 --mode pbs
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "learning_regularizers" / "dmo"))

from experiment_matrix import (EXPERIMENTS, BENCHMARK_CONFIG, LO_METHODS,
                               ALL_METHODS, lo_checkpoint_name)
from run_all_experiments import PBS_TEMPLATE, pbs_short_name


def fmt(v):
    s = ('%.12f' % float(v)).rstrip('0').rstrip('.')
    return (s or '0').replace('.', 'p').replace('-', 'm')


def build_jobs(args):
    e = EXPERIMENTS[args.experiment]
    te_ds, te_arch = e["test_dataset"], e["test_arch"]
    tr_ds, tr_arch = e["train_dataset"], e["train_arch"]
    ov = dict(e.get("eval_overrides") or {})
    ov["max_hours"] = 23.0
    jobs = []

    def setstr(extra):
        d = dict(ov); d.update(extra)
        d["experiment_name"] = args.experiment
        return " ".join(f"{k}={v}" for k, v in d.items())

    # ---- (2) lr grid for every AdamW/SGD baseline ----
    if not args.horizon:
        for m in args.methods:
            bcfg = BENCHMARK_CONFIG[m.split("_", 1)[1] if "_" in m else m]
            base = "SGD" if m == "sgd" else "AdamW"
            for lr in args.lrs:
                tag = f"{args.experiment}_{m}_lr{fmt(lr)}"
                cmd = (f"{args.python} experiments/dmo/evaluate_optimizee.py "
                       f"--if_normal_optim --regularizer {bcfg} "
                       f"--base_optimizer {base} --dataset {te_ds} "
                       f"--optimizee_model {te_arch} --profile {args.profile} "
                       f"--eval_tag {tag} --set "
                       + setstr({"experiment_group": "lr_sweep", "lr": lr}))
                jobs.append((tag, cmd))

    # ---- (3) one out-of-horizon point for ALL methods ----
    else:
        n = int(args.horizon)
        for m in ALL_METHODS:
            tag = f"{args.experiment}_{m}_h{n}"
            if m in LO_METHODS:
                ckpt = lo_checkpoint_name(m, tr_ds, tr_arch)
                cmd = (f"{args.python} experiments/dmo/evaluate_optimizee.py "
                       f"--saved_model {ckpt} --dataset {te_ds} "
                       f"--optimizee_model {te_arch} "
                       f"--lo_train_dataset {tr_ds} --lo_train_arch {tr_arch} "
                       f"--profile {args.profile} --eval_tag {tag} --set "
                       + setstr({"experiment_group": "horizon",
                                 "optimizee_epochs": n}))
            else:
                bcfg = BENCHMARK_CONFIG[m.split("_", 1)[1]]
                cmd = (f"{args.python} experiments/dmo/evaluate_optimizee.py "
                       f"--if_normal_optim --regularizer {bcfg} "
                       f"--base_optimizer AdamW --dataset {te_ds} "
                       f"--optimizee_model {te_arch} --profile {args.profile} "
                       f"--eval_tag {tag} --set "
                       + setstr({"experiment_group": "horizon",
                                 "optimizee_epochs": n}))
            jobs.append((tag, cmd))
    return jobs


def report(args):
    """Pick each baseline's best lr by FINAL VALIDATION accuracy (never by
    test accuracy — that would be model selection on the test set)."""
    best = defaultdict(lambda: (None, -1.0))
    for f in glob.glob(str(ROOT / "artifacts/dmo/results/json/*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        meta = d.get("metadata", {})
        tag = str(meta.get("eval_tag") or "")
        pre = f"{args.experiment}_"
        if not (tag.startswith(pre) and "_lr" in tag):
            continue
        m, lr_tok = tag[len(pre):].rsplit("_lr", 1)
        vals = [c[-1] for c in (d.get("instances_acc_val") or []) if c]
        if not vals:
            vals = [c[-1] for c in (d.get("instances_acc_ts") or []) if c]
            src = "TEST(!)"
        else:
            src = "val"
        score = sum(vals) / len(vals)
        if score > best[m][1]:
            best[m] = (lr_tok.replace("p", ".").replace("m", "-"), score, src)
    if not best:
        print("no *_lr* tagged results found — run the sweep first")
        return
    print(f"{'method':<16}{'best lr':<10}{'val acc':<10}selected on")
    for m, (lr, sc, src) in sorted(best.items()):
        print(f"{m:<16}{lr:<10}{sc:<10.4f}{src}")
        print(f"    final eval: re-run the exp eval for '{m}' with "
              f"--set lr={lr} (tag {args.experiment}_{m})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="exp2",
                    choices=[k for k in EXPERIMENTS if not
                             EXPERIMENTS[k].get("sweep")])
    ap.add_argument("--methods", nargs="+",
                    default=["adamw_no_reg", "adamw_sam", "adamw_gam",
                             "adamw_gsam"])
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--horizon", type=int, default=0,
                    help=">0: emit ONE eval per method at this many "
                         "optimizee steps (out-of-horizon check) instead "
                         "of the lr grid")
    ap.add_argument("--profile", default="production")
    ap.add_argument("--mode", choices=["dry", "pbs"], default="dry")
    ap.add_argument("--report", action="store_true",
                    help="pick each baseline's best lr from finished runs")
    ap.add_argument("--python", default='"$PY"')
    ap.add_argument("--queue", default="gpu-ext")
    ap.add_argument("--select", default="1:ncpus=8:mem=60gb")
    ap.add_argument("--walltime", default="24:00:00")
    ap.add_argument("--cuda-module", default="compilers/cuda/11.1")
    ap.add_argument("--pbs-python",
                    default="$HOME/.conda/envs/learn_reg/bin/python")
    args = ap.parse_args()

    if args.report:
        return report(args)

    jobs = build_jobs(args)
    kind = f"horizon h={args.horizon}" if args.horizon else "baseline lr grid"
    print(f"{kind}: {len(jobs)} jobs for {args.experiment}")
    if args.mode == "dry":
        for name, cmd in jobs:
            print(f"\n[{name}]\n  {cmd}")
        return
    outdir = ROOT / "pbs_scripts" / "baseline_lr"
    outdir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    for name, cmd in jobs:
        (outdir / f"{name}.pbs").write_text(PBS_TEMPLATE.format(
            job_name=name, short_name=pbs_short_name(name),
            queue=args.queue, select=args.select, walltime=args.walltime,
            cuda_module=args.cuda_module, python=args.pbs_python,
            command=cmd))
    print(f"wrote {len(jobs)} PBS scripts -> {outdir}")
    print(f"submit:  for f in {outdir}/*.pbs; do qsub $f; done")


if __name__ == "__main__":
    main()
