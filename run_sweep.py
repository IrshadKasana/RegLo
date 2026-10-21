#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sweep.py
============
Hyperparameter sweep over the L_smooth vs L_reg proportion (reviewer request:
"provide ablations: remove L_smooth; vary lambda_reg, lambda_smooth, rho").

For every (smooth_lamb, reg_strength) combination this driver:
  1. meta-trains an LO:   experiments/dmo/train.py    --regularizer R --set ...
  2. evaluates it:        experiments/dmo/evaluate_optimizee.py --set ...
and tags results so generate_paper_plots.py can build the sweep heatmap.

The "strength" knob differs per regularizer:
    sam / sam_opzee -> rho          (SAM perturbation radius)
    l2              -> optzee_wd    (L2 coefficient on the optimizee)
    gsam            -> rho_max      (rho_min is scaled proportionally)
    gam             -> grad_rho_max (grad_norm_rho_max scaled proportionally)

smooth_lamb = 0.0 disables L_smooth entirely (if_smooth=False), which IS the
"remove L_smooth" ablation - so include 0.0 in --smooth-values.

Usage (local, sequential - fine for mlp/mnist scale):
    python run_sweep.py --regularizer sam \
        --smooth-values 0.0 0.1 0.5 1.0 \
        --reg-values 0.01 0.05 0.1 0.5 \
        --dataset mnist --optimizee-model mlp --mode local

Usage (PBS - generates one job script per combo into pbs_scripts/sweep/):
    python run_sweep.py --regularizer sam \
        --smooth-values 0.0 0.1 0.5 1.0 \
        --reg-values 0.01 0.05 0.1 0.5 \
        --dataset mnist --optimizee-model mlp --mode pbs
    # then:  for f in pbs_scripts/sweep/*.pbs; do qsub $f; done
"""
from __future__ import annotations
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STRENGTH_KEY = {
    "sam": "rho",
    "sam_opzee": "rho",
    "l2": "optzee_wd",
    "gsam": "rho_max",
    "gam": "grad_rho_max",
}

def _short_jobname(tag, limit=15):
    """PBS -N is length-limited; keep the full tag on the .out/.err files."""
    s = (tag.replace("sw_", "").replace("_grad_rho_max", "gr")
            .replace("_rho_max", "rm").replace("_rho", "r")
            .replace("_sl", "s").replace("_lr", "l").replace("_se", "e")
            .replace("0p", ""))
    return ("sw" + s)[:limit]


PBS_TEMPLATE = """#!/bin/bash
#==========================
# PBS Configuration
#==========================
#PBS -N {short_name}
#PBS -q {queue}
#PBS -l select={select}
#PBS -l walltime={walltime}
#PBS -o logs/{job_name}.out
#PBS -e logs/{job_name}.err
#PBS -V

#==========================
# Enter Project Directory
#==========================
cd "$PBS_O_WORKDIR"
mkdir -p logs

#==========================
# Environment
#==========================
module purge
module load {cuda_module}
export PYTHONUNBUFFERED=1
PY="{python}"

#==========================
# GPU Validation
#==========================
echo "========================================"
echo "Sweep Cell  : {job_name}"
echo "Job Started : $(date)"
echo "Host        : $(hostname)"
echo "Directory   : $PWD"
echo "========================================"
nvidia-smi || echo "WARNING : nvidia-smi not found"

$PY - <<'PYCHECK'
import torch
print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("Available :", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not visible - torch/CUDA module mismatch. Fix the env on the LOGIN node before resubmitting."
print("GPU :", torch.cuda.get_device_name(0))
print("Device Count :", torch.cuda.device_count())
PYCHECK
if [ $? -ne 0 ]; then
    echo "GPU validation FAILED - aborting before walltime is consumed."
    exit 1
fi

#==========================
# Run Sweep Cell (meta-train -> evaluate)
#==========================
{command}
rc=$?

#==========================
# Exit
#==========================
echo "========================================"
echo "Finished    : $(date)"
echo "Return Code : $rc"
echo "========================================"
echo "NOTE: if the log ends with WALLTIME_EXIT, qsub this SAME script again."
echo "      When ALL cells are done: python pick_best_lambdas.py --apply"
exit $rc
"""



def fmt(v):
    """Decimal-safe token: 0.05 -> '0p05', 1e-05 -> '0p00001' (never
    scientific notation, which the plot regex silently dropped — item 46)."""
    s = ('%.12f' % float(v)).rstrip('0').rstrip('.')
    if s in ('', '-'):
        s = '0'
    return s.replace('.', 'p').replace('-', 'm')


def build_commands(reg, smooth, strength, ds, model, args, lam_reg=None,
                   smooth_eps=None):
    key = STRENGTH_KEY[reg]
    tag = f"sw_{reg}_sl{fmt(smooth)}_{key}{fmt(strength)}"
    if lam_reg is not None:
        tag += f"_lr{fmt(lam_reg)}"
    if smooth_eps is not None:
        tag += f"_se{fmt(smooth_eps)}"
    lo_name = f"SWEEP_{tag}_{ds}_{model}"

    train_overrides = [
        f"dataset={ds}",
        f"optimizee_model_name={model}",
        f"model_name={lo_name}",
        f"smooth_lamb={smooth}",
        f"if_smooth={'True' if smooth > 0 else 'False'}",
        f"{key}={strength}",
        # if_pretrained / pretrained_model_name are YAML defaults in every
        # regularizer config — no need to restate them here.
    ]
    if getattr(args, "max_hours", 0):
        train_overrides += [f"max_hours={args.max_hours}", "resume=True"]
    if lam_reg is not None:
        # design-v2 Phase-1 axis: sweep the Eq. 7 penalty weight in the
        # consolidated objective (Change.MD items 75c, 77)
        train_overrides += [f"lambda_reg={lam_reg}", "reg_mode=penalty"]
        # BUGFIX (2026-07-31): penalty mode (consolidated Alg. 1) reads the
        # shared ascent radius from `rho` for ALL three regularizers —
        # gsam's rho_max / gam's grad_rho_max are ONLY read by the legacy
        # ascent path and the meta-test optimizers. Without this line the
        # sweep's declared strength (e.g. rho0p005 in the tag) never reached
        # the objective and the ascent silently used the code default 0.05.
        train_overrides += [f"rho={strength}"]
    if smooth_eps is not None:
        train_overrides += [f"smooth_eps={smooth_eps}"]
    if reg == "gsam":
        train_overrides.append(f"rho_min={strength / 4.0}")
    if reg == "gam":
        train_overrides.append(f"grad_rho_min={strength / 2.0}")
        train_overrides.append(f"grad_norm_rho_max={strength * 10.0}")
        train_overrides.append(f"grad_norm_rho_min={strength * 5.0}")
    if args.train_set:
        train_overrides += args.train_set

    _prof = f" --profile {args.profile}" if getattr(args, "profile", None) else ""
    train_cmd = (
        f"{args.python} experiments/dmo/train.py --regularizer {reg}{_prof} "
        f"--set {' '.join(train_overrides)}"
    )

    eval_overrides = []
    if args.eval_set:
        eval_overrides += args.eval_set
    eval_cmd = (
        f"{args.python} experiments/dmo/evaluate_optimizee.py "
        f"--saved_model {lo_name} --dataset {ds} --optimizee_model {model} "
        f"--eval_tag {tag} --lo_train_dataset {ds}{_prof}"
    )
    if eval_overrides:
        eval_cmd += f" --set {' '.join(eval_overrides)}"

    return tag, train_cmd, eval_cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regularizer", "--regularizers", dest="regularizers",
                    nargs="+", required=True, choices=sorted(STRENGTH_KEY),
                    help="one or MORE regularizers, e.g. --regularizer sam gsam gam")
    ap.add_argument("--grid", choices=["full", "coarse"], default=None,
                    help="preset lambda grids. 'full' = the 6x6 paper grid "
                         "(108 PBS cells over 3 regularizers). 'coarse' = "
                         "log-spaced 4x4 {0, 0.01, 0.1, 1.0}^2 (48 cells, "
                         "-55%%): SAM-family strengths are conventionally "
                         "tuned on coarse log grids, the 0-anchors keep the "
                         "remove-L_smooth / remove-L_reg ablations, and the "
                         "heatmap still renders. Explicit --*-values "
                         "override the preset axis-by-axis.")
    ap.add_argument("--smooth-values", type=float, nargs='+', required=False,
                    help="lambda_smooth grid; include 0.0 for the no-smoothing ablation")
    ap.add_argument("--smooth-eps-values", type=float, nargs='+', default=None,
                    help="optional smooth_eps (PGA radius of the smoothing "
                         "term) grid — 'other required hyperparameters'")
    ap.add_argument("--lambda-reg-values", type=float, nargs='+', default=None,
                    help="optional lambda_reg grid (penalty mode); crossed "
                         "with --smooth-values and --reg-values")
    ap.add_argument("--reg-values", type=float, nargs='+', required=False,
                    help="regularizer-strength grid (rho / optzee_wd / rho_max / grad_rho_max)")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--optimizee-model", default="mlp")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help=">0: forwarded to every cell as max_hours=<v> resume=True "
                         "so 24h-walltime kills resume on resubmission")
    ap.add_argument("--mode", choices=["local", "pbs", "dry"], default="dry",
                    help="local = run sequentially now; pbs = emit job scripts; dry = print only")
    ap.add_argument("--queue", default="gpu-ext")
    ap.add_argument("--select", default="1:ncpus=8:mem=60gb",
                    help="PBS -l select line (gpu-ext grants the GPU)")
    ap.add_argument("--cuda-module", default="compilers/cuda/11.1")
    ap.add_argument("--pbs-python",
                    default="$HOME/.conda/envs/learn_reg/bin/python",
                    help="interpreter used INSIDE PBS scripts")
    ap.add_argument("--walltime", default="24:00:00")
    ap.add_argument("--python", default=str(PYTHON),
                    help="python executable used inside jobs "
                         "(e.g. ~/.conda/envs/learn_reg/bin/python)")
    ap.add_argument("--profile", default=None,
                    help="scale profile from configs/dmo/profiles/ "
                         "(production | smoke): keeps epochs/steps/periods "
                         "in YAML instead of on the command line. The swept "
                         "lambda/rho values stay on the CLI — varying them "
                         "per cell IS the sweep.")
    ap.add_argument("--train-set", nargs='*', default=[], metavar="KEY=VALUE",
                    help="extra --set overrides forwarded to meta-training "
                         "(e.g. max_epochs=2000 for a shorter sweep budget)")
    ap.add_argument("--eval-set", nargs='*', default=[], metavar="KEY=VALUE",
                    help="extra --set overrides forwarded to evaluation "
                         "(e.g. n_training=5 optimizee_epochs=1000)")
    args = ap.parse_args()

    # --grid presets fill any axis the user did not set explicitly.
    _PRESETS = {
        "full":   dict(smooth=[0.0, 0.001, 0.01, 0.1, 0.5, 1.0],
                       lam=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0], reg=[0.005]),
        "coarse": dict(smooth=[0.0, 0.01, 0.1, 1.0],
                       lam=[0.0, 0.01, 0.1, 1.0], reg=[0.005]),
    }
    if args.grid:
        g = _PRESETS[args.grid]
        args.smooth_values = args.smooth_values or g["smooth"]
        args.reg_values = args.reg_values or g["reg"]
        if args.lambda_reg_values is None:
            args.lambda_reg_values = g["lam"]
    if not args.smooth_values or not args.reg_values:
        ap.error("--smooth-values and --reg-values are required "
                 "(or pass --grid full|coarse)")

    lam_grid = args.lambda_reg_values if args.lambda_reg_values else [None]
    se_grid = args.smooth_eps_values if args.smooth_eps_values else [None]
    combos = [(reg, sm, st, lg, se)
              for reg in args.regularizers
              for se in se_grid
              for lg in lam_grid
              for (sm, st) in itertools.product(args.smooth_values, args.reg_values)]
    axis = "regularizer x smooth_lamb x strength" + (" x lambda_reg" if args.lambda_reg_values else "")
    print(f"Sweep: {len(combos)} combos of ({axis}) over {args.regularizers}")

    if args.mode == "pbs":
        args.python = '"$PY"'      # cluster conda env, defined by the template
    pbs_dir = ROOT / "pbs_scripts" / "sweep"
    if args.mode == "pbs":
        pbs_dir.mkdir(parents=True, exist_ok=True)

    for reg, smooth, strength, lam_reg, smooth_eps in combos:
        tag, train_cmd, eval_cmd = build_commands(
            reg, smooth, strength, args.dataset,
            args.optimizee_model, args, lam_reg=lam_reg,
            smooth_eps=smooth_eps)

        print(f"\n### {tag}")
        print(f"  train: {train_cmd}")
        print(f"  eval : {eval_cmd}")

        if args.mode == "local":
            subprocess.run(train_cmd, shell=True, check=True, cwd=str(ROOT))
            subprocess.run(eval_cmd, shell=True, check=True, cwd=str(ROOT))
        elif args.mode == "pbs":
            script = PBS_TEMPLATE.format(
                job_name=tag, short_name=_short_jobname(tag),
                queue=args.queue, select=args.select,
                walltime=args.walltime, cuda_module=args.cuda_module,
                python=args.pbs_python,
                command=f"{train_cmd} \\\n  && {eval_cmd}")
            out = pbs_dir / f"{tag}.pbs"
            out.write_text(script)
            print(f"  wrote {out}")

    if args.mode == "pbs":
        print(f"\nSubmit with:\n  for f in {pbs_dir}/*.pbs; do qsub $f; done")
    elif args.mode == "dry":
        print("\n(dry run - nothing executed; use --mode local or --mode pbs)")


if __name__ == "__main__":
    main()