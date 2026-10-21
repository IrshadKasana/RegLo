"""
run_all_experiments.py
======================
The unified experiment manager (the "one command" driver).

    Read Experiment Matrix -> Train -> Evaluate -> Benchmark -> Report

Examples
--------
# see everything that WOULD run for the whole matrix:
python run_all_experiments.py --mode dry

# smoke the pipeline locally on experiment 1, SAM LO + benchmark only:
python run_all_experiments.py --experiments 1 --methods sam adamw_sam \
    --mode local --smoke

# generate ready-to-qsub PBS jobs for experiments 1,2,5,6 at production scale
# (this REPLACES the hand-written setup_*.py / ev_*.pbs zoo):
python run_all_experiments.py --experiments 1 2 5 6 --mode pbs --production

Phases (all by default): train, eval, bench, report.
LO trainings shared by several experiments (e.g. SAMLO_mnist_mlp feeds
exp1, exp3 and exp7) are deduplicated automatically.
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

from experiment_matrix import (EXPERIMENTS, iter_experiments,
                               lo_checkpoint_name, lo_train_config,
                               BENCHMARK_CONFIG, LO_METHODS, ALL_METHODS,
                               COMPARISON_SETS)

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# ======================================================================
# Agastya PBS Pro template (gpu-ext / Tesla V100 32GB / CUDA 11.1 /
# conda env `learn_reg`: Python 3.8 + PyTorch 1.9.0+cu111).
#
# Matches the user's working script exactly:
#   * NO ngpus= on the select line — Agastya's PBS Pro does not define that
#     resource (`qsub: Unknown resource: ngpus`); the gpu-ext queue grants
#     the V100. This matches the user's proven-working submission exactly.
#   * module purge before loading CUDA, so the login env can't leak in
#   * PY points at the conda interpreter (never the login node's python3)
#   * GPU validation via heredoc, and — unlike the bare template — the
#     result is CHECKED: a bad env aborts in seconds instead of burning
#     24h of walltime on CPU.
# Walltime is 24h while --production sets max_hours=23, so every job exits
# cleanly with an hour to spare and RESUMES when the same script is
# re-submitted the next day.
# ======================================================================
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
echo "Job Started : $(date)"
echo "Job Name    : {job_name}"
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
# Run Experiment
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
echo "NOTE: if the log ends with WALLTIME_EXIT, just qsub this SAME script"
echo "      again — training resumes from the last epoch-block and"
echo "      evaluation from the last completed instance."
exit $rc
"""


# PBS -N is length-limited on many schedulers; abbreviate aggressively but
# keep the full name on the .out/.err files so logs stay greppable.
_ABBREV = [("train_", "t_"), ("eval_", "e_"), ("bench_", "b_"),
           ("report_", "r_"), ("cifar10c", "c10c"), ("cifar10", "c10"),
           ("mnist", "mn"), ("fmnist", "fmn"), ("resnet18", "r18"),
           ("resnet8", "r8"), ("vgg16", "v16"), ("adamw_", "aw_"),
           ("no_reg", "nr"), ("LO_", ""), ("_baseline", "")]


def pbs_short_name(name, limit=15):
    s = name
    for a, b in _ABBREV:
        s = s.replace(a, b)
    return s[:limit]


# ----------------------------------------------------------------------
# Run SCALE now lives in YAML, not here:
#     configs/dmo/profiles/{production,smoke}_{train,eval}.yaml
# --production / --smoke simply select the profile, so epochs, steps,
# periods, curriculum, seeds, patience and Hessian flags are editable
# WITHOUT touching Python — and the generated PBS scripts carry only
# scheduler settings plus model_name / dataset / architecture / resume /
# max_hours.  Merge order:
#     base yaml < regularizer yaml < profile yaml < --set  (--set wins)
# ----------------------------------------------------------------------
def profile_flag(args):
    if args.smoke:
        return " --profile smoke"
    if args.production:
        return " --profile production"
    return ""


def fmt_set(d):
    return " ".join(f"{k}={v}" for k, v in d.items())


def build_jobs(args):
    """Return (train_jobs, eval_jobs, bench_jobs, report_jobs); each job is
    (job_name, command). Train jobs are deduplicated on the checkpoint."""
    train_jobs, eval_jobs, bench_jobs, report_jobs = {}, [], [], []

    for key, e in iter_experiments(args.experiments):
        if e.get("sweep"):
            if args.smoke:
                # The lambda sweep is a production-scale search (dozens of
                # cells x full meta-training). At smoke scale it would
                # dominate the run and prove nothing, so we skip it and let
                # exp2-exp5 use the DEFAULT lambdas already committed in
                # configs/dmo/regularizers/*.yaml.
                print(f"[{key}] SKIPPED under --smoke — smoke uses the "
                      f"default lambdas from configs/dmo/regularizers/*.yaml. "
                      f"Run the sweep at production scale, then "
                      f"`python pick_best_lambdas.py --apply`.")
                continue
            if "eval" in args.phases or "train" in args.phases:
                # BUGFIX (2026-07-31): the sweep delegation never forwarded
                # the scale profile, so every generated sweep cell meta-
                # trained at the BASE (smoke-ish) yaml scale
                # (optimizee_steps=40, max_epochs=4, train_period=2,
                # if_cl=False) and evaluated at 60 epochs / 2 instances —
                # the frozen "winners" would have been noise. run_sweep.py
                # already accepts --profile and forwards it to BOTH the
                # train and eval commands of every cell.
                _sweep_cmd = e["command"].format(
                    mode=args.mode if args.mode != "dry" else "dry")
                if args.production:
                    _sweep_cmd += " --profile production"
                eval_jobs.append((f"{key}_sweep", _sweep_cmd))
            continue

        methods = [m for m in e["methods"] if m in args.methods]
        tr_ds, te_ds = e["train_dataset"], e["test_dataset"]
        tr_arch, te_arch = e["train_arch"], e["test_arch"]

        for m in methods:
            # ---------------- TRAIN (LO methods only, deduped) -------------
            if m in LO_METHODS:
                ckpt = lo_checkpoint_name(m, tr_ds, tr_arch)
                if ckpt not in train_jobs:
                    cfg = lo_train_config(m, tr_ds)
                    # ONLY cell identity + the walltime trio go on the CLI;
                    # if_pretrained/pretrained_model_name are YAML defaults
                    # in every regularizer config (verified), so inlining
                    # them was pure noise.
                    sets = {"model_name": ckpt, "dataset": tr_ds,
                            "optimizee_model_name": tr_arch}
                    bw = getattr(args, "_best_lambdas", {}).get(m)
                    if bw:                    # optional audit trail; the
                        # frozen lambdas are ALREADY in the regularizer YAML
                        # via `pick_best_lambdas.py --apply`.
                        sets = {**sets, **{k: v for k, v in bw.items()
                                           if k not in ("selected_by", "score",
                                                        "tag", "json")}}
                    cmd = (f"{args.python} experiments/dmo/train.py "
                           f"--regularizer {cfg}{profile_flag(args)} "
                           f"--set {fmt_set(sets)}"
                           + (f" {' '.join(args.train_set)}" if args.train_set else ""))
                    train_jobs[ckpt] = (f"train_{ckpt}", cmd)

            # ---------------- EVALUATE -------------------------------------
            tag = f"{key}_{m}"
            common = {"experiment_name": key, "experiment_group": e["group"],
                      **e["eval_overrides"]}
            if m.startswith("adamw_") or m == "sgd":
                _breg = (m.split("_", 1)[1] if "_" in m else m)
                bcfg = BENCHMARK_CONFIG[_breg]   # no_reg|sam|gam|gsam|sgd
                _base = "SGD" if m == "sgd" else "AdamW"
                # --base_optimizer must be passed EXPLICITLY: it is one of the
                # few keys that exists both as a CLI arg and in the YAML, and
                # eval-time merge precedence lets the CLI default ('sgd')
                # clobber the YAML's AdamW (Change.MD item 76d — verified
                # live in the exp1 smoke).
                cmd = (f"{args.python} experiments/dmo/evaluate_optimizee.py "
                       f"--if_normal_optim --regularizer {bcfg} "
                       f"--base_optimizer {_base} "
                       f"--dataset {te_ds} --optimizee_model {te_arch} "
                       f"{profile_flag(args).strip()} "
                       f"--eval_tag {tag} --set {fmt_set(common)}"
                       + (f" {' '.join(args.eval_set)}" if args.eval_set else ""))
            else:
                ckpt = lo_checkpoint_name(m, tr_ds, tr_arch)
                cmd = (f"{args.python} experiments/dmo/evaluate_optimizee.py "
                       f"--saved_model {ckpt} "
                       f"--dataset {te_ds} --optimizee_model {te_arch} "
                       f"--lo_train_dataset {tr_ds} --lo_train_arch {tr_arch} "
                       f"{profile_flag(args).strip()} "
                       f"--eval_tag {tag} --set {fmt_set(common)}"
                       + (f" {' '.join(args.eval_set)}" if args.eval_set else ""))
            eval_jobs.append((f"eval_{tag}", cmd))

        # ---------------- BENCHMARK (per experiment cell) ------------------
        lo_pairs = ",".join(
            f"{m}:{lo_checkpoint_name(m, tr_ds, tr_arch)}"
            for m in methods if m in LO_METHODS)
        bcmd = (f"{args.python} benchmark_compute.py "
                f"--dataset {te_ds} --model {te_arch} "
                f"--batch-size {args.bench_batch} "
                f"--methods {' '.join(methods)} "
                + (f"--lo-checkpoints {lo_pairs} " if lo_pairs else "")
                + f"--out results/benchmark/{key}"
                + (" --allow-cpu" if args.smoke else ""))
        bench_jobs.append((f"bench_{key}", bcmd))

        # ---------------- REPORT -------------------------------------------
        # Full 8-method figures, PLUS one folder per comparison set
        # (LO No-Reg vs LO+R vs AdamW+R) — "compare corresponding pairs
        # only". Sets land in plots/paper/<exp>/set_<R>/.
        rcmd = (f"{args.python} generate_paper_plots.py "
                f"--experiment {key}")
        report_jobs.append((f"report_{key}", rcmd))
        for _cs in COMPARISON_SETS:
            report_jobs.append((
                f"report_{key}_set_{_cs}",
                f"{args.python} generate_paper_plots.py --experiment {key} "
                f"--comparison-set {_cs}"))

    return list(train_jobs.values()), eval_jobs, bench_jobs, report_jobs


def run_local(jobs, keep_going=False):
    for name, cmd in jobs:
        print(f"\n=== {name} ===\n{cmd}")
        r = subprocess.run(cmd, shell=True, cwd=str(ROOT))
        if r.returncode != 0:
            msg = f"[FAIL] {name} (exit {r.returncode})"
            if keep_going:
                print(msg + " — continuing (--keep-going)")
            else:
                raise SystemExit(msg)


def write_pbs(jobs, args, subdir):
    outdir = ROOT / "pbs_scripts" / "experiments" / subdir
    outdir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    for name, cmd in jobs:
        (outdir / f"{name}.pbs").write_text(PBS_TEMPLATE.format(
            job_name=name, short_name=pbs_short_name(name),
            queue=args.queue, select=args.select, walltime=args.walltime,
            cuda_module=args.cuda_module,
            python=args.pbs_python, command=cmd))
    return outdir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="e.g. 1 2 5  or exp1 exp5 (default: all)")
    ap.add_argument("--methods", nargs="+",
                    default=list(ALL_METHODS) + ["sgd"],
                    help="subset of methods (user point 5: start with "
                         "'sam adamw_sam no_reg')")
    ap.add_argument("--phases", nargs="+",
                    default=["train", "eval", "bench", "report"],
                    choices=["train", "eval", "bench", "report"])
    ap.add_argument("--mode", choices=["local", "pbs", "dry"], default="dry")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU-friendly settings for a pipeline check")
    ap.add_argument("--production", action="store_true",
                    help="restore paper-scale settings (Change.MD items 1/4)")
    ap.add_argument("--train-set", nargs="*", default=[],
                    help="extra KEY=VALUE forwarded to every train --set")
    ap.add_argument("--eval-set", nargs="*", default=[],
                    help="extra KEY=VALUE forwarded to every eval --set")
    ap.add_argument("--use-best-lambdas", action="store_true",
                    help="append the frozen sweep winners "
                         "(configs/dmo/best_lambdas.json, written by "
                         "pick_best_lambdas --apply) to every LO train "
                         "command — explicit audit trail that the "
                         "generalization runs use ONLY the best combination")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="walltime budget forwarded to every train/eval job "
                         "(defaults to 23.0 with --production); jobs exit "
                         "cleanly and RESUME when resubmitted")
    ap.add_argument("--bench-batch", type=int, default=128)
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--queue", default="gpu-ext")
    ap.add_argument("--select", default="1:ncpus=8:mem=60gb",
                    help="PBS -l select line (Agastya: the gpu-ext queue "
                         "grants the GPU, so no ngpus= — matches the user's "
                         "working script). Override e.g. "
                         "'1:ncpus=8:ngpus=1:mem=60gb' if your queue needs it.")
    ap.add_argument("--cuda-module", default="compilers/cuda/11.1",
                    help="module load <this> inside every PBS script")
    ap.add_argument("--pbs-python", default="$HOME/.conda/envs/learn_reg/bin/python",
                    help="interpreter used INSIDE PBS scripts (the cluster "
                         "conda env), independent of --python for local runs")
    ap.add_argument("--walltime", default="24:00:00")
    ap.add_argument("--python", default=PYTHON)
    args = ap.parse_args()

    if args.mode == "pbs":
        # inside a PBS script the interpreter is the cluster conda env,
        # exported as $PY by the template — never the submitting machine's
        # sys.executable (which would be /usr/bin/python3 on the login node).
        args.python = '"$PY"'

    if args.use_best_lambdas:
        import json as _json
        bl = ROOT / "configs/dmo/best_lambdas.json"
        if not bl.exists():
            raise SystemExit("--use-best-lambdas: run `python "
                             "pick_best_lambdas.py --apply` first "
                             f"(missing {bl})")
        winners = _json.loads(bl.read_text())["winners"]
        args._best_lambdas = winners
        print(f"[best-lambdas] frozen winners loaded for: {list(winners)}")

    if args.max_hours is None and args.production:
        args.max_hours = 23.0
    if args.max_hours:
        args.train_set = list(args.train_set) + [f"max_hours={args.max_hours}"]
        args.eval_set = list(args.eval_set) + [f"max_hours={args.max_hours}"]
    train_jobs, eval_jobs, bench_jobs, report_jobs = build_jobs(args)
    phase_map = {"train": train_jobs, "eval": eval_jobs,
                 "bench": bench_jobs, "report": report_jobs}

    print(f"Experiment manager: {len(train_jobs)} train (deduped), "
          f"{len(eval_jobs)} eval, {len(bench_jobs)} bench, "
          f"{len(report_jobs)} report jobs | phases: {args.phases}")

    if args.mode == "dry":
        for ph in args.phases:
            print(f"\n########## PHASE: {ph} ##########")
            for name, cmd in phase_map[ph]:
                print(f"\n[{name}]\n  {cmd}")
        return

    if args.mode == "local":
        for ph in args.phases:
            print(f"\n########## PHASE: {ph} ##########")
            if ph == "report" and not (ROOT / "generate_experiment_report.py").exists():
                print("generate_experiment_report.py not present yet "
                      "(Step 3) — skipping report phase.")
                continue
            run_local(phase_map[ph], keep_going=args.keep_going)
        return

    # PBS: one job per command, grouped per phase so dependencies are simple
    # (submit train first; eval after checkpoints exist; then bench+report).
    for ph in args.phases:
        outdir = write_pbs(phase_map[ph], args, ph)
        print(f"wrote {len(phase_map[ph])} PBS jobs -> {outdir}")
    print("\nSubmit order:  train  ->  eval  ->  bench  ->  report")
    print("e.g.  for f in pbs_scripts/experiments/train/*.pbs; do qsub $f; done")


if __name__ == "__main__":
    main()