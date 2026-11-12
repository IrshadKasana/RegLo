#!/bin/bash -x
#SBATCH -J r18report
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -p cpu
#SBATCH -t 02:00:00
#SBATCH -o logs/report_%j.out
#SBATCH -e logs/report_%j.err
# Aggregation of results/<preset>/ -> summary.md, best_config.json, all_cells.csv, plots/, trial_table/.  CPU partition is fine.
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT" || exit 1
PRESET="${PRESET:-full}"
DATASETS="${DATASETS:-imagenet}"
REGS="${REGS:-l1 l2 sam gsam gam}"
. "$ROOT/experiments/resnet18_imagenet_sweep/slurm/env.sh" || exit 1
cd experiments/resnet18_imagenet_sweep
python run_sweep.py report --preset "$PRESET" --datasets $DATASETS --regs $REGS
python plot_sweep_results.py --preset "$PRESET"
python make_trial_table.py --preset "$PRESET" --datasets $DATASETS --regs $REGS
