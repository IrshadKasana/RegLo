#!/bin/bash -x
#SBATCH -J v3report
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
DATASETS="${DATASETS:-svhn cifar10 mnist fmnist}"
REGS="${REGS:-l1 l2 sam gsam gam}"
. "$ROOT/experiments/convnet4_sweep_v3_4datasets/slurm/env.sh" || exit 1
cd experiments/convnet4_sweep_v3_4datasets
python run_sweep.py report --preset "$PRESET" --datasets $DATASETS --regs $REGS
python plot_sweep_results.py --preset "$PRESET"
python make_trial_table.py --preset "$PRESET" --datasets $DATASETS --regs $REGS
