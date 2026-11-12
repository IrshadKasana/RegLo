# ResNet-18 / ImageNet-32 five-method sweep (M1–M5) on PARAM Rudra

Forked from `experiments/convnet4_sweep_v3_4datasets` (same code paths, same SLURM machinery incl. work-stealing `--dynamic`).
All sweep values are in `sweep_config.py`.

| | |
|---|---|
| optimizee | ResNet-18, CIFAR-style stem, 11.68 M parameters (1000-way head) |
| dataset | ImageNet-1k downsampled to 32×32; train+val pooled (1,331,167 imgs), 70/20/10 split, seed 42 |
| ID metrics | 10 % test split: accuracy, loss, macro one-vs-rest AUROC |
| OOD | ImageNet → SVHN, CIFAR-10, MNIST, FashionMNIST (their 20 % partitions): max-softmax AUROC |
| selection | first 50 000 images of the shuffled 20 % validation split, meta-seed 0 only |
| budget | 1000 optimizee steps, batch 128 (≈ 0.14 epoch) for every method; 2 optimizee inits per cell |

## Methods and sweeps (each axis strictly positive, 5–6 log-spaced values)
| | sweep |
|---|---|
| M1 AdamW | lr ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2} |
| M2 AdamW+λR (R = L1, L2, SAM, GSAM, GAM) | lr × λ full grid (λ grid per regulariser, see config), then ρ × λ at the best lr (SAM/GSAM/GAM, ρ ∈ {0.01, 0.02, 0.05, 0.1, 0.2}) |
| M3 LO | meta_lr (same 6 values) |
| M4a LO + λ_smooth·R_smooth | **λ_smooth fixed = 1.0** (`SMOOTH_FIXED`); sweep meta_lr (6 trials) |
| M5 M4a + λ_reg·R | **λ_smooth fixed = 1.0**; λ_reg ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, **1.0**} at M4a's meta_lr, then meta_lr, then **ρ × λ_reg** (SAM/GSAM/GAM: 5 ρ × 7 λ_reg at the best meta_lr); then 2 extra meta-seeds for the selected cells |

≈ 430 trials in total (M1 6, M2 252, M3 6, M4a 6, M5 ≈ 144 incl. the ρ × λ_reg sweep, final ≤ 14); estimated ≈ 36 A100-hours from the measured GPU-check timings
(M1 ≈ 76 s, M2-GAM ≈ 216 s, M5-GAM ≈ 490 s per trial). Note: for GSAM, λ_reg = 1.0 puts zero weight on the plain task loss (`1 − λ_reg`).

## IMPORTANT design decision: the learned optimizer is meta-trained on a proxy
Meta-training back-propagates through 20 unrolled steps of a coordinate-wise LSTM: ~1.2 KB of activations per coordinate and step
→ ~270 GB for ResNet-18's 11.7 M coordinates. It cannot be done on one 80 GB A100. So M3/M4a/M5 meta-train the LSTM on a
**ConvNet-4 proxy (1000-way head) on the same ImageNet-32 stream** (the repo's ConvNet-4 pipeline, incl. smooth / SAM / GSAM / GAM / L1 / L2
meta-losses) and are **meta-tested on ResNet-18** (the repo already treats ResNet-18 as the "meta-test target", `dmo/model.py`). Every LO
hyper-parameter is chosen by the ResNet-18 validation accuracy. λ_smooth, λ_reg and ρ of M4a/M5 therefore act on the proxy's meta-objective.
If you need LO meta-training directly on ResNet-18, that needs a different (memory-saving) algorithm, not a config change.

## What was tested (local, CPU only — NOT on a GPU, NOT on Rudra)
* Full pipeline `run_sweep.py all --preset smoke` (all phases, L2 + GAM, 20 k-image ImageNet subset, tiny ResNet width 8, 40 steps) → summary, plots, trial table OK.
* ResNet-18 flat-parameter order == `parameters()` order; LO trained on the proxy updates ResNet-18 weights.
* GPU check (job 115275, A100-80GB): full-width trial timings above; **peak 47.6 GB (M5) → `PROCS=1`**.
* `META_CLIP` (meta-gradient clipping) exists but is OFF: a CPU test at 1.0 did not help. Non-finite meta-gradients are skipped.
* **Known problem (LO probe, job 115374):** the 6 M3 trials (meta_lr 1e-4 … 3e-2) all ended at chance (val_acc ≈ 0.0011). The proxy-trained LO did not
  train ResNet-18 in that probe; the fixed λ_smooth = 1.0 of M4a/M5 is untested on the GPU. Look at the M4a results before the long M2/M5 phases finish.

## Upload and run on Rudra
Rudra already has `~/Rego/learn-regg_v5_consolidated_1` (src/, datasets/ with SVHN, CIFAR-10, MNIST, FashionMNIST). Only the new experiment folder and ImageNet are needed.
```bash
# on your PC (separate terminal), from /home/hp/Downloads/Reglo
scp -P 4422 resnet18_imagenet_sweep_code.zip imagenet32_parquet.zip aroofa_ext@paramrudra.iitp.ac.in:~/Rego/learn-regg_v5_consolidated_1/

# on Rudra
ssh -p 4422 aroofa_ext@paramrudra.iitp.ac.in
cd ~/Rego/learn-regg_v5_consolidated_1
unzip -q -o resnet18_imagenet_sweep_code.zip && unzip -q imagenet32_parquet.zip -d ~/imagenet32_parquet
. experiments/resnet18_imagenet_sweep/slurm/env.sh                     # python with torch
python -c "import pyarrow, PIL" || pip install --user pyarrow pillow
python experiments/resnet18_imagenet_sweep/prepare_imagenet32.py ~/imagenet32_parquet   # ~7 min on the login node, writes datasets/imagenet32 (3.9 GB)
bash experiments/resnet18_imagenet_sweep/slurm/setup_env.sh            # expect every dataset file OK
mkdir -p logs && sbatch experiments/resnet18_imagenet_sweep/slurm/gpu_check.sbatch
squeue --me; cat logs/gpucheck.<jobid>.out                              # SMOKE_OK, GPUCHECK_OK, per-trial seconds, PROCS <= N
```
(If `pyarrow` cannot be installed on Rudra, upload `imagenet32_npy.tar` instead — 3.9 GB — and untar it into `datasets/`.)

Production, after the GPU check looks right (adjust `PROCS`; 4 lanes × 48 h, 2 chained copies):
```bash
PROCS=1 bash experiments/resnet18_imagenet_sweep/slurm/submit_rudra.sh full 2 4 48:00:00
squeue --me
```
Re-running the same command resumes (finished trials are skipped). The final `report` job writes
`experiments/resnet18_imagenet_sweep/results/full/{summary.md,best_config.json,all_cells.csv,plots/,trial_table/}`.
Rudra gotchas from earlier runs: gpu QOS max 48 h and 5 running jobs; avoid node `ragpu004` (default `EXCLUDE`); never `set -u` in job scripts; compute nodes have no internet.
