# ConvNet-4 lambda sweep — SVHN + CIFAR-10

Four methods, ConvNet-4 optimizee (4 × [conv3×3-BN-ReLU-maxpool] + linear, 115,658 params at width 64):

| | method | what is swept |
|---|---|---|
| M1 | AdamW | lr ∈ {1e-3, 3e-3, 1e-2} (validation) |
| M2 | AdamW + λ·R(θ) | λ over GRID, R ∈ {L1, L2, SAM, GSAM, GAM} |
| M3 | optimizee + learned optimizer (LSTM LO) | – (= the (0,0) cell of M4) |
| M4 | LO + λ_smooth·L_smooth + λ_reg·R | (λ_smooth, λ_reg) over GRID × GRID, same R set |

GRID = {0, 0.001 … 0.009, 0.01 … 0.09, 0.1 … 1.0} — 29 values; 0 switches the term off.
Selection is on a held-out **validation** split; test accuracy is stored but never used to select.

## Audit: what the project had and what was constructed

| piece | in project? | action |
|---|---|---|
| SAM / GAM / GSAM (optimizee side) | yes, `regularizers/optimizers.py` | not reusable for M2 (they take a radius, no λ) → λ-weighted penalty forms written in `common.fill_grads` |
| SAM / GAM / GSAM (LO meta-loss) | yes, `utils.consolidated_reg_train_opt_func` | ported into `common.meta_unroll` (model factory swapped for ConvNet-4) |
| smoothing term, LSTM LO, MetaModel | yes | reused unchanged |
| **L1 penalty** | only standalone for an MLP in `experiments/l1_l2_mnist` | **new**: M2 and M4 for ConvNet-4 |
| **L2 penalty** | meta-trainer retired from `src/`; MLP-only standalone | **new**: M2 and M4 for ConvNet-4 |
| **ConvNet-4** | **no** — `conv`/`big_conv` retired from `model.py` | **new**: `common.ConvNet4` |
| **SVHN** | **no** (loader, norm stats, class count) | **new**: `common.load_data('svhn')`; files downloaded to `datasets/{train,test}_32x32.mat` |
| CIFAR-10 | yes (`datasets/cifar-10-batches-py`) | reused |
| sweep driver for this grid / model | no (`run_sweep.py` at root is MLP-era; existing sweeps use a 17-value grid) | **new**: `run_sweep.py` |

Nothing under `src/` was modified.

## Conventions you should know about

* **R(θ) for L1/L2 is a sum** (Σ|θ|, Σθ²), as in the repo's L2 trainer and `l1_l2_mnist`. For ConvNet-4 that makes
  λ ≈ 0.001–0.01 already a strong penalty; the interesting part of the grid is its low end.
* **M2 SAM / GSAM / GAM** use the same λ-weighted form as Alg. 1, applied at the optimizee:
  SAM `L + λ·L(θ+ε)`, GSAM `L + λ·(L(θ+ε) − L(θ))`, GAM `L + λ·ρ‖∇L(θ+ε)‖`.
* **Radius ρ is fixed per regularizer, not swept, and shared by M2 and M4:** SAM 0.001, GAM 0.04, GSAM 0.005
  (`Cfg.rho_sam / rho_gam / rho_gsam`).
* **GSAM is the true surrogate gap** (gap not detached). The repo's penalty mode detaches `L(θ)`, which makes
  GSAM's gradient identical to SAM's (see `FIXES_2026-07-31.md` §E.1). Here they differ; flag if you want the
  repo's behaviour instead.
* M4 L1/L2 act on the final unrolled optimizee parameters θ_T (as in `l1_l2_mnist`).
* BatchNorm running statistics are frozen during the perturbed pass of SAM/GSAM/GAM.

## Running on Agastya (GPU, PBS Pro)

Written against the HPC User Manual v1.4.4 (§4.3): `gpu-ext` = **≥ 40 cores/node, 1 V100 per node,
max 2 running jobs per user, ≤ 2 nodes/job, walltime ≤ 7 days**. So the sweep is two whole-node **lane**
jobs (`pbs/<preset>/lane0.pbs`, `lane1.pbs`), each running 2 worker processes that share the node's V100 (a GAM cell peaks at ~13 GB, so the count is capped by GPU memory: 2 on a 32 GB V100, 5 on an 80 GB A100; see `max_procs_per_gpu`)
(ConvNet-4 uses only a few GB). Lanes split the work by deterministic sharding and wait on each other's result
files at the two real dependencies (M1 → M2, axes → local); no PBS dependency is needed between them.

The code was verified end to end **on CPU** (including two lanes running concurrently); the GPU path itself
could not be exercised on the machine it was written on (driver mismatch), so do the check in step 2.

1. **Upload** the bundle (includes `datasets/cifar-10-batches-py/` and `datasets/{train,test}_32x32.mat`).
   Compute nodes may be offline; the worker aborts immediately with a clear message if a file is missing.
2. **GPU check** on a gpu-ext node (a few minutes; use `qsub -I -q gpu-ext -l select=1:ncpus=40` or a tiny job):
   ```bash
   cd experiments/convnet4_lambda_sweep_svhn_cifar10
   PY=$HOME/.conda/envs/learn_reg/bin/python      # any env with torch >= 1.9 + torchvision
   $PY run_sweep.py worker --phase m1 --preset smoke --datasets cifar10 svhn --require-gpu
   $PY run_sweep.py job-lo --preset smoke --dataset svhn --reg gam --ls 0.01 --lr 0.1
   ```
3. **Submit** (`pbs/fast/` or `pbs/full/`; regenerate with
   `$PY run_sweep.py pbs --preset full [--walltime 168:00:00 --procs 2 --python PATH --cuda-module MODULE]`):
   ```bash
   bash pbs/fast/submit_agastya.sh 3      # 3 chained copies per lane (day-chaining); default 1
   qstat -u $USER
   ```
   Check `module avail` for the CUDA module name and `which python` for the env path; both are flags above
   (defaults copy the repo's earlier scripts: `compilers/cuda/11.1`, `$HOME/.conda/envs/learn_reg`).
   Walltime defaults to 24 h (the value the earlier scripts used); the manual allows up to 7 days.
4. **Results**: `results/<preset>/summary.md` and `best_lambdas.json` (written by `report.pbs`, or run
   `$PY run_sweep.py report --preset fast` at any time for a partial view).

Resilience: a lane stops starting new cells when the next one would not fit in `--max-hours`
(walltime − 0.5 h) and logs `WALLTIME_EXIT`; the next chained copy resumes (finished cells skipped; a
crashing cell is logged and does not kill its worker). Worker logs: `results/<preset>/logs/node<lane>_<phase>_p<k>.log`.

Presets: `smoke` (width 16, tiny), `fast` (width 32, stages 100/200/400, 500 eval steps),
`full` (width 64, stages 100/200/500/1000, 1000 eval steps).
Local (CPU box) use: `run_sweep.py all --preset fast --workers 7 --threads 4`.

## Cost (measured on this CPU box, 4 threads/job, width 64 — GPU timings not measured)

| | |
|---|---|
| M1 / M2 L1,L2,SAM | 2.3 / 3.4 min per 1000-step run |
| M2 GAM (double backward) | 8 min per 1000-step run |
| LO meta-training | 0.4 – 0.9 s per unrolled step (GAM slowest) → ~45–60 min per cell at `full` |

Stages: M2 = 5 R × 28 λ + 1 per dataset; M4 axes = (29 + 5·28) LO cells per dataset; M4 local (radius 1) = 9 per
(dataset, R). A full 29×29 M4 grid (`--stage full`) is 5 × 812 cells per dataset and is not realistic on this CPU box.
