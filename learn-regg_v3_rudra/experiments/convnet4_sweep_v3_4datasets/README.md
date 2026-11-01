# Five-method, four-dataset sweep (v3) — ConvNet-4, SVHN / CIFAR-10 / MNIST / FashionMNIST

* **Protocol** (see `sweep_config.py`, the single place for every sweep value): each dataset's official train+test images are pooled and
  split once (fixed seed) 70 % train / 20 % validation / 10 % test. Hyper-parameters are selected on the *source* dataset's validation split
  only (seed 0). ID evaluation = source test split: accuracy, loss, ID AUROC (macro one-vs-rest AUROC of the softmax). OOD = the selected
  model, with no retraining, on each *other* dataset's 20 % partition: accuracy under the target's label ids (≈ chance) and OOD AUROC
  (ID test vs OOD by max-softmax confidence). MNIST / FashionMNIST are zero-padded 28→32 and replicated to 3 channels.
* **Grids** (all strictly positive): λ, AdamW lr and `meta_lr` ∈ {0.001…0.009, 0.01…0.06, 0.1…0.5} (20 values); ρ ∈ {0.001…0.008, 0.01…0.06} (14).
* **Methods / sweeps**: M1 AdamW (lr) · M2 AdamW+λR (full lr×λ grid at the default ρ, then ρ×λ at the best lr) · M3 learned optimizer (meta_lr) ·
  M4a LO+λ_smooth (full meta_lr×λ_smooth grid) · M5 LO+λ_smooth+λ_reg·R (full λ_smooth×λ_reg grid at M4a's meta_lr, then meta_lr, then ρ at the
  best cell); R ∈ {L1, L2, SAM, GSAM, GAM}. The selected M3/M4a/M5 cells are re-meta-trained with 2 more seeds. Every trial runs 3 inits.
  (The full Cartesian product is ~100 days of GPU time; `M5_MODE="block"` in `sweep_config.py` runs the M5 axis + a 7×7 block instead.)
* **Files**: `common.py` (model, data, trainers, meta-training), `run_sweep.py` (phases, selection, SLURM/PBS helpers, report),
  `plot_sweep_results.py` (12 source→target tables + figures), `make_trial_table.py` (evidence record: per-trial / per-init / OOD / curve /
  meta-stage tables, coverage, selection trace, data dictionary).
* **Outputs** in `results/<preset>/`: `summary.md`, `best_config.json`, `all_cells.csv`, `plots/`, `trial_table/`, plus per-trial `m1|m2|lo/*.json`,
  `ckpt/` (learned-optimizer checkpoints) and `weights/` (final optimizee weights of every trial).
* **Run**: on PBS (Agastya) `python run_sweep.py pbs --preset full ...`; on SLURM (Rudra) see `README_RUDRA.md` and `slurm/`.
