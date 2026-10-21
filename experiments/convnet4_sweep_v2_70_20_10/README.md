# Five-method sweep — ConvNet-4, SVHN + CIFAR-10, 70:20:10 split, ID test + cross-dataset OOD

This folder is a revision of `../convnet4_lambda_sweep_svhn_cifar10` (kept untouched, with its results, as the
backup of the previous 4-method / 50-45-5 protocol). Nothing under `src/` is modified.

| file | role |
|---|---|
| `sweep_config.py` | **every** sweep value and protocol constant (split, grids, seeds, budgets) — edit here, not in the code |
| `common.py` | ConvNet-4, data split + OOD sets, AdamW(+R) trainer, learned-optimizer meta-training, `report()` |
| `run_sweep.py` | jobs, phases, selection, PBS generation, report |
| `plot_sweep_results.py` | PNG figures from the results |

## 1. Data protocol (identical for every method)

For each dataset the official train and test images are **pooled** and split once, with a fixed seed, into
**70 % train / 20 % validation / 10 % test** (`common.split_indices`; the indices are written to
`results/<preset>/splits/<dataset>.npz`). CIFAR-10: 42 000 / 12 000 / 6 000. SVHN: 69 502 / 19 858 / 9 929.

| split | used for |
|---|---|
| 70 % train | optimizee training **and** the learned optimizer's meta-training stream |
| 20 % validation | **the only thing hyper-parameters are selected on** (source dataset, seed-0 runs) |
| 10 % test | ID evaluation of the selected configuration — never read by any selection code |

**Cross-dataset OOD, source → target** (SVHN → CIFAR-10 and CIFAR-10 → SVHN, reported separately): the configuration
selected on the *source* validation split is evaluated, with no retraining / fine-tuning / tuning, on the
**target dataset's 20 % partition** (its own validation partition from the same `split_indices` call), normalised with the
*source's* statistics. That partition is never seen during the source's training or selection; it is only used for
selection in the *other* experiment where the target is the source. OOD metrics are computed for every cell and stored,
but all selection functions in `run_sweep.py` (section "selection") read `val_acc` only.

OOD metrics: `acc` / `loss` under the target's own label ids, and `auroc_msp` = AUROC of ID-test vs OOD samples by max-softmax
confidence.

## 2. The five methods

| | method | what is swept (all values in `sweep_config.py`) |
|---|---|---|
| M1 | optimizee + AdamW | lr ∈ `ADAMW_LRS` = {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} (weight decay fixed 0.01) |
| M2 | AdamW + λ·R(θ), R ∈ {L1, L2, SAM, GSAM, GAM} | λ ∈ `GRID` (28 values); ρ ∈ `RHO_VALUES` = {0.001…0.008, 0.01…0.06} (14 radii) for SAM / GSAM / GAM; AdamW lr = M1's pick |
| M3 | optimizee + learned optimizer (LSTM, `θ ← θ + LSTM(∇)`) | **meta_lr** ∈ `META_LRS` = {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}: the Adam rate that trains the LO's own weights (1e-3 was the old fixed value) |
| M4a | learned optimizer + λ_smooth·L_smooth | λ_smooth ∈ `GRID`; meta_lr = M3's pick |
| M5 | learned optimizer + λ_smooth·L_smooth + λ_reg·R, R ∈ {L1, L2, SAM, GSAM, GAM} | λ_reg axis over `GRID` at M4a's λ_smooth; then a **joint** block of (2·`RADIUS`+1)² = 25 cells (λ_smooth, λ_reg) around (M4a's λ_smooth, that R's λ_reg winner); ρ fixed at the repo defaults; meta_lr = M3's pick |

`GRID = {0.001…0.009, 0.01…0.09, 0.1…1.0}` (28 values). **Rule for every grid: no zero value is swept** (λ, ρ, lr, meta_lr all take
non-zero values only), so each of M2 / M4a / M5 really contains its defining terms. The no-term baselines are separate methods
(M3 = learned optimizer without smooth/R term; M1 = AdamW without R). M5's λ_reg axis therefore runs at M4a's selected
λ_smooth (not at λ_smooth = 0), then the joint block is centred on (M4a's λ_smooth, that R's axis winner).

M3's learning rate: the code's learned optimizer has no learning rate in its *update* (`meta_optimizer.meta_update` adds the
raw LSTM output), so the only learning rate in the learned-optimizer pipeline is the Adam `meta_lr` that trains the LSTM —
that is what M3 sweeps. M4a and M5 inherit M3's chosen meta_lr (per source dataset) so that M3 → M4a → M5 differ only in the terms
added to the objective.

Objectives (M5, Alg. 1 penalty mode, ported unchanged from `utils.consolidated_reg_train_opt_func`):
`L_total = L_task + λ_smooth·L_smooth + λ_reg·R`; L1/L2 act on θ_T, SAM `L(φ+ε)`, GSAM the true surrogate gap, GAM `ρ‖∇L(φ+ε)‖`.
M2 uses the same λ-weighted forms on θ (see `common.fill_grads`).

## 3. Budget, seeds, selection

* Every optimizee: **1000 steps, batch 128**, no augmentation. **Every cell of every method runs all 3 optimizee initialisations**
  (seeds `EVAL_SEED0 + i`, i = 0,1,2, identical for every method; asserted in the job code; `all_cells.csv` has `n_inits` and
  `val_acc_init0..2` per row so this can be checked).
* Sweep + selection: meta-seed 0 (`SELECT_SEED`). M1/M2 cells average 3 inits.
* `final` phase: the selected M3 / M4a / M5 cells are re-meta-trained with `FINAL_SEEDS` = (1, 2). Reported test / OOD / train for those
  methods are therefore **mean ± std. error over 3 meta-seeds** (each seed = mean of its 3 inits); for M1/M2 ± is over the 3 inits.
  `val` is always the seed-0 selection value (a single number; it is optimistically biased by selection).
* Selection ties/noise: with ~12 000–20 000 validation images the validation standard error is ≈ 0.3–0.4 %; differences below about
  that are not meaningful.

## 4. Running on Agastya

```bash
cd ~/learn-regg_v5_consolidated_1/experiments/convnet4_sweep_v2_70_20_10
PY=$HOME/.conda/envs/learn_reg/bin/python
# 0. (login node, CPU) sanity check of the whole pipeline, ~25 min:
$PY run_sweep.py all --preset smoke --datasets cifar10 --regs l2 sam --workers 3 --threads 2
# 1. generate PBS scripts (2 lanes x 1 V100) and submit:
$PY run_sweep.py pbs --preset full --walltime 72:00:00 --procs 2
bash pbs/full/submit_agastya.sh 1          # add COPIES>1 to chain re-submissions
qstat -u $USER
# 2. results / figures (also written by report.pbs):
$PY run_sweep.py report --preset full
$PY plot_sweep_results.py --preset full
```
Phases run as `m1 → m2` and `m3 → m4a → axes → local → final` (lanes wait on each other's result files). Workers retry a job after a
CUDA out-of-memory (two workers share a V100; a GAM cell peaks ≈ 13 GB). A lane stops cleanly before its walltime with
`WALLTIME_EXIT`; `qsub` the same script again to resume (finished cells are skipped).

## 5. Outputs (`results/<preset>/`)

| file | content |
|---|---|
| `summary.md` | per source → target direction: selected config, val, ID test, OOD acc / AUROC for all methods |
| `best_config.json` | full selected-config records (± values, curves, files of the selected cells, protocol, cfg) |
| `all_cells.csv` | **the complete sweep**: one row per method × cell × seed with val / train / test / OOD metrics and run time |
| `m1/ m2/ lo/` | one JSON per cell (all inits, curves, meta-training history & seconds) |
| `splits/<dataset>.npz` | the split indices |
| `plots/*.png` | tables, ID vs OOD comparison, sweep curves, M5 blocks, training curves |

## 6. Assumptions / ambiguities

1. **Dataset = official train + test pooled** and re-split 70/20/10 (SVHN's `extra` split is not used).
2. **Meta-training data of the learned optimizer = the 70 % train split** (the previous protocol held a separate 50 % stream out of
   the train set). Validation / test / OOD never enter meta-training or the checkpoint score (which is a training-loss tail).
3. **OOD accuracy is not very informative here**: SVHN digits and CIFAR-10 objects share no label semantics, so accuracy on the other
   dataset is ≈ chance by construction. That is why the max-softmax AUROC is reported as well. The requested accuracy is still stored and tabulated.
4. M2's "SAM" = the five regularizers R ∈ {L1, L2, SAM, GSAM, GAM} (as in the project), each swept separately.
5. M5's joint sweep is staged (λ_reg axis at M4a's λ_smooth → joint block) rather than the full 28×28 grid per R (that would be ≈ 8 000 meta-trainings); `run_sweep.py worker --phase full`
   runs the full grid if wanted.
