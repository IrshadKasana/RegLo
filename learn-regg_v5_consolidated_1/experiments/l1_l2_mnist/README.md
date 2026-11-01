# Stand-alone experiment: L1 / L2 regularizers × {AdamW, learned optimizer} on MNIST

This is a **separate experiment**. Nothing under `src/`, `configs/` or the other
`experiments/` folders was modified; the LSTM learned optimizer (`MetaOptimizerLSTM`),
the MLP optimizee (`model.py`, 784-20-10 sigmoid MLP) and the smoothing term
(`smooth_regularize`) are *imported* from `src/learning_regularizers/dmo`.

## The four methods

| # | Method | What is regularized |
|---|---|---|
| 1 | MLP + AdamW | nothing explicit (AdamW decoupled `weight_decay=0.01`, the repo's `adamw_no_reg_baseline`) |
| 2 | MLP + AdamW + R(θ) | `loss = NLL + λ·R(θ)` at every optimizee step |
| 3 | MLP + learned optimizer | LO meta-trained with `L_task` only (repo `no_reg`) |
| 4 | MLP + learned optimizer + R(θ) | LO meta-trained with `L_task + λ_s·L_smooth + λ·R(θ_T)` — the proposed model's objective with SAM/GAM/GSAM replaced by R |

`R = L1: Σ|θ|` or `L2: Σθ²` over all optimizee parameters (weights + biases).
At meta-test time the LO only receives the task gradient — the penalty is never
applied to methods 3/4 when they train the MLP; whatever regularization they
show has been internalized during meta-training (the paper's question).

### What already existed vs. what was built

| Piece | Status in the repo | Here |
|---|---|---|
| MLP optimizee, AdamW baseline, LO w/o regularizer, smoothing term | existed | reused / re-implemented thinly |
| L2 meta-training | `L2_reg_train_opt_func` in `utils.py`, **retired** (not routed in `train.py`, `l2.yaml` in `extras/deprecated`) | rebuilt as a shared `penalty()` |
| L1 anywhere | **did not exist** | new (`common.penalty`) |
| AdamW + explicit L1/L2 penalty | did not exist | new (`common.run_adamw`) |

## Protocol (mirrors the repo's production settings)

* **Data** — `split_seed=42`, same 50/50 split as `dataloaders.py`:
  A (30k) meta-training stream; B_train (27k) optimizee pool *and* LO checkpoint
  selection; B_val (3k, never trained on) for choosing lr / λ; official 10k test
  set reported once, never tuned on.
* **LO meta-training** — 2-layer LSTM, hidden 20, Adam meta-lr 1e-3, batch 128,
  TBPTT 20, curriculum 100→200→500→1000 steps, 3 meta-iterations (20 optimizee
  initialisations each) per stage, best checkpoint by held-out loss at 1000 steps.
  Meta-loss = NLL of the post-update optimizee on the window's last batch (as in
  the repo).
* **Meta-test** — fresh MLP, **1000 steps**, batch 512, final-step test accuracy.
  All four methods share optimizee init and mini-batch order (paired design).
* **Hyper-parameters** — chosen on B_val: AdamW lr ∈ {1e-3,3e-3,1e-2,3e-2}
  (arm 1: lr; arm 2: lr × λ jointly), LO λ (arm 4) from the grid below with one
  tuning seed (seed 0; its checkpoints are re-used as final seed 0). Grids: L1 λ ∈ {1e-5,1e-4,1e-3,1e-2}, L2 λ ∈ {1e-4,1e-3,1e-2,1e-1}.
* **Seeds** — 3 meta-training seeds (0,1,2) × 3 optimizee inits each; mean ± std over seeds.

### Things to keep in mind when reading the numbers

* Method 4 includes the LO smoothing term (as the proposed model does). Method 3
  does not. So 4 vs 3 mixes "smoothing" and "R"; the extra **diagnostic** arm
  (LO + smoothing only, not one of the four methods) isolates R.
* Methods 1/3 are identical in the L1 and the L2 experiment (same runs).
* The LO path uses a flat-parameter functional MLP (`common.FastLO` & friends) instead of
  `MetaOptimizerLSTM.meta_update`, purely for speed; `test_equivalence.py` shows it is
  numerically identical to the repo path (trajectory, meta-gradient, smoothing term).
* Seed 0 is the tuning seed, so its LO numbers are mildly favoured by λ selection on
  *validation* (test is never used for selection).
* λ for the LO is tuned on a single tuning seed; the validation set is 3k images,
  so differences of a few tenths of a percent are inside selection noise.
* Only 3 seeds: std estimates are rough; treat gaps < ~0.3 pp as unresolved.

## Run

```bash
PY=/home/hp/miniconda3/envs/new_env/bin/python   # torch 2.9.1 + CUDA
cd experiments/l1_l2_mnist
$PY run_experiment.py all --smoke --seeds 0 1    # ~15 s pipeline check
$PY run_experiment.py all                        # full run (resumable, ~1.8 h on an RTX 4060)
$PY test_equivalence.py                          # fast path == repo path
$PY run_experiment.py report                     # re-aggregate
```

Outputs in `results/full/`: `RESULTS.md`, `results.json`, `test_accuracy.png`,
`selection.json`, per-job JSON in `tune/` and `final/`, logs in `logs/`,
LO checkpoints in `ckpt/`.

## Result (MNIST test accuracy %, mean ± std over 3 seeds; see `results/full/RESULTS.md`)

| | 1. AdamW | 2. AdamW + R | 3. LO | 4. LO + R |
|---|---|---|---|---|
| **L1** (λ_AdamW=1e-4, λ_LO=1e-3) | 93.13 ± 0.08 | 93.75 ± 0.31 | 93.27 ± 0.16 | 93.65 ± 0.48 |
| **L2** (λ_AdamW=1e-4, λ_LO=1e-3) | 93.13 ± 0.08 | 93.49 ± 0.34 | 93.27 ± 0.16 | 93.39 ± 0.48 |

Explicit L1/L2 helps AdamW consistently (3/3 seeds). For the LO the regularizer helps on
average (+0.38 pp L1, +0.12 pp L2) but not reliably across seeds, and method 4 is not
distinguishable from method 2 or from the LO+smoothing-only diagnostic at 3 seeds.
