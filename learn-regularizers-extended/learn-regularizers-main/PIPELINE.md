# Hierarchical L2O Regularization Pipeline

This extension adds a three-phase experimental pipeline to the base *Learning Regularizers*
code: **Meta-Training → Optimizer Evaluation → Optimizee Evaluation**. It compares six training
variants under five regularizers, on seen and unseen optimizees and datasets.

It lives in `src/learning_regularizers/pipeline/` and does **not** modify the base `dmo/` code
or any existing YAML file. `tests/test_pipeline.py` pins the hashes of the original YAMLs.

---

## 1. Important: base-code meta-gradient bug

In `dmo/meta_optimizer.py`, `MetaModel.set_flat_params` assigns `param.data = flat_params[...]`.
Writing through `.data` detaches the optimizee parameters from the LSTM output. As a result,
`meta_loss.backward()` never reaches the learned optimizer, and every LSTM `.grad` stays `None`:
the base `train.py` loops run but leave the LO at its random initialisation. With GSAM
regularization the base code would also fail, because `torch.stack([])` receives an empty list.
`tests/test_pipeline.py::test_base_meta_update_has_no_gradient_path` documents the bug.

The pipeline uses a functional unroll (`torch.func.functional_call`) whose parameters stay
connected to the LSTM updates. Checks:

* The meta-gradient matches a central finite difference to ~1e-9 relative error
  (`test_meta_gradient_matches_finite_differences`).
* On a synthetic task, 8 meta-iterations lower the LO's evaluation loss from 3.96 to 0.31.

`LSTMLearnedOptimizer` keeps the same sub-modules and preprocessing as `MetaOptimizerLSTM`.
Its outputs match the base class to within 5e-10, and **checkpoints load in both directions**,
so a pipeline LO can be evaluated with the base `optimizee_train.py --saved_model ...`.

---

## 2. Methods, optimizers, regularizers

| # | Method | Optimizer training (phase 1) | Optimizee training in phase 1 |
|---|---|---|---|
| 1 | Optimizee + B | – (B is hand-designed) | B |
| 2 | Optimizee + B + λR | – | B + λR |
| 3 | Optimizee + LO | meta-train LO | (throw-away unrolls) |
| 4 | Optimizee + LO + λ_smooth R_smooth | meta-train LO with R_smooth | – |
| 5 | Optimizee + LO + λ_reg R | meta-train LO with R | – |
| 6 | Optimizee + LO + λ_smooth R_smooth + λ_reg R (**proposed**) | meta-train LO with both | – |

* **B ∈ {Adam, SGD}** applies to methods 1–2 and is reported in separate tables. Methods 3–6 do
  not depend on B, and their rows repeat in both tables.
* **R ∈ {L1, L2, SAM, GAM, GSAM}** applies to methods 2, 5 and 6. Methods 1, 3 and 4 repeat across
  the R slices.
* **LO**: a coordinate-wise 2-layer LSTM (hidden 20) with log/sign gradient preprocessing,
  identical to the base code.

How R is applied **during training**:

| R | Baseline (method 2) | Learned optimizer (methods 5, 6) |
|---|---|---|
| L1 | loss + λ Σ\|w\| (weight tensors) | meta-loss + λ Σ\|θ_T\| (optimizee params at end of truncation) |
| L2 | loss + λ Σw² | meta-loss + λ Σθ_T² |
| SAM | base `SAM` on optimizee | base `SAM` on LO parameters φ (`sam.yaml`: `SAMonOptimizer: True`) |
| GAM | base `GAM` on optimizee | base `GAM` on φ (closure replays the truncation) |
| GSAM | base `GSAM` on optimizee | base `GSAM` on φ (closure replays the truncation) |

**R_smooth** comes from the SmoothedOptimizer reference: the MSE between LO(g) and LO(g+δ),
maximised over ‖δ‖∞ ≤ ε with 10 sign-PGA steps of size ε/5 from a 0.01-random start, with both
evaluations starting from the same LSTM state. As in the reference, δ is measured in
RMS-normalised gradient space, so ε = 1 is scale-free. It is added to the meta-loss at every
inner step by default (`meta_training.smooth_every`). Hyperparameters come from `smooth.yaml`
(λ = 1, ε = 1, 10 steps).

**During evaluation (phases 2 and 3), all regularizers are removed.** Learned optimizers always
run frozen and unregularized. Baselines run plain B when
`optimizer_evaluation.strip_regularizers_for_baselines: true`, which is the default. Because a
hand-designed optimizer carries nothing learned, **method 2 equals method 1 in the phase-2
columns** under this rule. The cache reuses method 1's runs, so nothing is recomputed. Method 2's
effect shows up in phase 3, which evaluates the optimizee it trained *with* R. Setting the flag
to `false` restores the original paper protocol (B + λR at evaluation).

---

## 3. The three phases

**Phase 1: Meta-Training**, on the seen task ConvNet-4 + SVHN, using the meta-train half of the
SVHN training split.
* Methods 3–6 meta-train the LO with truncated BPTT (`truncated_bptt_step` = 20) and the base
  curriculum (unroll 100 → 200 → 500 → 1000, `n_period` evaluations per stage). The checkpoint
  with the best evaluation loss is saved to `checkpoints/lo/<run_id>.pt`.
* Methods 1–2 have no optimizer to train. They train ConvNet-4 on SVHN with B (+R) for every seed.

**Phase 2: Optimizer Evaluation.** The optimizer is frozen and regularizers are removed. For each
optimizee (ConvNet-4 = seen; MLP, ResNet-18 = unseen), fresh networks are trained on the
*other* half of SVHN-train for `optimizee_epochs` steps, over `n_training` seeds. Those settings
come from `optimizee_eval.yaml`: 1000 steps, batch 512, 5 seeds. A frozen LO updates ResNet-18's
11M coordinates in chunks (`models.lo_chunk_size`), which is exact because the LO is
coordinate-wise.

**Phase 3: Optimizee Evaluation.** Optimizer and optimizee are both frozen. The evaluated network
is the method's *seen-task optimizee*: for methods 1–2, the phase-1 network (trained with R for
method 2); for methods 3–6, the phase-2 ConvNet-4 trained by the frozen LO. All of them use the
same data, step budget and seeds.
* **Seen dataset (SVHN)**: evaluated directly.
* **Unseen datasets** (MNIST, FMNIST, CIFAR-10, CIFAR-100, ImageNet): the label spaces differ, so
  the backbone stays frozen and only a new linear head is fitted on its features (a linear probe).
  By default the probe uses a fixed Adam optimizer, so the result measures only the frozen
  optimizee. `probe.optimizer: method` instead trains the head with the method's own frozen
  optimizer, without R.
* Every dataset is mapped to 3×32×32: MNIST/FMNIST are resized and copied to 3 channels, and
  ImageNet is resized and center-cropped.

---

## 4. Metrics and the 3D tables

| Metric | Definition | Direction |
|---|---|---|
| Test accuracy | % correct on the test split (ImageNet: val) | ↑ |
| Generalization gap | `train_acc − test_acc` (pp). Set `metrics.gap: loss` for `test_loss − train_loss` | ↓ |
| FLOPs | Counted by `torch.utils.flop_counter.FlopCounterMode` (matmul/conv, forward + backward). Phase 2: optimizee-training FLOPs. Phase 3: FLOPs to train the evaluated optimizee + evaluation/probe FLOPs | ↓ |

The LSTM is computed with explicit matmuls, because `nn.LSTM`/cuDNN kernels are invisible to the
counter, so the LO's overhead is counted on every device. Measured on a synthetic run, relative
to plain B per step: SAM 2×, GAM 4×, GSAM 2×, frozen LO ≈1.2×. Meta-training FLOPs are reported
separately in the Phase-1 table of `summary.md`.

`artifacts/pipeline/<experiment>/tables/` holds, for each metric and each B, a 3D table
**Regularizer (Z) × Method (Y) × Evaluation setting (X)**:

* `<metric>__B-<b>.npy`: array `[5, 6, 4]` with settings ordered seen optimizee, unseen optimizee,
  seen dataset, unseen dataset.
* `<metric>__B-<b>.csv`, `.png` (one heat-map per R slice), `summary.md`, `breakdown.md`
  (per optimizee and per dataset), `results_long.csv` (mean ± std over seeds).
* "Unseen optimizee" is the mean over MLP and ResNet-18. "Unseen dataset" is the mean over the
  unseen datasets that were available.

---

## 5. Configuration

`configs/pipeline/pipeline.yaml` is the only file you normally edit. Precedence, lowest to highest:

```
configs/dmo/base/meta_train.yaml | optimizee_eval.yaml     (base, untouched)
configs/dmo/regularizers/smooth.yaml                       (base, untouched; meta-training)
configs/dmo/regularizers/<R>.yaml                          (base sam/gam/gsam/no_reg + new l1/l2)
pipeline.yaml  `overrides:` blocks and task fields
--set key.path=value                                       (command line)
```

Hyperparameters therefore come from the existing YAMLs. Some examples:
* M5/M6 with GAM meta-train the LO with Adam, lr 1e-3, and `gam.yaml`'s ρ values.
* M2 with SAM trains the optimizee with B at `sam.yaml`'s lr 0.01 and ρ 0.05.
* M1 uses `no_reg.yaml`.

The new `l1.yaml` and `l2.yaml` copy `no_reg.yaml`'s optimizer settings and add `reg_lambda`
(1e-5 and 5e-4).

Choices made by `pipeline.yaml`:
* The default meta-training task is SVHN + ConvNet-4, as in the figure. It can be changed (below).
* ConvNet-4 width is 32. Set `models.convnet4.width: 64` for the classic Conv-4; it needs about
  4× the LO memory during meta-training.

### Choosing the meta-training task

`meta_training.optimizee` and `meta_training.dataset` define "seen" for the whole pipeline. The phase-2/3
fields (`optimizer_evaluation.dataset`, `.seen_optimizee`, `optimizee_evaluation.seen_dataset`,
`.optimizees`, `.unseen_datasets`) default to `null` and follow it:

| Field | `null` means |
|---|---|
| `optimizer_evaluation.dataset`, `optimizee_evaluation.seen_dataset` | `meta_training.dataset` |
| `optimizer_evaluation.seen_optimizee` | `meta_training.optimizee` (also added to `optimizer_evaluation.optimizees`) |
| `optimizee_evaluation.optimizees` | `[meta_training.optimizee]` |
| `optimizee_evaluation.unseen_datasets` | all six datasets except the seen one |

An explicit value that contradicts the meta-training task is rejected. Command-line shortcuts:

```bash
python experiments/pipeline/run_pipeline.py --meta-optimizee resnet18 --meta-dataset imagenet
python experiments/pipeline/run_pipeline.py --meta-optimizee mlp --meta-dataset mnist \
    --eval-optimizees convnet4 resnet18 --unseen-datasets fmnist cifar10
```

`meta_training.models` changes model settings **only during meta-training**, e.g.
`--set "meta_training.models={resnet18: {width: 16}}"` meta-trains on a narrow ResNet-18 proxy; phases 2-3
still use `models.resnet18` (width 64). The LO is coordinate-wise, so it transfers unchanged.

Meta-training memory grows with (optimizee weights x `truncated_bptt_step`): about 1.4 KB per weight per
unroll step (measured, method 3), because the unrolled LSTM graph is kept for BPTT. The pipeline prints the
estimate and warns when it exceeds free GPU memory.

| ResNet-18 width | weights | `truncated_bptt_step` 20 (default) | `truncated_bptt_step` 5 |
|---|---|---|---|
| 8  | 0.24 M | ~6.5 GB  | 1.7 GB (measured) |
| 16 | 0.83 M | ~23 GB   | 5.7 GB (measured) |
| 32 | 3.05 M | ~85 GB   | ~21 GB |
| 64 (full) | 11.7 M | ~315 GB (does not fit) | ~80 GB |

ConvNet-4 (width 32, ~29k weights) needs well under 1 GB. Methods 4/6 (R_smooth) need somewhat more.

A saved LO is re-used only if its settings fingerprint (task, model widths, hyperparameters) matches, so
changing the task under the same `experiment.name` re-trains instead of silently re-using an old LO.

---

## 6. Running

```bash
pip install -r requirements.txt          # torch>=2.1 (FlopCounterMode, torch.func)

# smoke test: all 6 methods x 5 R x 2 B on synthetic data (no downloads, CPU, ~10-20 min)
bash experiments/pipeline/smoke_test.sh
# same smoke grid on the real datasets (GPU, ~35 min on an RTX 4060)
bash experiments/pipeline/smoke_real.sh
python tests/test_pipeline.py            # or: python -m pytest tests/test_pipeline.py -q

# full pipeline (datasets go to ./datasets or $LSAIML_DATASETS_DIR)
bash experiments/pipeline/run_full.sh --device cuda:0

# subsets
python experiments/pipeline/run_pipeline.py --list
python experiments/pipeline/run_pipeline.py --phases 1 --methods 6 --regularizers gam
python experiments/pipeline/run_pipeline.py --phases 2 3 report --run-ids M6_LO_gam M5_LO_gam
python experiments/pipeline/run_pipeline.py --set meta_training.overrides.max_epochs=500 --set experiment.name=short

# SLURM: one array task per run (see header of the script)
sbatch --array=0-23 experiments/pipeline/slurm_array.sh 1
```

Every step can be resumed:
* LO checkpoints with a sidecar JSON are not re-trained unless you pass `--force-meta`.
* Optimizee runs are cached under `optimizee_runs/`, keyed by optimizer, effective regularizer,
  optimizee, dataset, seed, resolved arguments and LO-checkpoint hash.

**ImageNet** cannot be downloaded automatically by torchvision. The pipeline reads ImageNet-32 from
`<data_root>/imagenet32/{x,y}.npy` (written by `prepare_imagenet32.py`: 1,281,167 train images, then the
50,000 validation images, used as the test split), or `datasets.imagenet.npy_root`. Otherwise it falls back to an
ImageFolder layout (`train/<wnid>/…`, `val/<wnid>/…`) at `<data_root>/imagenet` or `datasets.imagenet.root`.
If neither exists, phase 3 skips ImageNet with a warning (`skip_missing_datasets`).

---

## 7. Compute notes

* The base meta-training budget (`max_epochs: 10000`, `train_period: 20`) is large. Each outer
  iteration meta-trains on 20 fresh optimizees, so use `meta_training.overrides` to shorten it
  while iterating.
* R_smooth adds 10 PGA steps per inner step, which makes M4/M6 meta-training about 3.8× slower
  (a measured FLOP ratio). `meta_training.smooth_every: k` amortizes it.
* A frozen LO on ResNet-18 keeps 2 × (2 × 11.2M × 20) floats of LSTM state, about 3.6 GB.

---

## 8. Files added

```
configs/pipeline/pipeline.yaml               pipeline configuration
configs/dmo/regularizers/{l1,l2}.yaml        new regularizer configs
src/learning_regularizers/pipeline/
    config.py            YAML merging, per-run arguments
    methods.py           the 6 methods and the run grid
    data.py              unified 3x32x32 datasets, splits, synthetic data
    models.py            MLP, ConvNet-4, ResNet-18 (features + classifier)
    learned_optimizer.py differentiable, checkpoint-compatible LSTM LO; R_smooth; frozen stepper
    regularizers.py      L1/L2 penalties; B and B+R steps (base SAM/GAM/GSAM reused)
    meta_train.py        phase 1 for methods 3-6
    trainers.py          optimizee training loop (B, B+R, frozen LO)
    optimizee_runs.py    cached optimizee trainings
    evaluation.py        metrics, generalization gap, linear probe
    flops.py             FLOP meter
    phases.py            phase 1 / 2 / 3 orchestration
    report.py            3D result tables
    cli.py, artifacts.py
experiments/pipeline/  run_pipeline.py, run_full.sh, smoke_test.sh, slurm_array.sh
tests/test_pipeline.py
```

### Other differences from the base code worth knowing

* The base `project_paths.repo_root` looks for an `L2O-CL/` folder that this repo no longer has.
  The pipeline therefore resolves data from `experiment.data_root`, `$LSAIML_DATASETS_DIR`, or
  `<repo>/datasets`.
* The base `GAM`/`GSAM.LO_set_closure` hard-codes `.cuda()`. The pipeline passes its own
  device-agnostic closure to `step(closure)` instead.
* The base Conv_NN applies ReLU to the logits. The new ConvNet-4 does not.
