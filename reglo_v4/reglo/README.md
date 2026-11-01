# RegLO — learned optimizers that internalise regularization

Experiments for the M.Tech thesis on *learned regularizers*, building on
**Learning Regularizers: Learning Optimizers that can Regularize** (Sahoo & Krishnan, arXiv:2510.08968).

A coordinate-wise LSTM optimizer is meta-trained with a regularizer inside its objective
(Algorithm 1). Afterwards it trains new networks **without computing any regularizer**, and we
measure whether the regularizer's effect — accuracy, generalization, sparsity, small weights, flat
minima — appears anyway, and at what computational cost. The optimizers are meta-trained in two
**rounds** on different source tasks (ConvNet-4 / SVHN and MLP / FMNIST) and every evaluation task
is trained with the optimizers of both rounds, so a task that is unseen for one round is seen — or
unseen in another way — for the other (cross-evaluation).

The same experiment files run at two scales: a **smoke test** that runs the whole study — every
stage, method, regularizer and evaluation setting, on the real datasets — in minutes on a laptop,
and **production**, the real numbers, on GPUs (a PARAM or Agastya cluster, or cloud GPU machines).
If the smoke test finishes cleanly, the production run differs only in scale.

## The four methods

| | Config | Optimizee is trained by | Regularizer *r* | When *r* is computed |
|---|---|---|---|---|
| **M1** | `m1_adamw` | AdamW | — | never |
| **M2** | `m2_adamw_reg` | AdamW | L1 · L2 · SAM · GAM · GSAM | at every step, in every stage: Stage 1 on each round's source pair, Stages 2–3 on every evaluation pair |
| **M3** | `m3_lstm` | LSTM optimizer φ, meta-trained on L_meta | — | never |
| **M4** *(proposed)* | `m4_lstm_reg` | LSTM optimizer φ*_r, meta-trained on L_meta + λ_smooth·L_smooth + λ_reg·L_reg | L1 · L2 · SAM · GAM · GSAM | only while meta-training φ*_r (Stage 1) |

Every comparison is made one regularizer at a time: **M1 · M2(r) · M3 · M4(r)** for r = L1,
then L2, SAM, GAM and GSAM. M2(r) is the explicit-regularization baseline everywhere: it computes r
at every step on every pair, in Stage 1 (the benchmark on the source pair) and in Stages 2–3. M4(r)
never computes r outside Stage-1 meta-training, so on every pair it is compared with M2(r), which
pays for r at every step. On a round's source pair M2's Stage-1 runs serve Stages 2–3 too (the
evaluation files' `from_stage1`): nothing is trained twice.

### M4's meta-objective (Algorithm 1)

```
L_meta,reg(φ) = Σ_t w_t·L(θ_t)  +  λ_smooth·L_smooth(φ)  +  λ_reg·L_reg(φ)

  θ_{t+1} = θ_t + F_φ(g_t, h_t)   F_φ: 2-layer LSTM, 20 units, per coordinate; 20-step truncated BPTT;
                                  outer Adam 3e-4; trajectories of up to 1,000 steps (curriculum)
  L_smooth   max over an ℓ∞ ball (ε = 0.01) of ‖F_φ(s) − F_φ(s')‖²₂ on the LSTM's normalised input,
             10 PGA steps, hidden state held fixed (Xiong & Hsieh 2022)           λ_smooth = 1.0
  L_reg      r applied to the learned optimizer's own weights φ (Algorithm 1, lines 21–22):
             L1 λ‖φ‖₁ · L2 ½λ‖φ‖² (biases excluded, M2's λ) ·                     λ_reg = 0.1
             SAM L_meta(φ+ε̂), ε̂ = ρ∇_φL_meta/‖∇_φL_meta‖ (the same truncation replayed at φ+ε̂) ·
             GAM ρ‖∇_φL_meta‖ · GSAM's direction ∇_φL_meta(φ+ε̂) − α∇⊥_φL_meta
  φ update   Adam (lr 3e-4) on ∇_φ L_meta,reg, ‖∇_φ‖ clipped at 1.0 — the same for M3 and M4
```

`meta.space: theta` in `regularizers/*.yaml` applies r to the optimizee's end point θ_T(φ)
instead. λ_reg = 0.1 and λ_smooth = 1.0 are the reference code's values and worth a short sweep
(`regularizers/*.yaml` → `meta.coef`, `regularizers/smooth.yaml` → `coef`); with M2's λ the L1/L2
terms on φ are small (≈ 6e-3 and 3e-2 at initialisation against L_meta ≈ 2.3).

## The study, stage by stage

```
 1 TRAINING                  2 OPTIMIZER EVALUATION       3 OPTIMIZEE EVALUATION       4 EVALUATION METRICS
   per round, on its source                                                              per round and metric
 ┌───────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐   ┌──────────────────────────┐
 │ φ    for M3           │   │ seen optimizee         │   │ seen dataset           │   │ test accuracy        ↑   │
 │ φ*_r for M4, one per  │──▶│   the source pair      │──▶│   the source pair      │──▶│ generalization gap   ↓   │
 │ r ∈ L1 L2 SAM GAM GSAM│   │ unseen optimizee       │   │ unseen datasets        │   │ FLOPs                ↓   │
 │ M2 = AdamW + r        │   │   other networks on    │   │   other datasets, same │   │ method × regularizer ×   │
 │ (M1: AdamW alone)     │   │   the source dataset   │   │   or other network     │   │ setting; + transfer gap  │
 └───────────────────────┘   └────────────────────────┘   └────────────────────────┘   └──────────────────────────┘
   stage1_meta_training       M1 · M2(r) · M3 · M4(r), with the optimizers of every round     make_report.py
```

**Rounds** (`configs/rounds.yaml`). A round meta-trains its optimizers on one source pair:

| Round | Source pair (seen optimizee and dataset) | Why |
|---|---|---|
| **A** | ConvNet-4 / SVHN | the paper's protocol |
| **B** | MLP (2 × 128 ReLU) / Fashion-MNIST | a second, different source — the FashionMnistRelu128x128 task of the learned-optimizer literature |

Every evaluation pair of the 3 × 3 grid {ConvNet-4, MLP, ResNet-18} × {SVHN, FMNIST, CIFAR-10}
is trained with M1 once and with M3 / M4(r) of **both** rounds. Its setting is derived from the
round's source, never listed by hand:

| Pair | Round A (ConvNet-4 / SVHN) | Round B (MLP / FMNIST) |
|---|---|---|
| ConvNet-4 / SVHN | **seen** (Stage 2 and 3) | unseen optimizee + dataset (3) |
| MLP / SVHN | unseen optimizee (2) | unseen dataset (3) |
| ResNet-18 / SVHN | unseen optimizee (2) | unseen optimizee + dataset (3) |
| ConvNet-4 / FMNIST | unseen dataset (3) | unseen optimizee (2) |
| MLP / FMNIST | unseen optimizee + dataset (3) | **seen** (2 and 3) |
| ResNet-18 / FMNIST | unseen optimizee + dataset (3) | unseen optimizee (2) |
| ConvNet-4 / CIFAR-10 | unseen dataset (3) | unseen optimizee + dataset (3) |
| MLP / CIFAR-10 | unseen optimizee + dataset (3) | unseen dataset (3) |
| ResNet-18 / CIFAR-10 | unseen optimizee + dataset (3) | unseen optimizee + dataset (3) |

Round A's column is the thesis figure (ConvNet-4 / SVHN seen; ResNet-18 / SVHN unseen optimizee;
ConvNet-4 / FMNIST and ResNet-18 / CIFAR-10 unseen datasets) plus the MLP and the full grid.
Another round is one entry in `configs/rounds.yaml` and its name in the experiment files' `rounds`.

| Stage | Experiment file | What it trains | Runs (production · smoke) |
|---|---|---|---|
| 1 | `stage1_meta_training` | per round: φ for M3, φ*_r for M4 and each r (meta half of the source dataset); M2(r) = AdamW + r on the source pair | 12 + 30 · 12 + 10 |
| 2 | `stage2_optimizer_evaluation` | the SVHN pairs: ConvNet-4, MLP, ResNet-18 | 147 · 49 |
| 3 | `stage3_optimizee_evaluation` | the FMNIST and CIFAR-10 pairs: ConvNet-4, MLP, ResNet-18 | 309 · 103 |
| 4 | `make_report.py`, `stage4_analysis` | the 3D tables per round, the benchmark, the cross-evaluation; audits and same-GPU timing | one audit per run |
| — | `extended_evaluation` *(optional, round A)* | WRN-28 and ViT-Lite-7/4 as more unseen optimizees, CIFAR-100 as a harder unseen dataset | 252 · 84 |
| — | `pilot` *(optional)* | M1 vs M3 (round A) for 10 epochs: does the LSTM hold up over long training? | 2 · 2 |

Each evaluation pair runs M1 + M2 × 5 r + (M3 + M4 × 5 r) × 2 rounds = 18 variants × 3 seeds
(smoke: 1 seed); the extended grid, with round A only, 12. On the two source pairs (ConvNet-4 /
SVHN in Stage 2, MLP / FMNIST in Stage 3) M2's 15 runs are Stage 1's, so they are not repeated:
3 × 54 − 15 = 147 and 6 × 54 − 15 = 309. In all, 740 optimizee runs (30 + 147 + 309 + 252 + 2)
and 12 meta-training runs:

| Grid | M1 | M2 | M3 | M4 | Runs |
|---|--:|--:|--:|--:|--:|
| Stage 1 (source pairs) | — | 30 | 2 φ | 10 φ*_r | 30 + 12 |
| Stage 2 | 9 | 45 − 15 = 30 | 18 | 90 | 147 |
| Stage 3 | 18 | 90 − 15 = 75 | 36 | 180 | 309 |
| Extended (round A) | 21 | 105 | 21 | 105 | 252 |
| Pilot | 1 | — | 1 | — | 2 |

The file names follow round A's view (Stage 2: its source dataset; Stage 3: the others); for
round B the same runs fall into its own Stage 2 and 3 (table above). Runs on SVHN and FMNIST — the
source datasets — train on the half that no optimizer was meta-trained on. Every pair is trained
once per *study* (`main`) and feeds every setting and round that uses it.

### Two generalization gaps

* **Generalization gap of a run** (Stage 4's metric, every run): train top-1 − test top-1 of the
  trained network — the clean training set (no augmentation) against the held-out test set, both
  at the end of training (`gen_gap_top1`; `gen_gap` is test loss − train loss). It needs no second
  round: it says how much a network trained by that optimizer overfits.
* **Transfer gap of a learned optimizer** (the cross-evaluation): how much worse an optimizer does
  on a task it was not meta-trained on. One round cannot tell a harder task from a transfer gap —
  that needs an optimizer that *did* see the task. With two rounds, on round B's source MLP / FMNIST,
  B's optimizers are seen and A's unseen: their difference there (and the reverse on ConvNet-4 /
  SVHN) is the transfer gap. `tables/<study>/transfer_*` lists every pair with both rounds'
  optimizers, and the advantage over M1 by role (seen → unseen optimizee → unseen dataset → both).

## Smoke and production

Every number that differs between the two scales lives in **two mirrored files**, and nowhere
else — the group files hold `???` for them, and a test checks that the smoke file sets every
number the production file sets:

| | `configs/profiles/production.yaml` | `configs/profiles/smoke.yaml` |
|---|---|---|
| device | CUDA GPU required (no silent CPU fallback) | GPU if there is one, else CPU |
| training images | all (SVHN, FMNIST: the optimizee half, 36,629 · 30,000) | the first 512 (test: 256) |
| epochs, Stages 2–3 | 50 (CIFAR-10 / 100: 100) | 2 |
| seeds per grid | 42, 43, 44 | 42 |
| networks | as in `configs/optimizees/` (ResNet-18: 11.2 M parameters; MLP 128-128) | same layouts, narrower (ResNet-18 width 8; MLP 16-16) |
| Stage-1 trajectories (per arm and round) | 10,000 | 4 |
| trajectory lengths (curriculum) | 100 → 200 → 500 → 1,000 steps | 20 → 40 |
| meta-validation | every 20 trajectories, 5 optimizees × 1,000 steps | every 2, 2 × 20 steps |
| checkpoints | every 5 epochs · every 10 trajectories | every epoch · every trajectory |
| Stage-4 audits | every run: 30 power iterations, 50 Hutchinson samples, 10 PGA steps | one per pair and method: 2, 2, 2 |
| same-GPU timing | 20 warm-up + 100 timed steps | 1 + 2 |

Identical in both: the LSTM, unroll, meta-learning rate, every λ, ε and ρ, AdamW, batch size 128,
the datasets and their meta split, and the experiment grids themselves. **`smoke_offline`** is the
smoke profile with random stand-in images of each dataset's shape — for a check without downloads.
Each profile writes to its own folder, `artifacts/<profile>/`, so smoke results never touch the
real ones. Precedence, lowest first: group files → profile → experiment file → command line.

## Quick start: the smoke test on your laptop (Windows, VS Code)

```powershell
python -m venv .venv
.venv\Scripts\activate                                  # macOS / Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # NVIDIA GPU: …/whl/cu121
pip install -e ".[dev,analysis]"

python scripts/prepare_data.py                          # once: FMNIST, SVHN, CIFAR-10/100 (~0.6 GB)
python scripts/run_study.py --profile smoke             # Stage 1 → 4, report, connectivity check
```

In VS Code the same steps are in *Run and Debug*: "1 · Download the datasets", then "2 · Smoke
test". The smoke test runs every stage, both rounds and every pair (about 150 runs, plus one
audit per pair and method) and ends with the report and `all checks passed`; its numbers mean
nothing — it proves that every stage runs. On a 4-core laptop CPU, `--jobs 4` finishes in about
15 minutes (each job needs up to ~1.7 GB of memory); without `--jobs` everything runs in one
process, one run after another, with breakpoints. Results:
`artifacts/smoke/tables/main/summary.md`. Without a download: `--profile smoke_offline`.
Started again, it skips what has finished; delete `artifacts/smoke/` to start over (also after
updating from an earlier version of this code: run folders now carry the round).

## Running the real study

Prepare the data once (`python scripts/prepare_data.py`, on a login node — compute nodes are
usually offline), check that the stages connect (`python scripts/check_study.py`), then:

**A cloud GPU machine** — one command, every GPU busy:

```bash
python scripts/run_study.py --gpus all                  # or --gpus 0,1,2,3; --per-gpu 2 for small runs
```

M1 and M2 runs start at once, M3/M4 runs as soon as the Stage-1 optimizer they load (their
round's) is trained; each run's output goes to `artifacts/production/logs/`. Interrupted? Start
the same command again. To have round A's results first: `--only round=A`, then the same command
without it (finished runs are skipped).

**PARAM (SLURM)**:

```bash
bash cluster/slurm/submit_pipeline.sh --dry-run         # the sbatch commands, nothing submitted
bash cluster/slurm/submit_pipeline.sh --partition gpu --time 2-00:00:00 [--account <project>]
squeue -u $USER                                          # one array task per run
python scripts/check_study.py                            # any time: what has finished, stage by stage
```

**Agastya (PBS Pro, queue `gpu-ext`)**:

```bash
bash cluster/pbs/submit_pipeline.sh --dry-run
bash cluster/pbs/submit_pipeline.sh                      # --walltime 96:00:00 if your queue allows
watch -n 60 qstat -u $USER -t
```

Both pipelines submit, per round, Stage 1's M3 job, its M4 array and its M2 array (at once), the
M1 and M2 arrays of Stages 2–3 at once, each round's M3 and M4 arrays with an `afterok` dependency on that round's
Stage-1 jobs, and the timing after Stage 1; `--rounds A` submits round A only (`--rounds B`
later), `--extended` and `--pilot` add the optional grids, `--profile smoke` runs the smoke test
as cluster jobs first. Edit `cluster/env.sh` once (the Python environment, data and artifacts
folders). When Stages 2–3 have finished:

```bash
bash cluster/slurm/submit.sh stage4_analysis --kind diagnostics     # (PBS: cluster/pbs/submit.sh)
python scripts/make_report.py
```

**Walltimes.** The submit scripts set `max_hours` from the walltime: a run that would not finish
its next epoch (or Stage-1 trajectory) in time saves a checkpoint and stops with exit code 75
instead of being killed. On SLURM the task requeues itself and continues; on PBS, submit the same
command again — finished runs return at once, interrupted ones resume. A run already being
trained by another job is left alone (exit code 4).

**Compute budget** (rough estimates for a V100; `python scripts/profile_compute.py` measures your
GPU and prints hours per run): Stage 1 dominates — per round, about half a day for M3's optimizer
and one to one and a half days for each M4 optimizer (round B's MLP has about as many weights as
ConvNet-4, so it costs about the same), where L_smooth's 10 PGA steps at every inner step cost most
(`regularizers/smooth.yaml` → `every: k` computes it every k-th step); the twelve arms run in
parallel on twelve GPUs. Stages 2–3 took on the order of 150–250 GPU-hours before M2 was added;
its 105 M2 runs (and the extended grid's 105) add more, since SAM, GAM and GSAM need two forward–backward passes per step
(`python scripts/profile_compute.py` gives the hours per run).

## Repository layout

```
reglo/
├── configs/                      every hyperparameter, as YAML
│   ├── default.yaml              profile, study namespace, round (meta_tag), paths, seed, device, training loop
│   ├── rounds.yaml               the meta-training rounds: A ConvNet-4 / SVHN, B MLP / FMNIST
│   ├── profiles/                 production · smoke · smoke_offline: every number that differs
│   ├── datasets/                 fmnist · svhn · cifar10 · cifar100 (· synthetic: unit tests)
│   ├── optimizees/               convnet4 · mlp · resnet18 · wrn28 · vit_lite7
│   ├── optimizers/
│   │   ├── adamw.yaml            conventional optimizer   → M1, M2
│   │   └── lstm.yaml             learned optimizer F_φ    → M3, M4
│   ├── regularizers/             none · l1 · l2 · sam · gam · gsam · smooth
│   │                             each r: an `explicit` block (M2) and a `meta` block (M4)
│   ├── methods/                  m1_adamw · m2_adamw_reg · m3_lstm · m4_lstm_reg
│   ├── meta_training/            Stage 1: base · m3_lstm · m4_lstm_reg
│   └── experiments/              the study: stage1 … stage4 · extended · pilot
│
├── src/reglo/
│   ├── study.py                  the study design: stages, evaluation settings per round, metrics
│   ├── rounds.py                 the rounds and what a pair is to each (seen / unseen …)
│   ├── profiles.py               smoke vs production, applied to runs and grids
│   ├── config.py                 composes the YAML files + profile + command-line overrides
│   ├── manifest.py               experiment file → run list (one line per run, validated)
│   ├── runner.py                 a whole study on one machine: dependencies, GPU slots
│   ├── checks.py                 stage-to-stage connectivity of a study, and its progress
│   ├── registry.py               YAML name → Python class
│   ├── data/                     datasets (+ offline stand-ins) · 50/50 meta split · transforms · loaders
│   ├── models/
│   │   ├── optimizees/           convnet4 · mlp · resnet · wide_resnet · vit
│   │   └── optimizers/           adamw (M1, M2) · lstm_optimizer + frozen_lstm (M3, M4) · input features
│   ├── regularizers/             base · l1 · l2 · sam · gam · gsam · smooth
│   ├── meta_training/            Stage 1 (Algorithm 1): unroll · objective · curriculum · trainer
│   ├── training/                 optimizee runs (Stage 1's M2, Stages 2–3): methods M1–M4 · trainer · evaluation
│   ├── diagnostics/              Stage 4 supplements: Hessian · sharpness · norms · FLOPs & timing · audit
│   ├── reporting/                Stage 4: result cubes per round · benchmark · cross-evaluation · figures · workbook
│   └── utils/                    paths · runtime (devices, exit codes) · logging · checkpoints · run locks
│
├── scripts/                      entry points — run_study.py does everything; make_workbook.py the Excel plan
├── cluster/                      env.sh · pbs/ (Agastya) · slurm/ (PARAM): submit.sh, submit_pipeline.sh, job script
├── artifacts/<profile>/          ALL results — layout in artifacts/README.md
├── datasets/                     downloaded data, caches, split indices (shared by the profiles)
├── notebooks/                    result exploration only
└── tests/                        fast checks (CPU, a few minutes), run before spending GPU hours
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# 1. PyTorch built for your GPU driver (Python 3.9–3.12). V100: cu118 · A100/H100: cu121 · no GPU: cpu
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
# 2. Everything else, with this package in editable mode
pip install -e ".[dev,analysis]"
# 3. Check
pytest
python scripts/check_study.py --profile smoke
```

On a cluster, do this once on the login node inside a fresh conda environment — the exact
commands are at the top of `cluster/env.sh`.

## Configuration

A run is composed from one YAML file per group, on top of `configs/default.yaml`, then the
profile, then command-line overrides:

| Form | Meaning | Example |
|---|---|---|
| `group=name` | choose a file | `method=m2_adamw_reg regularizer=sam` |
| `meta_tag=round` | whose optimizers M3/M4 load, or which round to meta-train (default A) | `meta_tag=B` |
| `key.path=value` | change an existing value | `training.epochs=100 optimizer.lr=5e-4` |
| `+key.path=value` | add a new key on purpose | `+notes=first_try` |
| `profile=name` | the scale (default `$REGLO_PROFILE`, else production) | `profile=smoke` |

Misspelt keys, missing choices and invalid combinations (M1 with a regularizer, M3 with AdamW,
meta-training on a dataset without a meta split, meta-training a round on another pair than its
source, an M3/M4 run loading a φ meta-trained on another pair than its round's, trajectory lengths
that are not whole unrolls, stand-in data in a profile that needs the real datasets) stop with a
message that lists the valid options.

```bash
python scripts/show_config.py method=m2_adamw_reg regularizer=sam optimizee=resnet18 dataset=svhn
python scripts/show_config.py meta_training=m4_lstm_reg regularizer=l1 profile=smoke
python scripts/show_config.py --list                     # every choice of every group, and the profiles
```

An experiment file lists what to train (`train.pairs` — Stage 1: `sources` —, methods, seeds) and
whose optimizers M3/M4 use (`rounds`); the evaluation settings follow from the rounds' sources;
`study` names the pool of runs. It holds no scale numbers:
those belong to the profiles (a `profile_overrides: {production: [...]}` entry gives one grid its
own budget, as the pilot does). Paths follow environment variables when set: `REGLO_DATA`
(datasets), `REGLO_ARTIFACTS` (results root), `REGLO_PROFILE`, `REGLO_ROOT` (repository root).

## The parts of a run

```bash
python scripts/meta_train.py meta_training=m4_lstm_reg regularizer=sam              # one Stage-1 arm (round A)
python scripts/meta_train.py meta_training=m4_lstm_reg regularizer=sam meta_tag=B   # … of round B: MLP / FMNIST
python scripts/train_optimizee.py method=m2_adamw_reg regularizer=sam optimizee=convnet4 dataset=svhn study=main
python scripts/train_optimizee.py method=m4_lstm_reg regularizer=sam optimizee=resnet18 dataset=svhn meta_tag=B study=main
python scripts/run_diagnostics.py --run artifacts/production/runs/main/resnet18-svhn/m4_lstm_reg-sam@B/seed42
python scripts/profile_compute.py --index 1                                         # timing of one pair
python scripts/make_manifest.py stage2_optimizer_evaluation --only optimizee=resnet18   # a run list
python scripts/run_manifest.py artifacts/production/manifests/stage2_optimizer_evaluation.txt 7
python scripts/run_study.py --stages 1 --regularizers sam                           # part of the study
```

Run folders of M3/M4 end in `@<round>` (`m4_lstm_reg-sam@B`): the same pair is trained with each
round's optimizers. Every run resumes when started again with the same command and returns at once
if it has already finished. `--only` takes `meta_training`, `method`, `regularizer`, `optimizee`,
`dataset`, `round` or `seed` (M1 runs have no round and stay in a `round=` selection). Exit codes:
0 finished · 1 crashed · 2 invalid configuration · 3 missing prerequisite (Stage-1 optimizer,
dataset, GPU) · 4 running elsewhere · 75 stopped at `max_hours` (resumable).

## The experiment workbook (Excel)

`python scripts/make_workbook.py` writes the whole plan as one workbook,
`artifacts/<profile>/workbook/reglo_experiments.xlsx`, built from the configs: Table 1 (optimizees,
optimizers, datasets, training protocol), Table 2 (regularizers), Table 3 (sweeps, with cells for
the best values), Table 4 (every experiment, stage by stage and round by round), the Stage-4 cubes,
the cross-evaluation, the Stage-1 optimizers, one row per run, and the metric definitions. Results
go in the blue cells of the *Runs* and *Stage 1 meta-training* sheets; seed means and spreads,
differences to M1, the cubes and the cross-evaluation are Excel formulas. `--fill` writes every
finished result from `artifacts/<profile>/` into a new workbook, `--fill my_copy.xlsx` into an
existing one (your notes and remarks stay).

Table 3's sweeps select on held-out training images, never on the test set: add
`training.validation_fraction=0.1` and a study of its own to each sweep run, e.g.
`python scripts/train_optimizee.py method=m2_adamw_reg regularizer=sam regularizer.rho=0.1
optimizee=convnet4 dataset=svhn training.validation_fraction=0.1 study=sweep_sam_rho0.1`. The study
grids keep 0 (`check_study.py` fails otherwise), so every reported number is a test-set number.

## Results

`python scripts/make_report.py [--profile smoke]` reads whatever has finished and writes, under
`artifacts/<profile>/`, per study:

| File | Content |
|---|---|
| `tables/<study>/summary.md` | start here: the study stage by stage and round by round, the 3D tables, the benchmark, the cross-evaluation, M4's differences (Welch p), missing or failed runs |
| `tables/<study>/round_<R>/cube_{accuracy,gen_gap,flops}.*` | **Stage 4**: method × regularizer × evaluation setting for round R, one table per metric (M2 trained with r on every pair) |
| `tables/<study>/round_<R>/benchmark.*` | **Stage 1**: on R's source pair, M1 against M2(r) trained with r (Δ top-1, Δ gap, FLOPs), beside M3 and M4(r) |
| `tables/<study>/round_<R>/cube_long.csv` | every cube cell as one row (for notebooks and plots) |
| `tables/<study>/round_<R>/<optimizee>-<dataset>/<r>.*` | M1 · M2(r) · M3 · M4(r) on one pair: top-1/top-5, gaps, r's property, passes, GFLOPs, time |
| `tables/<study>/transfer_{accuracy,gen_gap}.*` | the cross-evaluation: every pair with M1 and both rounds' M3/M4(r), each pair's role per round |
| `tables/<study>/transfer_summary_{accuracy,gen_gap}.*` | variant − M1 averaged by role, and own − other round on each source pair (the transfer gap) |
| `tables/overview.*` | M4(r) − M1, M4(r) − M2(r) and M4(r) − M3 for every round, pair and r |
| `tables/meta_training/<round>.*` | Stage 1: validation, final L_meta / L_smooth / L_reg, meta-gradient norms |
| `tables/compute/<optimizee>-<dataset>.*` | all 12 variants timed on one GPU (M2 with r) |
| `figures/<study>/round_<R>/cube_{accuracy,gen_gap,flops}` | the 3D tables as figures: a panel per setting, regularizers along x |
| `figures/<study>/round_<R>/<pair>/training_curves` | training loss and test accuracy per epoch, M1 · M2(r) · M3 · M4(r) |
| `figures/<study>/round_<R>/<pair>/pareto_{flops,time}` · `poc` · `weights` | accuracy against cost; point-of-convergence tests; log₁₀\|w\| |
| `figures/meta_training/<round>/{convergence,terms}` | Stage-1 curves |

Tables come as LaTeX (`\input{…}`, needs `booktabs` and `graphicx`) and figures as PDF for the
thesis; `tables/<study>/runs.csv` holds every seed for the notebooks. Each report regenerates
the folders of the studies it covers, so keep your own files elsewhere. A smoke report says on
its first line that it is not a result.

## References

- S. K. Sahoo, N. C. Krishnan. *Learning Regularizers: Learning Optimizers that can Regularize.* arXiv:2510.08968, 2025.
- M. Andrychowicz et al. *Learning to learn by gradient descent by gradient descent.* NeurIPS 2016.
- Y. Xiong, C.-J. Hsieh. *Learning to Learn with Smooth Regularization.* ECCV 2022.
- T. Chen et al. *Training Stronger Baselines for Learning to Optimize.* NeurIPS 2020.
- L. Metz et al. *VeLO: Training Versatile Learned Optimizers by Scaling Up.* 2022 (learned-optimization task suite, incl. FashionMnistRelu128x128).
- N. S. Keskar et al. *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.* ICLR 2017.
- J. Yang et al. *Learning to Generalize Provably in Learning to Optimize.* AISTATS 2023.
- M. Sucker, J. Fadili, P. Ochs. *Learning-to-Optimize with PAC-Bayesian Guarantees.* JMLR 2025.
- P. Foret et al. *Sharpness-Aware Minimization for Efficiently Improving Generalization.* ICLR 2021.
- J. Zhuang et al. *Surrogate Gap Minimization Improves Sharpness-Aware Training.* ICLR 2022.
- X. Zhang et al. *Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization.* CVPR 2023.
- A. Hassani et al. *Escaping the Big Data Paradigm with Compact Transformers.* 2021.
