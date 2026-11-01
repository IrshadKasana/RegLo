# sharpbench

A standalone, reproducible comparison of sharpness-aware regularizers —
**SAM, ASAM, GAM, GSAM** — against an AdamW control, on MNIST with a small MLP.

Covers the full pipeline: training → evaluation → hyperparameter selection →
significance testing → publication figures.

```
python -m sharpbench check      # verify GPU, benchmark this machine
python -m sharpbench fetch      # download MNIST
python -m sharpbench sweep      # run the grid (resumable, shardable)
python -m sharpbench analyze    # selection report + significance table
python -m sharpbench plot       # seven figures
```

See **QUICKSTART.md** for the exact command sequence.

---

## Contents

1. [What the methods actually are](#1-what-the-methods-actually-are)
2. [What to tune and what to fix](#2-what-to-tune-and-what-to-fix)
3. [How you actually know the best rho](#3-how-you-actually-know-the-best-rho)
4. [Protocol: epochs, seeds, statistical power](#4-protocol)
5. [What the pipeline produces](#5-what-the-pipeline-produces)
6. [Project layout](#6-project-layout)
7. [Known limitations](#7-known-limitations)

---

## 1. What the methods actually are

None of these is an additive loss penalty of the form `loss + lambda * term`.
They are **gradient-modification procedures**: the base optimizer never sees a
modified loss, only a modified gradient. This matters for how you write them up.

### SAM (Foret et al., ICLR 2021)

$$\min_w \max_{\lVert\epsilon\rVert_2 \le \rho} L(w+\epsilon)$$

$$g = \nabla L(w),\quad \epsilon = \rho\frac{g}{\lVert g\rVert},\quad
\tilde g = \nabla L(w+\epsilon),\quad w \leftarrow \mathrm{AdamW}(w,\tilde g)$$

`rho` is the **radius of a ball in weight space** — a length, not a mixing
coefficient. SAM has exactly one hyperparameter, and there is no lambda.

The additive intuition is not wrong, it is just first-order:

$$\max_{\lVert\epsilon\rVert\le\rho} L(w+\epsilon) \approx L(w) + \rho\lVert\nabla L(w)\rVert_2$$

So SAM *behaves like* a gradient-norm penalty of strength rho, but never
computes one. Write the min-max objective, then the two-step algorithm.

### ASAM (Kwon et al., ICML 2021) — `adaptive=True`

Rescales the perturbation elementwise by `|w|`, making it invariant to weight
rescaling. **The useful radius is 10–50× larger than vanilla SAM**: roughly
`[0.1, 4.0]` instead of `[0.01, 0.5]`. Running `adaptive=True` with `rho=0.05`
gives you very nearly plain AdamW at double the compute, and is the single most
common misconfiguration of this method.

### GSAM (Zhuang et al., ICLR 2022)

$$g_\perp = g - \frac{\langle g, g_{adv}\rangle}{\lVert g_{adv}\rVert^2}g_{adv},
\qquad g_{\mathrm{GSAM}} = g_{adv} - \alpha\, g_\perp$$

Descends the SAM objective *and* the surrogate gap $h(w)=L(w+\epsilon)-L(w)$.
Two hyperparameters: radius rho and gap coefficient alpha.

### GAM (Zhang et al., CVPR 2023) — genuinely additive

$$\min_w L(w) + \alpha\,R(w),\qquad
R(w) = \rho\max_{\lVert\epsilon\rVert\le\rho}\lVert\nabla L(w+\epsilon)\rVert_2$$

Here alpha *is* a penalty weight. Because grad R needs a Hessian-vector
product, it is approximated by finite differences — hence four
forward-backward passes per step.

| Method | radius | strength | passes/step |
|---|---|---|---|
| AdamW | — | — | 1 |
| SAM / ASAM | rho | *none* | 2 |
| GSAM | rho | alpha | 2 |
| GAM | grad_rho, grad_norm_rho | grad_gamma | 4 |

---

## 2. What to tune and what to fix

The governing principle reviewers apply: **the baseline must receive a search
budget at least as large as whatever you are comparing it to.** A method tuned
over 20 configurations against a baseline given one value of rho is not a
comparison.

### Tier A — sweep, per method (the method's own knobs)

Non-negotiable. These *are* the method.

| Method | Sweep | Grid used here |
|---|---|---|
| SAM | `rho` | `0.01, 0.02, 0.05, 0.1, 0.2, 0.5` |
| ASAM | `rho` | `0.1, 0.5, 1.0, 2.0, 4.0` — **different scale** |
| GSAM | `rho` × `alpha` | `0.02…0.2` × `0.01…0.4` |
| GAM | radius scale `s` **and** `grad_gamma` | `s ∈ 1…8` (keeping the published `grad_norm_rho = 5·grad_rho` ratio) × `grad_gamma ∈ 0.03…1.0`. **`grad_gamma` is the α in GAM's objective — it must be swept.** Pinning it at the default while sweeping only the radius starves the penalty term and makes GAM look inert. |

### Tier B — sweep, shared across every arm

**Learning rate.** This is the one people forget, and it is the most common
"unfair baseline" objection. SAM changes the effective gradient scale, so the
optimal LR can shift between arms. Two defensible protocols:

- *Shared grid* (what this repo does): every arm, including the control,
  searches `3e-4, 1e-3, 3e-3`. Symmetric and cheap.
- *Per-arm re-tuning*: tune LR independently for each method. Stricter, more
  expensive, and what a hostile reviewer would prefer.

Either is fine. **Silently fixing LR at the value that suits one arm is not.**
Whichever you use, state it in the paper.

### Tier C — fix, report the value, and check sensitivity once

These are real hyperparameters that interact with rho. Sweeping them jointly
explodes the grid and makes results uninterpretable, so fix them — but *say*
you fixed them, and ideally run a one-dimensional sensitivity check at the
winning rho.

| Constant | Why it matters | Handling |
|---|---|---|
| **weight decay** | AdamW's decoupled decay is a *different* regularizer. Sweeping it confounds "does sharpness-awareness help" with "does more L2 help". | Fixed at `0.01`. Run one sensitivity check at the winning rho. |
| **batch size** | The optimal rho scales roughly with gradient noise, so rho and batch size interact strongly. | Fixed at 128. Never vary it in the same table as rho. |
| **m-sharpness** (`m_sharpness`) | SAM's reported benefit depends on the *micro-batch* used for the ascent step, not just rho. Smaller m regularizes more. **Very often left unreported**, which makes SAM numbers hard to reproduce across papers. | Fixed at full-batch (`null`). State this value explicitly in your methods section. |
| **epochs / total budget** | Sharpness-aware methods separate from the control *late* in training. | Identical for every arm. |
| **LR schedule + warmup** | GSAM's published rho schedule is tied to the LR schedule; changing one changes the other. | Cosine, no warmup, identical everywhere. |
| **other regularizers** (dropout, label smoothing, augmentation) | Each independently flattens minima and will mask the effect you are measuring. | All off. If you enable one, enable it identically across arms. |
| **gradient clipping** | Interacts with SAM's normalized perturbation. | Off by default. |
| **TF32 / precision** | On Ampere+, TF32 matmuls silently reduce precision by an amount comparable to the effect size here. | **Disabled by default** in `device.py`. |

### Tier D — never sweep (part of the method's definition)

- **GAM's `grad_beta_0..3` = (1, 1, −1, 1)**. These define GAM's gradient
  combination. Sweeping them means you are comparing against a method you
  invented that resembles GAM, and a reviewer who knows the paper will say so.
  Note this is *not* the same as `grad_gamma`, which is the penalty weight α
  and is a legitimate Tier-A hyperparameter.
- **The perturbation normalization scheme** (global L2 vs elementwise).
- **Epsilon in denominators.**

### The extra row a good reviewer asks for

SAM and GSAM cost 2× the gradient evaluations per step; GAM costs 4×. An
epoch-matched table therefore hands the regularizers extra compute. Report
epoch-matched as primary (that is the convention), then add one
**compute-matched control**: AdamW at 2× and 4× the epochs. Every run records
`fwd_bwd_passes`, so you can build it directly. Volunteering this row is much
stronger than having it extracted from you in review.

---

## 3. How you actually know the best rho

This is a procedure, not a single number. Six steps:

**1. Grid, with at least 3 seeds per point.** A single-seed sweep is noise —
seed-to-seed std on this setup is comparable to the whole rho effect.

**2. Select on mean validation accuracy, never on test.** `analyze` enforces
this: it groups by (arm, hyperparameters), ranks by mean *validation*
accuracy, and reports the *test* accuracy of the validation-selected setting.
Picking the best test number is the most common methodological failure in this
literature.

**3. Check the peak is interior.** If the winner sits at the smallest or
largest grid value, the true optimum is outside your range and the baseline is
under-tuned. `analyze` prints an explicit warning:

```
GRID BOUNDARY WARNINGS - the search range is too narrow:
  [sam] best rho=0.5 is the LARGEST value in the grid [0.01, ..., 0.5]
        - extend the grid upward
```

Extend and re-run. Do not report a boundary-selected baseline.

**4. Check the peak is real, not noise.** Look at
`figures/01_rho_sensitivity.png`. If the gap between the best and second-best
rho is smaller than the seed standard deviation, rho is **not identified** by
your data. The honest report is then "performance was insensitive to rho over
`[a, b]`; we used the validation-selected value", not a confident single number.
This is a legitimate and publishable finding — it is also what the figure is
for.

**5. Confirm on held-out seeds.** The two-stage protocol exists for exactly
this: stage 1 selects on 3 seeds, stage 2 re-runs the winners on all 10. If the
winner does not survive the larger seed set, the selection was overfitting to
seed noise.

**6. Publish the sensitivity curve, not just the winner.** Reviewers want to
know whether the method is robust to rho or only works at one magic value.
Figure 01 answers that directly, and pre-empts "did you cherry-pick rho?".

One more check worth running: **does rho do what it claims?**
`figures/04_sharpness_vs_rho.png` plots measured sharpness against rho. If
increasing rho does not reduce the SAM gap, the regularizer is not working and
any accuracy difference has some other cause.

---

## 4. Protocol

### Epochs

100 epochs × 429 steps = 42,900 updates, identical for every arm. Sharpness
methods separate from the control late in training, so 50 epochs can understate
them; 100 costs about a minute extra per run here. Set `max_steps` instead of
`epochs` if you need to step-match rather than epoch-match.

### Seeds — and a warning about effect size

The default model has **15,910 parameters for 55,000 training examples**. It is
*underparameterized*: it cannot interpolate the training set, so it will not
reach the sharp interpolating minima that sharpness-aware methods exist to
avoid. **Expect small effects on clean MNIST** — plausibly 0.1–0.4% test
accuracy, against a seed-to-seed std of 0.1–0.2%.

Minimum detectable effect, paired *t*-test, alpha = .05, 80% power,
sigma_diff ≈ 0.15%:

| seeds | MDE |
|---|---|
| 3 | ≈ 0.35% |
| 5 | ≈ 0.25% |
| **10** | **≈ 0.15%** |
| 20 | ≈ 0.10% |

With 3 seeds you would likely conclude "SAM does not help" when it does. Use
10. They cost about 20 minutes.

Seed pairing is what makes those numbers achievable: every arm at a given seed
gets **bit-identical initial weights** (`models.make_model`), so the paired
test removes initialization variance. `analyze` reports the paired *t* p-value
and Cohen's *dz*.

### If the effect is still too small to resolve

That is a fact about the *setup*, not the methods. Two companion configs ship
with the repo:

- **`configs/wide.yaml`** — hidden=200 (159k params, overparameterized). The
  regime where sharpness methods have room to work. Running both widths lets
  you show the effect scales with capacity, which is a much stronger claim than
  a single width.
- **`configs/label_noise.yaml`** — 20% symmetric label noise on train only.
  The standard stress test for flat-minima claims; effect sizes go from tenths
  of a percent to several percent.

---

## 5. What the pipeline produces

### Tables

`analyze` prints the selection report (top-k per arm), boundary warnings, and
the final table, and exports LaTeX + CSV:

```
arm          test acc (%)    delta         p      dz   gen gap    |grad|   SAM gap      passes
adamw        93.41 +- 0.14    +0.00    1.0000    0.00      1.32    0.0891    0.0412       42900
sam          93.66 +- 0.11    +0.25    0.0031**  1.42      0.94    0.0412    0.0089       85800
```

### Figures

| File | Question it answers |
|---|---|
| `01_rho_sensitivity` | How was rho chosen, and is the choice robust? |
| `02_val_vs_test` | Is validation a reliable selector, or is selection noisy? |
| `03_training_curves` | Do arms differ in optimization or only generalization? |
| `04_sharpness_vs_rho` | Does the regularizer actually flatten the minimum? |
| `05_accuracy_vs_sharpness` | Is flatness associated with generalization *here*? |
| `06_gsam_heatmap` | The rho × alpha interaction a 1-D sweep would hide. |
| `07_summary` | Headline comparison, per-seed dots, 95% CIs. |
| `08_secondary_effects` | Accuracy, generalisation gap and sharpness side by side. **Lead with this when accuracy comes back null.** |

All in PNG (300 dpi) and PDF (vector, for `\includegraphics`), colour-blind-safe
palette, consistent arm colours across panels.

### Diagnostics recorded per run

Beyond accuracy: `final_grad_norm_train`, `sam_gap_rho{r}` (empirical sharpness
by projected gradient ascent), optional `hessian_lambda_max` (power iteration),
`generalization_gap`, `fwd_bwd_passes`, `peak_memory_mb`, `wall_seconds`, and
full per-epoch history.

**The sharpness diagnostics are the stronger scientific result.** An accuracy
delta of 0.25% is arguable; a 5× reduction in measured sharpness is not, and it
remains interpretable even when the accuracy difference is statistically
marginal.

---

## 6. Project layout

```
sharpbench-project/
├── sharpbench/
│   ├── cli.py            single entry point, all subcommands
│   ├── device.py         CUDA/memory/determinism, one place for GPU issues
│   ├── data.py           GPU-resident MNIST, fixed split, label noise
│   ├── models.py         MLP + sharpness diagnostics
│   ├── optimizers.py     SAM / ASAM / GSAM / GAM wrappers
│   ├── train.py          one run -> JSON record
│   ├── sweep.py          grid, resume, shard, two-stage
│   ├── analyze.py        selection, paired stats, boundary checks, export
│   └── plots.py          seven figures
├── configs/
│   ├── quick.yaml        5-minute pipeline validation
│   ├── sweep_grid.yaml   the main experiment
│   ├── refine.yaml       widened grids after boundary warnings
│   ├── compute_matched.yaml  AdamW at 2x/4x epochs
│   ├── wide.yaml         hidden=200 companion
│   └── label_noise.yaml  20% label noise companion
├── scripts/
│   ├── run_all.ps1       Windows parallel shards
│   └── run_all.sh        Linux/macOS parallel shards
├── .vscode/launch.json   debug configs for every stage
├── QUICKSTART.md
└── requirements.txt
```

### GPU handling

Everything CUDA-related goes through `device.py`:

- `--device auto|cpu|cuda|cuda:1|0`, with a **loud warning** on silent CPU
  fallback (the usual cause is the CPU-only wheel)
- free-memory check before allocating, with a legible error instead of a
  mid-sweep OOM
- `--memory-fraction` to cap a process so parallel shards do not starve
  each other
- TF32 disabled and cuDNN determinism enabled by default
- peak memory recorded in every result record

Peak usage is ~300 MB including the resident dataset, so a 6 GB card runs
several shards comfortably. `check` tells you exactly how many.

---

## 7. Known limitations

- **GAM** follows the published algorithm but has not been diffed
  line-by-line against the authors' released code. Verify against a reference
  implementation before publishing GAM numbers.
- **Statistics** use a paired *t*-test, which assumes roughly normal
  differences. With 10 seeds this is usually fine; for a non-parametric
  alternative, a Wilcoxon signed-rank test on the same paired differences would
  be the natural addition.
- **The clean-MNIST effect size is small** by construction — see §4. Do not
  build strong conclusions on the default config alone.
- **Single dataset.** Claims about sharpness-aware methods generally need more
  than one dataset; this repo is a controlled study of one setting, not a
  survey.
