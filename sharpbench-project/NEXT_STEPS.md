# NEXT_STEPS — reading your first full run, and what to fix

Your stage-2 table (10 seeds):

```
arm      test acc (%)   delta       p      dz   gen gap   |grad|   SAM gap    passes
adamw    95.99 ± 0.23   +0.00   1.0000    0.00     3.09   0.0129   0.0093     42900
asam     95.98 ± 0.22   -0.01   0.8863   -0.05     1.91   0.0665   0.0051     85800
gam      96.01 ± 0.14   +0.02   0.8205    0.07     1.77   0.0264   0.0027    171600
gsam     96.15 ± 0.27   +0.16   0.1725    0.47     1.67   0.0234   0.0022     85800
sam      96.11 ± 0.25   +0.12   0.3560    0.31     1.24   0.0404   0.0025     85800
```

---

## 1. You have a result — it just isn't in the accuracy column

Nothing reaches significance on accuracy (best p = 0.17). But look at the two
columns to the right:

| | AdamW | best regularized | change |
|---|---|---|---|
| generalization gap | 3.09 pp | 1.24 pp (SAM) | **cut by 60%** |
| empirical sharpness | 0.0093 | 0.0022 (GSAM) | **4.2× flatter** |

Every single arm cut the gap by 40–60% and flattened the minimum by 2–4×.
That is not noise, and it is exactly the mechanism these methods claim.

The honest framing: *on an underparameterised MLP at MNIST's accuracy ceiling,
sharpness-aware regularizers measurably flatten the minimum and roughly halve
the generalisation gap, but the effect does not convert into a statistically
detectable test-accuracy gain.* That is a publishable, defensible sentence.
Straining +0.16 pp at p = 0.17 into "GSAM improves accuracy" is not.

`analyze` now prints a **power analysis** and a **secondary-metrics table** so
you can state this precisely, and `plot` writes a new
`figures/08_secondary_effects.png` that shows all three metrics side by side.

---

## 2. The GAM design flaw — this one was mine

GAM's objective is

$$\min_w\; L(w) + \alpha\,R(w), \qquad R(w)=\rho\max_{\lVert\epsilon\rVert\le\rho}\lVert\nabla L(w+\epsilon)\rVert$$

and in the code **`grad_gamma` *is* that α**. My original config pinned it at
the published default `0.03` and swept only the radius. So GAM's entire
gradient-norm penalty entered every update at 3% weight, and the optimizer had
no way to turn it up.

You can see the consequence in your own output: GAM pushed `grad_rho` to
`0.16`, the grid maximum, and triggered a boundary warning. **That is what a
method does when its strength knob is frozen too low — it tries to compensate
with radius.** Radius and weight are not interchangeable, so it couldn't.

Result: GAM spent **4× the compute of AdamW (171,600 passes) for +0.02 pp.**

In README §2 I listed GAM's tunables as "radius *and* α (grad_gamma)" and then
didn't sweep α. `configs/refine.yaml` fixes it: `grad_gamma: [0.03, 0.1, 0.3, 1.0]`.

To be clear about what is still off-limits: `grad_beta_0..3` define GAM's
gradient combination and must stay at `(1, 1, −1, 1)`. `grad_gamma` is a
legitimate hyperparameter. Different things.

---

## 3. Every regularized arm is learning-rate-limited

```
[asam] best lr=0.0003 is the SMALLEST value in the grid
[gam]  best lr=0.0003 is the SMALLEST value in the grid
[gsam] best lr=0.0003 is the SMALLEST value in the grid
[sam]  best lr=0.0003 is the SMALLEST value in the grid
```

Four out of four, while the AdamW control picked `1e-3` — comfortably interior.
That is a systematic signal, not noise: these methods change the effective
gradient scale, so they want a smaller step. **All four were under-tuned**, and
their numbers above are a floor, not a ceiling. Extended down to `3e-5`.

GSAM's `alpha=0.01` was also at its grid minimum. Extended to `0.001`.

---

## 4. Commands

First, replace `sharpbench/`, `configs/`, and `scripts/` from the new zip.
**Keep your `results/` folder** — the sweep is keyed on the full hyperparameter
tuple, so everything you already ran is detected and skipped. I verified this:
re-expanding the `adamw` arm under the new config finds 6 of 9 already done.

```powershell
# 1. The result you already have, with power analysis + secondary metrics
python -m sharpbench analyze --results "results/stage2_*.jsonl" `
       --latex tables/main.tex --csv tables/main.csv

python -m sharpbench plot --results "results/stage1_*.jsonl" "results/stage2_*.jsonl" `
       --min-acc 90

# 2. Refinement -- cheap arms first (~1-2 h on 4 shards; most is skipped)
python -m sharpbench sweep --config configs/refine.yaml --stage1 `
       --arms adamw sam asam gsam --shard 0/4 --out results/stage1_0.jsonl
#   ...or all four shards:
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -Shards 4 -Stage 1 -Config configs/refine.yaml

# 3. GAM separately -- 4 passes/step, run it overnight (~2.5 h on 4 shards)
python -m sharpbench sweep --config configs/refine.yaml --stage1 --arms gam `
       --shard 0/4 --out results/stage1_0.jsonl

# 4. Re-select, confirm the boundary warnings are gone
python -m sharpbench analyze --results "results/stage1_*.jsonl" `
       --emit-winners configs/winners.yaml --top-k 5

# 5. New winners on 10 seeds
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -Shards 4 -Stage 2 -Config configs/refine.yaml

# 6. Final table + figures
python -m sharpbench analyze --results "results/stage2_*.jsonl" `
       --latex tables/main.tex --csv tables/main.csv
python -m sharpbench plot --results "results/stage1_*.jsonl" "results/stage2_*.jsonl" --min-acc 90
```

Refinement cost with per-arm LR grids (new feature — expensive arms no longer
search learning rates the cheap arms already ruled out):

| arm | configs | runs | serial |
|---|---|---|---|
| adamw | 3 | 9 | 0.2 h |
| sam | 24 | 72 | 3.6 h |
| asam | 20 | 60 | 3.0 h |
| gsam | 64 | 192 | 12.6 h |
| gam | 32 | 96 | 9.2 h |
| **total** | **143** | **429** | **7.1 h on 4 shards** |

minus roughly 300 runs already on disk and skipped automatically.

---

## 5. If you want a result that shows up in the accuracy column

Refinement will tighten the numbers but is unlikely to move a 0.16 pp effect
past significance — your MDE at 10 seeds is around 0.2 pp. Two options, in
order of value:

```powershell
# 20% label noise -- the standard stress test for flat-minima claims.
# Effect sizes go from tenths of a percent to several percent.
python -m sharpbench sweep --config configs/label_noise.yaml --stage1 --shard 0/4 --out results/noise_0.jsonl

# hidden=200 (159k params) -- overparameterised, so the model can actually
# reach the sharp interpolating minima these methods exist to avoid
python -m sharpbench sweep --config configs/wide.yaml --stage1 --shard 0/4 --out results/wide_0.jsonl
```

Reporting all three settings — clean/underparameterised, wide, noisy — is a
much stronger contribution than one null table, because it shows *when* the
methods pay off rather than just whether they did once.

There is also `configs/compute_matched.yaml` (AdamW at 200 and 400 epochs) for
the fairness question: SAM used 2× your control's gradient evaluations and GAM
used 4×. Right now GAM's +0.02 pp for 4× compute is the honest headline for
that method, and volunteering the comparison is stronger than having it
extracted from you.

---

## 6. Everything else that changed

| Fix | Why |
|---|---|
| `--min-acc` on `plot` | Your `--min-acc 90` failed — that flag was added after you downloaded. Splits the flatness-vs-accuracy figure into all-runs and converged-only panels, because the diverged runs flip the sign of the correlation. |
| Wilcoxon signed-rank | Non-parametric complement to the paired *t*. With 10 seeds, normality is not checkable; showing both agree closes off that criticism. Verified exact against 2/1024. |
| Power analysis / MDE | Makes a null result interpretable. "p = 0.36" says nothing; "p = 0.36, MDE = 0.23 pp" says the experiment was underpowered for that effect. |
| Secondary-metrics table | Gen gap and sharpness with paired significance — where your real result is. |
| Figure 08 | Accuracy, gap, and sharpness side by side, seed-paired. |
| Per-arm `lr_grid` | Lets GAM skip learning rates already ruled out. Difference between a 3-hour and a 40-hour refinement. |
| Sweep progress line | Now prints `rho`/`alpha`/`grad_rho`/`grad_gamma`. Previously two runs differing only in rho printed identically and you couldn't tell from the log which number belonged to which config. |

One thing worth noting from your throughput numbers: GAM measured **4.9×** the
cost of AdamW on your 3050, not the theoretical 4×, and GSAM measured 3.3×
rather than 2×. Small-kernel launch overhead on a laptop GPU. Budget from
`check`'s measured numbers rather than the pass counts.

Also: ASAM's `|grad|` of 0.0665 is 5× *higher* than AdamW's while its SAM gap
is 2× *lower*. Not a bug — ASAM optimises in a `|w|`-rescaled geometry, so the
plain Euclidean gradient norm isn't the right yardstick for it. Compare ASAM on
the SAM-gap column, not the gradient-norm column.
