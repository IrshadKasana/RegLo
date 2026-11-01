# Published hyperparameter values, and what they mean for your setup

## 1. What the papers actually report

All values below are **CIFAR-10/100 with ResNet-18 or WRN-28-10, base
optimizer SgdM (momentum 0.9)**. Sources: Foret et al. ICLR'21 (SAM),
Kwon et al. ICML'21 (ASAM), Zhuang et al. ICLR'22 (GSAM), Zhang et al.
CVPR'23 (GAM), Yue et al. KDD'23 (WSAM — whose Tables 3 and 5 are the
single most useful summary, since they tune every method on one protocol).

| method | knob | search range used | value selected |
|---|---|---|---|
| SAM | rho | {0.01, 0.02, 0.05, 0.1, 0.2, 0.5} | **0.05** (Foret default) / **0.2** (WSAM paper, ResNet18+WRN, both C10 and C100) |
| ASAM | rho | {0.1, 0.5, 1.0 … 6.0} | **0.5–1.0** typical; WSAM paper selected **5.0** on WRN/C10 |
| GSAM | rho | same scale as SAM | **0.05–0.2** |
| GSAM | alpha | {0.01, 0.02, 0.03, 0.1, 0.2, 0.3} | **0.02** (RN18/C10), 0.03 (RN18/C100), 0.01 (WRN/C10), 0.2 (WRN/C100) |
| WSAM | gamma | {0.5, 0.6, 0.7, 0.8, 0.82 … 0.96} | **0.88** (RN18/C10), 0.82 (RN18/C100), 0.88 (WRN/C10), 0.94 (WRN/C100) |
| GAM | grad_rho | ~0.02–0.05 | **0.02–0.04** |
| GAM | grad_norm_rho | ~0.1–0.2 | **0.1–0.2** (≈ 5 × grad_rho) |
| GAM | grad_beta_0..3 | **never swept** | **(1, 1, −1, 1)** — these define the method |

Under label noise the optimal rho **shrinks** as noise grows
(WSAM Table 5): rho = 0.2 at 20–40% noise, 0.1 at 60%, 0.05 at 80%.

If you want one number per method as a starting point, the most defensible
"community default" is:

```
SAM    rho = 0.05          (Foret's headline value)
ASAM   rho = 0.5           (~10x SAM, because of the |w| rescaling)
GSAM   rho = 0.05, alpha = 0.02
WSAM   rho = 0.05, gamma = 0.88   (lambda = 7.33)
GAM    grad_rho = 0.02, grad_norm_rho = 0.1, grad_gamma = 0.03
```

## 2. Why you still cannot just copy them

Two mismatches, both of which change the optimum:

**Base optimizer.** Every value above was tuned with **SgdM at lr 0.05–0.1**.
You use **AdamW at lr ~1e-3**. AdamW rescales each coordinate by its own
second moment, so the effective step geometry — and therefore the useful
perturbation radius — is not the same. Expect your optimum near, not on,
the published value.

**Model and dataset.** Nobody reports SAM/GAM/GSAM numbers for an MLP on
MNIST. Your model has 15,910 parameters and cannot interpolate 55,000
examples; CIFAR ResNet-18 has 11.2M and does. Sharpness-aware methods exist
to avoid sharp *interpolating* minima, so the regime is genuinely different.

That is exactly why the sweep exists. Use the published ranges as the grid
(which `configs/mnist_lambda.yaml` does), then report your own selected value
and say it was chosen on validation.

## 3. What rho means numerically on YOUR model

Measured on the MLP 784-20-10, ‖w‖₂ = 3.19 at initialisation:

| rho | ‖ε‖ / ‖w‖ | comment |
|---|---|---|
| 0.01 | 0.31 % | very weak |
| 0.02 | 0.63 % | |
| 0.05 | 1.57 % | Foret's default |
| 0.1 | 3.13 % | |
| 0.2 | 6.26 % | WSAM paper's choice |
| 0.5 | 15.7 % | too large — this is where SAM collapses |

**rho = 0.1 is a 3% radius, not 10%.** It is a Euclidean distance in weight
space, not a percentage of anything. Note ‖w‖ grows during training, so the
relative perturbation shrinks as training proceeds.

**Important exception:** for ASAM (`adaptive: True`) the perturbation is
rescaled elementwise by |w|, so its rho behaves like a *relative* scale.
That is why ASAM needs rho ≈ 0.5–5 while SAM needs 0.05. **The two numbers
are not comparable and must never share a grid.**

## 4. Your three baseline YAMLs — what is wrong

### `adamw_sam_baseline.yaml`
```yaml
adaptive: True      # <-- ASAM
rho: 0.05           # <-- vanilla-SAM scale
```
**Scale mismatch.** With `adaptive: True`, rho = 0.05 is roughly 1/10 to
1/100 of the useful ASAM radius, so this is close to running plain AdamW at
double the compute. Pick one:

```yaml
# Option A (recommended primary baseline: what "AdamW+SAM" normally means)
adaptive: False
rho: 0.05

# Option B (if and only if your method's SAM branch is also adaptive)
adaptive: True
rho: 0.5
```

### `adamw_gsam_c10_r18.yaml`
```yaml
adaptive: True
rho_min: 0.5        # \
rho_max: 2.0        #  > cannot all three apply
rho: 0.05           # /
alpha: 0.4
```
Three problems: the same adaptive/rho scale mismatch; `rho_min`/`rho_max`
coexisting with a constant `rho` (only one can be read — in this codebase the
schedule is **not implemented**, so the min/max are silently ignored); and
alpha = 0.4 sits above the published range, whose optimum is 0.01–0.03.

```yaml
adaptive: False
rho: 0.05
alpha: 0.02
# rho_min / rho_max removed
```

### `adamw_gam_baseline.yaml`
```yaml
grad_rho: 0.04
grad_rho_max: 0.04       # redundant with grad_rho
grad_rho_min: 0.02
grad_norm_rho: 0.2
grad_norm_rho_max: 0.4
grad_norm_rho_min: 0.2
grad_gamma: 0.03         # frozen
```
The min/max keys are dead. And `grad_gamma` **is GAM's penalty weight α** in
L(w) + α·R(w) — freezing it at 0.03 means the gradient-norm penalty enters
every update at 3 % strength, which makes GAM look inert. Sweep it.

```yaml
adaptive: True
grad_rho: 0.04
grad_norm_rho: 0.2
grad_beta_0: 1.0
grad_beta_1: 1.0
grad_beta_2: -1.0
grad_beta_3: 1.0
grad_gamma: 0.03    # SWEEP over [0.03, 0.1, 0.3, 1.0]
```

`build_optimizer` now **raises** on all of these dead keys rather than
ignoring them, so a config can no longer claim behaviour it never runs.
