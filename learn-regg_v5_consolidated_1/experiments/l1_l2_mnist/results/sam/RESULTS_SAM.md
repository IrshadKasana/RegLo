# MNIST, MLP optimizee: SAM as the regularizer

3 seeds (each = fresh LO meta-training and 3 optimizee inits); mean ± std over seeds; test accuracy after 1000 optimizee steps. Methods 1 and 3 are re-used from the L1/L2 run (`results/full`, same seeds and eval inits).

| Method | hyper-params | Test acc (%) | Test NLL | Val acc (%) | Train acc (%) | per-seed test acc |
|---|---|---|---|---|---|---|
| 1. MLP + AdamW | lr=0.01 | **93.16 ± 0.02** | 0.2320 ± 0.0042 | 92.57 ± 0.21 | 96.59 ± 0.10 | 93.14, 93.15, 93.18 |
| 2. MLP + AdamW + SAM | lr=0.01, ρ=1 (adaptive) | **93.09 ± 0.07** | 0.2345 ± 0.0048 | 92.84 ± 0.08 | 95.91 ± 0.06 | 93.06, 93.16, 93.04 |
| 3. MLP + learned optimizer | meta-lr=1e-3 | **93.25 ± 0.40** | 0.2597 ± 0.0291 | 92.87 ± 0.41 | 95.80 ± 0.30 | 93.64, 92.84, 93.28 |
| 4. MLP + LO + SAM | λ_r=1, ρ=0.05, + smoothing | **93.80 ± 0.23** | 0.2144 ± 0.0129 | 93.39 ± 0.09 | 96.19 ± 0.37 | 93.70, 94.06, 93.63 |

Paired test-accuracy differences (same optimizee init + batch order):

- (2) AdamW+SAM − (1) AdamW : -0.07 ± 0.08 pp (1/3 seeds better)
- (4) LO+SAM − (3) LO       : +0.54 ± 0.60 pp (3/3 seeds better)
- (3) LO − (1) AdamW        : +0.10 ± 0.41 pp (2/3 seeds better)
- (4) LO+SAM − (2) AdamW+SAM: +0.71 ± 0.16 pp (3/3 seeds better)

Diagnostic (LO with smoothing only, no SAM): 92.94 ± 0.94 %.

λ_r selection (mean validation accuracy over seeds): 0.01: 92.74, 0.1: 93.06, 1: 93.39
