# Full λ-sweep table — MNIST, MLP 784-20-10

Validation accuracy decides everything (3 000 held-out images); test accuracy (10 000) is shown for information only. Bold = best λ of that method on validation accuracy.

AdamW arms: ± = SE over 6 optimizee inits; best (lr, wd) chosen per λ on validation. LO arms: ± = SE across meta-training seeds (each seed = an independent LO).

## λ = 0 references

| Method | config | seeds | val acc % | val NLL | test acc % |
|---|---|---|---|---|---|
| 1. AdamW | lr=0.01, wd=0.1 | 6 inits | 92.76 ± 0.22 | 0.2415 | 93.27 ± 0.15 |
| 3. LO | meta-lr 1e-3 | 6 meta-seeds | 92.67 ± 0.12 | 0.2559 | 93.20 ± 0.10 |
| (diag) LO + smoothing only | meta-lr 1e-3 | 6 meta-seeds | 92.41 ± 0.15 | 0.2640 | 92.89 ± 0.16 |

## L1 regularizer

| λ | **M2** AdamW+L1: lr | wd | val acc % | val NLL | test acc % | **M4** LO+L1: val acc % (stage 1, n=2) | val acc % (fresh) | n fresh | val NLL (stage 1) | test acc % (stage 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-06 | 0.01 | 0.1 | 92.78 ± 0.17 | 0.2391 | 93.38 ± 0.09 | 93.01 ± 0.14 | 92.19 ± 0.36 | 4 | 0.2358 | 93.40 ± 0.10 |
| 3e-06 | 0.01 | 0.3 | 92.73 ± 0.14 | 0.2645 | 93.23 ± 0.12 | 92.21 ± 1.09 | – | – | 0.2632 | 92.35 ± 1.09 |
| 1e-05 | 0.01 | 0.3 | 92.81 ± 0.21 | 0.2658 | 93.22 ± 0.12 | 92.63 ± 0.62 | – | – | 0.2519 | 93.04 ± 0.54 |
| 3e-05 | 0.01 | 0.1 | 93.05 ± 0.11 | 0.2308 | 93.52 ± 0.15 | 92.48 ± 0.27 | – | – | 0.2685 | 92.82 ± 0.27 |
| **0.0001** | 0.01 | 0 | 93.65 ± 0.19 | 0.2105 | 93.86 ± 0.10 | 92.32 ± 0.01 | – | – | 0.2822 | 92.52 ± 0.17 |
| 0.0003 | 0.01 | 0 | 93.47 ± 0.18 | 0.2173 | 93.75 ± 0.08 | 92.92 ± 0.37 | **92.40 ± 0.21** | 4 | 0.2398 | 93.33 ± 0.31 |
| 0.001 | 0.01 | 0 | 92.23 ± 0.11 | 0.2813 | 92.56 ± 0.19 | 92.52 ± 0.34 | – | – | 0.2445 | 92.95 ± 0.37 |
| 0.003 | 0.01 | 0 | 89.49 ± 0.18 | 0.4259 | 90.22 ± 0.15 | 92.64 ± 0.19 | 91.91 ± 0.56 | 4 | 0.2480 | 93.14 ± 0.10 |

**Best λ — method 2: 0.0001** (lr=0.01, wd=0), val 93.65 ± 0.19 %. **Method 4: 0.0003** (fresh meta-seeds).

## L2 regularizer

| λ | **M2** AdamW+L2: lr | wd | val acc % | val NLL | test acc % | **M4** LO+L2: val acc % (stage 1, n=2) | val acc % (fresh) | n fresh | val NLL (stage 1) | test acc % (stage 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-05 | 0.01 | 0.1 | 92.94 ± 0.18 | 0.2391 | 93.47 ± 0.12 | 91.80 ± 0.08 | – | – | 0.2889 | 91.93 ± 0.02 |
| 3e-05 | 0.01 | 0.01 | 92.87 ± 0.19 | 0.2392 | 93.22 ± 0.16 | 92.54 ± 1.14 | – | – | 0.2614 | 92.83 ± 0.96 |
| 0.0001 | 0.01 | 0 | 92.94 ± 0.15 | 0.2365 | 93.31 ± 0.12 | 91.57 ± 0.89 | – | – | 0.3067 | 91.84 ± 0.97 |
| **0.0003** | 0.01 | 0 | 93.19 ± 0.16 | 0.2302 | 93.62 ± 0.17 | 92.98 ± 0.12 | 93.03 ± 0.15 | 4 | 0.2410 | 93.33 ± 0.29 |
| 0.001 | 0.01 | 0 | 92.55 ± 0.18 | 0.2779 | 93.03 ± 0.13 | 92.79 ± 0.07 | **93.07 ± 0.12** | 4 | 0.2358 | 93.34 ± 0.26 |
| 0.003 | 0.003 | 0 | 91.16 ± 0.10 | 0.3861 | 91.49 ± 0.09 | 92.99 ± 0.02 | 92.67 ± 0.39 | 4 | 0.2307 | 93.41 ± 0.03 |
| 0.01 | 0.003 | 0 | 89.18 ± 0.19 | 0.6224 | 89.74 ± 0.12 | 92.36 ± 0.12 | – | – | 0.2466 | 93.03 ± 0.03 |
| 0.03 | 0.003 | 0 | 84.21 ± 0.43 | 1.1058 | 84.64 ± 0.33 | 92.29 ± 0.12 | – | – | 0.2581 | 92.69 ± 0.09 |

**Best λ — method 2: 0.0003** (lr=0.01, wd=0), val 93.19 ± 0.16 %. **Method 4: 0.001** (fresh meta-seeds).

## Summary — best configuration per method (selected on validation)

| Method | regularizer | best λ | other hyper-params | val acc % | test acc % | basis |
|---|---|---|---|---|---|---|
| 1. AdamW | – | – | lr=0.01, wd=0.1 | 92.76 ± 0.22 | 93.27 ± 0.15 | 6 inits |
| 2. AdamW + L1 | L1 | **0.0001** | lr=0.01, wd=0 | 93.65 ± 0.19 | 93.86 ± 0.10 | 6 inits |
| 2. AdamW + L2 | L2 | **0.0003** | lr=0.01, wd=0 | 93.19 ± 0.16 | 93.62 ± 0.17 | 6 inits |
| 3. LO | – | – | meta-lr 1e-3 | 92.67 ± 0.12 | 93.20 ± 0.10 | 6 meta-seeds |
| 4. LO + L1 | L1 | **0.0003** | + smoothing | 92.40 ± 0.21 | 92.96 ± 0.24 | fresh meta-seeds (n=4) |
| 4. LO + L2 | L2 | **0.001** | + smoothing | 93.07 ± 0.12 | 93.44 ± 0.16 | fresh meta-seeds (n=4) |

*Method 4 contains the smoothing term (as the proposed model does); compare it with the 'LO + smoothing only' row to isolate the regularizer's effect.*
