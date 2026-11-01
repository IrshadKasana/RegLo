# λ sweep — MNIST, MLP 784-20-10, L1 and L2 regularizers

Selection criterion: **mean validation accuracy** (3 000 held-out images, never trained on); ties broken by validation NLL. AdamW arms: `±` is the SE over 6 optimizee inits. Learned-optimizer arms: `±` is the SE **across meta-training seeds** (each seed is an independent LO; its 3 inits share it). Method-4 finalists are re-trained on fresh meta-seeds and the reported best λ is decided on those fresh seeds only. Test accuracy is listed for the selected configuration only, as a sanity check.

## L1

### Method 2 — AdamW + λ·L1  (best lr, wd per λ; 160 configs)

| λ | best lr | best wd | val acc % | ± SE | val NLL | ‖θ‖₁ | frac(|θ|<1e-3) |
|---|---|---|---|---|---|---|---|
| 1e-06 | 0.01 | 0.1 | 92.78 | 0.17 | 0.2391 | 2240 | 0.009 |
| 3e-06 | 0.01 | 0.3 | 92.73 | 0.14 | 0.2645 | 1420 | 0.017 |
| 1e-05 | 0.01 | 0.3 | 92.81 | 0.21 | 0.2658 | 1291 | 0.033 |
| 3e-05 | 0.01 | 0.1 | 93.05 | 0.11 | 0.2308 | 1384 | 0.052 |
| 0.0001 | 0.01 | 0 | 93.65 | 0.19 | 0.2105 | 1057 | 0.085 | ◀ best
| 0.0003 | 0.01 | 0 | 93.47 | 0.18 | 0.2173 | 635 | 0.142 |
| 0.001 | 0.01 | 0 | 92.23 | 0.11 | 0.2813 | 370 | 0.208 |
| 0.003 | 0.01 | 0 | 89.49 | 0.18 | 0.4259 | 219 | 0.304 |

### Method 4 — LO + λ·L1 (+ smoothing)

| λ | meta-seeds | val acc % (all) | ± SE | fresh seeds | val acc % (fresh) | ± SE | val NLL (all) | ‖θ‖₁ |
|---|---|---|---|---|---|---|---|---|
| 1e-06 | 2 | 93.01 | 0.14 | – | – | – | 0.2358 | 2151 | ◀ best
| 3e-06 | 2 | 92.21 | 1.09 | – | – | – | 0.2632 | 2257 |
| 1e-05 | 2 | 92.63 | 0.62 | – | – | – | 0.2519 | 2851 |
| 3e-05 | 2 | 92.48 | 0.27 | – | – | – | 0.2685 | 3205 |
| 0.0001 | 2 | 92.32 | 0.01 | – | – | – | 0.2822 | 2759 |
| 0.0003 | 2 | 92.92 | 0.37 | – | – | – | 0.2398 | 1674 |
| 0.001 | 2 | 92.52 | 0.34 | – | – | – | 0.2445 | 2016 |
| 0.003 | 2 | 92.64 | 0.19 | – | – | – | 0.2480 | 885 |

**Best λ**

- Method 2: **λ = 0.0001** (lr=0.01, wd=0) — val 93.65 ± 0.19 %; within 1 SE: 0.0001, 0.0003
- Method 4: **λ = 1e-06** — val 93.01 ± 0.14 % (stage-1 meta-seeds (NOT confirmed)); within 1 SE: 1e-06, 0.0003
  - ⚠ best λ is at the **edge** of the grid — extend the grid before trusting it.

## L2

### Method 2 — AdamW + λ·L2  (best lr, wd per λ; 160 configs)

| λ | best lr | best wd | val acc % | ± SE | val NLL | ‖θ‖₁ | frac(|θ|<1e-3) |
|---|---|---|---|---|---|---|---|
| 1e-05 | 0.01 | 0.1 | 92.94 | 0.18 | 0.2391 | 2151 | 0.008 |
| 3e-05 | 0.01 | 0.01 | 92.87 | 0.19 | 0.2392 | 2502 | 0.007 |
| 0.0001 | 0.01 | 0 | 92.94 | 0.15 | 0.2365 | 1826 | 0.012 |
| 0.0003 | 0.01 | 0 | 93.19 | 0.16 | 0.2302 | 1218 | 0.018 | ◀ best
| 0.001 | 0.01 | 0 | 92.55 | 0.18 | 0.2779 | 779 | 0.027 |
| 0.003 | 0.003 | 0 | 91.16 | 0.10 | 0.3861 | 524 | 0.039 |
| 0.01 | 0.003 | 0 | 89.18 | 0.19 | 0.6224 | 367 | 0.076 |
| 0.03 | 0.003 | 0 | 84.21 | 0.43 | 1.1058 | 235 | 0.125 |

### Method 4 — LO + λ·L2 (+ smoothing)

| λ | meta-seeds | val acc % (all) | ± SE | fresh seeds | val acc % (fresh) | ± SE | val NLL (all) | ‖θ‖₁ |
|---|---|---|---|---|---|---|---|---|
| 1e-05 | 2 | 91.80 | 0.08 | – | – | – | 0.2889 | 4219 |
| 3e-05 | 2 | 92.54 | 1.14 | – | – | – | 0.2614 | 3393 |
| 0.0001 | 2 | 91.57 | 0.89 | – | – | – | 0.3067 | 4155 |
| 0.0003 | 2 | 92.98 | 0.12 | – | – | – | 0.2410 | 2243 |
| 0.001 | 2 | 92.79 | 0.07 | – | – | – | 0.2358 | 1564 |
| 0.003 | 2 | 92.99 | 0.02 | – | – | – | 0.2307 | 1595 | ◀ best
| 0.01 | 2 | 92.36 | 0.12 | – | – | – | 0.2466 | 1965 |
| 0.03 | 2 | 92.29 | 0.12 | – | – | – | 0.2581 | 1360 |

**Best λ**

- Method 2: **λ = 0.0003** (lr=0.01, wd=0) — val 93.19 ± 0.16 %; within 1 SE: 0.0003
- Method 4: **λ = 0.003** — val 92.99 ± 0.02 % (stage-1 meta-seeds (NOT confirmed)); within 1 SE: 0.0003, 0.003

## Paired validation-accuracy differences at the selected λ

AdamW rows: same optimizee inits and batch order on both sides. LO rows: paired by meta-seed (fresh seeds when available). ± is the SE of the paired differences.

- **L1**: (2) AdamW+L1 − (1) AdamW = +0.89 ± 0.35 pp  (5/6 paired runs better)
  - (4) LO+L1 − (3) LO = +0.21 ± 0.06 pp  (2/2 paired meta-seeds better)
  - (4) LO+L1 − LO+smoothing-only = +0.80 ± 0.21 pp  (2/2 paired meta-seeds better)
  - (4) LO+L1 − (2) AdamW+L1 = -0.64 ± 0.24 pp  (unpaired: different seed sets)
- **L2**: (2) AdamW+L2 − (1) AdamW = +0.43 ± 0.21 pp  (5/6 paired runs better)
  - (4) LO+L2 − (3) LO = +0.19 ± 0.22 pp  (1/2 paired meta-seeds better)
  - (4) LO+L2 − LO+smoothing-only = +0.78 ± 0.05 pp  (2/2 paired meta-seeds better)
  - (4) LO+L2 − (2) AdamW+L2 = -0.19 ± 0.16 pp  (unpaired: different seed sets)

## λ = 0 references (no explicit regularizer)

| Method | config | val acc % | ± SE | val NLL | test acc % |
|---|---|---|---|---|---|
| 1. AdamW | lr=0.01, wd=0.1 | 92.76 | 0.22 | 0.2415 | 93.27 |
| 3. LO | 2 meta-seeds | 92.80 | 0.20 | 0.2509 | 93.35 |
| (diagnostic) LO + smoothing only | 2 meta-seeds | 92.21 | 0.07 | 0.2812 | 92.95 |

Method 4 contains the smoothing term (as the proposed model does); the diagnostic row is its λ = 0 limit, so *method 4 vs diagnostic* isolates R.
