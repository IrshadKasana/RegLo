# Pooled curve fit of the λ sweep

Model: `val_acc = α_seed + b·log10λ + c·(log10λ)²`  (per-seed intercepts). Seeds used for method 4: **all available**. Bootstrap = 3000 residual resamples; 90 % interval.

Method 2 is fitted in a ±1-decade window around its empirical best (5 λ's); method 4 uses all 8 λ.

| Method | Reg. | n | resid. SD (pp) | p(λ matters) | p(curvature) | fitted optimum λ* | 90 % interval | P(interior max) | fitted peak val % | curve range over grid (pp) | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2. AdamW + R | L1 | 30 | 0.47 | 8.96e-05 | 2.75e-05 | 8.5e-05 | 6.7e-05 – 0.00011 | 1.00 | 93.59 | 1.24 | λ matters and the curve has an **interior optimum** |
| 4. LO + R | L1 | 28 | 0.70 | 0.943 | 0.87 | 7.6e-06 | 1e-06 – 0.003 | 0.48 | 92.36 | 0.13 | λ has **no detectable effect** (p>0.10): any λ in the grid is statistically equivalent |
| 2. AdamW + R | L2 | 30 | 0.34 | 6.98e-10 | 2.61e-07 | 0.00013 | 0.0001 – 0.00016 | 1.00 | 93.22 | 1.95 | λ matters and the curve has an **interior optimum** |
| 4. LO + R | L2 | 28 | 0.67 | 0.219 | 0.119 | 0.00099 | 0.00029 – 0.03 | 0.91 | 92.89 | 0.94 | λ has **no detectable effect** (p>0.10): any λ in the grid is statistically equivalent |

How to read it: `p(λ matters)` is an F-test of the quadratic-in-log λ model against a λ-independent model with the same per-seed intercepts; `P(interior max)` is the fraction of bootstrap curves whose maximum lies strictly inside the grid; `fitted peak` is averaged over seeds/inits. The residual SD includes both meta-training noise and any lack of fit of a quadratic.
