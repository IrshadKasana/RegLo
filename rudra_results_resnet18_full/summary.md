# Five-method sweep — ResNet-18 on ImageNet-32, 70:20:10 split, OOD on SVHN / CIFAR-10 / MNIST / FashionMNIST

Selection: source validation split (seed 0) only. ID test = source 10 % split; ID AUROC = macro one-vs-rest AUROC of the softmax output on it. OOD = selected model, no retraining, on the TARGET dataset's 20 % partition; OOD acc is accuracy under the *target's own label ids* (unrelated label spaces → ≈ chance expected); OOD AUROC = ID-test vs OOD separation by max-softmax confidence. ± = std. error over inits (M1/M2) or meta-seeds (M3/M4a/M5).


## ImageNet-32 (source) → SVHN (OOD target)

| method | R | config | val acc % | ID test acc % | ID AUROC | OOD acc % → SVHN | OOD AUROC | cells | n |
|---|---|---|---|---|---|---|---|---|---|
| M1 AdamW | - | lr=0.0003 | 3.18 | 3.18 ± 0.10 | 0.863 ± 0.001 | 0.36 ± 0.06 | 0.526 ± 0.009 | 6 | 2 inits |
| M2 AdamW+R | l1 | lr=0.0003, λ=1e-06 | 3.26 | 3.26 ± 0.36 | 0.863 ± 0.007 | 0.13 ± 0.12 | 0.478 ± 0.052 | 36 | 2 inits |
| M2 AdamW+R | l2 | lr=0.0003, λ=1e-06 | 3.25 | 3.25 ± 0.38 | 0.863 ± 0.007 | 0.12 ± 0.05 | 0.463 ± 0.012 | 36 | 2 inits |
| M2 AdamW+R | sam | lr=0.0003, λ=0.001, ρ=0.1 | 3.43 | 3.42 ± 0.18 | 0.867 ± 0.003 | 0.12 ± 0.08 | 0.450 ± 0.027 | 60 | 2 inits |
| M2 AdamW+R | gsam | lr=0.0003, λ=0.005, ρ=0.01 | 3.47 | 3.42 ± 0.34 | 0.866 ± 0.005 | 0.10 ± 0.06 | 0.523 ± 0.032 | 60 | 2 inits |
| M2 AdamW+R | gam | lr=0.0003, λ=0.001, ρ=0.1 | 3.39 | 3.39 ± 0.47 | 0.867 ± 0.008 | 0.24 ± 0.06 | 0.500 ± 0.039 | 60 | 2 inits |
| M3 LO | - | meta_lr=0.0003 | 0.12 | 0.19 ± 0.08 | 0.565 ± 0.061 | 1.39 ± 1.39 | 0.536 ± 0.032 | 6 | 3 meta-seeds |
| M4a LO+smooth | - | λ_smooth=1, meta_lr=0.0003 | 0.12 | 0.30 ± 0.16 | 0.579 ± 0.042 | 0.43 ± 0.43 | 0.545 ± 0.076 | 6 | 3 meta-seeds |
| M5 LO+smooth+R | l1 | λ_smooth=1, λ_reg=0.001, meta_lr=0.0003 | 0.82 | 0.40 ± 0.21 | 0.639 ± 0.065 | 0.06 ± 0.03 | 0.556 ± 0.055 | 12 | 3 meta-seeds |
| M5 LO+smooth+R | l2 | λ_smooth=1, λ_reg=0.005, meta_lr=0.0003 | 0.86 | 0.65 ± 0.12 | 0.727 ± 0.019 | 0.48 ± 0.17 | 0.551 ± 0.078 | 12 | 3 meta-seeds |
| M5 LO+smooth+R | sam | λ_smooth=1, λ_reg=1, meta_lr=0.0003, ρ=0.06 | 0.14 | 0.41 ± 0.24 | 0.628 ± 0.070 | 0.22 ± 0.21 | 0.503 ± 0.088 | 40 | 3 meta-seeds |
| M5 LO+smooth+R | gsam | λ_smooth=1, λ_reg=1, meta_lr=0.0003, ρ=0.01 | 0.37 | 0.25 ± 0.06 | 0.604 ± 0.033 | 0.01 ± 0.01 | 0.437 ± 0.082 | 40 | 3 meta-seeds |
| M5 LO+smooth+R | gam | λ_smooth=1, λ_reg=0.5, meta_lr=0.0003, ρ=0.01 | 0.49 | 0.48 ± 0.21 | 0.661 ± 0.079 | 0.38 ± 0.33 | 0.511 ± 0.046 | 40 | 3 meta-seeds |

## ImageNet-32 (source) → CIFAR-10 (OOD target)

| method | R | config | val acc % | ID test acc % | ID AUROC | OOD acc % → CIFAR-10 | OOD AUROC | cells | n |
|---|---|---|---|---|---|---|---|---|---|
| M1 AdamW | - | lr=0.0003 | 3.18 | 3.18 ± 0.10 | 0.863 ± 0.001 | 0.06 ± 0.01 | 0.485 ± 0.011 | 6 | 2 inits |
| M2 AdamW+R | l1 | lr=0.0003, λ=1e-06 | 3.26 | 3.26 ± 0.36 | 0.863 ± 0.007 | 0.01 ± 0.01 | 0.506 ± 0.003 | 36 | 2 inits |
| M2 AdamW+R | l2 | lr=0.0003, λ=1e-06 | 3.25 | 3.25 ± 0.38 | 0.863 ± 0.007 | 0.03 ± 0.01 | 0.497 ± 0.002 | 36 | 2 inits |
| M2 AdamW+R | sam | lr=0.0003, λ=0.001, ρ=0.1 | 3.43 | 3.42 ± 0.18 | 0.867 ± 0.003 | 0.03 ± 0.00 | 0.497 ± 0.002 | 60 | 2 inits |
| M2 AdamW+R | gsam | lr=0.0003, λ=0.005, ρ=0.01 | 3.47 | 3.42 ± 0.34 | 0.866 ± 0.005 | 0.02 ± 0.00 | 0.494 ± 0.014 | 60 | 2 inits |
| M2 AdamW+R | gam | lr=0.0003, λ=0.001, ρ=0.1 | 3.39 | 3.39 ± 0.47 | 0.867 ± 0.008 | 0.03 ± 0.00 | 0.490 ± 0.006 | 60 | 2 inits |
| M3 LO | - | meta_lr=0.0003 | 0.12 | 0.19 ± 0.08 | 0.565 ± 0.061 | 0.24 ± 0.24 | 0.467 ± 0.013 | 6 | 3 meta-seeds |
| M4a LO+smooth | - | λ_smooth=1, meta_lr=0.0003 | 0.12 | 0.30 ± 0.16 | 0.579 ± 0.042 | 0.11 ± 0.11 | 0.482 ± 0.022 | 6 | 3 meta-seeds |
| M5 LO+smooth+R | l1 | λ_smooth=1, λ_reg=0.001, meta_lr=0.0003 | 0.82 | 0.40 ± 0.21 | 0.639 ± 0.065 | 0.03 ± 0.03 | 0.491 ± 0.007 | 12 | 3 meta-seeds |
| M5 LO+smooth+R | l2 | λ_smooth=1, λ_reg=0.005, meta_lr=0.0003 | 0.86 | 0.65 ± 0.12 | 0.727 ± 0.019 | 0.12 ± 0.02 | 0.486 ± 0.013 | 12 | 3 meta-seeds |
| M5 LO+smooth+R | sam | λ_smooth=1, λ_reg=1, meta_lr=0.0003, ρ=0.06 | 0.14 | 0.41 ± 0.24 | 0.628 ± 0.070 | 0.03 ± 0.03 | 0.467 ± 0.012 | 40 | 3 meta-seeds |
| M5 LO+smooth+R | gsam | λ_smooth=1, λ_reg=1, meta_lr=0.0003, ρ=0.01 | 0.37 | 0.25 ± 0.06 | 0.604 ± 0.033 | 0.01 ± 0.01 | 0.475 ± 0.013 | 40 | 3 meta-seeds |
| M5 LO+smooth+R | gam | λ_smooth=1, λ_reg=0.5, meta_lr=0.0003, ρ=0.01 | 0.49 | 0.48 ± 0.21 | 0.661 ± 0.079 | 0.18 ± 0.09 | 0.475 ± 0.015 | 40 | 3 meta-seeds |

## ImageNet-32 (source) → MNIST (OOD target)

| method | R | config | val acc % | ID test acc % | ID AUROC | OOD acc % → MNIST | OOD AUROC | cells | n |
|---|---|---|---|---|---|---|---|---|---|
| M1 AdamW | - | lr=0.0003 | 3.18 | 3.18 ± 0.10 | 0.863 ± 0.001 | 0.00 ± 0.00 | 0.534 ± 0.025 | 6 | 2 inits |
| M2 AdamW+R | l1 | lr=0.0003, λ=1e-06 | 3.26 | 3.26 ± 0.36 | 0.863 ± 0.007 | 0.00 ± 0.00 | 0.526 ± 0.165 | 36 | 2 inits |
| M2 AdamW+R | l2 | lr=0.0003, λ=1e-06 | 3.25 | 3.25 ± 0.38 | 0.863 ± 0.007 | 0.00 ± 0.00 | 0.570 ± 0.022 | 36 | 2 inits |
| M2 AdamW+R | sam | lr=0.0003, λ=0.001, ρ=0.1 | 3.43 | 3.42 ± 0.18 | 0.867 ± 0.003 | 0.00 ± 0.00 | 0.467 ± 0.064 | 60 | 2 inits |
| M2 AdamW+R | gsam | lr=0.0003, λ=0.005, ρ=0.01 | 3.47 | 3.42 ± 0.34 | 0.866 ± 0.005 | 0.00 ± 0.00 | 0.623 ± 0.077 | 60 | 2 inits |
| M2 AdamW+R | gam | lr=0.0003, λ=0.001, ρ=0.1 | 3.39 | 3.39 ± 0.47 | 0.867 ± 0.008 | 0.00 ± 0.00 | 0.605 ± 0.031 | 60 | 2 inits |
| M3 LO | - | meta_lr=0.0003 | 0.12 | 0.19 ± 0.08 | 0.565 ± 0.061 | 0.00 ± 0.00 | 0.765 ± 0.023 | 6 | 3 meta-seeds |
| M4a LO+smooth | - | λ_smooth=1, meta_lr=0.0003 | 0.12 | 0.30 ± 0.16 | 0.579 ± 0.042 | 0.00 ± 0.00 | 0.442 ± 0.220 | 6 | 3 meta-seeds |
| M5 LO+smooth+R | l1 | λ_smooth=1, λ_reg=0.001, meta_lr=0.0003 | 0.82 | 0.40 ± 0.21 | 0.639 ± 0.065 | 0.02 ± 0.02 | 0.416 ± 0.132 | 12 | 3 meta-seeds |
| M5 LO+smooth+R | l2 | λ_smooth=1, λ_reg=0.005, meta_lr=0.0003 | 0.86 | 0.65 ± 0.12 | 0.727 ± 0.019 | 0.00 ± 0.00 | 0.574 ± 0.066 | 12 | 3 meta-seeds |
| M5 LO+smooth+R | sam | λ_smooth=1, λ_reg=1, meta_lr=0.0003, ρ=0.06 | 0.14 | 0.41 ± 0.24 | 0.628 ± 0.070 | 0.00 ± 0.00 | 0.594 ± 0.116 | 40 | 3 meta-seeds |
| M5 LO+smooth+R | gsam | λ_smooth=1, λ_reg=1, meta_lr=0.0003, ρ=0.01 | 0.37 | 0.25 ± 0.06 | 0.604 ± 0.033 | 0.00 ± 0.00 | 0.482 ± 0.050 | 40 | 3 meta-seeds |
| M5 LO+smooth+R | gam | λ_smooth=1, λ_reg=0.5, meta_lr=0.0003, ρ=0.01 | 0.49 | 0.48 ± 0.21 | 0.661 ± 0.079 | 0.00 ± 0.00 | 0.606 ± 0.077 | 40 | 3 meta-seeds |

## ImageNet-32 (source) → FashionMNIST (OOD target)

| method | R | config | val acc % | ID test acc % | ID AUROC | OOD acc % → FashionMNIST | OOD AUROC | cells | n |
|---|---|---|---|---|---|---|---|---|---|
| M1 AdamW | - | lr=0.0003 | 3.18 | 3.18 ± 0.10 | 0.863 ± 0.001 | 0.00 ± 0.00 | 0.564 ± 0.113 | 6 | 2 inits |
| M2 AdamW+R | l1 | lr=0.0003, λ=1e-06 | 3.26 | 3.26 ± 0.36 | 0.863 ± 0.007 | 0.00 ± 0.00 | 0.490 ± 0.065 | 36 | 2 inits |
| M2 AdamW+R | l2 | lr=0.0003, λ=1e-06 | 3.25 | 3.25 ± 0.38 | 0.863 ± 0.007 | 0.00 ± 0.00 | 0.585 ± 0.059 | 36 | 2 inits |
| M2 AdamW+R | sam | lr=0.0003, λ=0.001, ρ=0.1 | 3.43 | 3.42 ± 0.18 | 0.867 ± 0.003 | 0.00 ± 0.00 | 0.497 ± 0.071 | 60 | 2 inits |
| M2 AdamW+R | gsam | lr=0.0003, λ=0.005, ρ=0.01 | 3.47 | 3.42 ± 0.34 | 0.866 ± 0.005 | 0.00 ± 0.00 | 0.592 ± 0.032 | 60 | 2 inits |
| M2 AdamW+R | gam | lr=0.0003, λ=0.001, ρ=0.1 | 3.39 | 3.39 ± 0.47 | 0.867 ± 0.008 | 0.00 ± 0.00 | 0.550 ± 0.069 | 60 | 2 inits |
| M3 LO | - | meta_lr=0.0003 | 0.12 | 0.19 ± 0.08 | 0.565 ± 0.061 | 0.00 ± 0.00 | 0.626 ± 0.066 | 6 | 3 meta-seeds |
| M4a LO+smooth | - | λ_smooth=1, meta_lr=0.0003 | 0.12 | 0.30 ± 0.16 | 0.579 ± 0.042 | 0.00 ± 0.00 | 0.385 ± 0.165 | 6 | 3 meta-seeds |
| M5 LO+smooth+R | l1 | λ_smooth=1, λ_reg=0.001, meta_lr=0.0003 | 0.82 | 0.40 ± 0.21 | 0.639 ± 0.065 | 0.14 ± 0.14 | 0.425 ± 0.091 | 12 | 3 meta-seeds |
| M5 LO+smooth+R | l2 | λ_smooth=1, λ_reg=0.005, meta_lr=0.0003 | 0.86 | 0.65 ± 0.12 | 0.727 ± 0.019 | 0.00 ± 0.00 | 0.548 ± 0.041 | 12 | 3 meta-seeds |
| M5 LO+smooth+R | sam | λ_smooth=1, λ_reg=1, meta_lr=0.0003, ρ=0.06 | 0.14 | 0.41 ± 0.24 | 0.628 ± 0.070 | 0.00 ± 0.00 | 0.530 ± 0.091 | 40 | 3 meta-seeds |
| M5 LO+smooth+R | gsam | λ_smooth=1, λ_reg=1, meta_lr=0.0003, ρ=0.01 | 0.37 | 0.25 ± 0.06 | 0.604 ± 0.033 | 0.34 ± 0.34 | 0.663 ± 0.044 | 40 | 3 meta-seeds |
| M5 LO+smooth+R | gam | λ_smooth=1, λ_reg=0.5, meta_lr=0.0003, ρ=0.01 | 0.49 | 0.48 ± 0.21 | 0.661 ± 0.079 | 0.00 ± 0.00 | 0.492 ± 0.034 | 40 | 3 meta-seeds |
