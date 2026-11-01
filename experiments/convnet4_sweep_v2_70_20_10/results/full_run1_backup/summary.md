# Five-method sweep — ConvNet-4, 70:20:10 split

Selection: source validation split (seed 0) only. ID test = source 10 % split. OOD = selected model, no retraining, on the TARGET dataset's 20 % partition. ± = std. error over inits (M1/M2) or meta-seeds (M3/M4a/M5). OOD acc is accuracy under the *target's own label ids* (the two label spaces are unrelated, so ≈ chance is expected); OOD AUROC = ID-test vs OOD separation by max-softmax confidence.


## SVHN (source) → CIFAR-10 (OOD target)

| method | R | config | val acc % (selection) | ID test acc % | OOD acc % → CIFAR-10 | OOD AUROC | cells | n |
|---|---|---|---|---|---|---|---|---|
| M1 AdamW | - | lr=0.003 | 90.42 | 90.19 ± 0.34 | 9.76 ± 0.18 | 0.933 | - | 3 inits |
| M2 AdamW+R | l1 | λ=0.001 (lr=0.003) | 84.10 | 84.56 ± 0.61 | 9.76 ± 0.34 | 0.861 | 28 | 3 inits |
| M2 AdamW+R | l2 | λ=0.001 (lr=0.003) | 88.10 | 88.22 ± 0.35 | 9.25 ± 0.42 | 0.922 | 28 | 3 inits |
| M2 AdamW+R | sam | λ=0.007, ρ=0.05 (lr=0.003) | 90.94 | 90.65 ± 0.14 | 9.32 ± 0.43 | 0.934 | 84 | 3 inits |
| M2 AdamW+R | gsam | λ=0.8, ρ=0.05 (lr=0.003) | 90.78 | 90.83 ± 0.24 | 9.68 ± 0.39 | 0.943 | 84 | 3 inits |
| M2 AdamW+R | gam | λ=0.4, ρ=0.1 (lr=0.003) | 90.87 | 90.74 ± 0.26 | 10.04 ± 0.33 | 0.940 | 84 | 3 inits |
| M3 LO | - | meta_lr=0.0001 | 89.46 | 89.00 ± 0.36 | 9.74 ± 0.19 | 0.890 | 5 | 3 meta-seeds |
| M4a LO+smooth | - | λ_smooth=0.006, meta_lr=0.0001 | 89.72 | 88.16 ± 0.99 | 9.36 ± 0.11 | 0.859 | 28 | 3 meta-seeds |
| M5 LO+smooth+R | l1 | λ_smooth=0.004, λ_reg=0.7, meta_lr=0.0001 | 89.53 | 84.63 ± 3.54 | 9.71 ± 0.44 | 0.859 | 48 | 3 meta-seeds |
| M5 LO+smooth+R | l2 | λ_smooth=0.004, λ_reg=0.002, meta_lr=0.0001 | 89.52 | 88.51 ± 0.46 | 9.97 ± 0.21 | 0.878 | 48 | 3 meta-seeds |
| M5 LO+smooth+R | sam | λ_smooth=0.006, λ_reg=0.002, meta_lr=0.0001 | 89.90 | 89.09 ± 0.42 | 9.97 ± 0.12 | 0.892 | 44 | 3 meta-seeds |
| M5 LO+smooth+R | gsam | λ_smooth=0.006, λ_reg=0.007, meta_lr=0.0001 | 89.74 | 89.23 ± 0.32 | 9.84 ± 0.09 | 0.894 | 48 | 3 meta-seeds |
| M5 LO+smooth+R | gam | λ_smooth=0.008, λ_reg=0.001, meta_lr=0.0001 | 89.39 | 83.93 ± 4.88 | 9.29 ± 0.41 | 0.788 | 20 | 3 meta-seeds |

## CIFAR-10 (source) → SVHN (OOD target)

| method | R | config | val acc % (selection) | ID test acc % | OOD acc % → SVHN | OOD AUROC | cells | n |
|---|---|---|---|---|---|---|---|---|
| M1 AdamW | - | lr=0.001 | 73.68 | 73.59 ± 0.52 | 9.85 ± 1.13 | 0.820 | - | 3 inits |
| M2 AdamW+R | l1 | λ=0.001 (lr=0.001) | 66.89 | 66.54 ± 1.61 | 9.62 ± 0.87 | 0.790 | 28 | 3 inits |
| M2 AdamW+R | l2 | λ=0.001 (lr=0.001) | 72.32 | 71.83 ± 0.31 | 9.00 ± 0.87 | 0.802 | 28 | 3 inits |
| M2 AdamW+R | sam | λ=0.006, ρ=0.01 (lr=0.001) | 74.80 | 74.57 ± 0.43 | 9.11 ± 0.91 | 0.811 | 84 | 3 inits |
| M2 AdamW+R | gsam | λ=0.4, ρ=0.005 (lr=0.001) | 74.78 | 73.98 ± 0.19 | 9.93 ± 0.66 | 0.813 | 84 | 3 inits |
| M2 AdamW+R | gam | λ=1, ρ=0.01 (lr=0.001) | 74.75 | 74.23 ± 0.30 | 9.49 ± 0.48 | 0.824 | 84 | 3 inits |
| M3 LO | - | meta_lr=0.0001 | 71.88 | 69.63 ± 2.07 | 10.12 ± 0.31 | 0.830 | 5 | 3 meta-seeds |
| M4a LO+smooth | - | λ_smooth=0.8, meta_lr=0.0001 | 72.98 | 65.03 ± 6.76 | 10.69 ± 0.40 | 0.802 | 28 | 3 meta-seeds |
| M5 LO+smooth+R | l1 | λ_smooth=0.8, λ_reg=0.01, meta_lr=0.0001 | 69.84 | 68.61 ± 0.84 | 10.03 ± 0.30 | 0.788 | 48 | 3 meta-seeds |
| M5 LO+smooth+R | l2 | λ_smooth=0.6, λ_reg=0.001, meta_lr=0.0001 | 72.36 | 70.08 ± 1.43 | 9.60 ± 0.29 | 0.841 | 44 | 3 meta-seeds |
| M5 LO+smooth+R | sam | λ_smooth=0.8, λ_reg=0.09, meta_lr=0.0001 | 73.42 | 66.96 ± 4.72 | 10.56 ± 0.16 | 0.801 | 48 | 3 meta-seeds |
| M5 LO+smooth+R | gsam | λ_smooth=0.6, λ_reg=0.009, meta_lr=0.0001 | 73.25 | 69.26 ± 2.42 | 10.11 ± 0.42 | 0.788 | 48 | 3 meta-seeds |
| M5 LO+smooth+R | gam | λ_smooth=0.8, λ_reg=0.005, meta_lr=0.0001 | 73.27 | 67.65 ± 3.58 | 9.40 ± 0.15 | 0.836 | 28 | 3 meta-seeds |
