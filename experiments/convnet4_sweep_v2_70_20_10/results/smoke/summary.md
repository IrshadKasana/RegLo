# Five-method sweep — ConvNet-4, 70:20:10 split

Selection: source validation split (seed 0) only. ID test = source 10 % split. OOD = selected model, no retraining, on the TARGET dataset's 20 % partition. ± = std. error over inits (M1/M2) or meta-seeds (M3/M4a/M5). OOD acc is accuracy under the *target's own label ids* (the two label spaces are unrelated, so ≈ chance is expected); OOD AUROC = ID-test vs OOD separation by max-softmax confidence.


## CIFAR-10 (source) → SVHN (OOD target)

| method | R | config | val acc % (selection) | ID test acc % | OOD acc % → SVHN | OOD AUROC | cells | n |
|---|---|---|---|---|---|---|---|---|
| M1 AdamW | - | lr=0.01 | 37.43 | 38.77 | 9.75 | 0.666 | - | 1 inits |
| M2 AdamW+R | l2 | λ=0.01 (lr=0.01) | 24.00 | 24.15 | 9.59 | 0.566 | 2 | 1 inits |
| M2 AdamW+R | sam | λ=0.01, ρ=0.01 (lr=0.01) | 37.67 | 37.78 | 9.67 | 0.656 | 4 | 1 inits |
| M3 LO | - | meta_lr=0.01 | 31.74 | 26.46 ± 3.32 | 8.28 ± 0.50 | 0.672 | 2 | 3 meta-seeds |
| M4a LO+smooth | - | λ_smooth=0.1, meta_lr=0.01 | 31.81 | 27.72 ± 1.70 | 8.79 ± 0.40 | 0.579 | 2 | 3 meta-seeds |
| M5 LO+smooth+R | l2 | λ_smooth=0.1, λ_reg=0.1, meta_lr=0.01 | 29.14 | 27.33 ± 0.96 | 10.44 ± 0.31 | 0.591 | 4 | 3 meta-seeds |
| M5 LO+smooth+R | sam | λ_smooth=0.01, λ_reg=0.01, meta_lr=0.01 | 33.98 | 28.70 ± 3.89 | 8.63 ± 0.95 | 0.648 | 4 | 3 meta-seeds |
