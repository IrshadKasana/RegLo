# Trial status — preset `full`

3090 rows in trials_full.csv (3032 planned trials, 58 older results outside the current grid).

| source | method | planned | PASSED | PASSED_AFTER_RETRY | FAILED | MISSING | EXTRA_NOT_IN_CURRENT_GRID |
|---|---|---|---|---|---|---|---|
| cifar10 | M1 | 5 | 5 | 0 | 0 | 0 | 0 |
| cifar10 | M2 | 1232 | 1232 | 0 | 0 | 0 | 28 |
| cifar10 | M3 | 7 | 7 | 0 | 0 | 0 | 0 |
| cifar10 | M4a | 30 | 30 | 0 | 0 | 0 | 0 |
| cifar10 | M5 | 246 | 226 | 20 | 0 | 0 | 0 |
| svhn | M1 | 5 | 5 | 0 | 0 | 0 | 0 |
| svhn | M2 | 1232 | 1232 | 0 | 0 | 0 | 28 |
| svhn | M3 | 7 | 7 | 0 | 0 | 0 | 0 |
| svhn | M4a | 30 | 30 | 0 | 0 | 0 | 0 |
| svhn | M5 | 238 | 218 | 20 | 0 | 0 | 2 |

## FAILED or MISSING trials: 0

## Failed in a worker log, never completed, and no longer part of the plan: 0
(cells of an earlier local block whose centre moved after the axis was completed; nothing is missing from the current grid)

## Passed only after a failed attempt / OOM retry: 40
- svhn_gam_ls0.006_lr0.002_mlr0.0001_seed0: failed attempts=2, oom_retries=6
- svhn_gam_ls0.006_lr0.004_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr0.006_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr0.008_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr0.01_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr0.03_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr0.05_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr0.07_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr0.09_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr0.2_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr0.4_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr0.6_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr0.8_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.006_lr1_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.8_lr0.006_mlr0.0001_seed0: failed attempts=2, oom_retries=6
- cifar10_gam_ls0.8_lr0.008_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.8_lr0.01_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.8_lr0.03_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.8_lr0.05_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.8_lr0.07_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.8_lr0.09_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.8_lr0.2_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.8_lr0.4_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.8_lr0.6_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.8_lr0.8_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.8_lr1_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.004_lr0.002_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.005_lr0.001_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.005_lr0.003_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.007_lr0.001_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.007_lr0.003_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- svhn_gam_ls0.008_lr0.002_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.7_lr0.003_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.7_lr0.005_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.7_lr0.007_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.9_lr0.003_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.9_lr0.005_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls0.9_lr0.007_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls1_lr0.004_mlr0.0001_seed0: failed attempts=1, oom_retries=3
- cifar10_gam_ls1_lr0.006_mlr0.0001_seed0: failed attempts=1, oom_retries=3
