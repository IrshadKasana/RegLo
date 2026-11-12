# Trial status — preset `full`

22469 rows in trials_full.csv (21473 planned trials, 996 older results outside the current grid).

| source | method | planned | PASSED | PASSED_AFTER_RETRY | FAILED | MISSING | EXTRA_NOT_IN_CURRENT_GRID |
|---|---|---|---|---|---|---|---|
| cifar10 | M1 | 20 | 20 | 0 | 0 | 0 | 0 |
| cifar10 | M2 | 2780 | 2780 | 0 | 0 | 0 | 0 |
| cifar10 | M3 | 22 | 22 | 0 | 0 | 0 | 0 |
| cifar10 | M4a | 402 | 402 | 0 | 0 | 0 | 0 |
| cifar10 | M5 | 2145 | 2145 | 0 | 0 | 0 | 996 |
| fmnist | M1 | 20 | 20 | 0 | 0 | 0 | 0 |
| fmnist | M2 | 2780 | 2780 | 0 | 0 | 0 | 0 |
| fmnist | M3 | 22 | 22 | 0 | 0 | 0 | 0 |
| fmnist | M4a | 402 | 402 | 0 | 0 | 0 | 0 |
| fmnist | M5 | 2144 | 2141 | 3 | 0 | 0 | 0 |
| mnist | M1 | 20 | 20 | 0 | 0 | 0 | 0 |
| mnist | M2 | 2780 | 2780 | 0 | 0 | 0 | 0 |
| mnist | M3 | 22 | 22 | 0 | 0 | 0 | 0 |
| mnist | M4a | 402 | 402 | 0 | 0 | 0 | 0 |
| mnist | M5 | 2144 | 2144 | 0 | 0 | 0 | 0 |
| svhn | M1 | 20 | 20 | 0 | 0 | 0 | 0 |
| svhn | M2 | 2780 | 2780 | 0 | 0 | 0 | 0 |
| svhn | M3 | 22 | 22 | 0 | 0 | 0 | 0 |
| svhn | M4a | 402 | 402 | 0 | 0 | 0 | 0 |
| svhn | M5 | 2144 | 2144 | 0 | 0 | 0 | 0 |

## FAILED or MISSING trials: 0

## Failed in a worker log, never completed, no longer part of the plan: 0

## Passed only after a failed attempt / OOD retry: 3
- lo_fmnist_l2_ls0.006_lr0.002_mlr0.002_seed0: failed attempts=0, oom_retries=2
- lo_fmnist_l2_ls0.006_lr0.003_mlr0.002_seed0: failed attempts=0, oom_retries=1
- lo_fmnist_l2_ls0.006_lr0.004_mlr0.002_seed0: failed attempts=0, oom_retries=4

## Passed trials without a weights file: 0
