| Method | L1: mean ± std (%) | L1: seeds 0, 1, 2 | L2: mean ± std (%) | L2: seeds 0, 1, 2 |
|---|---|---|---|---|
| 1. MLP + AdamW | **93.13 ± 0.08** | 93.11, 93.06, 93.21 | **93.13 ± 0.08** | 93.11, 93.06, 93.21 |
| 2. MLP + AdamW + regularizer | **93.75 ± 0.31** | 93.90, 93.40, 93.95 | **93.49 ± 0.34** | 93.43, 93.17, 93.85 |
| 3. MLP + learned optimizer | **93.27 ± 0.16** | 93.37, 93.08, 93.35 | **93.27 ± 0.16** | 93.37, 93.08, 93.35 |
| 4. MLP + learned optimizer + regularizer | **93.65 ± 0.48** | 94.08, 93.74, 93.13 | **93.39 ± 0.48** | 93.93, 93.00, 93.25 |

MNIST test accuracy after 1000 optimizee steps; mean ± std over 3 seeds (the bar height and error bar in `test_accuracy.png`). Methods 1 and 3 are the same runs in both columns.
