"""
sweep_config.py -- EVERY sweep value and protocol constant of the five-method, four-dataset experiment, in one place.

Edit this file (or override on the command line where the flag exists) -- no training code has to change.

PROTOCOL
    datasets  : SVHN, CIFAR-10, MNIST, FashionMNIST.  Every dataset is a SOURCE once and an OOD TARGET for the other three,
                so there are 4 x (1 ID + 3 OOD) evaluations (12 ordered source -> target pairs).
    split     : each dataset's official train + test images are pooled and re-split ONCE with a fixed seed into
                70 % train / 20 % validation / 10 % test (indices are deterministic and written to
                results/<preset>/splits/<dataset>.npz).  MNIST / FashionMNIST are zero-padded 28 -> 32 and replicated to
                3 channels so that one ConvNet-4 serves every dataset (needed for cross-dataset evaluation).
    selection : ONLY the source dataset's 20 % validation split (seed-0 runs).
    ID test   : the source dataset's 10 % test split (never used for selection): accuracy, loss and ID AUROC
                (macro one-vs-rest multiclass AUROC of the softmax output).
    OOD       : source -> target.  The selected model (no retraining, no tuning) is evaluated on the TARGET dataset's
                20 % partition: accuracy / loss under the target's label ids (chance by construction: unrelated label
                spaces) and OOD AUROC = ID-test vs OOD separation by max-softmax confidence.

METHODS
    M1  Optimizee + AdamW                                         sweep: lr
    M2  Optimizee + AdamW + lambda*R                              sweep: lr x lambda (full grid), then rho x lambda at the best lr
    M3  Optimizee + learned optimizer (LSTM)                      sweep: meta_lr (Adam rate that trains the LO)
    M4a Optimizee + learned optimizer + lambda_smooth*R_smooth    sweep: meta_lr x lambda_smooth (full grid)
    M5  M4a + lambda_reg*R                                        sweep: lambda_smooth x lambda_reg (full grid at M4a's
                                                                         meta_lr), then meta_lr, then rho at the best cell
    The full Cartesian product of (lr | meta_lr) x lambda_smooth x lambda_reg x rho is ~100 days of GPU time on the two
    V100s a user may hold, so the axes that interact most (the lambda pair; lr x lambda; meta_lr x lambda_smooth) are swept
    as complete grids and the remaining axes (meta_lr and rho of M5, rho of M2) are swept coordinate-wise at the best cell.
"""

SPLIT = (0.70, 0.20, 0.10)             # train / validation / test
SPLIT_SEED = 42
DATASETS = ("svhn", "cifar10", "mnist", "fmnist")
REGS = ("l1", "l2", "sam", "gsam", "gam")
PAD_GRAY_TO = 32                       # MNIST / FashionMNIST: 28 -> 32 zero padding, 1 -> 3 channels

# ONE value list for every non-zero hyper-parameter grid: lambda (smooth / reg / M2), AdamW lr, LO meta_lr.
# RULE: no zero value is ever part of a sweep grid.
GRID20 = (0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
          0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
          0.1, 0.2, 0.3, 0.4, 0.5)
GRID = GRID20                          # lambda_smooth, lambda_reg, M2 lambda
ADAMW_LRS = GRID20                     # M1 / M2 learning rate
META_LRS = GRID20                      # M3 / M4a / M5 meta learning rate
ADAMW_WD = 0.01

# perturbation radius rho of SAM / GSAM / GAM (M2 and M5): 14 radii.
RHO_VALUES = (0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
RHO_GRID = {"sam": RHO_VALUES, "gsam": RHO_VALUES, "gam": RHO_VALUES}
RHO_DEFAULT = {"sam": 0.001, "gsam": 0.005, "gam": 0.04}      # used while the other axes are being swept

# --- M5 staging ---
M5_MODE = "full"                       # "full": all len(GRID)^2 (lambda_smooth, lambda_reg) cells per regulariser
                                       # "block": lambda_reg axis at M4a's lambda_smooth + (2*RADIUS+1)^2 joint block
RADIUS = 3                             # half-width of the block (only used when M5_MODE == "block")

# --- budgets, seeds ---
EVAL_STEPS = 1000                      # optimizee training steps (same for every method)
EVAL_BATCH = 128
ADAMW_INITS = 3                        # optimizee initialisations per M1/M2 cell  (EVERY cell runs all of them)
LO_INITS = 3                           # optimizee initialisations per LO cell     (EVERY cell runs all of them)
SELECT_SEED = 0                        # meta-seed used for ALL selection
FINAL_SEEDS = (1, 2)                   # extra meta-seeds, run for the SELECTED M3/M4a/M5 cells only
CURVE_EVERY = 100                      # steps between curve points (a step-0 point is also recorded)
CURVE_TRAIN_SUBSET = 5000              # train images used for the train loss / accuracy of each curve point
SAVE_WEIGHTS = True                    # keep every trial's final optimizee weights (all inits) in results/<p>/weights/

# --- smoke preset (CPU sanity check only) ---
SMOKE_GRID = (0.01, 0.5)
SMOKE_LRS = (1e-3, 0.5)
SMOKE_META_LRS = (1e-3, 0.5)
SMOKE_RHO_GRID = {"sam": (0.001, 0.06), "gsam": (0.005, 0.06), "gam": (0.04, 0.06)}
