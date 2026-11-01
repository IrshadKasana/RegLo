"""
sweep_config.py -- EVERY sweep value and protocol constant of the five-method experiment, in one place.

Edit this file (or override on the command line where the flag exists) -- no training code has to change.

PROTOCOL
    split     : each dataset's official train + test images are pooled and re-split ONCE with a fixed seed into
                70 % train / 20 % validation / 10 % test  (indices are deterministic and written to
                results/<preset>/splits/<dataset>.npz).
    selection : ONLY the source dataset's 20 % validation split (seed-0 runs).
    ID test   : the source dataset's 10 % test split (never used for selection).
    OOD       : source -> target.  The selected model (no retraining, no tuning) is evaluated on the TARGET
                dataset's 20 % partition.  That partition plays no role in the source's selection; it is the
                target's own validation partition, so it is only ever used for selection when the target is
                itself the source of a *different* experiment (the two directions are reported separately).

METHODS
    M1  Optimizee + AdamW                                         sweep: lr
    M2  Optimizee + AdamW + lambda*R                              sweep: lambda (GRID), rho (SAM/GSAM/GAM); R in REGS
    M3  Optimizee + learned optimizer (LSTM)                      sweep: meta_lr (Adam rate that trains the LO)
    M4a Optimizee + learned optimizer + lambda_smooth*R_smooth    sweep: lambda_smooth (GRID), meta_lr = M3's pick
    M5  M4a + lambda_reg*R                                        sweep: lambda_reg axis at M4a's lambda_smooth, then a
                                                                         (2*RADIUS+1)^2 joint block, R in REGS
"""

SPLIT = (0.70, 0.20, 0.10)             # train / validation / test
SPLIT_SEED = 42
DATASETS = ("svhn", "cifar10")         # every ordered pair (source -> target) is evaluated
REGS = ("l1", "l2", "sam", "gsam", "gam")

# lambda grid (lambda_smooth, lambda_reg, M2's lambda): 28 STRICTLY POSITIVE values.
# RULE (all grids): a zero value is never part of a sweep grid -- every swept parameter takes non-zero values only.
GRID = (0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# --- M1 / M2 (AdamW) ---
ADAMW_LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)       # M1 sweep; M2 uses the lr M1 picked on validation
ADAMW_WD = 0.01
# M2 perturbation-radius sweep: the same 14 radii for SAM, GSAM and GAM. M5 keeps the fixed defaults
# below (a radius sweep inside a meta-training sweep would multiply its cost by len(RHO_VALUES)).
RHO_VALUES = (0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06)   # 14 radii
RHO_GRID = {"sam": RHO_VALUES, "gsam": RHO_VALUES, "gam": RHO_VALUES}
RHO_DEFAULT = {"sam": 0.001, "gsam": 0.005, "gam": 0.04}

# --- M3 / M4a / M5 (learned optimizer) ---
META_LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)        # M3 sweep (Adam lr of the LO's own weights); 1e-3 = old fixed value
META_LR_DEFAULT = 1e-3
RADIUS = 2                                        # M5 joint block: +-RADIUS grid steps around the axis winners

# --- budgets, seeds ---
EVAL_STEPS = 1000                                 # optimizee training steps (same for every method)
EVAL_BATCH = 128
ADAMW_INITS = 3                                   # optimizee initialisations per M1/M2 cell  (EVERY cell runs all of them)
LO_INITS = 3                                      # optimizee initialisations per LO cell     (EVERY cell runs all of them)
SELECT_SEED = 0                                   # meta-seed used for ALL selection
FINAL_SEEDS = (1, 2)                              # extra meta-seeds, run for the SELECTED M3/M4a/M5 cells only
CURVE_EVERY = 100                                 # steps between train-loss / val-acc curve points

# --- smoke preset (CPU sanity check only) ---
SMOKE_GRID = (0.01, 0.1)
SMOKE_META_LRS = (1e-3, 1e-2)
SMOKE_ADAMW_LRS = (1e-3, 1e-2)
SMOKE_RHO_GRID = {"sam": (0.001, 0.01), "gsam": (0.005, 0.02), "gam": (0.04, 0.1)}
