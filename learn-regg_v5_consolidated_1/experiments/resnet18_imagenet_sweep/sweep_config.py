"""
sweep_config.py -- EVERY sweep value and protocol constant of the five-method ResNet-18 / ImageNet experiment, in one place.

Edit this file (or override on the command line where the flag exists) -- no training code has to change.

PROTOCOL
    optimizee : ResNet-18 (CIFAR-style stem for 32x32 inputs, 11.2 M parameters + a 1000-way linear head).
    dataset   : ImageNet-1k downsampled to 32x32 (Chrabaszcz et al. 2017), 1000 classes.  The official train (1,281,167) and
                validation (50,000) images are pooled and re-split ONCE with a fixed seed into 70 % train / 20 % validation /
                10 % test (indices are written to results/<preset>/splits/imagenet.npz).
    selection : ONLY the source dataset's 20 % validation split (a fixed random VAL_SELECT_CAP-image subset of it), seed-0 runs.
    ID test   : the 10 % test split (never used for selection): accuracy, loss and ID AUROC (macro one-vs-rest multiclass AUROC).
    OOD       : ImageNet -> {SVHN, CIFAR-10, MNIST, FashionMNIST}.  The selected model (no retraining, no tuning) is evaluated on
                each target's 20 % partition: accuracy / loss under the target's own label ids (chance by construction) and
                OOD AUROC = ID-test vs OOD separation by max-softmax confidence.  MNIST / FashionMNIST are zero-padded 28 -> 32
                and replicated to 3 channels.

METHODS
    M1  ResNet-18 + AdamW                                          sweep: lr
    M2  ResNet-18 + AdamW + lambda*R                               sweep: lr x lambda (full grid), then rho x lambda at the best lr
    M3  ResNet-18 + learned optimizer (LSTM)                       sweep: meta_lr (Adam rate that trains the LO)
    M4a ResNet-18 + learned optimizer + lambda_smooth*R_smooth     sweep: meta_lr x lambda_smooth (full grid)
    M5  M4a + lambda_reg*R                                         sweep: lambda_smooth x lambda_reg (full grid at M4a's meta_lr),
                                                                          then meta_lr, then rho, at the best cell

    HOW THE LEARNED OPTIMIZER IS TRAINED (M3 / M4a / M5).  The LO is a coordinate-wise LSTM.  Meta-training back-propagates through
    `tbptt` unrolled optimizer steps, which stores ~1.2 KB of LSTM activations per optimizee coordinate and step: for the 11.2 M
    coordinates of ResNet-18 that is ~270 GB (tbptt = 20), far beyond one 80 GB GPU.  Following the repository's own convention
    (ResNet-18 is the "meta-test target", model.py), the LO is therefore META-TRAINED on a small PROXY optimizee -- ConvNet-4 with
    a 1000-way head on the SAME ImageNet-32 training stream, the pipeline of the ConvNet-4 sweep -- and META-TESTED on ResNet-18
    (no_grad, chunked LSTM).  Every LO hyper-parameter (meta_lr, lambda_smooth, lambda_reg, rho, regulariser) is selected by the
    validation accuracy of the resulting ResNet-18 run.  Set PROXY_WIDTH below to change the proxy.

    GRID SIZES.  The ConvNet-4 sweep used 20-value grids; one ResNet-18 / ImageNet trial costs minutes instead of seconds, so every
    axis here is a 5-6 point log-spaced grid (all strictly positive).  Estimated total ~100 A100-hours (README_RESNET18.md);
    widen a grid here and re-submit -- finished trials are skipped.
"""

SPLIT = (0.70, 0.20, 0.10)             # train / validation / test
SPLIT_SEED = 42
DATASETS = ("imagenet",)               # SOURCE datasets (each is trained on and gets an ID + OOD table)
OOD_TARGETS = ("svhn", "cifar10", "mnist", "fmnist")
ALL_DATASETS = DATASETS + OOD_TARGETS  # every dataset that must be on disk
REGS = ("l1", "l2", "sam", "gsam", "gam")
PAD_GRAY_TO = 32                       # MNIST / FashionMNIST: 28 -> 32 zero padding, 1 -> 3 channels
NUM_CLASSES = {"imagenet": 1000}       # source datasets; targets are only used as OOD inputs
IMAGENET_DIR = "imagenet32"            # datasets/<IMAGENET_DIR>/{x.npy,y.npy}  (made by prepare_imagenet32.py)

# --- AdamW learning rates (M1 / M2) and the Adam rate that trains the LO (M3 / M4a / M5) ---
ADAMW_LRS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03)
META_LRS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03)
ADAMW_WD = 0.01

# --- lambda grids ---
# M2 acts on the 11.2 M ResNet-18 weights with the SUM convention (l1: sum|theta|, l2: sum theta^2), so the useful scale of
# lambda is far smaller than for ConvNet-4; SAM / GSAM / GAM weights are O(1) mixing coefficients.
M2_LAMBDA = {"l1": (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
             "l2": (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
             "sam": (0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
             "gsam": (0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
             "gam": (0.001, 0.005, 0.01, 0.05, 0.1, 0.5)}
# M4a / M5 act on the PROXY meta-objective (ConvNet-4).  lambda_smooth is NOT swept: it is fixed at SMOOTH_FIXED for M4a and M5;
# only lambda_reg (M5) is swept, over GRID (strictly positive, up to 1.0).
SMOOTH_FIXED = 1.0                                # lambda_smooth of M4a and M5
GRID = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)  # lambda_reg

# --- perturbation radius rho of SAM / GSAM / GAM ---
# M2: radius in the ResNet-18 weight space (SAM's usual rho = 0.05).  M5: radius in the LO-parameter space of the proxy (ConvNet-4 scale).
M2_RHO_VALUES = (0.01, 0.02, 0.05, 0.1, 0.2)
M2_RHO_GRID = {"sam": M2_RHO_VALUES, "gsam": M2_RHO_VALUES, "gam": M2_RHO_VALUES}
M2_RHO_DEFAULT = {"sam": 0.05, "gsam": 0.05, "gam": 0.05}
RHO_VALUES = (0.001, 0.005, 0.01, 0.04, 0.06)
RHO_GRID = {"sam": RHO_VALUES, "gsam": RHO_VALUES, "gam": RHO_VALUES}
RHO_DEFAULT = {"sam": 0.001, "gsam": 0.005, "gam": 0.04}      # M5; used while the other axes are being swept

# --- M5 staging ---
M5_MODE = "full"                       # unused while lambda_smooth is fixed (the lambda_reg axis at SMOOTH_FIXED is the whole stage)
RADIUS = 1                             # unused (see M5_MODE)

# --- budgets, seeds ---
EVAL_STEPS = 1000                      # optimizee training steps (same for every method) -- 0.14 epoch of ImageNet-32
EVAL_BATCH = 128
ADAMW_INITS = 2                        # optimizee initialisations per M1/M2 cell  (EVERY cell runs all of them)
LO_INITS = 2                           # optimizee initialisations per LO cell     (EVERY cell runs all of them)
SELECT_SEED = 0                        # meta-seed used for ALL selection
FINAL_SEEDS = (1, 2)                   # extra meta-seeds, run for the SELECTED M3/M4a/M5 cells only
CURVE_EVERY = 100                      # steps between curve points (a step-0 point is also recorded)
CURVE_TRAIN_SUBSET = 5000              # train images used for the train loss / accuracy of each curve point
CURVE_VAL_SUBSET = 20000               # validation images used for each curve point (the final selection metric uses VAL_SELECT_CAP)
VAL_SELECT_CAP = 50000                 # validation images used for val_acc / val_loss in report() (fixed random subset of the 20 % split)
PROXY_WIDTH = 64                       # channels of the ConvNet-4 proxy the LO is meta-trained on
META_CLIP = 0.0                        # clip the L2 norm of the LO's meta-gradient (0 = off)
SAVE_WEIGHTS = False                   # a ResNet-18 state is 45 MB: keeping every trial's weights would need ~100 GB

# --- smoke preset (CPU sanity check only) ---
SMOKE_GRID = (0.01, 1.0)
SMOKE_LRS = (1e-3, 0.03)
SMOKE_META_LRS = (1e-3, 0.03)
SMOKE_M2_LAMBDA = {"l1": (1e-5, 1e-3), "l2": (1e-4, 1e-2), "sam": (0.01, 0.5), "gsam": (0.01, 0.5), "gam": (0.01, 0.5)}
SMOKE_M2_RHO_GRID = {"sam": (0.02, 0.1), "gsam": (0.02, 0.1), "gam": (0.02, 0.1)}
SMOKE_RHO_GRID = {"sam": (0.001, 0.06), "gsam": (0.005, 0.06), "gam": (0.04, 0.06)}
