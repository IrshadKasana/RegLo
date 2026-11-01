"""
experiment_matrix.py
====================
Single source of truth for the paper's experiment matrix (Experiments 1-9).

Each cell declares WHAT to run; run_all_experiments.py decides HOW (train ->
evaluate -> benchmark -> report), dedupes shared LO trainings, and emits
local commands or PBS jobs.

Method names (uniform across the repo):
    adamw_sam  : Model 1 benchmark  — explicit AdamW+SAM on the optimizee
    no_reg     : No-Reg LO          — vanilla learned optimizer
    sam        : SAM LO             — Eq. 7 penalty-mode meta-training
    gam        : GAM LO
    gsam       : GSAM LO

LO checkpoints are shared across experiments via a canonical name
    {METHOD}LO_{train_dataset}_{train_arch}
so e.g. Experiments 1, 3 and 7 all reuse SAMLO_mnist_mlp.

Edit policy: everything an experiment needs should be expressible by editing
THIS file only. `python experiment_matrix.py` prints the matrix;
`python experiment_matrix.py --export-yaml` writes per-experiment snapshot
YAMLs to configs/experiments/ (informational; the matrix stays canonical).
"""
from pathlib import Path

# Final framework (user poster, 2026-07-13): each regularizer R gets its
# MATCHED three-way ladder  AdamW+R  vs  No-Reg LO  vs  R LO.
# ======================================================================
# COMMON COMPARISON SETUP (final experimental plan, 2026-07-15)
#
# Every primary/generalization experiment trains & evaluates the SAME 8
# methods; figures then compare "corresponding pairs only" by slicing the
# 8 into three dedicated 3-method COMPARISON SETS.
# ======================================================================
LO_METHODS = ["no_reg", "sam", "gam", "gsam"]              # learned optimizers
BENCH_METHODS = ["adamw_no_reg", "adamw_sam",              # AdamW references
                 "adamw_gam", "adamw_gsam"]
ALL_METHODS = LO_METHODS + BENCH_METHODS                   # the 8

# Each set: LO No-Reg  vs  LO+R (best lambda* from Exp 1)  vs  AdamW+R
COMPARISON_SETS = {
    "sam":  ["no_reg", "sam", "adamw_sam"],
    "gam":  ["no_reg", "gam", "adamw_gam"],
    "gsam": ["no_reg", "gsam", "adamw_gsam"],
}

# meta-training config per (method, train_dataset): falls back to the base
# method config + --set overrides when no dedicated YAML exists.
LO_TRAIN_CONFIG = {
    ("sam", "mnist"): "sam",
    ("sam", "fmnist"): "sam_fmnist_mlp",
    ("sam", "cifar10"): "sam_cifar10_r8",
    # no_reg / gam / gsam reuse their base configs on every dataset
}

# benchmark (Model 1) eval config per test_dataset
BENCHMARK_CONFIG = {
    # regularizer -> Model-1 (AdamW) benchmark config. Dataset-specific
    # settings (e.g. augment for CIFAR-10) come from eval_overrides, so one
    # config per regularizer serves every dataset.
    "no_reg": "adamw_no_reg_baseline",
    "sgd": "sgd_baseline",     # MetaReg paper baseline optimiser
    "sam": "adamw_sam_baseline",
    "gam": "adamw_gam_baseline",
    "gsam": "adamw_gsam_baseline",
}

# NOTE on architectures: meta-training THROUGH ResNet-18 (11.2M params,
# T-step BPTT) does not fit a V100, so every cifar10 LO is meta-trained on
# the resnet8 PROXY and meta-TESTED on ResNet-18 (verified chunked path).
PROXY_TRAIN_ARCH = {"resnet18": "resnet8"}

RESNET18_EVAL_OVERRIDES = {
    "param_chunk_size": 262144, "eval_every": 10, "augment": True,
}
# same treatment for any big meta-test net (vgg16 = 14.72M params)
BIGNET_EVAL_OVERRIDES = RESNET18_EVAL_OVERRIDES


def _exp(name, group, train_ds, test_ds, train_arch, test_arch,
         methods=None, eval_overrides=None, flags=None, notes=""):
    return dict(
        name=name, group=group,
        train_dataset=train_ds, test_dataset=test_ds,
        train_arch=train_arch, test_arch=test_arch,
        methods=list(methods or ALL_METHODS),
        eval_overrides=dict(eval_overrides or {}),
        flags=dict(flags or {}),
        notes=notes,
    )


EXPERIMENTS = {
    # ============ EXPERIMENT 1: HYPERPARAMETER SWEEP =================
    # Independently grid-search (lambda_reg x lambda_smooth) for LOSAM,
    # LOGAM, LOGSAM. The winners are FROZEN (pick_best_lambdas --apply)
    # and reused unchanged in Experiments 2-5 — never retuned.
    "exp1": dict(
        name="sweep_lambda_reg_x_lambda_smooth", group="sweep", sweep=True,
        command=("python run_sweep.py --regularizer sam gam gsam "
                 "--smooth-values 0.0 0.001 0.01 0.1 0.5 1.0 "
                 "--reg-values 0.005 "
                 "--lambda-reg-values 0.0 0.01 0.05 0.1 0.5 1.0 "
                 "--dataset mnist --optimizee-model mlp "
                 "--max-hours 23 --mode {mode}"),
        notes="One PBS job PER CELL -> fully parallel. Then: "
              "`python pick_best_lambdas.py --apply` to freeze the winners "
              "before running exp2-exp5."),

    # ============ EXPERIMENT 2: MNIST + MLP ==========================
    "exp2": _exp("mnist_mlp", "primary",
                 "mnist", "mnist", "mlp", "mlp",
                 methods=ALL_METHODS,
                 notes="All 8 methods; LO rows use the frozen lambda* from "
                       "Exp 1 (--use-best-lambdas). Evaluated on the MNIST "
                       "test set."),

    # ============ EXPERIMENT 3: CIFAR-10 + ResNet-18 =================
    "exp3": _exp("cifar10_resnet18", "primary",
                 "cifar10", "cifar10", "resnet8", "resnet18",
                 methods=ALL_METHODS,
                 eval_overrides=RESNET18_EVAL_OVERRIDES,
                 notes="Identical protocol to Exp 2; only dataset+model "
                       "differ. LOs meta-train on the resnet8 family proxy "
                       "(BPTT through ResNet-18 is intractable) and "
                       "meta-TEST on the full ResNet-18 (chunked, exact)."),

    # ============ EXPERIMENT 4: CROSS-DATASET GENERALIZATION =========
    # Reuses the Exp-2 / Exp-3 checkpoints verbatim: NOTHING is retrained
    # (identical lo_checkpoint_name -> the manager dedupes).
    "exp4a": _exp("xds_mnist_to_fmnist", "cross_dataset",
                  "mnist", "fmnist", "mlp", "mlp",
                  methods=ALL_METHODS,
                  notes="PART A: Exp-2 models (trained on MNIST) evaluated "
                        "on Fashion-MNIST. AdamW rows = the reference "
                        "diagonal (AdamW run directly on FMNIST)."),
    "exp4b": _exp("xds_cifar10_to_cifar10c", "cross_dataset",
                  "cifar10", "cifar10", "resnet8", "resnet18",
                  methods=ALL_METHODS,
                  eval_overrides={**RESNET18_EVAL_OVERRIDES,
                                  "eval_cifar10c": True},
                  notes="PART B: Exp-3 models evaluated on CIFAR-10-C (15 "
                        "corruptions x 5 severities). Needs "
                        "datasets/CIFAR-10-C (zenodo 2535967)."),

    # ============ EXPERIMENT 5: CROSS-ARCHITECTURE GENERALIZATION ====
    "exp5": _exp("xarch_resnet18_to_vgg16", "cross_architecture",
                 "cifar10", "cifar10", "resnet8", "vgg16",
                 methods=ALL_METHODS,
                 eval_overrides=BIGNET_EVAL_OVERRIDES,
                 notes="Exp-3 models (CIFAR-10, ResNet family) evaluated on "
                       "CIFAR-10 with VGG-16 (14.72M params, chunked, "
                       "exact). Add further architectures by cloning this "
                       "cell with a different test_arch."),
}


# ---------------- EXPERIMENT 6: PACS DOMAIN GENERALIZATION ----------------
# MetaReg-style leave-one-domain-out (Balaji et al. 2018, Table 2): the
# optimizee is torchvision ResNet-18 (ImageNet-pretrained), trained on the
# 3 SOURCE domains with each optimiser, evaluated on the HELD-OUT domain.
# LOs REUSE the exp3 checkpoints (resnet8/cifar10 meta-training) — the
# strongest cross-everything generalization claim; 'sgd' is the paper's
# own baseline (SGD lr 1e-3, momentum 0.9). Hessian off (11.2M @ 224px);
# PGD / eps-sharpness metrics stay on.
PACS_EVAL_OVERRIDES = {"param_chunk_size": 262144, "eval_every": 10,
                       "batch_size": 64, "calc_hessian": False,
                       "pacs_img_size": 224}
for _i, _dom in enumerate(["art_painting", "cartoon", "photo", "sketch"]):
    EXPERIMENTS["exp6" + "abcd"[_i]] = _exp(
        "pacs_holdout_" + _dom, "domain_generalization",
        "cifar10", "pacs", "resnet8", "resnet18",
        methods=ALL_METHODS + ["sgd"],
        eval_overrides=dict(PACS_EVAL_OVERRIDES, pacs_holdout=_dom),
        notes=("Train ResNet-18 on the other 3 domains, evaluate on "
               + _dom + ". Compare against MetaReg Table 2."))


def lo_checkpoint_name(method, train_ds, train_arch):
    return f"{method.upper().replace('_','')}LO_{train_ds}_{train_arch}"


def lo_train_config(method, train_ds):
    return LO_TRAIN_CONFIG.get((method, train_ds), method)


def iter_experiments(selection=None):
    keys = list(EXPERIMENTS)
    if selection:
        want = set()
        for s in selection:
            s = str(s)
            want.add(s if s.startswith("exp") else f"exp{s}")
        # prefix match so "4"/"exp4" selects exp4a AND exp4b (Parts A/B)
        keys = [k for k in keys if k in want or
                any(k.startswith(w) and k[len(w):].isalpha() for w in want)]
    for k in keys:
        yield k, EXPERIMENTS[k]


def export_yaml(outdir="configs/experiments"):
    import yaml
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for k, e in EXPERIMENTS.items():
        (out / f"{k}.yaml").write_text(yaml.safe_dump({k: e}, sort_keys=False))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-yaml", action="store_true")
    a = ap.parse_args()
    if a.export_yaml:
        print("wrote", export_yaml())
    for k, e in EXPERIMENTS.items():
        if e.get("sweep"):
            print(f"{k:5s} {e['name']:22s} [{e['group']}]  -> run_sweep delegation")
            continue
        print(f"{k:5s} {e['name']:22s} [{e['group']:18s}] "
              f"train=({e['train_dataset']},{e['train_arch']}) "
              f"test=({e['test_dataset']},{e['test_arch']}) "
              f"methods={','.join(e['methods'])}")
