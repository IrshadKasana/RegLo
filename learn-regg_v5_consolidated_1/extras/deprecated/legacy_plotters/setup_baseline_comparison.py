#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_baseline_comparison.py
============================
Sets up the reviewer-requested baseline comparison:

    AdamW + SAM   (explicit sharpness-aware training at meta-test, NO learned optimizer)
    No Reg        (your learned optimizer, meta-trained without regularization)
    with SAM      (your learned optimizer, meta-trained with SAM regularization)

This script:
  1. Registers the 'adamw_sam_baseline' regularizer in experiment_config.py
  2. Generates a PBS script to run the AdamW+SAM baseline eval
  3. Generates a PBS script for a quick compute-benchmark (throughput / passes / time)

The SAM optimizer + AdamW base + if_normal_optim eval path ALREADY EXIST in the
repo. This script only wires up the missing config registration and run scripts.

Run once from the project root:
    python setup_baseline_comparison.py
"""

from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parent
PBS_DIR = ROOT / "pbs_scripts"
EXP_CONFIG = ROOT / "src" / "learning_regularizers" / "dmo" / "experiment_config.py"
PBS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Register the baseline config in REGULARIZER_CONFIGS
# ----------------------------------------------------------------------------
def patch_experiment_config():
    src = EXP_CONFIG.read_text()
    if '"adamw_sam_baseline"' in src:
        print("  [OK] experiment_config.py already has adamw_sam_baseline - skipping")
        return
    marker = 'REGULARIZER_CONFIGS = {'
    if marker not in src:
        print(f"  ! Could not find REGULARIZER_CONFIGS in {EXP_CONFIG}")
        return
    start = src.index(marker)
    end = src.index('}', start)
    block = src[start:end + 1]
    new_block = block.rstrip('}').rstrip() + '\n    "adamw_sam_baseline": "adamw_sam_baseline.yaml",\n}'
    EXP_CONFIG.write_text(src.replace(block, new_block))
    print(f"  [OK] Registered adamw_sam_baseline in {EXP_CONFIG.name}")


# ----------------------------------------------------------------------------
# 2. PBS script - run the AdamW + SAM baseline evaluation
#    Key flags: --if_normal_optim (bypasses LO) + --regularizer + --base_optimizer
# ----------------------------------------------------------------------------
EVAL_BASELINE_PBS = """\
#!/bin/bash
#PBS -N ev_adamw_sam
#PBS -q gpu-test
#PBS -l nodes=1:ppn=40
#PBS -l walltime=12:00:00
#PBS -l mem=60gb
#PBS -o eval_adamw_sam.out
#PBS -e eval_adamw_sam.err
#PBS -V

cd $PBS_O_WORKDIR
module load compilers/cuda/11.1

echo "==== Job started at $(date) on $(hostname) ===="
echo "==== Baseline: AdamW + SAM directly on optimizee (no LO) ===="
nvidia-smi || echo "no GPU visible"

# --if_normal_optim makes the eval bypass the learned optimizer and run
# a hand-coded AdamW + SAM directly on the optimizee.
~/.conda/envs/learn_reg/bin/python experiments/dmo/evaluate_optimizee.py \\
    --dataset=mnist \\
    --optimizee_model=mlp \\
    --if_normal_optim \\
    --base_optimizer=AdamW \\
    --regularizer=adamw_sam_baseline

echo "==== Job ended at $(date) ===="
"""

# ----------------------------------------------------------------------------
# 3. PBS script - compute benchmark (reviewer asked for throughput/passes/time)
# ----------------------------------------------------------------------------
BENCH_PBS = """\
#!/bin/bash
#PBS -N bench_compute
#PBS -q gpu-test
#PBS -l nodes=1:ppn=40
#PBS -l walltime=02:00:00
#PBS -l mem=60gb
#PBS -o bench_compute.out
#PBS -e bench_compute.err
#PBS -V

cd $PBS_O_WORKDIR
module load compilers/cuda/11.1

echo "==== Compute benchmark started at $(date) on $(hostname) ===="
nvidia-smi || echo "no GPU visible"

~/.conda/envs/learn_reg/bin/python benchmark_compute.py

echo "==== Job ended at $(date) ===="
"""


def write_pbs():
    (PBS_DIR / "eval_adamw_sam.pbs").write_text(EVAL_BASELINE_PBS)
    print(f"  + pbs_scripts/eval_adamw_sam.pbs")
    (PBS_DIR / "bench_compute.pbs").write_text(BENCH_PBS)
    print(f"  + pbs_scripts/bench_compute.pbs")


def write_submit():
    submit = ROOT / "submit_baseline_eval.sh"
    submit.write_text(
        "#!/bin/bash\n"
        "# Submits the AdamW+SAM baseline eval + compute benchmark.\n"
        "set -e\n"
        'cd "$(dirname "$0")"\n'
        "qsub pbs_scripts/eval_adamw_sam.pbs\n"
        "qsub pbs_scripts/bench_compute.pbs\n"
        "echo\necho 'Submitted. Check with: qstat -u 2025pcs0034'\n"
    )
    submit.chmod(submit.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  + submit_baseline_eval.sh")


OPTIMIZEE_TRAIN = ROOT / "src" / "learning_regularizers" / "dmo" / "optimizee_train.py"


def patch_sam_branch():
    """
    The SAM step in fit_w_normal_optimizer checks `args.regularizer=='sam'`.
    Our baseline uses regularizer='adamw_sam_baseline' so it would skip SAM and
    fall into the plain optimizer.step() branch. Make the branch also accept the
    baseline name so the baseline genuinely runs SAM.
    """
    src = OPTIMIZEE_TRAIN.read_text()
    changed = False

    # Patch the optimizer-construction check
    old_ctor = "        if args.regularizer=='sam':\n            sam_optimizer = SAM(model.parameters(),args = args)"
    new_ctor = "        if args.regularizer in ('sam', 'adamw_sam_baseline'):\n            sam_optimizer = SAM(model.parameters(),args = args)"
    if old_ctor in src and new_ctor not in src:
        src = src.replace(old_ctor, new_ctor)
        changed = True

    # Patch the step check
    old_step = "        if args.regularizer=='sam':\n            sam_optimizer.first_step(zero_grad=True)"
    new_step = "        if args.regularizer in ('sam', 'adamw_sam_baseline'):\n            sam_optimizer.first_step(zero_grad=True)"
    if old_step in src and new_step not in src:
        src = src.replace(old_step, new_step)
        changed = True

    # Patch the result-filename label so the baseline JSON is identifiable
    old_label = "    if args.regularizer=='sam':\n        optim_ += '_SAM_'"
    new_label = "    if args.regularizer in ('sam', 'adamw_sam_baseline'):\n        optim_ += '_SAM_'"
    if old_label in src and new_label not in src:
        src = src.replace(old_label, new_label)
        changed = True

    if changed:
        OPTIMIZEE_TRAIN.write_text(src)
        print(f"  [OK] Patched SAM branch in {OPTIMIZEE_TRAIN.name} to accept adamw_sam_baseline")
    else:
        print(f"  [OK] {OPTIMIZEE_TRAIN.name} SAM branch already handles the baseline")


OPTIMIZERS_PY = ROOT / "src" / "learning_regularizers" / "dmo" / "regularizers" / "optimizers.py"


def patch_sam_adamw_support():
    """
    The SAM optimizer class only builds base_optimizer for 'sgd' and 'adam'.
    Passing base_optimizer='AdamW' leaves self.base_optimizer unset and crashes.
    Add an AdamW branch so the baseline can use a true AdamW base optimizer.
    """
    src = OPTIMIZERS_PY.read_text()

    anchor = """        elif args.base_optimizer == 'adam':
            self.base_optimizer = optim.Adam(
                self.param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
                **kwargs
            )
"""
    adamw_branch = anchor + """
        elif args.base_optimizer == 'AdamW':
            self.base_optimizer = optim.AdamW(
                self.param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
                **kwargs
            )
"""
    # Detect whether the AdamW branch already exists INSIDE the SAM class
    # (the anchor is the SAM-class adam branch; only it has 'self.base_optimizer').
    if adamw_branch in src:
        print(f"  [OK] {OPTIMIZERS_PY.name} SAM class already supports AdamW base")
        return

    if anchor in src:
        src = src.replace(anchor, adamw_branch)
        OPTIMIZERS_PY.write_text(src)
        print(f"  [OK] Added AdamW base-optimizer branch to SAM class in {OPTIMIZERS_PY.name}")
    else:
        print(f"  ! Could not find the SAM-class adam branch in {OPTIMIZERS_PY.name}")


def main():
    print("=" * 60)
    print("Baseline Comparison Setup - AdamW+SAM vs No Reg vs LO-SAM")
    print("=" * 60)
    print("\n[1/5] Registering baseline config...")
    patch_experiment_config()
    print("\n[2/5] Adding AdamW support to SAM optimizer...")
    patch_sam_adamw_support()
    print("\n[3/5] Patching SAM branch to run baseline correctly...")
    patch_sam_branch()
    print("\n[4/5] Generating PBS scripts...")
    write_pbs()
    print("\n[5/5] Generating submit script...")
    write_submit()
    print("\n" + "=" * 60)
    print("DONE. Next steps:")
    print("=" * 60)
    print("  # 1. Run the AdamW+SAM baseline eval + compute benchmark:")
    print("  ./submit_baseline_eval.sh")
    print("")
    print("  # 2. Monitor:")
    print("  qstat -u 2025pcs0034")
    print("")
    print("  # 3. When done, generate the 3-bar comparison:")
    print("  ~/.conda/envs/learn_reg/bin/python generate_baseline_plots.py")
    print("")
    print("  NOTE: 'No Reg' and 'with SAM' bars reuse your EXISTING LO eval JSONs.")
    print("        Make sure your LO-SAM checkpoint is properly trained first")
    print("        (resume chaining) or that bar will be misleading.")


if __name__ == "__main__":
    main()