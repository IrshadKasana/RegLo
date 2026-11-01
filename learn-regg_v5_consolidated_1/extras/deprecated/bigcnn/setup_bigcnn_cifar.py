#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_bigcnn_cifar.py
=====================
Adds a BIGGER CNN (~1.28M params, under the 2M LO ceiling) and sets up
CIFAR-10 experiments for three methods:

    1. No Reg     (learned optimizer)
    2. SAM        (learned optimizer)
    3. AdamW+SAM  (explicit baseline, no LO)

The bigger CNN (BigConvNN) uses ONLY Conv2d + Linear layers (each with clean
weight+bias), so it is fully compatible with the LO's get_flat_params /
set_flat_params machinery. Verified: 1,276,234 params, ~16GB LO graph memory
(fits a 40GB GPU). No BatchNorm (keeps the flat-param flow simple for the LO).

Architecture:
    conv1: 3->32  (3x3)   -> pool 16x16
    conv2: 32->64 (3x3)   -> pool 8x8
    conv3: 64->128(3x3)   -> pool 4x4
    fc1: 2048->512  fc2: 512->256  fc3: 256->10

This script:
  1. Patches model.py: appends BigConvNN class + registers model_name='big_conv'
  2. Creates {no_reg,sam,adamw_sam}_bigcnn.yaml (dataset=cifar10, model=big_conv)
  3. Registers them in experiment_config.py
  4. Generates train + eval PBS scripts + submit scripts

Prereq: setup_baseline_comparison.py must have been run once (AdamW support in
SAM class). You did this for MNIST.

Run from project root:
    python setup_bigcnn_cifar.py
"""

from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parent
MODEL_PY = ROOT / "src" / "learning_regularizers" / "dmo" / "model.py"
REG_CFG = ROOT / "configs" / "dmo" / "regularizers"
PBS_DIR = ROOT / "pbs_scripts"
EXP_CONFIG = ROOT / "src" / "learning_regularizers" / "dmo" / "experiment_config.py"
REG_CFG.mkdir(parents=True, exist_ok=True)
PBS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. The BigConvNN class (appended to model.py) + dispatch registration
# ----------------------------------------------------------------------------
BIGCNN_CLASS = '''

# ============================================================
# BigConvNN: a larger CNN (~1.28M params) for CIFAR-10 under the
# learned optimizer. Conv2d + Linear only (LO-compatible flat-params).
# Added by setup_bigcnn_cifar.py
# ============================================================
class BigConvNN(nn.Module):
    def __init__(self, num_classes, data):
        super(BigConvNN, self).__init__()
        in_ch = 3 if data == 'cifar10' else 1
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # -> 16x16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # -> 8x8
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)   # -> 4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)
'''

def patch_model_py():
    src = MODEL_PY.read_text()
    # a) append the class if not present
    if "class BigConvNN" not in src:
        src = src + BIGCNN_CLASS
        print("  [OK] Appended BigConvNN class to model.py")
    else:
        print("  [OK] BigConvNN class already in model.py")
    # b) register in initialize_model dispatch
    anchor = "    elif (model_name == 'conv'):\n        return Conv_NN(num_classes, dataset)\n"
    addition = anchor + "    elif (model_name == 'big_conv'):\n        return BigConvNN(num_classes, dataset)\n"
    if "model_name == 'big_conv'" not in src:
        if anchor in src:
            src = src.replace(anchor, addition)
            print("  [OK] Registered 'big_conv' in initialize_model")
        else:
            print("  ! Could not find the conv dispatch anchor in model.py")
    else:
        print("  [OK] 'big_conv' already registered in initialize_model")
    MODEL_PY.write_text(src)


# ----------------------------------------------------------------------------
# 2. Configs
# ----------------------------------------------------------------------------
YAMLS = {
    "no_reg": """\
train_arguments:
  base_optimizer: 'adam'
  lr: 0.01
  weight_decay: 0.0
  momentum: 0.0
  model_name: 'DMO_adam_bigcnn'
  regularizer: 'no_reg'
  optzee_wd: 0.0
  if_avreg: False
  if_pretrained: False
  pretrained_model_name: 'no_name'
  dataset: 'cifar10'
  optimizee_model_name: 'big_conv'
""",
    "sam": """\
train_arguments:
  adaptive: True
  SAMonOptimizer: True
  base_optimizer: 'sgd'
  lr: 0.01
  rho: 0.05
  model_name: 'AdSAM_sgd_bigcnn'
  regularizer: 'sam'
  optzee_wd: 0.0
  weight_decay: 0.0
  momentum: 0.0
  if_avreg: True
  if_pretrained: False
  pretrained_model_name: 'no_name'
  dataset: 'cifar10'
  optimizee_model_name: 'big_conv'
""",
    "adamw_sam": """\
train_arguments:
  adaptive: True
  SAMonOptimizer: True
  base_optimizer: 'AdamW'
  lr: 0.001
  rho: 0.05
  weight_decay: 0.01
  momentum: 0.0
  optzee_wd: 0.0
  model_name: 'AdamW_SAM_bigcnn'
  regularizer: 'sam'
  if_avreg: False
  if_pretrained: False
  pretrained_model_name: 'no_name'
  dataset: 'cifar10'
  optimizee_model_name: 'big_conv'
""",
}

LO_CKPT = {"no_reg": "DMO_adam_bigcnn", "sam": "AdSAM_sgd_bigcnn"}


def write_configs():
    for reg, body in YAMLS.items():
        (REG_CFG / f"{reg}_bigcnn.yaml").write_text(body)
        print(f"  + configs/dmo/regularizers/{reg}_bigcnn.yaml")


def patch_experiment_config():
    src = EXP_CONFIG.read_text()
    marker = "REGULARIZER_CONFIGS = {"
    if marker not in src:
        print(f"  ! REGULARIZER_CONFIGS not found in {EXP_CONFIG.name}")
        return
    needed = {f"{r}_bigcnn": f"{r}_bigcnn.yaml" for r in YAMLS}
    if all(f'"{k}"' in src for k in needed):
        print(f"  [OK] {EXP_CONFIG.name} already has bigcnn entries")
        return
    start = src.index(marker); end = src.index("}", start)
    block = src[start:end + 1]
    new = "\n".join(f'    "{k}": "{v}",' for k, v in needed.items() if f'"{k}"' not in block)
    EXP_CONFIG.write_text(src.replace(block, block.rstrip("}").rstrip() + "\n" + new + "\n}"))
    print(f"  [OK] Registered bigcnn configs in {EXP_CONFIG.name}")


# ----------------------------------------------------------------------------
# 3. PBS scripts
# ----------------------------------------------------------------------------
TRAIN_PBS = """\
#!/bin/bash
#PBS -N tr_{reg}_bigcnn
#PBS -q gpu-test
#PBS -l nodes=1:ppn=40
#PBS -l walltime=23:00:00
#PBS -l mem=60gb
#PBS -o train_{reg}_bigcnn.out
#PBS -e train_{reg}_bigcnn.err
#PBS -V

cd $PBS_O_WORKDIR
module load compilers/cuda/11.1

echo "==== Job started at $(date) on $(hostname) ===="
echo "==== {reg}-LO  |  CIFAR-10 + BigCNN (~1.28M params) ===="
nvidia-smi || echo "no GPU visible"

~/.conda/envs/learn_reg/bin/python experiments/dmo/train.py \\
    --regularizer={reg}_bigcnn

echo "==== Job ended at $(date) ===="
"""

LO_EVAL_PBS = """\
#!/bin/bash
#PBS -N ev_{reg}_bigcnn
#PBS -q gpu-test
#PBS -l nodes=1:ppn=40
#PBS -l walltime=12:00:00
#PBS -l mem=60gb
#PBS -o eval_{reg}_bigcnn.out
#PBS -e eval_{reg}_bigcnn.err
#PBS -V

cd $PBS_O_WORKDIR
module load compilers/cuda/11.1

echo "==== Eval {reg}-LO  |  CIFAR-10 + BigCNN  $(date) ===="
nvidia-smi || echo "no GPU visible"

~/.conda/envs/learn_reg/bin/python experiments/dmo/evaluate_optimizee.py \\
    --dataset=cifar10 \\
    --optimizee_model=big_conv \\
    --saved_model={ckpt}

echo "==== Job ended at $(date) ===="
"""

BASELINE_EVAL_PBS = """\
#!/bin/bash
#PBS -N ev_adamw_bigcnn
#PBS -q gpu-test
#PBS -l nodes=1:ppn=40
#PBS -l walltime=12:00:00
#PBS -l mem=60gb
#PBS -o eval_adamw_sam_bigcnn.out
#PBS -e eval_adamw_sam_bigcnn.err
#PBS -V

cd $PBS_O_WORKDIR
module load compilers/cuda/11.1

echo "==== Baseline AdamW+SAM (no LO)  |  CIFAR-10 + BigCNN  $(date) ===="
nvidia-smi || echo "no GPU visible"

~/.conda/envs/learn_reg/bin/python experiments/dmo/evaluate_optimizee.py \\
    --dataset=cifar10 \\
    --optimizee_model=big_conv \\
    --if_normal_optim \\
    --base_optimizer=AdamW \\
    --regularizer=adamw_sam_bigcnn

echo "==== Job ended at $(date) ===="
"""


def write_pbs():
    for reg in LO_CKPT:
        (PBS_DIR / f"train_{reg}_bigcnn.pbs").write_text(TRAIN_PBS.format(reg=reg))
        print(f"  + pbs_scripts/train_{reg}_bigcnn.pbs")
    for reg in LO_CKPT:
        (PBS_DIR / f"eval_{reg}_bigcnn.pbs").write_text(
            LO_EVAL_PBS.format(reg=reg, ckpt=LO_CKPT[reg]))
        print(f"  + pbs_scripts/eval_{reg}_bigcnn.pbs")
    (PBS_DIR / "eval_adamw_sam_bigcnn.pbs").write_text(BASELINE_EVAL_PBS)
    print("  + pbs_scripts/eval_adamw_sam_bigcnn.pbs")


def write_submit():
    (ROOT / "submit_train_bigcnn.sh").write_text(
        "#!/bin/bash\nset -e\n"
        'cd "$(dirname "$0")"\n'
        "qsub pbs_scripts/train_no_reg_bigcnn.pbs\n"
        "qsub pbs_scripts/train_sam_bigcnn.pbs\n"
        "qsub pbs_scripts/eval_adamw_sam_bigcnn.pbs\n"
        "echo\necho 'Submitted 2 trainings + baseline eval. Check: qstat -u 2025pcs0034'\n")
    (ROOT / "submit_eval_bigcnn.sh").write_text(
        "#!/bin/bash\nset -e\n"
        'cd "$(dirname "$0")"\n'
        "qsub pbs_scripts/eval_no_reg_bigcnn.pbs\n"
        "qsub pbs_scripts/eval_sam_bigcnn.pbs\n"
        "echo\necho 'Submitted. Check: qstat -u 2025pcs0034'\n")
    for sh in ["submit_train_bigcnn.sh", "submit_eval_bigcnn.sh"]:
        p = ROOT / sh
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print("  + submit_train_bigcnn.sh")
    print("  + submit_eval_bigcnn.sh")


def main():
    print("=" * 62)
    print("Bigger CNN (~1.28M params) + CIFAR-10  (No Reg / SAM / AdamW+SAM)")
    print("=" * 62)
    print("\n[1/5] Patching model.py (add BigConvNN + register big_conv)...")
    patch_model_py()
    print("\n[2/5] Writing configs...")
    write_configs()
    print("\n[3/5] Registering configs in experiment_config.py...")
    patch_experiment_config()
    print("\n[4/5] Writing PBS scripts...")
    write_pbs()
    print("\n[5/5] Writing submit scripts...")
    write_submit()
    print("\n" + "=" * 62)
    print("DONE. Next steps:")
    print("=" * 62)
    print("""
  PREREQ: setup_baseline_comparison.py already run (AdamW in SAM class). Done.

  1. Pre-download CIFAR-10 (login node):
       ~/.conda/envs/learn_reg/bin/python -c "from torchvision import datasets; datasets.CIFAR10('datasets', download=True)"

  2. Submit both LO trainings + baseline eval:
       ./submit_train_bigcnn.sh
       qstat -u 2025pcs0034

  3. BigCNN (1.28M params) is ~60x bigger than the paper's CNN -> the LSTM
     per-param overhead makes training SLOWER and uses ~16GB GPU. Expect to
     hit 23h walltime -> resume-chain each (if_pretrained: True + the _bigcnn
     checkpoint name in the *_bigcnn.yaml), resubmit until plateaued.

  4. After training, eval:
       ./submit_eval_bigcnn.sh

  Result JSONs:
    No Reg   -> ...DMO_adam_bigcnn...json
    SAM      -> ...AdSAM_sgd_bigcnn...json
    Baseline -> ...AdamW_SAM_big_conv_cifar10...json

  WATCH the first run's .err for OOM. If OOM, lower truncated_bptt_step in
  meta_train.yaml (20 -> 10) to halve LO graph memory, or reduce batch_size.
""")


if __name__ == "__main__":
    main()