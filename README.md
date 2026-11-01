# Learning Regularizers: Experiments and Reproducibility Guide


## What Is In This Repository

This is a research workspace rather than a cleaned software package. The code relevant to the paper is concentrated in:

- [`src/learning_regularizers/dmo`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/src/learning_regularizers/dmo): active implementation used for the paper experiments.
- [`configs/dmo`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/configs/dmo): active DMO experiment configs.
- [`experiments/dmo`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/experiments/dmo): runnable DMO experiment entrypoints and SLURM examples.
- [`variants`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/variants): related SDMO and Scale variants.
- [`experiments/resnet_task`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/experiments/resnet_task): additional experiments on ResNet-style models.
- [`extras`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/extras): upstream comparison code and reference implementations.
- [`archive/l2l`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/archive/l2l) and [`archive/inactive-code`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/archive/inactive-code): older prototypes and archived implementations.

For reproducing the paper, start with `experiments/dmo`.

## Main Experiment Pipeline

The active implementation is under [`src/learning_regularizers/dmo`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/src/learning_regularizers/dmo). The old `L2O-CL/DMO/train.py` and `L2O-CL/DMO/optimizee_train.py` files are compatibility wrappers.

- [`src/train.py`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/src/learning_regularizers/dmo/train.py): meta-trains the learned optimizer.
- [`src/optimizee_train.py`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/src/learning_regularizers/dmo/optimizee_train.py): evaluates downstream optimizee training using either a learned optimizer or a standard optimizer.
- [`src/meta_optimizer.py`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/src/learning_regularizers/dmo/meta_optimizer.py): learned optimizer architecture.
- [`src/model.py`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/src/learning_regularizers/dmo/model.py): optimizee models such as `mlp`, `mlp_Relu`, `deep_mlp`, and `conv`.
- [`src/dataloaders.py`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/src/learning_regularizers/dmo/dataloaders.py): dataset loading for MNIST, FashionMNIST, and CIFAR-10.
- [`src/learning_regularizers/dmo/regularizers`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/src/learning_regularizers/dmo/regularizers): active SAM, GAM, GSAM, smooth-regularization, and scheduler implementations.
- [`src/run`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/src/run): main shell scripts for the DMO experiment pipeline.
- [`configs/dmo/base`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/configs/dmo/base): shared meta-training and optimizee-evaluation configs.
- [`configs/dmo/regularizers`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/configs/dmo/regularizers): configs for no-reg, SAM, SAM-on-optimizee, GAM, GSAM, and smooth regularization.
- [`extras/dmo/gam_reference`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/extras/dmo/gam_reference): archived upstream/reference GAM code. The active DMO flow does not import this folder.
- [`experiments/dmo/slurm_train.sh`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/experiments/dmo/slurm_train.sh) and [`experiments/dmo/slurm_eval.sh`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/experiments/dmo/slurm_eval.sh): SLURM submission examples.

## Experiments Covered By The Paper

The code and configs indicate the following main experiment families.

### 1. Meta-training learned optimizers

Train a learned optimizer:

- without explicit regularization,
- with SAM-style regularization,
- with SAM applied on optimizee parameters,
- with GAM,
- with GSAM.

These are controlled by:

- [`configs/dmo/base/meta_train.yaml`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/configs/dmo/base/meta_train.yaml)
- [`configs/dmo/regularizers/no_reg.yaml`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/configs/dmo/regularizers/no_reg.yaml)
- [`configs/dmo/regularizers/sam.yaml`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/configs/dmo/regularizers/sam.yaml)
- [`configs/dmo/regularizers/sam_opzee.yaml`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/configs/dmo/regularizers/sam_opzee.yaml)
- [`configs/dmo/regularizers/gam.yaml`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/configs/dmo/regularizers/gam.yaml)
- [`configs/dmo/regularizers/gsam.yaml`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/configs/dmo/regularizers/gsam.yaml)

### 2. Downstream optimizee evaluation

Evaluate trained learned optimizers on downstream tasks across:

- datasets: `mnist`, `fmnist`, `cifar10`
- optimizee models: `mlp`, `mlp_Relu`, `deep_mlp`, `conv`

The downstream runner is:

- [`optimizee_train.py`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/L2O-CL/DMO/optimizee_train.py)

### 3. Baseline optimizer comparisons

Compare learned optimizers against standard optimizers:

- `sgd`
- `adam`

with:

- no regularizer,
- `sam`,
- `gam`,
- `gsam`

This is handled in the same `optimizee_train.py` script using `--if_normal_optim`.

### 4. Neighborhood / sharpness analysis

The evaluation script also computes:

- max loss in an epsilon-neighborhood,
- max gradient norm in an epsilon-neighborhood,
- optionally over a range of epsilon values using `--hyper_eps`.

This corresponds to the paper’s claim that learned optimizers retain and transfer regularization effects by finding minima similar to those favored by explicit regularizers.

## Before Running Anything

The repository currently contains many machine-specific hardcoded paths such as:

- `/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/...`

These appear in:

- `L2O-CL/DMO/train.py`
- `L2O-CL/DMO/optimizee_train.py`
- `src/learning_regularizers/dmo/dataloaders.py`
- `experiments/dmo/slurm_train.sh`
- `experiments/dmo/slurm_eval.sh`
- `configs/dmo/**/*.yaml`
- similar files in `variants/`, `extras/`, and other experiment folders

You will need to replace these with paths valid on your machine.

### Recommended path replacement

Set:

- project root: `/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML`
- datasets root: `/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/datasets`
- checkpoints root: `/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/artifacts/dmo/checkpoints`
- results root: `/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/artifacts/dmo/results`

You can locate hardcoded references with:

```bash
rg -n "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML" /Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML
```

## Environment Setup

There is no single clean pinned environment file for the full DMO pipeline, so setup is partly manual.

At minimum, the code uses:

- Python
- PyTorch
- torchvision
- numpy
- tqdm
- pyyaml
- `hessian_eigenthings`
- `pyhessian`

Suggested starting point:

```bash
cd /Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy tqdm pyyaml hessian_eigenthings pyhessian
```

You can also install from the repository root with:

```bash
pip install -r /Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/requirements.txt
```

If you are running on a cluster, adapt the original SLURM scripts rather than running directly.

## How To Reproduce The Main DMO Experiments

Change into:

```bash
cd /Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/L2O-CL/DMO
```

### A. Train an unregularized learned optimizer

```bash
python train.py --regularizer=no_reg
```

This uses:

- `configs/dmo/base/meta_train.yaml` for shared meta-training settings
- `configs/dmo/regularizers/no_reg.yaml` for the unregularized learned optimizer setup

### B. Train a learned optimizer regularized with SAM

```bash
python train.py --regularizer=sam
```

### C. Train a learned optimizer with SAM on the optimizee

```bash
python train.py --regularizer=sam_opzee
```

### D. Train a learned optimizer regularized with GAM

```bash
python train.py --regularizer=gam
```

### E. Train a learned optimizer regularized with GSAM

```bash
python train.py --regularizer=gsam
```

Checkpoints are saved under the path defined by `args.save`, which currently comes from the YAML files.

## Run The Full Paper Matrix

Once your Python environment is ready, the repository root now includes:

- [`run_paper_experiments.sh`](/Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/experiments/dmo/run_paper_experiments.sh)

This script:

- trains the learned optimizers for `no_reg`, `sam`, `sam_opzee`, `gam`, and `gsam`,
- evaluates the saved learned optimizers across datasets and optimizee models,
- evaluates SGD/Adam baselines with and without explicit regularizers.

Run it from the repository root with:

```bash
bash /Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML/experiments/dmo/run_paper_experiments.sh
```

## Evaluate A Trained Learned Optimizer

To evaluate a learned optimizer on a downstream task:

```bash
python optimizee_train.py \
  --dataset=mnist \
  --optimizee_model=mlp \
  --saved_model=DMO_adam_280424
```

Example variants:

```bash
python optimizee_train.py --dataset=mnist --optimizee_model=mlp_Relu --saved_model=AdSAM_sgd_LO_140424
python optimizee_train.py --dataset=fmnist --optimizee_model=conv --saved_model=testpGAM_sgd_LO_300424
python optimizee_train.py --dataset=cifar10 --optimizee_model=mlp --saved_model=testpGSAM_sgd_LO_300424
```

## Evaluate Standard Optimizers With Explicit Regularizers

Examples:

### SGD / Adam without regularization

```bash
python optimizee_train.py --dataset=mnist --optimizee_model=mlp --if_normal_optim --base_optimizer=sgd --regularizer=no_reg
python optimizee_train.py --dataset=mnist --optimizee_model=mlp --if_normal_optim --base_optimizer=adam --regularizer=no_reg
```

### SAM

```bash
python optimizee_train.py --dataset=mnist --optimizee_model=mlp --if_normal_optim --base_optimizer=sgd --regularizer=sam
python optimizee_train.py --dataset=mnist --optimizee_model=mlp --if_normal_optim --base_optimizer=adam --regularizer=sam
```

### GAM

```bash
python optimizee_train.py --dataset=mnist --optimizee_model=mlp --if_normal_optim --base_optimizer=sgd --regularizer=gam
python optimizee_train.py --dataset=mnist --optimizee_model=mlp --if_normal_optim --base_optimizer=adam --regularizer=gam
```

### GSAM

```bash
python optimizee_train.py --dataset=mnist --optimizee_model=mlp --if_normal_optim --base_optimizer=sgd --regularizer=gsam
python optimizee_train.py --dataset=mnist --optimizee_model=mlp --if_normal_optim --base_optimizer=adam --regularizer=gsam
```

The same pattern applies to:

- `--dataset=fmnist`
- `--dataset=cifar10`
- `--optimizee_model=mlp_Relu`
- `--optimizee_model=conv`

These combinations are also reflected in the original batch scripts and output logs.

## Inspect Neighborhood Sharpness

To compute PGD-based neighborhood behavior during evaluation:

```bash
python optimizee_train.py \
  --dataset=mnist \
  --optimizee_model=mlp \
  --saved_model=AdSAM_sgd_LO_140424 \
  --eps=0.1
```

To sweep multiple epsilon values:

```bash
python optimizee_train.py \
  --dataset=mnist \
  --optimizee_model=mlp \
  --saved_model=AdSAM_sgd_LO_140424 \
  --hyper_eps
```

## Expected Outputs

The code writes outputs mainly to:

- checkpoint files under `artifacts/dmo/checkpoints`
- JSON result files under `artifacts/dmo/results/json`
- SLURM stdout/stderr logs under `artifacts/dmo/console/outs` and `artifacts/dmo/console/errors`

The JSON files produced by `optimizee_train.py` contain:

- training loss trajectories
- test loss trajectories
- test accuracies
- neighborhood max-loss statistics
- neighborhood max-gradient-norm statistics

### Professional Artifact Management

Checkpoints and results are managed by the `artifact_manager.py` module in `src/learning_regularizers/dmo/`. This provides:

- **Centralized path management** - all artifact paths defined in one place
- **Timestamped results** - JSON files automatically timestamped for reproducibility
- **Consistent structure** - checkpoints always saved to `artifacts/dmo/checkpoints/`, results to `artifacts/dmo/results/json/`
- **Helper functions** - list saved artifacts, verify paths, print artifact directory structure

Key functions available from `artifact_manager.py`:

```python
from artifact_manager import (
    get_checkpoint_path(model_name),              # Get full path for checkpoint
    get_checkpoints_dir(),                         # Get checkpoints directory
    get_timestamped_result_file(base_name),       # Get timestamped result file path
    get_result_file_path(filename),                # Get raw result file path
    list_checkpoints(),                            # List all saved checkpoints
    list_results(),                                # List all saved results
    print_artifact_structure()                     # Print artifact organization
)
```

Both `train.py` and `optimizee_train.py` use this module to ensure reproducible artifact storage.

## Repository Caveats

This code reproduces the original work, but it is not yet a cleaned release. A few important caveats:

- many paths are hardcoded to the original cluster filesystem,
- the repository contains datasets, checkpoints, console logs, caches, and virtual environments mixed with source code,
- several experiment directories are near-duplicates of each other,
- the batch scripts are cluster-specific,
- some files contain copy-paste drift across experiment variants.

For that reason, the safest reproduction route is:

1. fix all hardcoded paths,
2. start with `L2O-CL/DMO`,
3. run one small MNIST experiment first,
4. then scale to FMNIST and CIFAR-10,
5. then compare learned optimizer runs against SGD/Adam with SAM/GAM/GSAM.

## Suggested Minimal Reproduction

If you want the shortest path to re-running the core result:

1. Train one unregularized learned optimizer on MNIST.
2. Train one regularized learned optimizer, for example `--regularizer=sam` or `--regularizer=gam`.
3. Evaluate both on the same downstream optimizee such as `mnist + mlp`.
4. Compare against `sgd` and `adam` baselines with the matching explicit regularizer.
5. Use `--hyper_eps` to inspect whether the learned optimizer converges to flatter or more regularized neighborhoods.

## Notes

- The `DMO` folder appears to be the primary implementation for the paper.
- `SDMO` and `Scale` are useful related variants, but they do not look like the cleanest entry point for the main paper claims.
- The local PDF was referenced for this README, and the experiment summary here is aligned with the paper title, arXiv identifier, abstract, and the runnable code present in this repository.
