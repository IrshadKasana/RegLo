# Project Structure

The active project code now follows a root-level source layout.

## Active Source

- `src/learning_regularizers/dmo/`
  Active DMO implementation for training learned optimizers and evaluating optimizee tasks.

- `src/learning_regularizers/dmo/regularizers/`
  Active SAM, GAM, GSAM, smooth-regularization, and scheduler implementations.

## Source Organization

- `src/learning_regularizers/dmo/artifact_manager.py`
  Centralized artifact path manager for professional checkpoint and result storage.
  Provides functions: `get_checkpoints_dir()`, `get_results_dir()`, `get_timestamped_result_file()`, etc.

## Configuration

- `configs/dmo/base/`
  Shared meta-training and optimizee-evaluation YAML configs.

- `configs/dmo/regularizers/`
  Regularizer-specific YAML configs for `no_reg`, `sam`, `sam_opzee`, `gam`, `gsam`, and `smooth`.

## Experiment Entrypoints

- `experiments/dmo/train.py`
  Runs DMO meta-training.

- `experiments/dmo/evaluate_optimizee.py`
  Runs optimizee evaluation.

- `experiments/dmo/run_paper_experiments.sh`
  Runs the DMO paper experiment matrix.

- `experiments/dmo/slurm_*.sh`
  Cluster submission examples.

## Artifacts And Data

- `artifacts/dmo/checkpoints/`
  DMO learned optimizer checkpoint files (.pt). Managed by `artifact_manager.py`.

- `artifacts/dmo/results/json/`
  JSON output files with metrics and evaluation results. Timestamped for reproducibility.

- `artifacts/dmo/results/plots/`
  Plot and visualization files from analysis notebooks.

- `datasets/`
  Local dataset cache (MNIST, FashionMNIST, CIFAR-10).

## Notebooks

- `notebooks/dmo/`
  DMO analysis notebooks.

## Variants And Extras

- `variants/`
  Related active-ish variants such as SDMO and Scale.

- `experiments/resnet_task/`
  ResNet-specific experiment code.

- `extras/`
  External comparison code, upstream reference code, and old environments.

## Archive

- `archive/l2l/`
  Older L2L prototypes and baselines. Not imported by the active DMO path.

- `archive/inactive-code/`
  Legacy implementations (old DMO, old learning regularizer code). Preserved for reference.

## Compatibility Wrappers

- `L2O-CL/DMO/train.py`
- `L2O-CL/DMO/optimizee_train.py`

These are thin wrappers that call the root-level `src/learning_regularizers/dmo` implementation, so older commands still work.
