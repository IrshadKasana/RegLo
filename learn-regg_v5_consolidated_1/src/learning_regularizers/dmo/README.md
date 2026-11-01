# DMO Implementation

This folder contains the active DMO implementation used by the paper experiments.

Key files:
- `train.py`: meta-trains the learned optimizer.
- `optimizee_train.py`: evaluates downstream optimizee training.
- `meta_optimizer.py`: learned optimizer architecture and update logic.
- `model.py`: optimizee model definitions such as `mlp`, `mlp_Relu`, `deep_mlp`, and `conv`.
- `dataloaders.py`: dataset loading for MNIST, FashionMNIST, and CIFAR-10.
- `utils.py`: helper utilities for training, evaluation, and PGD analysis.
- `regularizers/`: explicit regularizer implementations (SAM, GAM, GSAM, smooth regularization, schedulers).

Compatibility wrappers in `L2O-CL/DMO/` and `experiments/dmo/` call this implementation.
