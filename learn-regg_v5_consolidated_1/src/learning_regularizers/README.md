# Learning Regularizers - DMO (Differentiable Meta-Optimizer)

This folder contains the implementation of **DMO (Differentiable Meta-Optimizer)** - a learned optimizer trained via meta-learning to incorporate explicit regularizers (SAM, GAM, GSAM, smooth regularization) for improved generalization.

## Overview

The DMO framework consists of two main stages:

1. **Meta-Training** (`train.py`): Train an LSTM-based meta-optimizer on a dataset using curriculum learning
2. **Optimizee Evaluation** (`optimizee_train.py`): Evaluate the trained meta-optimizer on downstream tasks

## Quick Start

### 1. Setup

```bash
# Navigate to workspace root
cd /Users/surajsahoo/Documents/2024_thesis/All_Code_Suraj_LSAIML

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### 1.5 GPU Device Configuration

**Supported Devices:**
- **NVIDIA CUDA**: Automatically detected and used if available
- **Apple Silicon (M1/M2/M3)**: Metal Performance Shaders (MPS) support enabled
- **CPU**: Fallback device if no GPU is available

The code automatically prioritizes hardware acceleration in this order:
1. NVIDIA CUDA GPU (if `--cuda_` flag enabled and CUDA is available)
2. Apple Silicon GPU via Metal Performance Shaders (MPS)
3. CPU (default fallback)

**For Apple Silicon users:**
The code will automatically detect and use Metal Performance Shaders when running on M1/M2/M3 chips. No additional configuration needed—just run the training commands normally!

### 2. Meta-Train a DMO with SAM Regularizer (Recommended First Experiment)

```bash
cd src/learning_regularizers/dmo
python train.py --regularizer sam
```

This will:
- Load meta-training configuration from `configs/dmo/base/meta_train.yaml` and `configs/dmo/regularizers/sam.yaml`
- Train an LSTM-based meta-optimizer with curriculum learning (100 → 200 → 500 → 1000 optimization steps)
- Save checkpoints to `artifacts/dmo/checkpoints/`
- Display training loss, validation loss, and curriculum progression

**Expected runtime**: ~2-4 hours on GPU (depends on hardware)

### 3. Evaluate the Trained DMO

```bash
python optimizee_train.py --saved_model dmo_sam_model --base_optimizer sgd
```

This will:
- Load the trained meta-optimizer from checkpoints
- Run multiple downstream optimizee training runs (default: 5 runs)
- Compare against baseline optimizers (SGD)
- Save results to `artifacts/dmo/results/json/`

---

## File Structure

```
dmo/
├── train.py                          # Meta-training script
├── optimizee_train.py                # Downstream evaluation script
├── meta_optimizer.py                 # LSTM-based learned optimizer
├── model.py                          # Optimizee model definitions (MLP, Conv, etc.)
├── dataloaders.py                    # Dataset loading utilities
├── utils.py                          # Training loops, PGD analysis, helper functions
├── experiment_config.py              # Configuration loading from YAML
├── artifact_manager.py               # Centralized path management for artifacts
├── regularized_optimizer_helper.py   # Regularizer integration utilities
├── hyper_tuning.py                   # Hyperparameter search utilities
├── analysis.py                       # Post-hoc analysis utilities
├── regularizers/                     # Explicit regularizer implementations
│   ├── sam.py                        # Sharpness Aware Minimization
│   ├── gam.py                        # Gradient Ascent Minimization
│   ├── gsam.py                       # Group SAM
│   ├── smooth.py                     # Smooth regularization
│   └── scheduler.py                  # Learning rate schedulers
└── README.md                         # Detailed documentation
```

---

## Experiment Parameters

### Available Regularizers

The DMO can be trained with different explicit regularizers. Each has its own YAML config:

```bash
python train.py --regularizer sam        # Sharpness Aware Minimization
python train.py --regularizer gam        # Gradient Ascent Minimization
python train.py --regularizer gsam       # Group SAM
python train.py --regularizer smooth     # Smooth regularization
python train.py --regularizer no_reg     # No explicit regularization (baseline)
```

### Available Datasets

Configured in `configs/dmo/base/meta_train.yaml` and `configs/dmo/base/optimizee_eval.yaml`:

- `mnist` - MNIST (default, fastest)
- `cifar10` - CIFAR-10 (more challenging)
- `fmnist` - FashionMNIST

### Available Optimizee Models

- `mlp` - Simple multi-layer perceptron
- `deep_mlp` - Deeper MLP variant
- `conv` - Convolutional network
- `conv_large` - Larger convolutional network

### Key Configuration Options

**Meta-Training (`configs/dmo/base/meta_train.yaml`):**
- `batch_size`: Training batch size (default: 128)
- `optimizee_steps`: Max optimization steps for meta-optimizer (default: 1000)
- `truncated_bptt_step`: Backprop-through-time truncation (default: 20)
- `max_epochs`: Max meta-training epochs (default: 10000)
- `hidden_size`: LSTM hidden dimension (default: 20)
- `if_cl`: Enable curriculum learning (default: True)
- `seed`: Random seed for reproducibility (default: 1)

**Optimizee Evaluation (`configs/dmo/base/optimizee_eval.yaml`):**
- `optimizee_epochs`: Number of downstream training steps (default: 1000)
- `n_training`: Number of repeated optimizee training runs (default: 5)
- `base_optimizer`: Baseline optimizer (default: "sgd")
- `if_adaptive`: Use adaptive learning rates with regularizer (default: True)
- `eps`: Epsilon ball radius for PGD analysis (default: 0.1)
- `calc_hessian`: Compute Hessian eigenvalues (default: False)

---

## Running the Full Paper Experiments

### Experiment 1: Train DMO with Different Regularizers (Main Results)

```bash
#!/bin/bash
cd src/learning_regularizers/dmo

# Train DMO with each regularizer
for regularizer in no_reg sam gam gsam smooth; do
    echo "Training DMO with $regularizer..."
    python train.py --regularizer $regularizer
done
```

**Output:** Checkpoints saved as `dmo_<regularizer>_model.pt` in `artifacts/dmo/checkpoints/`

### Experiment 2: Evaluate All Trained DMOs

```bash
#!/bin/bash
cd src/learning_regularizers/dmo

# Evaluate each trained DMO
for regularizer in no_reg sam gam gsam smooth; do
    echo "Evaluating DMO with $regularizer..."
    python optimizee_train.py --saved_model dmo_${regularizer}_model --base_optimizer sgd
done
```

**Output:** Results saved as timestamped JSON files in `artifacts/dmo/results/json/`

### Experiment 3: Cross-Dataset Evaluation

```bash
#!/bin/bash
cd src/learning_regularizers/dmo

# Edit configs/dmo/base/optimizee_eval.yaml to change dataset
# Then run evaluation on each dataset

for dataset in mnist cifar10 fmnist; do
    echo "Evaluating on $dataset..."
    # Modify dataset in config, then run:
    python optimizee_train.py --saved_model dmo_sam_model --base_optimizer sgd
done
```

### Experiment 4: Curriculum Learning Ablation

```bash
# Train with curriculum learning disabled
python train.py --regularizer sam  # Edit config to set if_cl: False
```

---

## Understanding the Output

### Checkpoint Directory: `artifacts/dmo/checkpoints/`

```
artifacts/dmo/checkpoints/
├── dmo_sam_model.pt            # Trained meta-optimizer with SAM
├── dmo_gam_model.pt            # Trained meta-optimizer with GAM
├── dmo_no_reg_model.pt         # Trained meta-optimizer without regularizer
└── ...
```

Each checkpoint is a complete LSTM meta-optimizer state that can be loaded for evaluation.

### Results Directory: `artifacts/dmo/results/json/`

```
artifacts/dmo/results/json/
├── optimizee_results_20260419_143022.json   # DMO evaluation results
├── baseline_results_20260419_143045.json    # Baseline optimizer results
└── ...
```

**Result JSON Structure:**
```json
{
  "regularizer": "sam",
  "dataset": "mnist",
  "model": "mlp",
  "optimizer": "dmo",
  "training_losses": [...],
  "validation_losses": [...],
  "test_accuracy": 0.9847,
  "final_loss": 0.0245,
  "convergence_speed": 542,
  "run_timestamp": "2026-04-19_143022"
}
```

---

## Reproducing Specific Results

### Reproduce Paper Result: DMO-SAM on MNIST with MLP

```bash
# Exact reproduction
cd src/learning_regularizers/dmo
python train.py --regularizer sam
python optimizee_train.py --saved_model dmo_sam_model --base_optimizer sgd
```

### Analyze Results

```python
import json
from pathlib import Path

results_dir = Path("artifacts/dmo/results/json")
for result_file in sorted(results_dir.glob("*.json")):
    with open(result_file) as f:
        results = json.load(f)
    print(f"{results['regularizer']}: Final Loss = {results['final_loss']:.4f}, "
          f"Test Acc = {results['test_accuracy']:.4f}")
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or model complexity:
```yaml
# In configs/dmo/base/meta_train.yaml
batch_size: 64  # Reduce from 128
hidden_size: 10  # Reduce from 20
```

### Checkpoint Not Found

Verify checkpoint exists:
```bash
ls artifacts/dmo/checkpoints/
```

If missing, run meta-training first with `train.py`.

---

## Artifact Management

All outputs are centralized through the **artifact_manager** system:

```python
# In code: All paths handled automatically via artifact_manager.py
from artifact_manager import (
    get_checkpoint_path,           # Load/save checkpoints
    get_checkpoints_dir,           # Checkpoint directory
    get_timestamped_result_file,   # Timestamped result files
    get_results_dir                # Results directory
)

# Automatic directory creation and timestamping ensures reproducibility
```

---

## Directory Structure

- `dmo/`: Main DMO implementation (this folder)
- `dmo/regularizers/`: Explicit regularizer implementations (SAM, GAM, GSAM, smooth)
- `configs/dmo/base/`: Base meta-training and evaluation configurations
- `configs/dmo/regularizers/`: Regularizer-specific configurations

The active DMO entrypoints are in `src/learning_regularizers/dmo`.
