#!/bin/bash

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"

if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

cd "${ROOT_DIR}"

regularizers=("no_reg" "sam" "sam_opzee" "gam" "gsam" "l2")
datasets=("mnist" "fmnist" "cifar10")
models=("mlp" "mlp_Relu" "conv")
base_opts=("sgd" "adam")

for reg in "${regularizers[@]}"; do
  python3 experiments/dmo/train.py --regularizer="${reg}"
done

declare -A learned_optimizer_names=(
  ["no_reg"]="DMO_adam_280424"
  ["sam"]="AdSAM_sgd_LO_140424"
  ["sam_opzee"]="AdSAMopzee_sgd_LO_140424"
  ["gam"]="testpGAM_sgd_LO_300424"
  ["gsam"]="testpGSAM_sgd_LO_300424"
  ["l2"]="L2_adam_LO_140424"
)

for reg in "${!learned_optimizer_names[@]}"; do
  saved_model="${learned_optimizer_names[$reg]}"
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      python3 experiments/dmo/evaluate_optimizee.py --dataset="${dataset}" --optimizee_model="${model}" --saved_model="${saved_model}"
    done
  done
done

for base_opt in "${base_opts[@]}"; do
  for reg in "no_reg" "sam" "gam" "gsam"; do
    for dataset in "${datasets[@]}"; do
      for model in "${models[@]}"; do
        python3 experiments/dmo/evaluate_optimizee.py --dataset="${dataset}" --optimizee_model="${model}" --base_optimizer="${base_opt}" --if_normal_optim --regularizer="${reg}"
      done
    done
  done
done