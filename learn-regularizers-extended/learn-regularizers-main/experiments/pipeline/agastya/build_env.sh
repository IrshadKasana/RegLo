#!/bin/bash
# Build the conda env for the learn-regularizers pipeline on Agastya (login node; needs internet).
set -x
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
E=$HOME/.conda/envs/lreg_pipe
[ -x $E/bin/python ] || conda create -y -p $E python=3.10
$E/bin/python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
$E/bin/python -m pip install --no-cache-dir "numpy<2" scipy pyyaml matplotlib tqdm joblib
$E/bin/python -c "import torch,torchvision,numpy,scipy,yaml,matplotlib;from torch.func import functional_call;from torch.utils.flop_counter import FlopCounterMode;print('ENV_OK',torch.__version__,torchvision.__version__,numpy.__version__,torch.version.cuda,torch.cuda.get_arch_list())"
