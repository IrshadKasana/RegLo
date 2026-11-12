#!/bin/bash 
#SBATCH --job-name=r-mlp_mnist      # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=1      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1 
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=testp
#SBATCH --output=console/outs/26test_mlp_mnist_reg.out
#SBATCH --error=console/errors/26test_mlp_mnist_reg.err

cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/suraj_env1
source ./bin/activate 
cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/L2O-CL/DMO
python3 optimizee_train.py  --optimizee_epochs=1000 --n_training=10 --dataset=mnist --optimizee_model=mlp --patience=100  --eps=0.1 --if_avreg
