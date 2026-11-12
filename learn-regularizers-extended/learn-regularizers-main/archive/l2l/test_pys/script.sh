#!/bin/bash 
#SBATCH --job-name=AVhesSDM      # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=1      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --time=36:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=dgxnp
#SBATCH --output=console/outs/AVtesting_hessian_SDMO.out
#SBATCH --error=console/errors/AVtesting_hessian_SDMO.err

cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/suraj_env1
source ./bin/activate 
cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/test_pys
python3 testing_hessian_SDMO.py
