#!/bin/bash 
#SBATCH --job-name=AVhesDM     # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=1      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:8 
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=dgxnp
#SBATCH --output=console/outs/AVtesting_hessian.out
#SBATCH --error=console/errors/AVtesting_hessian.err

cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/suraj_env1
source ./bin/activate 
cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/DMO/w_o_curriculum
python3 testing_hessian.py
