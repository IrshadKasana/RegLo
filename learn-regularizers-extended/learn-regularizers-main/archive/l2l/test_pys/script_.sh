#!/bin/bash 
#SBATCH --job-name=tsSdmsgd      # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=1      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1 
#SBATCH --time=23:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=testp
#SBATCH --output=console/outs/test_better_min_SDM.out
#SBATCH --error=console/errors/test_better_min_SDM.err

cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/suraj_env1
source ./bin/activate 
cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/test_pys
python3 test_better_min_DM_SDM.py
