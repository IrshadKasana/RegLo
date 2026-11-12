#!/bin/bash 
#SBATCH --job-name=SmoDMsgd      # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=2      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1 
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=testp
#SBATCH --output=console/outs/1test_smoothDM_better_min.out
#SBATCH --error=console/errors/1test_smoothDM_better_min.err

cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/suraj_env1
source ./bin/activate 
cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/smooth_regularize/DMLO_Smooth
python3 test_smoothDM_better_min.py
